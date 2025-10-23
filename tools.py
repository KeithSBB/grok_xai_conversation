"""Grok-Homeassistant tool module. This file covers all of the tools that Grok
can call on the Homeassistant platform integration."""

import asyncio
import fnmatch
import inspect
import json
import logging
import re
import typing
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    get_args,
    get_origin,
)

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import State  # Import State for type checking
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_call_later
from xai_sdk.chat import tool as xai_tool

# removed: from xai_sdk.chat import tool_result (This was causing Bug #2)
from xai_sdk.proto import chat_pb2  # For Tool proto only

from .const import MAX_SEARCH_RESULTS

_LOGGER = logging.getLogger(__name__)

# Note: This is a prompt instruction, not a tool definition.
# It's better placed in const.py or conversation.py as part of the system prompt.
TOOL_INSTRUCTIONS = ["'call_ha_service' with parameters: domain, service, target, data"]

JSON_TYPE_MAPPING = {
    "null": "null",
    "boolean": "boolean",
    "number": "number",
    "integer": "integer",
    "string": "string",
    "array": "array",
    "object": "object",
}

TYPE_MAPPING = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    tuple: "array",
    dict: "object",
    Any: "object",  # Fallback for Any
}


def parse_docstring_descriptions(doc: str) -> Dict[str, str]:
    """Parse param descriptions from docstring (PEP 257 style).

    Args:
        doc: The docstring to parse.

    Returns:
        Dict of param names to descriptions.
    """
    descriptions = {}
    if doc:
        lines = doc.splitlines()
        current_param = None
        for line in lines:
            if line.strip().startswith(":param"):
                parts = line.split(":", 2)
                if len(parts) > 2:
                    param_name = parts[1].strip().split()[0]
                    desc = parts[2].strip()
                    descriptions[param_name] = desc
                    current_param = param_name
            elif current_param and line.strip():
                descriptions[current_param] += " " + line.strip()
    return descriptions


def type_to_schema(
    ann: Type[Any],
    descriptions: Dict[str, str] | None = None,
    param_name: str | None = None,
) -> Dict[str, Any]:
    """Recursively convert type annotation to JSON Schema dict.

    Args:
        ann: Type annotation to convert.
        descriptions: Param descriptions from docstring.
        param_name: Name of the parameter for description lookup.

    Returns:
        JSON Schema dict.
    """
    origin = get_origin(ann)
    args = get_args(ann)
    schema: Dict[str, Any] = {}
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(args) > len(non_none_args):  # Has None, make nullable
            if len(non_none_args) == 1:
                sub_schema = type_to_schema(non_none_args[0], descriptions, param_name)
                schema["anyOf"] = [sub_schema, {"type": "null"}]
            else:
                schemas = [
                    type_to_schema(a, descriptions, param_name) for a in non_none_args
                ]
                schemas.append({"type": "null"})
                schema["anyOf"] = schemas
        else:
            # Union without null
            schema["anyOf"] = [
                type_to_schema(a, descriptions, param_name) for a in args
            ]
    elif origin in (list, Sequence, tuple):
        item_type = args[0] if args else Any
        schema = {
            "type": "array",
            "items": type_to_schema(item_type, descriptions, param_name),
        }
    elif origin in (dict, Dict):
        key_type, value_type = args if args else (str, Any)
        if get_origin(key_type) is not None or key_type is not str:
            raise ValueError("Dict keys must be str")
        schema = {
            "type": "object",
            "additionalProperties": type_to_schema(
                value_type, descriptions, param_name
            ),
        }
    else:
        type_str = TYPE_MAPPING.get(ann, "object")
        schema = {"type": JSON_TYPE_MAPPING[type_str]}
    if descriptions and param_name:
        schema["description"] = descriptions.get(param_name, "")
    return schema


def make_json_serializable(
    obj: Any,
) -> Union[Dict[str, Any], List[Any], str, int, float, bool, None]:
    """Recursively convert an object to be JSON serializable."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, State):  # <-- Explicitly handle State objects
        _LOGGER.warning(
            "make_json_serializable called on raw State object. "
            "This should be avoided. Use .as_dict() first."
        )
        return make_json_serializable(obj.as_dict())
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback for other non-serializable types
        try:
            return str(obj)
        except Exception:
            return "Unserializable Object"


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class GrokHaTool:
    """Class for managing HA tools and registries.

    An instance of this class is created once on integration setup.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the GrokHaTool instance."""
        _LOGGER.debug("Initializing GrokHaTool")
        self.hass = hass
        self.ent_reg = er.async_get(hass)
        self.area_reg = ar.async_get(hass)
        self.floor_reg = fr.async_get(hass)
        self.label_reg = lr.async_get(hass)
        self.dev_reg = dr.async_get(hass)

        self._exposed_entities: Optional[List[Dict[str, Any]]] = None
        self._debounce_call: Optional[CALLBACK_TYPE] = None

        # Store listeners as instance attributes
        self._registry_remove = self.hass.bus.async_listen(
            "entity_registry_updated", self._async_invalidate_cache
        )
        self._state_remove = self.hass.bus.async_listen(
            "state_changed", self._async_debounced_invalidate_cache
        )

    async def _async_invalidate_cache(self, event: Event) -> None:
        """Invalidate exposed entities cache on registry update."""
        self._exposed_entities = None
        _LOGGER.debug(
            "Exposed entities cache invalidated by event: %s", event.event_type
        )

    async def _async_debounced_invalidate_cache(self, event: Event) -> None:
        """Debounced cache invalidation on state changes."""
        if self._debounce_call:
            self._debounce_call()

        async def _delayed_invalidate(_now: datetime) -> None:
            await self._async_invalidate_cache(event)

        self._debounce_call = async_call_later(self.hass, 30, _delayed_invalidate)

    def async_unload_listeners(self) -> None:
        """Unload event listeners and reset cache."""
        if self._registry_remove:
            self._registry_remove()
            self._registry_remove = None
        if self._state_remove:
            self._state_remove()
            self._state_remove = None
        if self._debounce_call:
            self._debounce_call()
            self._debounce_call = None

        self._exposed_entities = None
        self.get_tool_schemas.cache_clear()  # Clear LRU cache
        _LOGGER.debug("Unloaded GrokHaTool listeners and reset cache")

    @classmethod
    @lru_cache(maxsize=1)
    def get_tool_schemas(cls) -> Sequence[chat_pb2.Tool]:
        """Generate xAI tool schemas from class methods."""
        schemas = []
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            # *** FIX #3: Added 'async_get_ha_context_summary' to exclusion list ***
            if name.startswith("_") or name in (
                "get_tool_schemas",
                "__init__",
                "async_unload_listeners",
                "get_area_name",
                "async_get_ha_context_summary",  # This is not for the LLM
            ):
                continue
            if not asyncio.iscoroutinefunction(method):
                continue
            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or ""
            if not doc:
                _LOGGER.warning("Tool method %s has no docstring, skipping.", name)
                continue

            description = doc.splitlines()[0].strip()
            param_descs = parse_docstring_descriptions(doc)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                ann = param.annotation
                if ann is inspect.Parameter.empty:
                    _LOGGER.warning(
                        "Tool %s parameter %s has no type annotation, skipping.",
                        name,
                        param_name,
                    )
                    continue
                schema = type_to_schema(ann, param_descs, param_name)
                properties[param_name] = schema
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            parameters_dict = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
            try:
                schemas.append(xai_tool(name, description, parameters_dict))
                _LOGGER.debug(f"Added tool schema for {name}")
            except Exception as e:
                _LOGGER.error(f"Failed to create tool schema for {name}: {e}")
        return schemas

    async def async_get_ha_context_summary(self) -> str:
        """Generate a summary of the Home Assistant configuration for the LLM."""
        try:
            # Use the class's existing registry instances
            floors = {f.floor_id: f.name for f in self.floor_reg.async_list_floors()}
            areas = self.area_reg.async_list_areas()

            # Get all entity domains
            domains = sorted(
                list(set(entry.domain for entry in self.ent_reg.entities.values()))
            )

            summary_lines = ["Home Assistant Configuration Summary:"]

            # 1. Floors and Areas
            if floors or areas:
                summary_lines.append("\nFloors and Areas:")
                floor_areas: dict[int, list[Any]] = {fid: [] for fid in floors}
                unassigned_areas = []

                for area in areas:
                    if area.floor_id in floor_areas:
                        floor_areas[area.floor_id].append(area.name)
                    else:
                        unassigned_areas.append(area.name)

                for floor_id, floor_name in floors.items():
                    area_list = ", ".join(sorted(floor_areas[floor_id])) or "No areas"
                    summary_lines.append(f"- Floor '{floor_name}': [{area_list}]")

                if unassigned_areas:
                    summary_lines.append(
                        "- Areas with no floor: "
                        f"[{', '.join(sorted(unassigned_areas))}]"
                    )
            else:
                summary_lines.append("\nNo Floors or Areas are configured.")

            # 2. Available Domains
            if domains:
                summary_lines.append("\nAvailable Device Domains:")
                summary_lines.append(f"[{', '.join(domains)}]")
            else:
                summary_lines.append("\nNo device domains found.")

            summary = "\n".join(summary_lines)
            _LOGGER.debug("Generated HA context summary: %s", summary)
            return summary
        except Exception as e:
            _LOGGER.error("Failed to generate HA context summary: %s", e)
            return "Could not retrieve HA context."

    def get_area_name(self, entry: er.RegistryEntry) -> str:
        """Get area name from an entity registry entry."""
        if not entry or not entry.area_id:
            return ""
        area_entry = self.area_reg.async_get_area(entry.area_id)
        if not area_entry:
            return ""
        return area_entry.name or ""

    async def call_ha_service(
        self,
        domain: str,
        service: str,
        target: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a Home Assistant service.

        :param domain: The domain of the service (e.g., 'light').
        :param service: The service to call (e.g., 'turn_on').
        :param target: Optional target entities (e.g.,
                       {'entity_id': 'light.living_room'}).
        :param data: Optional service data (e.g., {'brightness_pct': 50}).
        """
        try:
            await self.hass.services.async_call(domain, service, target, data)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_exposed_entities(self) -> List[Dict[str, Any]]:
        """Get list of exposed entities with details like name, aliases, entity_id."""
        if self._exposed_entities is None:
            try:
                _LOGGER.debug("Fetching fresh exposed entities (cache miss)")
                apis = llm.async_get_apis(self.hass)

                # Use DOMAIN from const.py for robust assistant_id matching
                assistant_id = typing.cast(
                    str,
                    next(
                        (api.id for api in apis if api.id == "grok_xai_conversation"),
                        "conversation",  # Fallback
                    ),
                )

                device_id_to_name = {
                    entry.id: entry.name or ""
                    for entry in self.dev_reg.devices.values()
                }  # Coalesce None to ""
                exposed = []
                for ent in self.ent_reg.entities.values():
                    if async_should_expose(self.hass, assistant_id, ent.entity_id):
                        exposed.append(
                            {
                                "entity_id": ent.entity_id,
                                "name": ent.name or ent.original_name or ent.entity_id,
                                "aliases": ent.aliases,
                                "domain": ent.domain,
                                "area_name": self.get_area_name(ent) or "",
                                "device_name": (
                                    device_id_to_name.get(ent.device_id, "")
                                    if ent.device_id
                                    else ""
                                ),
                            }
                        )
                self._exposed_entities = exposed
            except Exception as e:
                _LOGGER.error(f"Failed to fetch exposed entities: {e}")
                raise HomeAssistantError(str(e))
        else:
            _LOGGER.debug("Using cached exposed entities")
        return self._exposed_entities

    async def find_entities_advanced(
        self,
        name: Optional[str] = None,
        simple_mode: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = MAX_SEARCH_RESULTS,
        offset: int = 0,
        entity_globs: Optional[Union[str, List[str]]] = None,
        domains: Optional[Union[str, List[str]]] = None,
        area_globs: Optional[Union[str, List[str]]] = None,
        floor_globs: Optional[Union[str, List[str]]] = None,
        tag_globs: Optional[Union[str, List[str]]] = None,
        device_globs: Optional[Union[str, List[str]]] = None,
    ) -> Union[List[Dict[str, Any]], str]:
        """Advanced entity search with glob patterns and filters.
           Use simple_mode for name-based fuzzy search.

        Note: Always fetches fresh states for accuracy;
              static entity info may be cached.

        :param name: The name to search for in simple mode.
        :param simple_mode: If true, use fuzzy name search; else advanced globs.
        :param filters: Additional filters (e.g.,
                       domain, attributes like {'device_class': 'temperature'}).
        :param limit: Max results (default 10).
        :param offset: Result offset.
        :param entity_globs: Glob patterns for entity IDs.
        :param domains: Domains to filter (e.g., 'sensor' or ['light', 'switch']).
        :param area_globs: Glob patterns for areas.
        :param floor_globs: Glob patterns for floors.
        :param tag_globs: Glob patterns for tags.
        :param device_globs: Glob patterns for devices.
        :example filters: JSON object, e.g., {'domain': 'light',
                         'attributes': {'device_class': 'temperature'}}.
        """
        _LOGGER.debug(
            "find_entities_advanced called with: name=%s, simple_mode=%s, "
            "filters=%s, entity_globs=%s, domains=%s, area_globs=%s, "
            "floor_globs=%s, tag_globs=%s, device_globs=%s",
            name,
            simple_mode,
            filters,
            entity_globs,
            domains,
            area_globs,
            floor_globs,
            tag_globs,
            device_globs,
        )
        try:
            if isinstance(filters, str):
                filters = json.loads(filters)
        except json.JSONDecodeError:
            _LOGGER.warning(
                "Invalid filters JSON in find_entities_advanced; using empty dict"
            )
            filters = {}
        except Exception as e:
            _LOGGER.error(f"Filter processing error in find_entities_advanced: {e}")
            return f"Invalid filters provided: {e}"
        filters = filters or {}  # Default to empty dict to avoid NoneType errors

        # *** MODIFIED ensure_list function ***
        def ensure_list(value: Optional[Union[str, List[str]]]) -> List[str]:
            """Ensure value is a list, parsing JSON strings if needed."""
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                stripped_value = value.strip()
                # Check if it looks like a JSON list
                if stripped_value.startswith("[") and stripped_value.endswith("]"):
                    try:
                        parsed = json.loads(stripped_value)
                        if isinstance(parsed, list):
                            _LOGGER.debug("Parsed JSON string to list: %s", parsed)
                            return parsed
                    except json.JSONDecodeError:
                        # It wasn't valid JSON, fall back to treating it
                        # as a single string
                        _LOGGER.debug(
                            "String looked like list but failed to parse: %s", value
                        )
                        pass
                # Not a JSON list or failed to parse, treat as a single-item list
                return [value]
            # Fallback for other non-list, non-str types
            return [str(value)]

        def glob_match(string: str, pattern: str) -> bool:
            """Case-insensitive glob matching."""
            try:
                # fnmatch.translate converts glob to regex
                regex_pattern = fnmatch.translate(pattern)
                regex = re.compile(regex_pattern, re.IGNORECASE)
                return regex.match(string) is not None
            except re.error as e:
                _LOGGER.warning(
                    "Invalid glob pattern '%s' (regex error: %s)", pattern, e
                )
                return False

        # --- Simple Mode Logic ---
        if simple_mode:
            if not name:
                _LOGGER.warning("Simple mode search called without 'name'")
                return [{"message": "Name required for simple search"}]
            words = name.lower().split()
            domain_filter = filters.get("domain")
            if not domain_filter and domains:
                # Use first domain if provided
                domain_list = ensure_list(domains)
                if domain_list:
                    domain_filter = domain_list[0]

            attr_filters = filters.get("attributes", {})
            matching_entity_ids = set()
            try:
                exposed = await self.get_exposed_entities()
            except HomeAssistantError as e:
                return f"Error getting entities: {e}"

            area_globs_list = ensure_list(area_globs)
            device_globs_list = ensure_list(device_globs)

            for ent in exposed:
                if domain_filter and ent["domain"] != domain_filter:
                    continue

                try:
                    # Apply area/device globs even in simple mode
                    if area_globs_list and not any(
                        glob_match(ent["area_name"], g) for g in area_globs_list
                    ):
                        continue
                    if device_globs_list and not any(
                        glob_match(ent["device_name"], g) for g in device_globs_list
                    ):
                        continue
                except TypeError as te:
                    _LOGGER.error(f"TypeError in simple_mode glob matching: {te}")
                    continue

                # Score based on name, entity_id, area, and aliases
                score = sum(
                    1
                    for word in words
                    if word in ent["name"].lower()
                    or word in ent["entity_id"].lower()
                    or word in ent["area_name"].lower()
                    or any(word in alias.lower() for alias in ent.get("aliases", []))
                )
                if score > 0:
                    matching_entity_ids.add(ent["entity_id"])

            # Fetch fresh states for matching IDs
            matching = []
            for entity_id in sorted(list(matching_entity_ids)):
                fresh_state = self.hass.states.get(entity_id)
                if not fresh_state:
                    continue

                # Apply attribute filters on fresh state
                if attr_filters:
                    attrs = fresh_state.attributes
                    if not all(attrs.get(k) == v for k, v in attr_filters.items()):
                        _LOGGER.debug(
                            "Skipping %s, attr_filter mismatch. Wanted: %s, Got: %s",
                            entity_id,
                            attr_filters,
                            attrs,
                        )
                        continue

                ent_info = next(
                    (e for e in exposed if e["entity_id"] == entity_id), None
                )
                if ent_info:
                    matching.append(
                        {
                            "entity_id": entity_id,
                            "name": ent_info["name"],
                            "area_name": ent_info["area_name"],
                            "state": fresh_state.state,
                            "unit": fresh_state.attributes.get("unit_of_measurement"),
                            # Add all attributes for better context
                            "attributes": make_json_serializable(
                                fresh_state.attributes
                            ),
                        }
                    )

            _LOGGER.debug(f"Simple mode matched {len(matching)} entities")
            return (
                matching[offset : offset + limit]
                if matching
                else [{"message": "No entities found matching criteria"}]
            )

        # --- Advanced Mode Logic ---
        entity_globs_list = ensure_list(entity_globs)
        domains_list = ensure_list(domains)
        area_globs_list = ensure_list(area_globs)
        floor_globs_list = ensure_list(floor_globs)
        tag_globs_list = ensure_list(tag_globs)
        device_globs_list = ensure_list(device_globs)

        # Merge filters.domain if no domains
        if not domains_list and filters.get("domain"):
            domains_list = ensure_list(filters.get("domain"))

        # Pre-computation maps for faster lookups
        floor_id_to_name = {
            entry.floor_id: entry.name or ""
            for entry in self.floor_reg.async_list_floors()
        }
        area_id_to_name = {
            entry.id: entry.name or "" for entry in self.area_reg.async_list_areas()
        }
        label_id_to_name = {
            entry.label_id: entry.name or ""
            for entry in self.label_reg.async_list_labels()
        }
        device_id_to_name = {
            entry.id: entry.name or "" for entry in self.dev_reg.devices.values()
        }

        all_entity_entries = self.ent_reg.entities.values()
        matching_entity_ids = set()
        attr_filters = filters.get("attributes", {})

        for entry in all_entity_entries:
            # Domain filter
            if domains_list and entry.domain not in domains_list:
                continue

            # Attribute filters (requires state)
            if attr_filters:
                state = self.hass.states.get(entry.entity_id)
                if not state or not all(
                    state.attributes.get(k) == v for k, v in attr_filters.items()
                ):
                    continue

            # Entity ID glob
            if entity_globs_list and not any(
                glob_match(entry.entity_id, g) for g in entity_globs_list
            ):
                continue

            # Device glob
            match_device = not device_globs_list
            if device_globs_list and entry.device_id:
                device_name = device_id_to_name.get(entry.device_id, "")
                if device_name and any(
                    glob_match(device_name, g) for g in device_globs_list
                ):
                    match_device = True
            if not match_device:
                continue

            # Area glob
            area_id = entry.area_id
            device = None
            if not area_id and entry.device_id:
                device = self.dev_reg.async_get(entry.device_id)
                if device:
                    area_id = device.area_id

            area_name = area_id_to_name.get(area_id, "") if area_id else ""
            match_area = not area_globs_list
            if area_globs_list:
                if area_name and any(glob_match(area_name, g) for g in area_globs_list):
                    _LOGGER.debug(
                        "Entity %s matched area '%s' via glob",
                        entry.entity_id,
                        area_name,
                    )
                    match_area = True
                else:
                    _LOGGER.debug(
                        "Entity %s skipped, area '%s' did not match globs: %s",
                        entry.entity_id,
                        area_name,
                        area_globs_list,
                    )
            if not match_area:
                continue

            # Floor glob
            match_floor = not floor_globs_list
            if floor_globs_list:
                if area_id:
                    area_entry = self.area_reg.async_get_area(area_id)
                    if area_entry and area_entry.floor_id:
                        floor_name = floor_id_to_name.get(area_entry.floor_id, "")
                        if floor_name and any(
                            glob_match(floor_name, g) for g in floor_globs_list
                        ):
                            match_floor = True
            if not match_floor:
                continue

            # Tag/label glob
            match_tag = not tag_globs_list
            if tag_globs_list and entry.labels:
                if any(
                    glob_match(label_id_to_name.get(label_id, ""), g)
                    for label_id in entry.labels
                    for g in tag_globs_list
                ):
                    match_tag = True
            if not match_tag:
                continue

            # If all checks passed
            matching_entity_ids.add(entry.entity_id)

        _LOGGER.debug(f"Advanced mode matched {len(matching_entity_ids)} entities")

        # Retrieve states only for matches
        filtered_states = [
            self.hass.states.get(entity_id)
            for entity_id in sorted(list(matching_entity_ids))
            if self.hass.states.get(entity_id)
        ]

        filtered_states = filtered_states[offset : offset + limit]

        if not filtered_states:
            return "No matches were found"

        # *** FIX #1: Convert State objects to serializable dicts ***
        results = [
            typing.cast(Dict[str, Any], make_json_serializable(state.as_dict()))
            for state in filtered_states
        ]

        _LOGGER.debug("Returning %d serialized entities", len(results))
        return results

    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Fetch the full state of an entity, including attributes and unit.

        :param entity_id: The entity ID to fetch.
        :example entity_id: 'sensor.bedroom_temperature'.
        """
        state = self.hass.states.get(entity_id)
        if not state:
            return {"message": "Entity not found"}
        return typing.cast(Dict[str, Any], make_json_serializable(state.as_dict()))

    # Example new tool: set_thermostat_temp (as suggested enhancement)
    async def set_thermostat_temp(
        self, entity_id: str, temperature: float
    ) -> Dict[str, Any]:
        """Set the temperature of a thermostat entity.

        :param entity_id: The entity ID of the thermostat.
        :param temperature: The temperature to set.
        :example entity_id: 'climate.living_room_thermostat'.
        :example temperature: 72.5.
        """
        try:
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                target={"entity_id": entity_id},
                data={"temperature": temperature},
            )

            # *** FIX #2: Removed tool_result() wrapper. Return a simple dict. ***
            return {
                "status": "success",
                "message": f"Set {entity_id} to {temperature} degrees",
            }

        except Exception as e:
            _LOGGER.error(f"Failed to set thermostat: {str(e)}")
            return {"status": "error", "message": str(e)}
