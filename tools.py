"""Grok-Homeassistant tool module.

This module defines tools that Grok can call within the Home Assistant integration.
Tools follow xAI SDK schema (chat_pb2.Tool) and return JSON-serializable dicts.
"""

import fnmatch
import inspect
import json
import logging
import typing
from dataclasses import dataclass, field
from datetime import datetime
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
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_call_later
from xai_sdk.proto import chat_pb2  # Only for Tool proto; no 'tool' import

from .const import MAX_SEARCH_RESULTS

_LOGGER = logging.getLogger(__name__)

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
    Any: "object",
}


def parse_docstring_descriptions(doc: str) -> Dict[str, str]:
    """Parse parameter descriptions from docstring (PEP 257 style).

    Args:
        doc: The docstring to parse.

    Returns:
        Dictionary of parameter names to descriptions.
    """
    descriptions = {}
    if doc:
        lines = doc.splitlines()
        current_param = None
        for line in lines:
            line = line.strip()
            if line.startswith(":param"):
                parts = line.split(":", 2)
                if len(parts) > 2:
                    param_name = parts[1].strip().split()[0]
                    desc = parts[2].strip()
                    descriptions[param_name] = desc
                    current_param = param_name
            elif current_param and line:
                descriptions[current_param] += " " + line
    return descriptions


def type_to_schema(
    ann: Type[Any],
    descriptions: Optional[Dict[str, str]] = None,
    param_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Recursively convert type annotation to JSON Schema dict.

    Args:
        ann: Type annotation to convert.
        descriptions: Parameter descriptions from docstring.
        param_name: Name of the parameter for description lookup.

    Returns:
        JSON Schema dictionary.
    """
    origin = get_origin(ann)
    args = get_args(ann)
    schema: Dict[str, Any] = {}
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(args) > len(non_none_args):
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
    """Recursively convert an object to be JSON serializable.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable object.
    """
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
    elif isinstance(obj, State):
        return make_json_serializable(obj.as_dict())
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        try:
            return str(obj)
        except Exception as e:
            _LOGGER.debug("Failed to serialize object: %s", e)
            return None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class GrokHaTool:
    """Manages Home Assistant tools, registries, and context summaries.

    This class follows single-responsibility: Handles tool schemas,
    executions, and HA queries.
    Instance created once on setup for sharing across conversations.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the GrokHaTool instance.

        Args:
            hass: Home Assistant core instance.
        """
        _LOGGER.debug("Initializing GrokHaTool")
        self.hass = hass
        self.ent_reg = er.async_get(hass)
        self.area_reg = ar.async_get(hass)
        self.floor_reg = fr.async_get(hass)
        self.label_reg = lr.async_get(hass)
        self.dev_reg = dr.async_get(hass)
        self._exposed_entities: Optional[List[Dict[str, Any]]] = None
        self._debounce_call: Optional[CALLBACK_TYPE] = None
        # Cache for optimization
        self._ha_context_summary_cache: Optional[str] = None

        # Setup listeners
        self._registry_remove = self.hass.bus.async_listen(
            "entity_registry_updated", self._async_invalidate_cache
        )
        self._state_remove = self.hass.bus.async_listen(
            "state_changed", self._async_debounced_invalidate_cache
        )

    async def _async_invalidate_cache(self, event: Event) -> None:
        """Invalidate caches on registry/state updates.

        Args:
            event: HA event triggering invalidation.
        """
        self._exposed_entities = None
        self._ha_context_summary_cache = None
        _LOGGER.debug("Caches invalidated by event: %s", event.event_type)

    async def _async_debounced_invalidate_cache(self, event: Event) -> None:
        """Debounced cache invalidation for state changes.

        Args:
            event: HA event.
        """
        if self._debounce_call:
            self._debounce_call()

        async def _delayed_invalidate(_now: datetime) -> None:
            await self._async_invalidate_cache(event)

        self._debounce_call = async_call_later(self.hass, 30, _delayed_invalidate)

    def async_unload_listeners(self) -> None:
        """Unload event listeners and clear caches."""
        if self._registry_remove:
            self._registry_remove()
            self._registry_remove = None
        if self._state_remove:
            self._state_remove()
            self._state_remove = None
        if self._debounce_call:
            self._debounce_call()
        self._exposed_entities = None
        self._ha_context_summary_cache = None
        _LOGGER.debug("GrokHaTool listeners unloaded")

    def get_tool_schemas(self) -> Sequence[chat_pb2.Tool]:
        """Generate xAI-compatible tool schemas from public async methods.

        Auto-detects tools as non-private async methods;
        generates schemas from annotations/docstrings.

        Returns:
            List of chat_pb2.Tool protos.
        """
        tools = []
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if name.startswith("_") or not inspect.iscoroutinefunction(method):
                continue  # Skip private or sync methods
            doc = inspect.getdoc(method) or ""
            descriptions = parse_docstring_descriptions(doc)
            params = inspect.signature(method).parameters
            param_schemas = {
                p.name: type_to_schema(p.annotation, descriptions, p.name)
                for p in params.values()
                if p.name != "self"
            }
            schema = {
                "type": "object",
                "properties": param_schemas,
                "required": [
                    p.name
                    for p in params.values()
                    if p.default == inspect.Parameter.empty and p.name != "self"
                ],
            }
            tool_desc = doc.splitlines()[0] if doc else f"Tool: {name}"
            tools.append(
                chat_pb2.Tool(
                    function=chat_pb2.Function(
                        name=name,
                        description=tool_desc,
                        parameters=json.dumps(schema),
                    )
                )
            )
            _LOGGER.debug("Generated tool schema for: %s", name)
        if not tools:
            _LOGGER.warning("No tool schemas generated; check for public async methods")
        return tools

    async def async_get_ha_context_summary(self) -> str:
        """Generate a summary of HA configuration for Grok context.

        Includes floors, areas, domains, etc. Cached for performance.

        Returns:
            Formatted string summary.
        """
        if self._ha_context_summary_cache:
            _LOGGER.debug("Returning cached HA context summary")
            return self._ha_context_summary_cache

        try:
            floors = self.floor_reg.async_list_floors()
            areas = self.area_reg.async_list_areas()
            entities = list(
                self.ent_reg.entities.values()
            )  # Optimized: values() for efficiency
            domains = {entry.domain for entry in entities if entry.domain}

            summary = "Home Assistant Context:\n"
            summary += (
                f"- Floors: {', '.join(f.name for f in floors if f.name) or 'None'}\n"
            )
            for floor in floors:
                floor_areas = [a for a in areas if a.floor_id == floor.floor_id]
                summary += (
                    f"  - {floor.name or 'Unnamed'}: "
                    f"Areas - {', '.join(a.name for a in floor_areas
                                         if a.name) or 'None'}\n"
                )
            summary += f"- Domains: {', '.join(sorted(domains)) or 'None'}\n"
            summary += "- Use tools like find_entities_advanced for detailed queries.\n"

            self._ha_context_summary_cache = summary
            _LOGGER.info("Generated HA context summary")
            return summary
        except Exception as e:
            _LOGGER.error("Failed to generate HA context summary: %s", e)
            raise HomeAssistantError(f"Context summary error: {str(e)}")

    async def call_ha_service(
        self,
        domain: str,
        service: str,
        target: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a Home Assistant service.

        Args:
            domain: Service domain (e.g., 'light').
            service: Service name (e.g., 'turn_on').
            target: Target entity/area/device
                    (JSON dict, e.g., {'entity_id': 'light.bedroom'}).
            data: Service data (JSON dict, e.g., {'brightness_pct': 50}).

        Returns:
            Dict with status and message.
        """
        try:
            _LOGGER.debug(
                "Calling service: %s.%s with target=%s, data=%s",
                domain,
                service,
                target,
                data,
            )
            await self.hass.services.async_call(
                domain, service, target=target, service_data=data
            )
            return {"status": "success", "message": f"Called {domain}.{service}"}
        except Exception as e:
            _LOGGER.error("Service call failed: %s", e)
            return {"status": "error", "message": str(e)}

    async def async_search_entities(
        self,
        name: Optional[str] = None,
        entity_globs: Optional[Union[str, List[str]]] = None,
        domains: Optional[Union[str, List[str]]] = None,
        area_globs: Optional[Union[str, List[str]]] = None,
        floor_globs: Optional[Union[str, List[str]]] = None,
        tag_globs: Optional[Union[str, List[str]]] = None,
        device_globs: Optional[Union[str, List[str]]] = None,
        filters: Optional[Union[str, Dict[str, Any]]] = None,
        limit: int = MAX_SEARCH_RESULTS,
        offset: int = 0,
        simple_mode: bool = False,
    ) -> Union[List[Dict[str, Any]], str]:
        """Search for exposed entities with filters.

        Args:
            name: Simple name search (simple_mode only).
            entity_globs: Glob patterns for entity IDs.
            domains: Domain(s) to match (e.g., 'light').
            area_globs: Glob patterns for areas.
            floor_globs: Glob patterns for floors.
            tag_globs: Glob patterns for labels.
            device_globs: Glob patterns for devices.
            filters: Dict with 'domain' or 'attributes'.
            limit: Max results (default 10).
            offset: Result offset.
            simple_mode: Basic name-based search.

        Returns:
            List of matching entity dicts or error message.
        """
        try:
            exposed = await self.get_exposed_entities()
        except HomeAssistantError as e:
            _LOGGER.error("Error fetching exposed entities: %s", e)
            return f"Error: {str(e)}"

        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
                _LOGGER.debug("Successfully parsed filters from string to dict")
            except json.JSONDecodeError as e:
                _LOGGER.warning("Invalid filters JSON: %s", e)
                return f"Invalid filters: {str(e)}"

        # Type narrowing: After parsing, filters is either None or dict
        if filters is not None and not isinstance(filters, dict):
            _LOGGER.error("Unexpected parsed filters type: %s", type(filters))
            return "Error: Invalid filters type"

        def ensure_list(value: Optional[Union[str, List[str]]]) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value.strip()]
            return value

        domains_list = ensure_list(domains) or ensure_list(
            filters.get("domain") if filters else None
        )
        entity_globs_list = ensure_list(entity_globs)
        area_globs_list = ensure_list(area_globs)
        floor_globs_list = ensure_list(floor_globs)
        tag_globs_list = ensure_list(tag_globs)
        device_globs_list = ensure_list(device_globs)
        attr_filters = filters.get("attributes", {}) if filters else {}

        # Cached maps
        floor_id_to_name = {
            entry.floor_id: entry.name or ""
            for entry in self.floor_reg.async_list_floors()
        }

        label_id_to_name = {
            entry.label_id: entry.name or ""
            for entry in self.label_reg.async_list_labels()
        }

        matching = []
        for exp in exposed:
            entity_id = exp["entity_id"]
            entry = self.ent_reg.async_get(entity_id)
            if not entry:
                continue

            if domains_list and entry.domain not in domains_list:
                continue

            state = self.hass.states.get(entity_id)
            if not state:
                continue
            if attr_filters and not all(
                state.attributes.get(k) == v for k, v in attr_filters.items()
            ):
                continue

            if simple_mode:
                if not name:
                    return [{"message": "Name required for simple search"}]
                words = name.lower().split()
                score = sum(
                    1
                    for word in words
                    if word in exp["name"].lower()
                    or word in entity_id.lower()
                    or word in exp["area_name"].lower()
                )
                if score == 0:
                    continue
            else:
                if entity_globs_list and not any(
                    fnmatch.fnmatch(entity_id, g) for g in entity_globs_list
                ):
                    continue
                if device_globs_list and not any(
                    fnmatch.fnmatch(exp["device_name"], g) for g in device_globs_list
                ):
                    continue
                if area_globs_list and not any(
                    fnmatch.fnmatch(exp["area_name"], g) for g in area_globs_list
                ):
                    continue
                area_id = (
                    entry.area_id
                    or (
                        self.dev_reg.async_get(entry.device_id).area_id
                        if entry.device_id
                        else None
                    )
                    if entry.device_id
                    else None
                )
                floor_name = ""
                if area_id:
                    area = self.area_reg.async_get_area(area_id)
                    if area and area.floor_id:
                        floor_name = floor_id_to_name.get(area.floor_id, "")
                if floor_globs_list and not any(
                    fnmatch.fnmatch(floor_name, g) for g in floor_globs_list
                ):
                    continue
                if (
                    tag_globs_list
                    and entry.labels
                    and not any(
                        fnmatch.fnmatch(label_id_to_name.get(lbl, ""), g)
                        for lbl in entry.labels
                        for g in tag_globs_list
                    )
                ):
                    continue

            matching.append(
                {
                    "entity_id": entity_id,
                    "name": exp["name"],
                    "area_name": exp["area_name"],
                    "device_name": exp["device_name"],
                    "state": state.state,
                    "unit": state.attributes.get("unit_of_measurement"),
                    "attributes": make_json_serializable(state.attributes),
                }
            )

        if not matching:
            return "No matching entities found"
        return matching[offset : offset + limit]

    async def get_exposed_entities(self) -> List[Dict[str, Any]]:
        """Fetch list of exposed entities with details (cached).

        Returns:
            List of entity dicts.
        """
        if self._exposed_entities is not None:
            _LOGGER.debug("Returning cached exposed entities")
            return self._exposed_entities

        try:
            # Use DOMAIN from const.py for robust assistant_id matching
            apis = llm.async_get_apis(self.hass)
            assistant_id = typing.cast(
                str,
                next(
                    (api.id for api in apis if api.id == "grok_xai_conversation"),
                    "conversation",  # Fallback
                ),
            )

            exposed = []
            for (
                entry
            ) in self.ent_reg.entities.values():  # Optimized: values() for all entities
                if async_should_expose(self.hass, assistant_id, entry.entity_id):
                    area_name = ""
                    device_name = ""
                    if entry.area_id:
                        area = self.area_reg.async_get_area(entry.area_id)
                        area_name = area.name if area else ""
                    if entry.device_id:
                        device = self.dev_reg.async_get(entry.device_id)
                        device_name = device.name if device else ""
                    exposed.append(
                        {
                            "entity_id": entry.entity_id,
                            "name": entry.name or entry.original_name,
                            "area_name": area_name,
                            "device_name": device_name,
                            "aliases": entry.aliases,
                        }
                    )
            self._exposed_entities = exposed
            _LOGGER.info("Fetched %d exposed entities", len(exposed))
            return exposed
        except Exception as e:
            _LOGGER.error("Failed to fetch exposed entities: %s", e)
            raise HomeAssistantError(str(e))

    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Fetch full state of an entity.

        Args:
            entity_id: Entity ID (e.g., 'sensor.bedroom_temperature').

        Returns:
            Dict with state details or error message.
        """
        state = self.hass.states.get(entity_id)
        if not state:
            _LOGGER.warning("Entity not found: %s", entity_id)
            return {"message": "Entity not found"}
        # Cast to Dict[str, Any] since we know as_dict()
        # produces a dict and serialization preserves it
        return typing.cast(Dict[str, Any], make_json_serializable(state.as_dict()))

    async def set_thermostat_temp(
        self, entity_id: str, temperature: float
    ) -> Dict[str, Any]:
        """Set thermostat temperature.

        Args:
            entity_id: Thermostat entity ID (e.g., 'climate.living_room').
            temperature: Target temperature (e.g., 72.5).

        Returns:
            Dict with status and message.
        """
        try:
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {"entity_id": entity_id},
                {"temperature": temperature},
            )
            return {
                "status": "success",
                "message": f"Set {entity_id} to {temperature} degrees",
            }
        except Exception as e:
            _LOGGER.error("Failed to set thermostat: %s", e)
            return {"status": "error", "message": str(e)}
