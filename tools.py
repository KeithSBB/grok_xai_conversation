"""Grok-Homeassistant tool module.

This module defines tools that Grok can call within the Home Assistant integration.
Tools follow xAI SDK schema (chat_pb2.Tool) and return JSON-serializable dicts.
"""

import fnmatch
import inspect
import logging
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

# HA Imports
from homeassistant.components import history  # New import for history tool
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers.event import async_call_later

# xAI Imports
from xai_sdk.chat import tool

from .const import MAX_SEARCH_RESULTS

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
)


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

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the GrokHaTool instance.

        Args:
            hass: Home Assistant core instance.
            entry: Config entry for this integration (used for assistant_id).
        """
        _LOGGER.debug("Initializing GrokHaTool")
        self.hass = hass
        self.entry = entry
        # Use conversation.DOMAIN for exposed entities lookup in modern HA
        self.assistant_id = conversation.DOMAIN
        _LOGGER.debug(
            "Computed assistant_id for exposed_entities: %s", self.assistant_id
        )
        self.ent_reg = er.async_get(hass)
        self.area_reg = ar.async_get(hass)
        self.floor_reg = fr.async_get(hass)
        self.label_reg = lr.async_get(hass)
        self.dev_reg = dr.async_get(hass)
        self._exposed_entities: Optional[List[Dict[str, Any]]] = None
        self._debounce_call: Optional[CALLBACK_TYPE] = None
        # Cache for optimization
        self._ha_context_summary_cache: Optional[str] = None

        # Setup listeners (assuming the listener setup/cleanup logic remains the same)
        self._registry_remove = self.hass.bus.async_listen(
            "entity_registry_updated", self._async_invalidate_cache
        )
        self._state_remove = self.hass.bus.async_listen(
            "state_changed", self._async_debounced_invalidate_cache
        )

    async def _notify_exposed_entities_fallback(self) -> None:
        """Notify user via persistent notification if fallback is active."""
        try:
            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "Grok xAI Integration Warning",
                    "message": (
                        "Exposed entities module import failed; assuming"
                        "all entities are exposed. "
                        "This may affect security—check HA logs and core installation."
                    ),
                },
            )
        except Exception as e:
            _LOGGER.error(
                "Failed to create notification for exposed_entities fallback: %s", e
            )

    async def _async_invalidate_cache(self, event: Event) -> None:
        """Invalidate caches on registry/state updates.

        Args:
            event: HA event triggering invalidation.
        """
        try:
            self._exposed_entities = None
            self._ha_context_summary_cache = None
            _LOGGER.debug("Caches invalidated by event: %s", event.event_type)
        except Exception as e:
            _LOGGER.error("Error invalidating caches: %s", e)

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
            self._debounce_call = None
        self._exposed_entities = None
        self._ha_context_summary_cache = None
        _LOGGER.debug("Listeners unloaded and caches cleared")

    async def async_get_ha_context_summary(self) -> str:
        """Generate a structured summary of HA context (floors, areas, domains).

        Returns:
            Formatted string summary.
        """
        if self._ha_context_summary_cache is not None:
            _LOGGER.debug("Returning cached HA context summary")
            return self._ha_context_summary_cache

        _LOGGER.debug("Generating new HA context summary")
        try:
            summary_lines = ["Home Assistant Context:"]

            # Create a map of floor_id -> floor_name
            floor_map = {
                f.floor_id: f.name
                for f in self.floor_reg.async_list_floors()
                if f.floor_id and f.name
            }

            # Create a structured map of floor_id -> list of area_names
            floor_areas: Dict[Optional[str], List[str]] = {
                floor_id: [] for floor_id in floor_map
            }
            floor_areas[None] = []  # For areas not assigned to a floor

            # Populate the floor_areas map
            for area in self.area_reg.async_list_areas():
                if area.name:
                    if area.floor_id in floor_areas:
                        floor_areas[area.floor_id].append(area.name)
                    else:
                        floor_areas[None].append(area.name)  # Add to unassigned

            # Build the structured summary string
            if not floor_map and not floor_areas[None]:
                summary_lines.append("- No floors or areas are configured.")

            for floor_id, floor_name in floor_map.items():
                summary_lines.append(f'- Floor: "{floor_name}"')
                if floor_areas[floor_id]:
                    areas_str = ", ".join(sorted(floor_areas[floor_id]))
                    summary_lines.append(f"  - Areas: {areas_str}")
                else:
                    summary_lines.append("  - Areas: None")

            if floor_areas[None]:
                unassigned_areas_str = ", ".join(sorted(floor_areas[None]))
                summary_lines.append(f"- Areas with no floor: {unassigned_areas_str}")

            # Add available domains
            domains = list(
                set(
                    e.domain
                    for e in self.ent_reg.entities.values()
                    if hasattr(e, "domain") and isinstance(e.domain, str)
                )
            )
            if domains:
                summary_lines.append(
                    f"- Available Domains: {', '.join(sorted(domains))}"
                )

            summary_lines.append(
                "Use this context to disambiguate entities and locations."
            )

            summary = "\n".join(summary_lines)
            self._ha_context_summary_cache = summary
            _LOGGER.debug("Generated HA context summary:\n%s", summary)
            return summary
        except Exception as e:
            _LOGGER.error("Failed to generate HA context summary: %s", e, exc_info=True)
            return "HA context summary unavailable due to error."

    def get_tool_schemas(self) -> Sequence[Any]:
        """Generate tool schemas from methods using the xai_sdk.chat.tool helper.

        Returns:
            List of xAI tool helper objects.
        """
        _LOGGER.debug("Starting tool schema generation...")
        tools = []

        # Methods to explicitly exclude from being exposed as tools
        EXCLUDE_METHODS = {
            "async_get_ha_context_summary",
            "get_tool_schemas",
            "async_unload_listeners",
            "get_exposed_entities",
        }

        for name, method in inspect.getmembers(self):
            # Filter 1: Must be callable
            if not callable(method):
                continue

            # Filter 2: Skip private/dunder methods
            if name.startswith("_"):
                continue

            # Filter 3: Skip methods on the exclusion list
            if name in EXCLUDE_METHODS:
                _LOGGER.debug("Skipping '%s' (in EXCLUDE_METHODS)", name)
                continue

            # We found a valid tool method. Proceed with schema generation.
            _LOGGER.debug("Found potential tool method: %s", name)
            doc = inspect.getdoc(method)
            descriptions = parse_docstring_descriptions(doc or "")

            try:
                params = inspect.signature(method).parameters
            except (ValueError, TypeError) as e:
                # Happens on some built-in callables
                _LOGGER.debug("Could not get signature for %s: %s", name, e)
                continue

            param_schemas = {}
            required_params = []

            for param_name, param in params.items():
                if param.annotation is inspect.Parameter.empty:
                    _LOGGER.warning(
                        (
                            "Skipping parameter '%s' in tool '%s': "
                            "missing type annotation."
                        ),
                        param_name,
                        name,
                    )
                    continue

                param_schemas[param_name] = type_to_schema(
                    param.annotation, descriptions, param_name
                )
                if param.default is inspect.Parameter.empty:
                    required_params.append(param_name)

            try:
                # This is the JSON Schema structure (OpenAI style)
                # that the tool() helper function expects.
                parameters_dict = {
                    "type": "object",
                    "properties": param_schemas,
                    "required": required_params,
                }

                # Use the xai_sdk.chat.tool helper function.
                # This is the key change to fix the TypeError.
                tools.append(
                    tool(
                        name=name,
                        description=doc or f"No description for {name}",
                        parameters=parameters_dict,
                    )
                )
                _LOGGER.debug(
                    "Created tool schema for %s using SDK helper: %s",
                    name,
                    parameters_dict,
                )

            except Exception as e:
                _LOGGER.error(
                    "Failed to create tool schema for %s: %s", name, e, exc_info=True
                )
                continue  # Skip invalid tools to avoid full failure

        _LOGGER.info("Generated %d tool schemas", len(tools))
        return tools

    async def async_search_entities(
        self,
        name: Optional[str] = None,
        entity_globs: Optional[str] = None,
        device_globs: Optional[str] = None,
        area_globs: Optional[str] = None,
        floor_globs: Optional[str] = None,
        tag_globs: Optional[str] = None,
        limit: int = MAX_SEARCH_RESULTS,
        offset: int = 0,
        simple_mode: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """Search for entities matching criteria.

        Args:
            name: Entity name or partial name.
            entity_globs: Glob patterns for entity IDs (comma-separated).
            device_globs: Glob patterns for device names (comma-separated).
            area_globs: Glob patterns for area names (comma-separated).
            floor_globs: Glob patterns for floor names (comma-separated).
            tag_globs: Glob patterns for labels (comma-separated).
            limit: Max results to return.
            offset: Offset for pagination.
            simple_mode: If true, use simple keyword search.

        Returns:
            List of matching entity dicts or message if none found.
        """
        try:
            exposed = await self.get_exposed_entities()
            matching = []
            entity_globs_list = [
                g.strip() for g in (entity_globs or "").split(",") if g.strip()
            ]
            device_globs_list = [
                g.strip() for g in (device_globs or "").split(",") if g.strip()
            ]
            area_globs_list = [
                g.strip() for g in (area_globs or "").split(",") if g.strip()
            ]
            floor_globs_list = [
                g.strip() for g in (floor_globs or "").split(",") if g.strip()
            ]
            tag_globs_list = [
                g.strip() for g in (tag_globs or "").split(",") if g.strip()
            ]

            floor_id_to_name = {
                f.id: f.name
                for f in self.floor_reg.async_list_floors()
                if hasattr(f, "id") and hasattr(f, "name")
            }
            label_id_to_name = {
                lbl.label_id: lbl.name
                for lbl in self.label_reg.async_list_labels()
                if hasattr(lbl, "label_id") and hasattr(lbl, "name")
            }

            for exp in exposed:
                entity_id = exp["entity_id"]
                entry = self.ent_reg.async_get(entity_id)
                if not entry:
                    continue
                state = self.hass.states.get(entity_id)
                if not state:
                    continue

                # --- RESTRUCTURED FILTER LOGIC ---
                # All filters are now AND conditions.
                # If a filter is provided, it must pass.

                # Filter 1: Simple Mode Name Search
                if simple_mode and name:
                    words = name.lower().split()
                    score = sum(
                        1
                        for word in words
                        if word in exp["name"].lower()
                        or word in entity_id.lower()
                        or word in exp["area_name"].lower()
                    )
                    if score == 0:
                        continue  # Failed simple search, skip this entity

                # Filter 2: Entity Globs
                if entity_globs_list and not any(
                    fnmatch.fnmatch(entity_id.lower(), g.lower())
                    for g in entity_globs_list
                ):
                    continue  # Failed entity glob, skip

                # Filter 3: Device Globs
                if device_globs_list and not any(
                    fnmatch.fnmatch(exp["device_name"].lower(), g.lower())
                    for g in device_globs_list
                ):
                    continue  # Failed device glob, skip

                # Filter 4: Area Globs
                if area_globs_list and not any(
                    fnmatch.fnmatch(exp["area_name"].lower(), g.lower())
                    for g in area_globs_list
                ):
                    continue  # Failed area glob, skip

                # Filter 5: Floor Globs
                area_id = entry.area_id or (
                    self.dev_reg.async_get(entry.device_id).area_id
                    if entry.device_id
                    else None
                )
                floor_name = ""
                if area_id:
                    area = self.area_reg.async_get_area(area_id)
                    if area and area.floor_id:
                        floor_name = floor_id_to_name.get(area.floor_id, "")

                if floor_globs_list and not any(
                    fnmatch.fnmatch(floor_name.lower(), g.lower())
                    for g in floor_globs_list
                ):
                    continue  # Failed floor glob, skip

                # Filter 6: Tag/Label Globs
                if (
                    tag_globs_list
                    and entry.labels
                    and not any(
                        fnmatch.fnmatch(label_id_to_name.get(lbl, ""), g.lower())
                        for lbl in entry.labels
                        for g in tag_globs_list
                    )
                ):
                    continue  # Failed tag glob, skip

                # --- END RESTRUCTURED FILTER LOGIC ---

                # If we got here, all provided filters passed.
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
                # Add improved logging
                _LOGGER.warning(
                    (
                        "Search with args (name=%s, area_globs=%s, simple_mode=%s) "
                        "found 0 entities."
                    ),
                    name,
                    area_globs,
                    simple_mode,
                )
                return "No matching entities found"

            _LOGGER.debug(
                "Search found %d matches. Returning up to %d.", len(matching), limit
            )
            return matching[offset : offset + limit]

        except Exception as e:
            _LOGGER.error("Error in async_search_entities: %s", e, exc_info=True)
            return "Error searching entities"

    async def get_exposed_entities(self) -> List[Dict[str, Any]]:
        """Fetch list of exposed entities with details (cached).

        Returns:
            List of entity dicts.
        """
        if self._exposed_entities is not None:
            _LOGGER.debug("Returning cached exposed entities")
            return self._exposed_entities

        try:
            exposed = []
            _LOGGER.debug("Checking exposure for assistant ID: %s", conversation.DOMAIN)

            for entry in self.ent_reg.entities.values():  # Optimized: values()
                if async_should_expose(self.hass, conversation.DOMAIN, entry.entity_id):
                    area_name = ""
                    device_name = ""
                    device = None  # Local var for device
                    if entry.device_id:
                        device = self.dev_reg.async_get(entry.device_id)
                        device_name = device.name if device else ""

                    area = None  # Local var for area
                    if entry.area_id:
                        area = self.area_reg.async_get_area(entry.area_id)
                        area_name = area.name if area else ""
                    elif (
                        entry.device_id and device and device.area_id
                    ):  # Use cached device
                        area = self.area_reg.async_get_area(device.area_id)
                        area_name = area.name if area else ""

                    exposed.append(
                        {
                            "entity_id": entry.entity_id,
                            "name": entry.name
                            or entry.original_name
                            or entry.entity_id,  # Fallback to entity_id
                            "area_name": area_name,
                            "device_name": device_name,
                            "aliases": entry.aliases,
                        }
                    )

            self._exposed_entities = exposed
            _LOGGER.info(
                "Fetched %d exposed entities for assistant_id %s",
                len(exposed),
                self.assistant_id,
            )
            return exposed
        except Exception as e:
            _LOGGER.error(
                "Failed to fetch exposed entities for assistant_id %s: %s",
                self.assistant_id,
                e,
                exc_info=True,
            )
            raise HomeAssistantError(f"Error fetching exposed entities: {str(e)}")

    async def async_get_entity_history(
        self,
        entity_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Union[str, list[dict[str, Any] | list[Any] | str | int | float | None]]:
        """Retrieve the historical state of an entity over a period of time.

        The history component must be enabled for this to work.
        Args:
            entity_id: The ID of the entity to retrieve history for
                       (e.g., 'sensor.temperature').
            start_time: Optional ISO 8601 timestamp string for
                        the start of the period.
            end_time: Optional ISO 8601 timestamp string for the end of the period.

        Returns:
            List of lists of state dictionaries, or an error message string.
        """
        try:
            _start = datetime.fromisoformat(start_time) if start_time else None
            _end = datetime.fromisoformat(end_time) if end_time else None

            # History component returns [entity_id: [StateObject, ...]]
            history_data = await history.get_period(
                self.hass,
                end_time=_end,
                start_time=_start,
                entity_id=entity_id,
                include_start_time=True,
                no_attributes=False,
            )

            if not history_data or not history_data.get(entity_id):
                return (
                    f"No history found for entity: {entity_id} in the specified period."
                )

            # Convert HA State objects to a serializable list of dictionaries
            serialized_history = [
                make_json_serializable(state) for state in history_data[entity_id]
            ]

            _LOGGER.debug(
                "Retrieved %d history entries for %s",
                len(serialized_history),
                entity_id,
            )

            # History.get_period returns a dict, but we simplify the output for Grok
            return serialized_history

        except ValueError as e:
            _LOGGER.error("Invalid time format for history tool: %s", e)
            return (
                "Error: Invalid time format. Ensure start_time "
                "and end_time are in valid ISO 8601 format."
            )
        except Exception as e:
            _LOGGER.error(
                "Error retrieving entity history for %s: %s",
                entity_id,
                e,
                exc_info=True,
            )
            return f"Error retrieving history: {str(e)}"

    async def async_run_automation(
        self,
        entity_id: str,
    ) -> Dict[str, str]:
        """Trigger a Home Assistant automation entity.

        This is useful for activating pre-configured complex actions.
        Args:
            entity_id: The entity ID of the automation to trigger
                        (e.g., 'automation.morning_routine').

        Returns:
            A status dictionary indicating success or failure.
        """
        domain = entity_id.split(".")[0]
        if domain != "automation":
            return {
                "status": "error",
                "message": (
                    "Entity ID {entity_id} is not an automation."
                    "Only 'automation' domain entities can be run."
                ),
            }

        try:
            # Service call to trigger the automation
            await self.hass.services.async_call(
                domain="automation",
                service="trigger",
                target={"entity_id": entity_id},
                blocking=True,  # Wait for the service to complete
                context=None,
            )

            _LOGGER.info("Successfully triggered automation: %s", entity_id)
            return {
                "status": "success",
                "message": f"Successfully triggered automation: {entity_id}",
            }
        except HomeAssistantError as e:
            _LOGGER.error(
                "Home Assistant error running automation %s: %s", entity_id, e
            )
            return {
                "status": "error",
                "message": (
                    "Home Assistant Error: Could not trigger"
                    f"automation {entity_id}. Reason: {str(e)}"
                ),
            }
        except Exception as e:
            _LOGGER.error(
                "Unexpected error running automation %s: %s",
                entity_id,
                e,
                exc_info=True,
            )
            return {
                "status": "error",
                "message": f"Unexpected error while triggering automation {entity_id}.",
            }
