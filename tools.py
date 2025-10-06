"""Grok-Homeassistant tool module. This file covers all of the tools that Grok
can call on the Homeassistant platform integration."""

import re
from datetime import datetime
import fnmatch
import inspect
import json
import asyncio
import logging
import typing
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, get_origin, get_args
from dataclasses import dataclass, field

from xai_sdk.chat import tool
from homeassistant.core import HomeAssistant, State, CALLBACK_TYPE
from homeassistant.components import conversation
from homeassistant.helpers import (
    llm,
    area_registry as ar,
    entity_registry as er,
    floor_registry as fr,
    label_registry as lr,
    device_registry as dr,
    event as ev,
)
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.exceptions import HomeAssistantError
from .const import DEFAULT_LOG_LEVEL

_LOGGER = logging.getLogger(__name__)

# Hard-coded to avoid const import cycle
MAX_SEARCH_RESULTS = 10

# Fallback tool instructions (dynamic gen moved to tools.py to avoid circular imports)
TOOL_INSTRUCTIONS = ["'call_ha_service' with parameters: domain, service, target, data"]

@dataclass
class ToolSchema:
    """Dataclass for defining tool schemas in a structured way.

    Attributes:
        name: The name of the tool.
        description: A brief description of what the tool does.
        parameters: JSON schema for the tool's parameters.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class GrokHaTool:
    """Encapsulates tools with common data and methods for HA interactions.

    This class follows SOLID principles: single-responsibility for tool management.
    """

    _instance: Optional["GrokHaTool"] = None  # Singleton pattern
    _exposed_entities: Optional[List[Dict[str, Any]]] = None  # Manual cache for async method
    _debounce_call: Optional[CALLBACK_TYPE] = None  # For canceling debounce
    _registry_remove: Optional[CALLBACK_TYPE] = None  # For unsubscribing registry listener
    _state_remove: Optional[CALLBACK_TYPE] = None  # For unsubscribing state listener

    def __new__(cls, hass: HomeAssistant):
        if cls._instance is None:
            cls._instance = super(GrokHaTool, cls).__new__(cls)
            cls._instance.hass = hass
            cls._instance.entity_registry = er.async_get(hass)
            cls._registry_remove = hass.bus.async_listen('entity_registry_updated', cls._async_invalidate_cache)
            cls._state_remove = hass.bus.async_listen('state_changed', cls._async_debounced_invalidate_cache)
        return cls._instance
        
    @classmethod
    async def _async_invalidate_cache(cls, event):
        """Async handler for cache invalidation."""
        try:
            cls._exposed_entities = None
            _LOGGER.log(getattr(logging, DEFAULT_LOG_LEVEL),
            "Exposed entities cache invalidated on registry update")
        except Exception as e:
            _LOGGER.error(f"Error invalidating cache: {e}")

    @classmethod
    async def _async_debounced_invalidate_cache(cls, event):
        """Async debounced handler for state changes."""
        try:
            if cls._debounce_call is not None:
                cls._debounce_call()  # Call the cancel function
            async def _delayed_invalidate(_now):
                await cls._async_invalidate_cache(event)
                _LOGGER.log(getattr(logging, DEFAULT_LOG_LEVEL),
                f"Debounced cache invalidation on state change for {event.data.get('entity_id', 'unknown')}") 
            cls._debounce_call = ev.async_call_later(cls._instance.hass, 30, _delayed_invalidate)
        except Exception as e:
            _LOGGER.error(f"Error in debounced invalidation: {e}")

    @classmethod
    def async_unload_listeners(cls):
        """Unload event listeners and reset cache. Call from async_unload_entry."""
        if cls._registry_remove:
            cls._registry_remove()
            cls._registry_remove = None
        if cls._state_remove:
            cls._state_remove()
            cls._state_remove = None
        if cls._debounce_call:
            cls._debounce_call()
            cls._debounce_call = None
        cls._exposed_entities = None
        _LOGGER.debug("Unloaded GrokHaTool listeners and reset cache")

    @classmethod
    def generate_tool_instructions(cls) -> List[str]:
        """Generate tool instructions (moved from const.py to break circular import)."""
        try:
            schemas = cls.get_tool_schemas()
            instructions = []
            for schema in schemas:
                name = schema.name
                params = ', '.join(schema.parameters['properties'].keys())
                instructions.append(f"'{name}' with parameters: {params}")
            return instructions if instructions else ["'call_ha_service' with parameters: domain, service, target, data"]
        except Exception as e:
            _LOGGER.error(f"Failed to generate tool instructions: {e}")
            return ["'call_ha_service' with parameters: domain, service, target, data"]

    @classmethod
    @lru_cache(maxsize=1)
    def get_tool_schemas(cls) -> List[Dict[str, Any]]:
        """Generate valid flat xAI tool schemas with enhanced descriptions.

        Returns:
            List of tool schemas compatible with xAI SDK.
        """
        schemas = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith('_') or name in ('get_tool_schemas', '__init__', 'generate_tool_instructions'):
                continue
            if not asyncio.iscoroutinefunction(method):
                continue

            sig = inspect.signature(method)
            try:
                doc = inspect.getdoc(method) or ''
            except (ValueError, AttributeError) as e:
                _LOGGER.warning(f"Failed to get doc for {name}: {e}")
                continue
            if not doc:
                continue
            description = doc.splitlines()[0].strip()

            # Parse :param Name: desc and :example Name: value from docstrings
            param_descs = {}
            param_examples = {}
            if doc:
                for match in re.finditer(r':param (\w+): (.*?)(?=\n|:param|:example|$)', doc, re.DOTALL | re.IGNORECASE):
                    param_name = match.group(1)
                    raw_desc = match.group(2).strip()
                    param_descs[param_name] = raw_desc
                for match in re.finditer(r':example (\w+): (.*?)(?=\n|:example|$)', doc, re.DOTALL | re.IGNORECASE):
                    param_name = match.group(1)
                    raw_example = match.group(2).strip()
                    param_examples[param_name] = raw_example

            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                ann = param.annotation
                schema = {"type": "string"}  # Default
                desc = param_descs.get(param_name, f"{param_name} for {name}")
                if param_examples.get(param_name):
                    desc += f" (example: {param_examples[param_name]})"

                if ann != inspect.Parameter.empty:
                    origin = get_origin(ann)
                    args = get_args(ann)

                    if origin is Optional:
                        inner_schema = cls._build_json_schema(args[0])
                        schema = {**inner_schema, "nullable": True}  # xAI nullable for Optional
                    elif origin is Union:
                        schema = {"type": "string"}  # Fallback
                    elif origin is typing.Dict:
                        schema = {
                            "type": "object",
                            "additionalProperties": True
                        }
                    elif origin is typing.List:
                        item_schema = {"type": "string"}
                        if args:
                            item_schema = cls._build_json_schema(args[0])
                        schema = {"type": "array", "items": item_schema}
                    else:
                        schema = cls._build_json_schema(ann)

                properties[param_name] = {**schema, "description": desc}

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            schemas.append(tool(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            ))
            _LOGGER.debug(f"Generated schema for {name}")

        return schemas

    @classmethod
    def _build_json_schema(cls, ann: typing.Any) -> Dict[str, Any]:
        """Build JSON schema (no nulls here; handled in Optional)."""
        if ann == str:
            return {"type": "string"}
        elif ann == int:
            return {"type": "integer"}
        elif ann == float:
            return {"type": "number"}
        elif ann == bool:
            return {"type": "boolean"}
        else:
            return {"type": "string"}


    def get_area_name(self, entry: er.RegistryEntry) -> str:
        """Retrieves the area name for a given entity ID in Home Assistant."""
        if not entry or not entry.area_id:
            return ""

        area_reg = ar.async_get(self.hass)
        area_entry = area_reg.async_get_area(entry.area_id)

        if not area_entry:
            return ""

        return area_entry.name or ""  # Coalesce None to ""

    async def call_ha_service(
        self,
        domain: str,
        service: str,
        target: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        """Call a Home Assistant service with optional target and data.
        
        :param domain: The domain of the service (e.g., 'light').
        :param service: The service to call (e.g., 'turn_on').
        :param target: The target for the service (e.g., entity_id or area_id).
        :param data: Additional data for the service (e.g., brightness).
        :example target: JSON object, e.g., {'entity_id': 'light.bedroom'} or {'area_id': 'living_room'}.
        :example data: JSON object for params, e.g., {'brightness_pct': 50, 'rgb_color': [0, 255, 0], 'effect': 'rainbow'}.
        """
        try:
            if isinstance(target, str):
                target = json.loads(target)
            if isinstance(data, str):
                data = json.loads(data)
            await self.hass.services.async_call(domain, service, target=target, service_data=data)
            return {"status": "success"}
        except json.JSONDecodeError as e:
            _LOGGER.error(f"Invalid JSON in call_ha_service: {e}")
            return {"status": "error", "message": "Invalid JSON input"}
        except Exception as e:
            _LOGGER.error(f"Failed to call service {domain}.{service}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_exposed_entities(self) -> List[Dict[str, Any]]:
        """Get list of exposed entities with details like name, aliases, entity_id."""
        if self._exposed_entities is None:
            try:
                _LOGGER.debug("Fetching fresh exposed entities (cache miss)")
                apis = llm.async_get_apis(self.hass)
                assistant_id = next((api.id for api in apis if api.id == "grok_xai_conversation"), "conversation")  # Retrieve assistant ID dynamically; fallback to built-in if not registered
                ent_reg = er.async_get(self.hass)
                dev_reg = dr.async_get(self.hass)
                device_id_to_name = {entry.id: entry.name or "" for entry in dev_reg.devices.values()}  # Coalesce None to ""
                exposed = []
                for ent in ent_reg.entities.values():
                    if async_should_expose(self.hass, assistant_id, ent.entity_id):
                        exposed.append({
                            "entity_id": ent.entity_id,
                            "name": ent.name or ent.original_name or ent.entity_id,
                            "aliases": ent.aliases,
                            "domain": ent.domain,
                            "area_name": self.get_area_name(ent) or "",
                            "device_name": device_id_to_name.get(ent.device_id, "") if ent.device_id else ""
                        })
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
        device_globs: Optional[Union[str, List[str]]] = None
    ) -> Union[List[Dict[str, Any]], str]:
        """Advanced entity search with glob patterns and filters. Use simple_mode for name-based fuzzy search.
        
        Note: Always fetches fresh states for accuracy; static entity info may be cached.
        
        :param name: The name to search for in simple mode.
        :param simple_mode: If true, use fuzzy name search; else advanced globs.
        :param filters: Additional filters (e.g., domain, attributes like {'device_class': 'temperature'}).
        :param limit: Max results (default 10).
        :param offset: Result offset.
        :param entity_globs: Glob patterns for entity IDs.
        :param domains: Domains to filter.
        :param area_globs: Glob patterns for areas.
        :param floor_globs: Glob patterns for floors.
        :param tag_globs: Glob patterns for tags.
        :param device_globs: Glob patterns for devices.
        :example filters: JSON object, e.g., {'domain': 'light', 'attributes': {'device_class': 'temperature'}}.
        """
        try:
            if isinstance(filters, str):
                filters = json.loads(filters)
        except json.JSONDecodeError:
            _LOGGER.warning("Invalid filters JSON in find_entities_advanced; using empty dict")
            filters = {}
        except Exception as e:
            _LOGGER.error(f"Filter processing error in find_entities_advanced: {e}")
            return "Invalid filters provided"
        filters = filters or {}  # Default to empty dict to avoid NoneType errors
        
        def ensure_list(value: Optional[Union[str, List[str]]]) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            return value
            
        def glob_match(string: str, pattern: str) -> bool:
            regex_pattern = fnmatch.translate(pattern)
            regex = re.compile(regex_pattern, re.IGNORECASE)
            return regex.match(string) is not None

        entity_globs_list = ensure_list(entity_globs)
        area_globs_list = ensure_list(area_globs)
        device_globs_list = ensure_list(device_globs)

        if simple_mode:
            if not name:
                return [{"message": "Name required for simple search"}]
            domain_filter = filters.get("domain")
            if not domain_filter and domains:
                domain_filter = domains if isinstance(domains, str) else domains[0] if domains else None
            attr_filters = filters.get("attributes", {})
            matching_entity_ids = set()  # Collect IDs first, fetch fresh later
            exposed = await self.get_exposed_entities()
            words = name.lower().split()
            for ent in exposed:
                if domain_filter and ent["domain"] != domain_filter:
                    continue
                # Note: attr_filters require fresh attributes; defer to post-fetch if needed
                try:
                    if entity_globs_list and not any(glob_match(ent["entity_id"], g) for g in entity_globs_list):
                        continue
                    if area_globs_list and not any(glob_match(ent["area_name"], g) for g in area_globs_list):
                        continue
                    if device_globs_list and not any(glob_match(ent["device_name"], g) for g in device_globs_list):
                        continue
                except TypeError as te:
                    _LOGGER.error(f"TypeError in glob matching: {te}")
                    continue
                score = sum(1 for word in words if word in ent["name"].lower() or word in ent["entity_id"].lower() or word in ent["area_name"].lower() or any(word in alias.lower() for alias in ent.get("aliases", [])))
                if score > 0:
                    matching_entity_ids.add(ent["entity_id"])
            
            # Fetch fresh states for matching IDs
            matching = []
            for entity_id in sorted(list(matching_entity_ids)):  # Sort for consistency
                fresh_state = self.hass.states.get(entity_id)
                if not fresh_state:
                    continue
                # Apply attr_filters on fresh attributes
                if attr_filters and not all(fresh_state.attributes.get(k) == v for k, v in attr_filters.items()):
                    continue
                ent = next((e for e in exposed if e["entity_id"] == entity_id), None)
                if ent:
                    matching.append({
                        "entity_id": entity_id,
                        "name": ent["name"],
                        "score": sum(1 for word in words if word in ent["name"].lower() or word in entity_id.lower() or word in ent["area_name"].lower()),  # Recalc score if needed
                        "area_name": ent["area_name"],
                        "state": fresh_state.state,
                        "unit": fresh_state.attributes.get("unit_of_measurement")
                    })
            matching.sort(key=lambda x: x["score"], reverse=True)
            _LOGGER.debug(f"Matching entities:\n {matching}")
            return matching[offset:offset+limit] if matching else [{"message": "No entities found"}]  

        def make_json_serializable(obj: Any) -> Any:
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
            else:
                return obj


        entity_globs = ensure_list(entity_globs)
        domains = ensure_list(domains) or [filters.get("domain")] if filters.get("domain") else []  # Merge filters.domain if no domains
        area_globs = ensure_list(area_globs)
        floor_globs = ensure_list(floor_globs)
        tag_globs = ensure_list(tag_globs)
        device_globs = ensure_list(device_globs)

        ent_reg = er.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        floor_reg = fr.async_get(self.hass)
        label_reg = lr.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)

        # Pre-computation maps for faster lookups, coalescing None to ""
        floor_id_to_name = {entry.floor_id: entry.name or "" for entry in floor_reg.async_list_floors()}
        area_id_to_name = {entry.id: entry.name or "" for entry in area_reg.async_list_areas()}
        label_id_to_name = {entry.label_id: entry.name or "" for entry in label_reg.async_list_labels()}
        device_id_to_name = {entry.id: entry.name or "" for entry in dev_reg.devices.values()}

        all_entity_entries = ent_reg.entities.values()
        matching_entity_ids = set()

        for entry in all_entity_entries:
            # Domain filter (optimized: skip early if no match)
            if domains and entry.domain not in domains:
                continue

            state = self.hass.states.get(entry.entity_id)
            attr_filters = filters.get("attributes", {})
            if attr_filters and state and not all(state.attributes.get(k) == v for k, v in attr_filters.items()):
                continue

            # Entity ID glob
            if entity_globs and not any(glob_match(entry.entity_id, g) for g in entity_globs):
                continue

            # Device glob
            match_device = not device_globs
            if device_globs and entry.device_id:
                device_name = device_id_to_name.get(entry.device_id, '')
                match_device = any(glob_match(device_name, g) for g in device_globs)

            if not match_device:
                continue

            # Area glob
            match_area = not area_globs
            if area_globs:
                area_id = entry.area_id or (dev_reg.async_get(entry.device_id).area_id if entry.device_id else None)
                area_name = area_id_to_name.get(area_id, '') if area_id else ''
                match_area = bool(area_name) and any(glob_match(area_name, g) for g in area_globs)

            if not match_area:
                continue

            # Floor glob
            match_floor = not floor_globs
            if floor_globs:
                area_id = entry.area_id or (dev_reg.async_get(entry.device_id).area_id if entry.device_id else None)
                if area_id:
                    area_entry = area_reg.async_get_area(area_id)
                    if area_entry and area_entry.floor_id:
                        floor_name = floor_id_to_name.get(area_entry.floor_id, '')
                        match_floor = bool(floor_name) and any(glob_match(floor_name, g) for g in floor_globs)

            if not match_floor:
                continue

            # Tag/label glob
            match_tag = not tag_globs or (entry.labels and any(
                any(glob_match(label_id_to_name.get(label_id, ''), g) for g in tag_globs)
                for label_id in entry.labels
            ))

            if match_tag:
                matching_entity_ids.add(entry.entity_id)
                
        _LOGGER.debug(f"Advanced mode matched {len(matching_entity_ids)} entities")

        # Retrieve states only for matches (optimization)
        filtered = [self.hass.states.get(entity_id) for entity_id in matching_entity_ids if self.hass.states.get(entity_id)]
        filtered = filtered[offset:offset+limit]
        if not filtered:
            return "No matches were found"
        return [make_json_serializable(state.as_dict()) for state in filtered]
        
    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Fetch the full state of an entity, including attributes and unit.
        
        :param entity_id: The entity ID to fetch.
        :example entity_id: 'sensor.bedroom_temperature'.
        """
        state = self.hass.states.get(entity_id)
        if not state:
            return {"message": "Entity not found"}
        return make_json_serializable(state.as_dict())
        
    @dataclass
    class ToolResult:
        status: str = "success"
        message: str = ""

    # Example new tool: set_thermostat_temp (as suggested enhancement)
    async def set_thermostat_temp(self, entity_id: str, temperature: float) -> Dict[str, Any]:
        """Set the temperature of a thermostat entity.
        
        :param entity_id: The entity ID of the thermostat.
        :param temperature: The temperature to set.
        :example entity_id: 'climate.living_room_thermostat'.
        :example temperature: 72.5.
        """
        try:
            await self.hass.services.async_call(
                "climate", "set_temperature", {"entity_id": entity_id, "temperature": temperature}
            )
         
            return ToolResult(status="success", message=f"Set {entity_id} to {temperature} degrees").__dict__

        except Exception as e:
            _LOGGER.error(f"Failed to set thermostat: {str(e)}")
            return ToolResult(status="error", message=str(e)).__dict__
    
    