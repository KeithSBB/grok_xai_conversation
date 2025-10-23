import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries

try:
    from homeassistant.config_entries import AbortFlow, ConfigFlow, OptionsFlow
except ImportError:
    from homeassistant.config_entries import ConfigFlow, OptionsFlow

    class AbortFlow(Exception):  # type: ignore[no-redef]
        def __init__(self, reason: str):
            self.reason = reason


from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    TextSelectorType,  # Ensure TextSelectorType is imported
)
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)
from typing_extensions import ParamSpec, TypeVar

from .const import CONF_API_KEY, DEFAULT_MODEL, DEFAULT_PROMPT, DOMAIN

_LOGGER = logging.getLogger(__name__)

P = ParamSpec("P")  # Represents parameter spec
R = TypeVar("R")  # Represents return type


class GrokConfigFlow(ConfigFlow, domain=DOMAIN):  # type: ignore[misc,call-arg]
    """Grok xAI conversation config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial user configuration step."""
        if user_input is not None:
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title="Grok xAI Conversation", data=user_input
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            ),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle reconfiguration of an existing entry."""
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            return self.async_update_reload_and_abort(
                entry,
                data_updates=user_input,
            )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required("api_key"): str,
                    }
                ),
                entry.data,
            ),
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create and return the options flow handler."""
        return GrokOptionsFlow()


class GrokOptionsFlow(OptionsFlow):  # type: ignore[misc]
    """Grok xAI conversation options flow.

    Manages customizable options like model selection and prompts.
    """

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the initial options step."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        try:
            options_schema = await self._get_options_schema(
                self.hass
            )  # Await the async method
            # Suggest values: Prefer options, fallback to data or defaults
            suggested = dict(self.config_entry.options)
            if "model" not in suggested:
                suggested["model"] = self.config_entry.data.get("model", DEFAULT_MODEL)
            if "prompt" not in suggested:
                suggested["prompt"] = self.config_entry.data.get(
                    "prompt", DEFAULT_PROMPT
                )

            return self.async_show_form(
                step_id="init",
                data_schema=self.add_suggested_values_to_schema(
                    options_schema, suggested
                ),
            )
        except Exception as err:
            _LOGGER.error("Error retrieving options schema: %s", err)
            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema({}),  # Empty schema as fallback
                errors={"base": "schema_error"},  # User-friendly feedback
            )

    @staticmethod
    async def _get_options_schema(hass: HomeAssistant) -> vol.Schema:  # Made async
        """Generate the schema for options,
        including available LLM APIs asynchronously."""
        try:
            apis: list[SelectOptionDict] = [
                SelectOptionDict(
                    label=api.name,
                    value=api.id,
                )
                for api in llm.async_get_apis(hass)  # Await the async call
            ]
        except Exception as err:
            _LOGGER.warning("Failed to load LLM APIs: %s", err)
            apis = []

        return vol.Schema(
            {
                vol.Optional("model", default=DEFAULT_MODEL): str,
                vol.Optional("prompt", default=DEFAULT_PROMPT): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)
                ),
                vol.Optional(
                    CONF_LLM_HASS_API,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=apis,
                        multiple=True,
                        mode=SelectSelectorMode.LIST,
                    )
                ),
            }
        )
