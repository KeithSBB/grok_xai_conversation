import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector, 
    TextSelectorConfig, 
    TextSelectorType
)

from .const import DOMAIN, DEFAULT_MODEL, DEFAULT_PROMPT

from typing import Any



class GrokConfigFlow(config_entries.ConfigFlow, domain="grok_xai_conversation"):
    """Grok xAI conversation config flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> config_entries.ConfigFlowResult:
        """Handle the initial step."""
        if user_input is not None:
            self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title="Grok xAI Conversation", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required("api_key"): str,
                }
            ),
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> config_entries.ConfigFlowResult:
        """Handle reconfiguration."""
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
                entry.data
            ),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return GrokOptionsFlow(config_entry)


class GrokOptionsFlow(config_entries.OptionsFlow):
    """Grok xAI conversation options flow."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.entry_id = config_entry.entry_id
        self.data = config_entry.data.copy()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options_schema = self._get_options_schema(self.hass)
        # Suggest values: Prefer options, fallback to data or defaults
        suggested = dict(self.config_entry.options)
        if "model" not in suggested:
            suggested["model"] = self.config_entry.data.get("model", DEFAULT_MODEL)
        if "prompt" not in suggested:
            suggested["prompt"] = self.config_entry.data.get("prompt", DEFAULT_PROMPT)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                options_schema, suggested
            ),
        )

    @staticmethod
    def _get_options_schema(hass: HomeAssistant) -> vol.Schema:
        """Return the options schema."""
        apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(hass)
        ]

        return vol.Schema(
            {
                vol.Optional("model", default=DEFAULT_MODEL): str,
                vol.Optional("prompt", default=DEFAULT_PROMPT): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT, multiline=True)),
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



