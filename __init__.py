import asyncio
import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .conversation import GrokConversationAgent
from .tools import GrokHaTool

_LOGGER = logging.getLogger(__name__)  # This uses the module name as the logger

# Example log statement (add one temporarily if needed to force emission)
_LOGGER.info("Your Component loaded successfully")


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Grok xAI Conversation from a config entry."""
    # Merge data and options, preferring options
    config = {**entry.data, **entry.options}

    # Now use config for everything
    api_key = config["api_key"]
    model = config.get("model", "grok-4-fast-non-reasoning")
    prompt = config.get("prompt", "Your default prompt here...")
    # TODO: remove: exposed_apis = config.get(CONF_LLM_HASS_API, [])

    _LOGGER.debug("Setting up Grok xAI conversation agent with model: %s", model)

    # Register the LLM API
    agent = GrokConversationAgent(hass, entry, api_key, model, prompt)
    conversation.async_set_agent(hass, entry, agent)
    # llm.async_register_api(hass, llm_api)

    # Handle exposed APIs if needed
    # e.g., for api_id in exposed_apis:
    #     llm.async_expose_api(hass, api_id)  # Or whatever logic

    # Schedule the periodic cleanup task
    cleanup_task = hass.loop.create_task(agent._async_cleanup_loop())
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {"cleanup_task": cleanup_task}

    # Add listener for options updates to reload
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    conversation.async_unset_agent(hass, entry)
    GrokHaTool.async_unload_listeners()
    # Cancel the cleanup task
    if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
        cleanup_task = hass.data[DOMAIN][entry.entry_id].pop("cleanup_task", None)
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation

    return True
