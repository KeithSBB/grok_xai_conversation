"""The Grok xAI Conversation integration."""

import asyncio
import logging

import xai_sdk
from homeassistant.components import conversation as ha_conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_API_KEY, DEFAULT_MODEL, DEFAULT_PROMPT, DOMAIN
from .conversation import GrokConversationAgent
from .tools import GrokHaTool

_LOGGER = logging.getLogger(__name__)
_LOGGER.info(f"Grok integration is loading. xAI SDK version: {xai_sdk.__version__}")


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Grok xAI Conversation from a config entry."""
    # Merge data and options, preferring options
    config = {**entry.data, **entry.options}

    # Create a single instance of GrokHaTool and store it
    ha_tools = GrokHaTool(hass)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "ha_tools": ha_tools,
    }

    # Now use config for everything
    api_key = config[CONF_API_KEY]
    model = config.get("model", DEFAULT_MODEL)
    prompt = config.get("prompt", DEFAULT_PROMPT)

    _LOGGER.info("Setting up Grok xAI conversation agent with model: %s", model)

    try:
        agent = GrokConversationAgent(
            hass=hass,
            entry=entry,
            api_key=api_key,
            model=model,
            prompt=prompt,
            ha_tools=ha_tools,  # Pass the instance
        )
    except Exception as e:
        _LOGGER.error(
            "Failed to initialize GrokConversationAgent: %s", e, exc_info=True
        )
        # Clean up the partial data
        hass.data[DOMAIN].pop(entry.entry_id)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)
        return False

    ha_conversation.async_set_agent(hass, entry, agent)

    # Schedule the periodic cleanup task
    cleanup_task = hass.async_create_task(agent.async_run_cleanup_loop())
    hass.data[DOMAIN][entry.entry_id]["cleanup_task"] = cleanup_task

    # Add listener for options updates to reload
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    _LOGGER.debug("Reloading Grok integration entry.")
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Grok integration entry.")

    # Unset the conversation agent
    ha_conversation.async_unset_agent(hass, entry)

    # Retrieve and clean up integration data
    if DOMAIN not in hass.data or entry.entry_id not in hass.data[DOMAIN]:
        return True  # Already unloaded

    entry_data = hass.data[DOMAIN].pop(entry.entry_id)

    # Unload tool listeners
    if "ha_tools" in entry_data:
        ha_tools: GrokHaTool = entry_data["ha_tools"]
        ha_tools.async_unload_listeners()

    # Cancel the cleanup task
    if "cleanup_task" in entry_data:
        cleanup_task: asyncio.Task[ConfigEntry] = entry_data["cleanup_task"]
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            _LOGGER.debug("Conversation cleanup task cancelled successfully.")

    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)

    return True
