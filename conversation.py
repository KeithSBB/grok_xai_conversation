import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict

from homeassistant.components import conversation as haconv
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat
from xai_sdk.chat import system, tool_result, user

from .const import (
    AREA_CONTEXT_TEMPLATE,
    DEFAULT_API_TIMEOUT,
    DEFAULT_PROMPT,
    MAX_API_CALLS_PER_MIN,
    MAX_RETRIES,
    MAX_TOOL_LOOPS,
)
from .tools import GrokHaTool

_LOGGER = logging.getLogger(__name__)


def log_response(
    response: Any, logger: logging.Logger = _LOGGER, level: str = "DEBUG"
) -> None:
    """
    Logs key parts of an xAI SDK chat response (content and tool_calls)
    without verbose details like tool schemas.

    Args:
        response: The xAI SDK Message or response object from chat.sample().
        logger: Optional logger instance (defaults to module _LOGGER).
        level: Log level (e.g., "DEBUG", "INFO").
    """
    content = getattr(response, "content", "No content")
    tool_calls = getattr(response, "tool_calls", None)

    log_msg = f"Response Content: {content}"
    if tool_calls:
        log_msg += f"\nTool Calls: {[tc.function.name + ' -> '
                                     + tc.function.arguments[:100]
                                     + '...' if len(tc.function.arguments) > 100
                                     else tc.function.arguments
                                     for tc in tool_calls]}"

    getattr(logger, level.lower())(log_msg)


@dataclass
class ConversationContext:
    """Simple data structure for conversation tracking."""

    chat: Chat  # Or the specific derived class like Chat
    last_update_time: datetime


class GrokConversationAgent(haconv.AbstractConversationAgent):  # type: ignore[misc]
    """Grok xAI conversation agent with function calling using xAI SDK."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_key: str,
        model: str,
        prompt: str,
    ) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        _LOGGER.debug(
            f"Entry data keys: {list(entry.data.keys())}"
        )  # Masked sensitive data
        self.api_key = api_key
        self.model = model
        self.ha_tools = GrokHaTool(self.hass)  # Instantiate early
        tool_instructions = "; ".join(self.ha_tools.generate_tool_instructions())
        self.prompt = (prompt or DEFAULT_PROMPT).replace(
            "use available tools as specified.",
            f"use these tools: {tool_instructions}.",
        )
        self.client = AsyncClient(self.api_key)
        self.conversations: Dict[str, ConversationContext] = {}
        self._lock = asyncio.Lock()
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(MAX_API_CALLS_PER_MIN)
        self.tools = self.ha_tools.get_tool_schemas()

    async def _async_cleanup_loop(self) -> None:
        """Periodic cleanup of aged conversations."""
        while True:
            await asyncio.sleep(60)
            async with self._lock:
                to_remove = [
                    conv_id
                    for conv_id, ctx in list(self.conversations.items())
                    if datetime.now() - ctx.last_update_time > timedelta(minutes=5)
                ]
                for conv_id in to_remove:
                    del self.conversations[conv_id]
                    _LOGGER.debug(f"Cleaned up inactive conversation: {conv_id}")

    @property
    def supported_languages(self) -> list[str] | str:
        """Return a list of supported languages."""
        return ["en"]

    async def async_process(
        self, user_input: haconv.ConversationInput
    ) -> haconv.ConversationResult:
        """Process a sentence with tool-calling loop and optimizations."""
        conversation_id = user_input.conversation_id or str(uuid.uuid4())
        _LOGGER.debug(
            f"Processing input for conversation_id: {conversation_id},"
            f"text: {user_input.text}"
        )

        try:  # Main try for overall process
            # Fetch area from context (e.g., voice satellite device_id)
            area_name = "unknown"
            try:  # Nested for area fetch
                if user_input.context:
                    device_id = getattr(
                        user_input.context, "device_id", None
                    )  # Safe getattr
                    if not device_id:
                        _LOGGER.debug(
                            "Context present but no device_id; fallback to unknown"
                        )
                    if device_id:
                        dev_reg = dr.async_get(self.hass)
                        device = dev_reg.async_get(user_input.context.device_id)
                        if device and device.area_id:
                            area_reg = ar.async_get(self.hass)
                            area = area_reg.async_get_area(device.area_id)
                            if area:
                                area_name = area.name
                                _LOGGER.debug(
                                    f"Fetched area from device_id: {area_name}"
                                )

            except AttributeError as ae:
                _LOGGER.warning(f"Attribute error in context fetch: {ae}")
                area_name = "unknown"
            except Exception as e:  # Broad fallback
                _LOGGER.warning(
                    f"Failed to fetch area context: {str(e)}"
                )  # Graceful fallback

            area_context = (
                AREA_CONTEXT_TEMPLATE.format(area=area_name)
                if area_name != "unknown"
                else ""
            )

            async with self._lock:
                if conversation_id not in self.conversations:
                    _LOGGER.debug(f"Grok model: {self.model}")
                    # Format prompt with area context
                    formatted_prompt = self.prompt
                    chat = self.client.chat.create(
                        model=self.model,
                        messages=[system(formatted_prompt)],
                        tools=self.tools,
                        tool_choice="auto",
                        store_messages=True,
                    )
                    if area_context:
                        chat.append(system(area_context))
                    self.conversations[conversation_id] = ConversationContext(
                        chat=chat, last_update_time=datetime.now()
                    )
                else:
                    # For existing convos, append area context if changed/new
                    ctx = self.conversations[conversation_id]
                    if area_context:
                        ctx.chat.append(
                            system(area_context)
                        )  # Append as system message for ongoing context

                ctx = self.conversations[conversation_id]
                ctx.last_update_time = datetime.now()

                ctx.chat.append(user(user_input.text))

                # Initial sample with rate limiting and retries
                async with self.rate_limiter:
                    for attempt in range(MAX_RETRIES):
                        try:
                            response = await asyncio.wait_for(
                                ctx.chat.sample(), timeout=DEFAULT_API_TIMEOUT
                            )
                            break
                        except asyncio.TimeoutError:
                            if attempt == MAX_RETRIES - 1:
                                raise HomeAssistantError("API timeout after retries")
                            await asyncio.sleep(2**attempt)
                        except Exception as e:
                            _LOGGER.error(f"API error (attempt {attempt}): {str(e)}")
                            if attempt == MAX_RETRIES - 1:
                                raise HomeAssistantError(f"API failed: {str(e)}")

                _LOGGER.debug(f"=== Raw response: {response}")

                loop_count = 0
                while True:
                    loop_count += 1
                    if loop_count > MAX_TOOL_LOOPS:
                        _LOGGER.warning(
                            f"Tool loop exceeded max iterations"
                            f"({MAX_TOOL_LOOPS}); aborting"
                        )
                        break

                    if not (hasattr(response, "tool_calls") and response.tool_calls):
                        break

                    for tool_call in response.tool_calls:
                        try:
                            tool_name = tool_call.function.name
                            raw_args = tool_call.function.arguments
                            params = json.loads(raw_args)
                            method = getattr(self.ha_tools, tool_name, None)
                            if not callable(method):
                                raise ValueError(f"Unknown tool: {tool_name}")
                            result = await method(**params)
                            ctx.chat.append(tool_result(json.dumps(result)))
                        except json.JSONDecodeError as je:
                            error_result = {
                                "status": "error",
                                "message": f"Invalid tool args: {str(je)}",
                            }
                            ctx.chat.append(tool_result(json.dumps(error_result)))
                            _LOGGER.error(error_result["message"])
                        except ValueError as ve:
                            error_result = {"status": "error", "message": str(ve)}
                            ctx.chat.append(tool_result(json.dumps(error_result)))
                            _LOGGER.error(error_result["message"])
                        except Exception as e:
                            error_result = {
                                "status": "error",
                                "message": f"Tool execution failed: {str(e)}",
                            }
                            ctx.chat.append(tool_result(json.dumps(error_result)))
                            _LOGGER.error(error_result["message"])

                    # Sample next only if tools were called, with retries/rate limit
                    async with self.rate_limiter:
                        for attempt in range(MAX_RETRIES):
                            try:
                                response = await asyncio.wait_for(
                                    ctx.chat.sample(), timeout=DEFAULT_API_TIMEOUT
                                )
                                break
                            except asyncio.TimeoutError:
                                if attempt == MAX_RETRIES - 1:
                                    raise HomeAssistantError("API timeout in tool loop")
                                await asyncio.sleep(2**attempt)
                            except Exception as e:
                                _LOGGER.error(
                                    f"Tool sample error (attempt {attempt}): {str(e)}"
                                )
                                if attempt == MAX_RETRIES - 1:
                                    raise HomeAssistantError(
                                        f"API failed in tool loop: {str(e)}"
                                    )

                    log_response(response)

            if not hasattr(response, "content") or response.content is None:
                raise HomeAssistantError("Invalid API response: missing content")

            async with self._lock:
                ctx.last_update_time = datetime.now()

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(
                response.content or "Sorry, no response received."
            )
            return haconv.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

        except asyncio.TimeoutError as te:
            _LOGGER.error(f"Timeout in processing: {te}", exc_info=True)
            intent_response.async_set_speech("Request timed out; try again later.")
        except Exception as e:
            _LOGGER.critical(f"Unexpected error: {e}", exc_info=True)
            intent_response.async_set_speech("An error occurred; check logs.")
            return haconv.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
