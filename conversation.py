"""Grok xAI Conversation Agent."""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence

import grpc  # For RpcError
from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import intent
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat, Response

# Explicitly import the helper functions as requested
from xai_sdk.chat import assistant, system, tool_result, user
from xai_sdk.proto import chat_pb2

from .const import (
    AREA_CONTEXT_TEMPLATE,
    CLEANUP_INTERVAL,
    DEFAULT_API_TIMEOUT,
    DEFAULT_PROMPT,
    MAX_CONCURRENT_API_CALLS,
    MAX_RETRIES,
    MAX_TOOL_LOOPS,
)
from .tools import GrokHaTool

_LOGGER = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Data structure for tracking conversation state."""

    chat: Chat
    last_update_time: datetime
    formatted_prompt: str
    # Cache HA context summary to avoid re-generating
    ha_context_summary: str


class GrokConversationAgent(AbstractConversationAgent):  # type: ignore
    """Grok xAI conversation agent using xAI SDK."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        api_key: str,
        model: str,
        ha_tools: GrokHaTool,  # Accept the instance
        prompt: Optional[str] = None,
    ) -> None:
        """Initialize the conversation agent."""
        self.hass = hass
        self.entry = entry
        self.api_key = api_key
        self.model = model
        self.ha_tools = ha_tools  # Store the instance
        self.base_prompt = prompt or DEFAULT_PROMPT

        # Get tool schemas from the passed instance
        self.tools: Sequence[chat_pb2.Tool] = self.ha_tools.get_tool_schemas()

        self.client = AsyncClient(api_key=self.api_key)
        self.conversations: Dict[str, ConversationContext] = {}
        self._lock = asyncio.Lock()

        # Use the correctly named constant
        self.rate_limiter = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

        # Use standard logger
        _LOGGER.info("Initialized GrokConversationAgent with model: %s", self.model)

    async def async_run_cleanup_loop(self) -> None:
        """Periodic async task to clean up inactive conversations."""
        _LOGGER.debug("Starting conversation cleanup loop.")
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = datetime.now()
            async with self._lock:
                to_remove = [
                    conv_id
                    for conv_id, ctx in self.conversations.items()
                    if now - ctx.last_update_time > timedelta(minutes=5)
                ]
                for conv_id in to_remove:
                    try:
                        del self.conversations[conv_id]
                        _LOGGER.debug("Cleaned up inactive conversation: %s", conv_id)
                    except KeyError:
                        _LOGGER.debug("Conversation %s already removed.", conv_id)

    @property
    def supported_languages(self) -> list[str] | str:
        """Return list of supported languages."""
        return ["en"]

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process user input with tool-calling loop."""
        conversation_id = user_input.conversation_id or str(uuid.uuid4())
        _LOGGER.debug(
            "Processing input for conversation_id: %s, text: %s",
            conversation_id,
            user_input.text,
        )

        # 1. Fetch area context
        area_name = "unknown"
        if user_input.device_id:
            device_id = user_input.device_id
            if device_id:
                dev_reg = dr.async_get(self.hass)
                device = dev_reg.async_get(device_id)
                if device and device.area_id:
                    area_reg = ar.async_get(self.hass)
                    area = area_reg.async_get_area(device.area_id)
                    if area and area.name:
                        area_name = area.name
                        _LOGGER.debug("Fetched area context: %s", area_name)

        area_context_prompt = (
            AREA_CONTEXT_TEMPLATE.format(area=area_name)
            if area_name != "unknown"
            else ""
        )

        # 2. Get or create conversation context
        async with self._lock:
            if conversation_id not in self.conversations:
                _LOGGER.debug("Creating new conversation: %s", conversation_id)

                # --- THIS IS THE MODIFIED LINE ---
                # Call the method from the ha_tools instance instead
                ha_context_summary = await self.ha_tools.async_get_ha_context_summary()
                # --- END MODIFICATION ---

                chat = self.client.chat.create(
                    model=self.model,
                    messages=[
                        system(self.base_prompt),
                        system(ha_context_summary),
                    ],
                    tools=self.tools,
                    tool_choice="auto",
                    store_messages=True,
                )

                if area_context_prompt:
                    chat.append(system(area_context_prompt))

                self.conversations[conversation_id] = ConversationContext(
                    chat=chat,
                    last_update_time=datetime.now(),
                    formatted_prompt=self.base_prompt,
                    ha_context_summary=ha_context_summary,
                )
            else:
                # For existing, append updated area context
                _LOGGER.debug("Using existing conversation: %s", conversation_id)
                ctx = self.conversations[conversation_id]
                if area_context_prompt:
                    ctx.chat.append(system(area_context_prompt))

            ctx = self.conversations[conversation_id]
            ctx.last_update_time = datetime.now()
            ctx.chat.append(user(user_input.text))  # Append user's message

        # 3. Initial API call
        try:
            response = await self._async_sample_with_retries(ctx.chat)
        except HomeAssistantError as e:
            _LOGGER.error("API error after retries: %s", e)
            return self._create_error_response(
                user_input.language, f"An API error occurred: {e}"
            )

        # 4. Tool-calling loop
        loop_count = 0
        while hasattr(response, "tool_calls") and response.tool_calls:
            loop_count += 1
            if loop_count > MAX_TOOL_LOOPS:
                _LOGGER.warning(
                    "Tool loop exceeded max iterations (%d) for %s; aborting",
                    MAX_TOOL_LOOPS,
                    conversation_id,
                )
                break  # Exit loop

            _LOGGER.debug("Tool call(s) detected: %s", response.tool_calls)

            # Append tool results in a single batch
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                try:
                    params_str = tool_call.function.arguments
                    _LOGGER.debug("Tool call: %s, Args: %s", tool_name, params_str)
                    params = json.loads(params_str)

                    method = getattr(self.ha_tools, tool_name, None)
                    if not callable(method):
                        raise ValueError(f"Unknown tool: {tool_name}")

                    result = await method(**params)
                    tool_results.append(tool_result(json.dumps(result)))

                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Tool {tool_name} invalid args/unknown: {str(e)}"
                    _LOGGER.error(error_msg)
                    tool_results.append(
                        tool_result(
                            json.dumps({"status": "error", "message": error_msg})
                        )
                    )
                except Exception as e:
                    error_msg = f"Tool {tool_name} execution failed: {str(e)}"
                    _LOGGER.error(error_msg, exc_info=True)
                    tool_results.append(
                        tool_result(
                            json.dumps({"status": "error", "message": error_msg})
                        )
                    )

            # Append all results
            for res in tool_results:
                ctx.chat.append(res)

            # Sample next response
            try:
                response = await self._async_sample_with_retries(ctx.chat)
            except HomeAssistantError as e:
                _LOGGER.error("API error during tool loop: %s", e)
                return self._create_error_response(
                    user_input.language, f"An API error occurred during tool use: {e}"
                )

        # 5. Final Response
        final_content = response.content or "Sorry, I couldn't process that."

        # Use assistant() to store the final response in history
        ctx.chat.append(assistant(final_content))

        async with self._lock:
            ctx.last_update_time = datetime.now()

        _LOGGER.debug("Final response for %s: %s", conversation_id, final_content)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(final_content)
        return ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _create_error_response(
        self, language: str, error_message: str
    ) -> ConversationResult:
        """Create a standard error response for the user."""
        intent_response = intent.IntentResponse(language=language)
        intent_response.async_set_speech(error_message)
        return ConversationResult(response=intent_response)

    async def _async_sample_with_retries(self, chat: Chat) -> Response:
        """Helper to sample chat with retries and timeout."""
        async with self.rate_limiter:
            last_exception: Exception = Exception("Unknown error")
            for attempt in range(MAX_RETRIES):
                try:
                    _LOGGER.debug("Sampling chat (Attempt %d)...", attempt + 1)
                    return await asyncio.wait_for(
                        chat.sample(), timeout=DEFAULT_API_TIMEOUT
                    )
                except asyncio.TimeoutError as e:
                    _LOGGER.warning("API timeout (Attempt %d)", attempt + 1)
                    last_exception = e
                    if attempt == MAX_RETRIES - 1:
                        raise HomeAssistantError("API timeout after retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                except grpc.RpcError as grpc_err:
                    _LOGGER.error(
                        "gRPC error on sample (Attempt %d): %s", attempt + 1, grpc_err
                    )
                    last_exception = grpc_err
                    if attempt == MAX_RETRIES - 1:
                        raise HomeAssistantError(f"gRPC failed: {grpc_err.details()}")
                    await asyncio.sleep(2**attempt)
                except Exception as e:
                    _LOGGER.error(
                        "API sample error (Attempt %d): %s",
                        attempt + 1,
                        str(e),
                        exc_info=True,
                    )
                    last_exception = e
                    if attempt == MAX_RETRIES - 1:
                        raise HomeAssistantError(f"API failed: {str(e)}")
                    await asyncio.sleep(2**attempt)

        # This line should not be reachable, but raises for type safety
        raise HomeAssistantError(
            f"Unexpected failure in sample retries: {last_exception}"
        )
