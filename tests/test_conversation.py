# tests/test_conversation.py
from unittest.mock import AsyncMock, patch

import pytest
from custom_components.grok_xai_conversation.conversation import GrokConversationAgent
from homeassistant.components import conversation
from xai_sdk.chat import Chat


@pytest.mark.asyncio
async def test_async_process_success(hass) -> None:
    """Test successful conversation processing."""
    entry = AsyncMock()
    entry.data = {"api_key": "test_key"}
    agent = GrokConversationAgent(hass, entry, "test_key", "grok-beta", "Test prompt")

    with patch("xai_sdk.AsyncClient.chat.create") as mock_chat_create:
        mock_chat = AsyncMock(spec=Chat)
        mock_chat.sample.return_value = AsyncMock(content="Hello, world!")
        mock_chat_create.return_value = mock_chat

        user_input = conversation.ConversationInput(
            text="Hello", conversation_id=None, context=None, language="en"
        )

        result = await agent.async_process(user_input)

        assert result.response.speech == "Hello, world!"
        mock_chat.sample.assert_called_once()


# Add more tests: e.g., for tool-calling loop, error handling, cleanup.
