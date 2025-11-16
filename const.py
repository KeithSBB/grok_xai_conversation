"""Constants for the Grok xAI Conversation integration.

This file centralizes all static values used across the integration,
such as domain, default models, prompts, and API configurations.
"""

# Integration domain identifier
DOMAIN = "grokzilla"

# Default xAI model (fallback for config)
DEFAULT_MODEL = "grok-beta"

# List of supported models for options flow
MODEL_OPTIONS = [
    "grok-beta",
    "grok-4",
    "grok-code-fast-1",
    "grok-4-fast-non-reasoning",
]

# Default system prompt for Grok conversation agent
# Tool instructions and HA Context are injected dynamically.
DEFAULT_PROMPT = (
    "You are Grok, a helpful home automation assistant built by xAI. "
    "You are integrated into a Home Assistant instance."
    "For device control, use available tools as specified. "
    "If multiple entities match (e.g., lights in different areas),"
    "use the provided area context to disambiguate or ask the user for clarification. "
    "Example: For lights, use 'call_ha_service' with domain 'light', "
    "service 'turn_on', "
    "target as a JSON object with entity_id set to 'light.bedroom',"
    "data as a JSON object with brightness_pct set to 50. "
    "For general queries, provide concise, accurate answers."
).strip()

# Default API timeout in seconds
DEFAULT_API_TIMEOUT = 30

# Maximum concurrent API calls (for asyncio.Semaphore)
MAX_CONCURRENT_API_CALLS = 10
MAX_RETRIES = 3
# Maximum loops for tool-calling to prevent infinite loops
MAX_TOOL_LOOPS = 5

# Maximum number of results to return with search tool
MAX_SEARCH_RESULTS = 10

# Interval for conversation cleanup task in seconds
CLEANUP_INTERVAL = 60  # Note: This is 60 seconds (1 minute)

# Template for area context (used in conversation.py)
AREA_CONTEXT_TEMPLATE = (
    "The user is speaking from the {area} area."
    "Use this to disambiguate entities if needed."
)

# Config keys
GRPC_API_KEY_HEADER = "x-api-key"
CONF_API_KEY = "api_key"
MAX_PROMPT_LENGTH = 2000  # For validation
CONF_MODEL_OPTIONS = MODEL_OPTIONS
