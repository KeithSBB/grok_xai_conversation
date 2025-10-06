"""Constants for the Grok xAI Conversation integration.

This file centralizes all static values used across the integration, such as domain,
default models, prompts, and API configurations. This promotes modularity and reduces
hard-coding in other files like conversation.py and config_flow.py.
"""

# Integration domain identifier
DOMAIN: str = "grok_xai_conversation"

# Default xAI model (updated to official xAI SDK-compatible value; fallback for config)
DEFAULT_MODEL: str = "grok-beta"

# List of supported models for options flow (enables multi-model toggle)
MODEL_OPTIONS = ["grok-beta", "grok-4", "grok-code-fast-1", "grok-4-fast-non-reasoning"]  # From xAI docs; extend as new models release


# Default system prompt for Grok conversation agent
# Dynamically includes tool instructions; area_context is appended separately at runtime
# Enhanced with disambiguation instruction
DEFAULT_PROMPT: str = (
    f"You are Grok, a helpful home automation assistant built by xAI. "
    f"For device control, use these tools: {'; '.join(TOOL_INSTRUCTIONS)}. "
    f"If multiple entities match (e.g., lights in different areas), use provided area context to disambiguate or ask user for clarification. "
    f"Example: For lights, use 'call_ha_service' with domain 'light', service 'turn_on', "
    f"target as a JSON object with entity_id set to 'light.bedroom', data as a JSON object with brightness_pct set to 50. "
    f"For general queries, provide concise, accurate answers."
).strip()  # Clean whitespace

# xAI API base URL (for client initialization in conversation.py)
API_BASE_URL: str = "https://api.x.ai/v1"

# Default API timeout in seconds (used in async calls for robustness)
DEFAULT_API_TIMEOUT = 30

# Maximum loops for tool-calling in conversation.py to prevent infinite loops
MAX_API_CALLS_PER_MIN = 10
MAX_RETRIES = 3
MAX_TOOL_LOOPS = 5

# Maximum number of results to return with search tool
MAX_SEARCH_RESULTS = 10

# Interval for conversation cleanup task in seconds
CLEANUP_INTERVAL = 60

# Default log level for integration (can be overridden in config)
DEFAULT_LOG_LEVEL: str = "DEBUG"

# Template for area context (used in conversation.py)
AREA_CONTEXT_TEMPLATE: str = "The user is speaking from the {area} area. Use this to disambiguate entities if needed."

# Config keys
CONF_API_KEY = "api_key"
CONF_LOG_LEVEL = DEFAULT_LOG_LEVEL
MAX_PROMPT_LENGTH = 2000  # For validation
FALLBACK_MODEL = "grok-beta"
CONF_MODEL_OPTIONS = MODEL_OPTIONS