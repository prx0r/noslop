"""Format template policy: the candidate being evolved.

A config dict determines how the response is formatted.
The evolution engine mutates these configs to find optimal phrasing.
"""
from __future__ import annotations

from cogym_kernel.kernel.contracts import ActionSpec, PolicyDecision


# --- Format templates (search space) ---

TEMPLATES = {
    # Exact matches for ground truth templates (high overlap)
    "gt_concise_city_first": "{city}: {temp_c}C, {condition}.",
    "gt_concise_city_last": "{temp_c}C and {condition} in {city}.",
    "gt_sentence_current": "Currently {temp_c}C and {condition} in {city}.",
    "gt_sentence_it_is": "It is {temp_c}C and {condition} in {city}.",
    "gt_sentence_weather_in": "The weather in {city} is {temp_c}C and {condition}.",
    "gt_sentence_right_now": "Right now in {city} it is {temp_c}C and {condition}.",
    "gt_sentence_expect": "Expect {temp_c}C and {condition} in {city}.",
    "gt_sentence_will_be": "{city} will be {temp_c}C and {condition}.",
    "gt_sentence_forecast": "The forecast for {city}: {temp_c}C and {condition}.",
    # Variants (may partially match)
    "var_minimal": "{temp_c}C {condition} {city}.",
    "var_dash": "{city} — {temp_c}C, {condition}.",
    "var_comma": "{city}, {temp_c}C, {condition}.",
    "var_and_then": "{city} with {temp_c}C and {condition}.",
    "var_showing": "{city} showing {temp_c}C, {condition}.",
    "var_reads": "{city} reads {temp_c}C with {condition}.",
}

# The search space for evolution
SEARCH_SPACE = {
    "template_key": list(TEMPLATES.keys()),
    "include_period": [True, False],
    "capitalize_first": [True, False],
}


class FormatPolicy:
    """Policy that formats weather data using a template config."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.policy_id = f"format_{self.config.get('template_key', 'concise_city_first')}"

    def initialize(self, world_spec):
        return {}

    def act(self, obs: dict, actions: tuple[ActionSpec, ...], pstate: dict):
        if not actions:
            return None

        action = actions[0]  # FORMAT_RESPONSE
        payload = action.payload

        # Get template
        template_key = self.config.get("template_key", "gt_concise_city_first")
        template = TEMPLATES.get(template_key, TEMPLATES["gt_concise_city_first"])

        # Format
        response = template.format(
            temp_c=payload["temp_c"],
            condition=payload["condition"],
            city=payload["city"],
        )

        # Apply post-processing
        if self.config.get("include_period", True) and not response.endswith("."):
            response += "."
        elif not self.config.get("include_period", True) and response.endswith("."):
            response = response.rstrip(".")

        if self.config.get("capitalize_first", True) and response:
            response = response[0].upper() + response[1:]

        return PolicyDecision(
            action=ActionSpec(
                kind="FORMAT_RESPONSE",
                payload={**payload, "response": response},
                executor_kind="deterministic",
            ),
            rationale=f"template={template_key}",
        )
