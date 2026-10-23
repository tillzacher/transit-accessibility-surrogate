"""Background-controlled opportunity attenuation."""

from modules.opportunity_effects import (
    SENSITIVITY_GATE,
    score_attenuation,
    summarize_increment,
)

VERSION = "opportunity-attenuation-1"
__all__ = ["VERSION", "SENSITIVITY_GATE", "score_attenuation", "summarize_increment"]
