"""
What we pull out of a claim description, as a strict schema.

Free text like "Burst pipe in the bathroom flooded the hallway." becomes:
    damage_type=water, urgency=high, injury_mentioned=False, ...

A claims handler can route on these fields: water damage to the home team,
high urgency to the top of the queue.

The field descriptions matter twice:
  * they're sent to the LLM, so it knows the rules for each field
  * they're the rules the evaluation cases (llm/evaluate.py) were labelled by
One definition, used in both places. Otherwise we'd measure the model against
rules it was never told.
"""

from typing import Literal

from pydantic import BaseModel, Field

DamageType = Literal[
    "collision", "theft", "water", "fire", "weather",
    "lost_luggage", "travel_disruption", "medical", "other",
]
Urgency = Literal["low", "medium", "high"]


class ClaimInfo(BaseModel):
    """The fields the LLM has to fill in. Literal types: anything outside the list fails validation."""

    damage_type: DamageType = Field(description=(
        "Main cause of the loss. collision: a vehicle hit something or was hit. "
        "theft: something was stolen or broken into. water: leaks, burst pipes, flooding from inside. "
        "fire: fire or smoke. weather: storm, lightning, hail, fallen trees, rain getting in. "
        "lost_luggage: bags lost, delayed or damaged in transit. travel_disruption: cancelled or "
        "delayed transport. medical: illness, injury or treatment. other: none of these."
    ))
    urgency: Urgency = Field(description=(
        "high: damage is still happening (water or rain still coming in, spreading fire) or someone "
        "needed emergency medical care. low: minor damage, or only a cost to pay back with nothing "
        "at risk now (cancelled flight, routine appointment, small dent or crack). medium: everything else."
    ))
    injury_mentioned: bool = Field(description=(
        "True if the text says a person was physically hurt (a fall, a broken bone, pain after an "
        "accident). Illness alone, like an infection, does not count."
    ))
    third_party_involved: bool = Field(description=(
        "True if another person caused the loss or was part of it: another driver, a neighbour, "
        "a thief. Companies like airlines do not count."
    ))
    summary: str = Field(max_length=200, description="One short sentence in English describing what happened.")


class Extraction(ClaimInfo):
    """What our code returns: the LLM's fields plus how we got them."""

    extractor: str = Field(description='Which extractor produced this, e.g. "llm:qwen2.5:3b" or "fake"')
    needs_manual_review: bool = Field(
        default=False,
        description="True if the extractor couldn't produce a valid answer and a person must read the text",
    )
