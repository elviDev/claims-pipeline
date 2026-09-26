"""
Turn a free-text claim description into structured fields (llm/schema.py).

Two extractors with the same interface, extract(description) -> Extraction:

  FakeExtractor  keyword rules. Used by tests and CI (fast, free, offline,
                 same answer every time), when no LLM is configured, and as
                 the baseline the LLM has to beat.
  LLMExtractor   asks a real LLM through the openai package. Works with any
                 OpenAI-compatible server: Ollama locally, OpenAI, Azure OpenAI.

get_extractor() picks one from the environment, so the rest of the code
never needs to know which it has.
"""

import hashlib
import logging
import os
from typing import Protocol

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from llm.schema import ClaimInfo, Extraction

log = logging.getLogger("llm.extract")


class Extractor(Protocol):
    """Anything with a name and an extract() method counts as an extractor."""
    name: str

    def extract(self, description: str) -> Extraction: ...


# ---------------------------------------------------------------------------
# Fake extractor: keyword rules
# ---------------------------------------------------------------------------

# Every keyword comes from the generator's 18 description templates, and
# nothing else. That's what someone writing rules against the real data would
# have. The evaluation's harder cases are text these rules were NOT written
# for, which is the point: they show how rules cope with something new.
# (Adding keywords from the evaluation cases would be tuning on the test set,
# and the baseline would look better than it really is.)

# Checked in order; the first match wins. "stolen" comes before "trip", so
# "phone stolen during the trip" is theft, not travel disruption.
DAMAGE_KEYWORDS = [
    ("theft", ["stolen", "broken into", "break-in"]),
    ("lost_luggage", ["luggage"]),
    ("travel_disruption", ["cancelled"]),
    ("fire", ["fire"]),
    ("weather", ["storm", "rain"]),
    ("water", ["leak", "pipe", "flood"]),
    ("collision", ["rear-ended", "collision", "hit", "dented"]),
    ("medical", ["doctor", "x-ray", "physio", "dental", "tooth", "emergency room",
                 "specialist", "surgery", "infection"]),
]
HIGH_URGENCY = ["flooded", "coming in", "emergency"]
LOW_URGENCY = ["cancelled", "consultation", "physiotherapy", "headlight", "windshield"]
INJURY = ["a fall", "x-ray", "cast", "broken tooth"]
THIRD_PARTY = ["other driver", "neighbour", "stolen", "rear-ended", "break-in", "broken into"]


def _any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


class FakeExtractor:
    """
    Keyword rules. Deliberately simple: it will get tricky cases wrong
    ("tree fell on the car" is weather, not collision). The evaluation shows
    how often, and the LLM has to do better to be worth its cost.
    """
    name = "fake"

    def extract(self, description: str) -> Extraction:
        text = description.lower()
        damage = next((d for d, words in DAMAGE_KEYWORDS if _any(text, words)), "other")

        if _any(text, HIGH_URGENCY):
            urgency = "high"
        elif _any(text, LOW_URGENCY):
            urgency = "low"
        else:
            urgency = "medium"

        return Extraction(
            damage_type=damage,
            urgency=urgency,
            injury_mentioned=_any(text, INJURY),
            third_party_involved=_any(text, THIRD_PARTY),
            # First sentence, cut to the schema's limit
            summary=description.strip().split(".")[0][:200],
            extractor=self.name,
        )


# ---------------------------------------------------------------------------
# Real extractor: an LLM behind an OpenAI-compatible API
# ---------------------------------------------------------------------------

# Bump this whenever the prompt changes. The evaluation logs it to MLflow, so
# every accuracy number can be traced back to the exact prompt that produced it.
PROMPT_VERSION = "v1"


def _field_rules() -> str:
    """The rules for each field, taken from the schema descriptions (one definition, see schema.py)."""
    return "\n".join(f"- {name}: {field.description}" for name, field in ClaimInfo.model_fields.items())


SYSTEM_PROMPT = f"""You read insurance claim descriptions and fill in a form.
The description may be in any language; always answer in English.
Reply with a JSON object with exactly these fields:
{_field_rules()}
"""

# Sent with every request. Ollama and OpenAI use it to constrain what the model
# can generate, so the answer almost always has the right shape.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "claim_info", "schema": ClaimInfo.model_json_schema()},
}

DEFAULT_TIMEOUT_SECONDS = 180   # a 3B model on a laptop CPU can take a while


class LLMExtractor:
    def __init__(self, model: str, base_url: str | None = None, api_key: str = "none",
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, client=None):
        self.model = model
        self.name = f"llm:{model}"
        # max_retries=0: the openai package normally retries failed calls twice
        # by itself. On a CPU, that could turn one 3-minute timeout into 9
        # minutes. We do one retry ourselves in extract(), where we can see it.
        # `client` can be passed in so tests can use a stand-in with canned answers.
        self.client = client or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self._cache: dict[str, Extraction] = {}

    def extract(self, description: str) -> Extraction:
        # Same text, same answer (temperature 0), so don't pay for it twice.
        # The generator reuses 18 templates; real claims repeat a lot too.
        key = hashlib.sha256(description.strip().encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        for attempt in (1, 2):
            answer = None
            try:
                answer = self._ask(messages)
                info = ClaimInfo.model_validate_json(answer)
            except (OpenAIError, ValidationError) as err:
                log.warning("Extraction attempt %d failed: %s", attempt, str(err)[:300])
                if answer is not None:
                    # Temperature 0 means asking the same question again gives
                    # the same wrong answer. So the retry shows the model its
                    # answer and what was wrong with it.
                    messages += [
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": f"That answer was invalid: {err}. Reply again with corrected JSON."},
                    ]
                continue
            result = Extraction(**info.model_dump(), extractor=self.name)
            self._cache[key] = result
            return result

        # Two failures: never pass a guess downstream. Send it to a person.
        # Not cached, so a timeout now doesn't stick to this text forever.
        return Extraction(
            damage_type="other", urgency="medium", injury_mentioned=False,
            third_party_involved=False, summary="Could not be read automatically.",
            extractor=self.name, needs_manual_review=True,
        )

    def _ask(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,              # same description, same answer
            response_format=RESPONSE_FORMAT,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Pick one from the environment
# ---------------------------------------------------------------------------

def get_extractor() -> Extractor:
    """
    The LLM if LLM_API_KEY and LLM_MODEL are set (from .env), otherwise the
    fake one. So tests, CI, and anyone without an LLM can still run everything.
    """
    api_key, model = os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if not api_key or not model:
        return FakeExtractor()
    return LLMExtractor(
        model=model,
        base_url=os.getenv("LLM_BASE_URL") or None,     # None = OpenAI itself
        api_key=api_key,
        timeout=float(os.getenv("LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)),
    )
