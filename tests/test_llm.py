"""
Tests for the LLM extraction (step 5).

None of these call a real LLM. Tests have to be fast, free, work offline and
give the same result every time, and a real model is none of those. The real
extractor is tested with a stand-in client that returns canned answers.
"""

import mlflow
import pytest
from openai import APITimeoutError
from pydantic import ValidationError

from generate_claims import DESCRIPTIONS
from llm import evaluate
from llm.evaluate import CASES, FIELDS, N_TEMPLATES
from llm.extract import FakeExtractor, LLMExtractor, get_extractor
from llm.schema import ClaimInfo


# --- The schema ---------------------------------------------------------------

def test_schema_rejects_a_damage_type_outside_the_list():
    """If an LLM answers "flooding" instead of "water", it must fail, not slip through."""
    with pytest.raises(ValidationError):
        ClaimInfo(damage_type="flooding", urgency="high", injury_mentioned=False,
                  third_party_involved=False, summary="Flooded hallway.")


def test_schema_rejects_a_long_summary():
    with pytest.raises(ValidationError):
        ClaimInfo(damage_type="water", urgency="high", injury_mentioned=False,
                  third_party_involved=False, summary="x" * 201)


# --- Fake extractor -----------------------------------------------------------

@pytest.mark.parametrize("description, damage_type, urgency", [
    ("Burst pipe in the bathroom flooded the hallway.", "water", "high"),
    ("Car was broken into overnight, window smashed and radio stolen.", "theft", "medium"),
    ("Flight cancelled, had to pay for an extra hotel night.", "travel_disruption", "low"),
    ("Small kitchen fire, cabinets and extractor fan damaged.", "fire", "medium"),
    ("Something odd happened.", "other", "medium"),
])
def test_fake_extractor_keyword_rules(description, damage_type, urgency):
    result = FakeExtractor().extract(description)
    assert (result.damage_type, result.urgency) == (damage_type, urgency)
    assert result.extractor == "fake"
    assert result.needs_manual_review is False


def test_fake_extractor_flags_injury_and_third_party():
    assert FakeExtractor().extract("Emergency room visit after a fall, wrist X-ray and cast.").injury_mentioned
    assert FakeExtractor().extract("Water leak from the upstairs neighbour.").third_party_involved


# --- Real extractor, with a stand-in client -----------------------------------

GOOD_ANSWER = (
    '{"damage_type": "water", "urgency": "high", "injury_mentioned": false,'
    ' "third_party_involved": false, "summary": "A burst pipe flooded the hallway."}'
)


class StandInClient:
    """
    Looks like openai.OpenAI() to LLMExtractor, but returns canned answers in
    order and records what it was asked. An Exception in the list is raised
    instead, like a timeout from a real server.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        self.chat = self                 # so client.chat.completions.create(...) works
        self.completions = self

    def create(self, **request):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        message = type("Message", (), {"content": answer})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


def make_extractor(*answers):
    client = StandInClient(answers)
    return LLMExtractor(model="test-model", client=client), client


def test_valid_answer_is_parsed():
    extractor, client = make_extractor(GOOD_ANSWER)
    result = extractor.extract("Burst pipe in the bathroom flooded the hallway.")

    assert result.damage_type == "water"
    assert result.extractor == "llm:test-model"
    assert result.needs_manual_review is False
    # What we sent: temperature 0 and the JSON schema
    assert client.requests[0]["temperature"] == 0
    assert client.requests[0]["response_format"]["type"] == "json_schema"


def test_invalid_answer_is_retried_with_the_error():
    extractor, client = make_extractor("this is not json", GOOD_ANSWER)
    result = extractor.extract("Burst pipe.")

    assert result.damage_type == "water"
    assert len(client.requests) == 2
    retry_messages = client.requests[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "this is not json"}
    assert "invalid" in retry_messages[-1]["content"]


def test_answer_outside_the_schema_counts_as_invalid():
    """Valid JSON, but "flooding" isn't an allowed damage_type."""
    bad = GOOD_ANSWER.replace('"water"', '"flooding"')
    extractor, _ = make_extractor(bad, bad)
    assert extractor.extract("Burst pipe.").needs_manual_review is True


def test_two_failures_give_a_fallback_for_manual_review():
    extractor, client = make_extractor("junk", "more junk", GOOD_ANSWER)
    result = extractor.extract("Burst pipe.")

    assert result.needs_manual_review is True
    assert result.damage_type == "other"
    # The fallback isn't cached: asking again calls the model again, and works
    assert extractor.extract("Burst pipe.").needs_manual_review is False
    assert len(client.requests) == 3


def test_server_error_gives_the_fallback():
    extractor, _ = make_extractor(APITimeoutError(request=None), APITimeoutError(request=None))
    assert extractor.extract("Burst pipe.").needs_manual_review is True


def test_cache_avoids_a_second_call():
    extractor, client = make_extractor(GOOD_ANSWER)
    first = extractor.extract("Burst pipe in the bathroom flooded the hallway.")
    second = extractor.extract("  Burst pipe in the bathroom flooded the hallway.  ")   # same text, extra spaces

    assert first == second
    assert len(client.requests) == 1


# --- Choosing an extractor ----------------------------------------------------

def test_no_api_key_means_fake_extractor(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert isinstance(get_extractor(), FakeExtractor)


def test_api_key_and_model_mean_real_extractor(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    extractor = get_extractor()          # builds the client; doesn't call anything
    assert isinstance(extractor, LLMExtractor)
    assert extractor.name == "llm:qwen2.5:3b"


# --- Evaluation ---------------------------------------------------------------

def test_the_first_cases_are_the_generator_templates():
    """The template/new_text split is by position, so the first N_TEMPLATES must be exactly the templates."""
    templates = {text for texts in DESCRIPTIONS.values() for text in texts}
    assert {c["description"] for c in CASES[:N_TEMPLATES]} == templates


def test_every_expected_answer_is_a_valid_value():
    """A typo in the labels ("colision") would silently make every extractor look worse."""
    for case in CASES:
        ClaimInfo(**{f: case[f] for f in FIELDS}, summary="x")


def test_evaluation_scores_and_logs_to_mlflow(tmp_path):
    metrics = evaluate.run(FakeExtractor(), tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_tracking_uri(None)

    assert set(metrics) >= {"accuracy_damage_type", "accuracy_all_fields", "seconds_per_claim",
                            "accuracy_all_fields_template", "accuracy_all_fields_new_text"}
    assert all(0 <= metrics[f"accuracy_{f}"] <= 1 for f in FIELDS)
    # The keyword rules get the easy cases but not all the tricky ones
    assert 0.5 < metrics["accuracy_damage_type"] < 1
