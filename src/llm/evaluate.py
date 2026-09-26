"""
Measure how well an extractor reads claim descriptions, and log it to MLflow.

"How do you know the LLM works?" "I measured it." This script is that answer:
a set of descriptions with the right answers written down, the extractor's
answers compared field by field, and the results logged next to the fraud
model's runs so every model and prompt can be compared.

Run it (the model comes from LLM_MODEL in .env):
    python -m llm.evaluate                         # the configured LLM
    python -m llm.evaluate --fake                  # the keyword-rules baseline
    LLM_MODEL=gemma3:1b python -m llm.evaluate     # another model, same cases
"""

import argparse
import time

import mlflow
import pandas as pd

from llm.extract import PROMPT_VERSION, Extractor, FakeExtractor, get_extractor
from model.train import DEFAULT_TRACKING_URI

EXPERIMENT = "claims-llm-extraction"

# The fields we score. summary is free text, so it has no single right answer.
FIELDS = ["damage_type", "urgency", "injury_mentioned", "third_party_involved"]

# Labelled by the rules in llm/schema.py. The first 18 are the generator's
# templates (what the real data contains). The rest are harder cases the
# generator never produces, to see how the extractor copes with new text.
# The two groups are scored separately too: doing well on the templates but
# badly on new text is exactly what hand-written rules look like.
N_TEMPLATES = 18
CASES = [
    # --- the generator's templates ---
    {"description": "Rear-ended at a traffic light, bumper and trunk damaged.",
     "damage_type": "collision", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Hit a pole while parking, front left headlight broken.",
     "damage_type": "collision", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Windshield cracked by a stone on the highway.",
     "damage_type": "collision", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Car was broken into overnight, window smashed and radio stolen.",
     "damage_type": "theft", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Collision at a roundabout, other driver did not stop. Door dented.",
     "damage_type": "collision", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Water leak from the upstairs neighbour, kitchen ceiling stained.",
     "damage_type": "water", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Burst pipe in the bathroom flooded the hallway.",
     "damage_type": "water", "urgency": "high", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Storm blew tiles off the roof, rain coming in to the bedroom.",
     "damage_type": "weather", "urgency": "high", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Break-in while on holiday, laptop and jewellery missing.",
     "damage_type": "theft", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Small kitchen fire, cabinets and extractor fan damaged.",
     "damage_type": "fire", "urgency": "medium", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Luggage lost on connecting flight, still not returned after 5 days.",
     "damage_type": "lost_luggage", "urgency": "medium", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Flight cancelled, had to pay for an extra hotel night.",
     "damage_type": "travel_disruption", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Phone stolen at the train station during the trip.",
     "damage_type": "theft", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Needed a doctor abroad for a stomach infection.",
     "damage_type": "medical", "urgency": "medium", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Emergency room visit after a fall, wrist X-ray and cast.",
     "damage_type": "medical", "urgency": "high", "injury_mentioned": True, "third_party_involved": False},
    {"description": "Physiotherapy sessions after knee surgery.",
     "damage_type": "medical", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Dental emergency, broken tooth needed a crown.",
     "damage_type": "medical", "urgency": "high", "injury_mentioned": True, "third_party_involved": False},
    {"description": "Specialist consultation and blood tests.",
     "damage_type": "medical", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    # --- harder cases the generator never produces ---
    {"description": "Rear-ended on the motorway, I have neck pain and went to A&E.",
     "damage_type": "collision", "urgency": "high", "injury_mentioned": True, "third_party_involved": True},
    {"description": "Someone hit my parked car and left a note with their number. Scratch on the door.",
     "damage_type": "collision", "urgency": "low", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Lightning struck the house, the TV and router are fried.",
     "damage_type": "weather", "urgency": "medium", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Washing machine hose split, water all over the kitchen floor and it is still leaking.",
     "damage_type": "water", "urgency": "high", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Grease fire in the kitchen spread to the ceiling, the fire brigade came.",
     "damage_type": "fire", "urgency": "high", "injury_mentioned": False, "third_party_involved": False},
    # A car, but the cause is weather
    {"description": "A tree fell on the car during the storm and crushed the roof.",
     "damage_type": "weather", "urgency": "medium", "injury_mentioned": False, "third_party_involved": False},
    {"description": "My son fell off his bike and broke his arm, X-ray and cast at the hospital.",
     "damage_type": "medical", "urgency": "high", "injury_mentioned": True, "third_party_involved": False},
    # "late" sounds like travel disruption, but it's about the bag
    {"description": "Suitcase arrived two days late with a broken handle.",
     "damage_type": "lost_luggage", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    {"description": "Train was 3 hours late, missed the connection and had to buy a new ticket.",
     "damage_type": "travel_disruption", "urgency": "low", "injury_mentioned": False, "third_party_involved": False},
    # Spanish: "My phone was stolen on the metro." Real claims in Barcelona won't all be in English.
    {"description": "Me han robado el móvil en el metro.",
     "damage_type": "theft", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
    {"description": "Bag with my passport was stolen from the hotel room.",
     "damage_type": "theft", "urgency": "medium", "injury_mentioned": False, "third_party_involved": True},
]


def evaluate(extractor: Extractor, cases: list[dict] = CASES) -> tuple[dict, pd.DataFrame]:
    """Run the extractor on every case. Returns (metrics, one row per case with expected vs got)."""
    rows = []
    for i, case in enumerate(cases):
        start = time.perf_counter()
        got = extractor.extract(case["description"])
        seconds = time.perf_counter() - start
        print(f"  {seconds:5.1f}s  {case['description'][:60]}")

        row = {"description": case["description"], "group": "template" if i < N_TEMPLATES else "new_text",
               "seconds": round(seconds, 2),
               "needs_manual_review": got.needs_manual_review}
        for field in FIELDS:
            row[f"expected_{field}"] = case[field]
            row[f"got_{field}"] = getattr(got, field)
            row[f"{field}_correct"] = case[field] == getattr(got, field)
        row["all_correct"] = all(row[f"{f}_correct"] for f in FIELDS)
        row["summary"] = got.summary
        rows.append(row)

    results = pd.DataFrame(rows)
    metrics = {f"accuracy_{field}": results[f"{field}_correct"].mean() for field in FIELDS}
    metrics["accuracy_all_fields"] = results["all_correct"].mean()
    for group, part in results.groupby("group"):
        metrics[f"accuracy_all_fields_{group}"] = part["all_correct"].mean()
    metrics["manual_review_rate"] = results["needs_manual_review"].mean()
    metrics["seconds_per_claim"] = results["seconds"].mean()
    return {k: round(float(v), 4) for k, v in metrics.items()}, results


def run(extractor: Extractor, tracking_uri: str = DEFAULT_TRACKING_URI) -> dict:
    """Evaluate one extractor and log it as an MLflow run. Returns the metrics."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    print(f"Evaluating {extractor.name} on {len(CASES)} cases...")
    with mlflow.start_run(run_name=extractor.name):
        mlflow.log_params({
            "extractor": extractor.name,
            "model": getattr(extractor, "model", "none"),
            # So every score can be traced back to the exact prompt that produced it
            "prompt_version": PROMPT_VERSION,
            "n_cases": len(CASES),
        })
        # Warm-up, not timed. The first call to a local model includes loading
        # it into memory (~30 s on this laptop). A running service pays that
        # once, not per claim, so it shouldn't count in seconds_per_claim.
        extractor.extract("Warm-up call, not part of the evaluation.")
        metrics, results = evaluate(extractor)
        mlflow.log_metrics(metrics)
        # Every case, expected vs got. The first place to look when a score drops.
        mlflow.log_table(results, "results.json")

    print(f"\n{extractor.name}")
    for name, value in metrics.items():
        print(f"  {name:28s} {value}")
    wrong = results[~results["all_correct"]]
    if len(wrong):
        print(f"\nCases with at least one wrong field ({len(wrong)}):")
        for _, row in wrong.iterrows():
            misses = [f"{f}: expected {row[f'expected_{f}']}, got {row[f'got_{f}']}"
                      for f in FIELDS if not row[f"{f}_correct"]]
            print(f"  {row['description'][:55]:55s} | " + "; ".join(misses))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate claim-description extraction.")
    parser.add_argument("--fake", action="store_true", help="Evaluate the keyword-rules baseline")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    run(FakeExtractor() if args.fake else get_extractor(), args.tracking_uri)


if __name__ == "__main__":
    main()
