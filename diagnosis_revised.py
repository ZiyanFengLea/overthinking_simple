from typing import Any, Dict, List, Optional

from answer_utils import answers_match


def safe_float(value: Any) -> Optional[float]:
    """Convert numeric-like values to float; return None for missing/bad values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile for q in [0, 1]."""
    if not values:
        return None

    q = max(0.0, min(1.0, q))
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def find_stability_step(
    results: List[Dict[str, Any]],
    gold_answer: str,
    tail_percentile: float = 0.1,
    consecutive_required: int = 2,
) -> Dict[str, Any]:
    """
    Earliest stability/post-sufficiency point.

    Stability begins at the smallest index where `consecutive_required`
    consecutive prefixes have low-tail content length and correct induced answers.
    """
    valid_lengths = [
        length
        for row in results
        for length in [safe_float(row.get("content_token_len"))]
        if length is not None
    ]
    threshold = percentile(valid_lengths, tail_percentile)

    if threshold is None:
        return {
            "stable": False,
            "stability_step": None,
            "stability_content_token_lens": [],
        }

    consecutive_required = max(1, consecutive_required)
    for idx in range(0, len(results) - consecutive_required + 1):
        window = results[idx:idx + consecutive_required]
        content_lens = [safe_float(row.get("content_token_len")) for row in window]
        induced_answers = [row.get("induced_answer") for row in window]

        if any(content_len is None for content_len in content_lens):
            continue

        low_tail = all(content_len <= threshold for content_len in content_lens)
        correct = all(answers_match(answer, gold_answer) for answer in induced_answers)

        if low_tail and correct:
            return {
                "stable": True,
                "stability_step": window[0].get("step", idx + 1),
                "stability_content_token_lens": content_lens,
            }

    return {
        "stable": False,
        "stability_step": None,
        "stability_content_token_lens": [],
    }


def likelihood_values(
    results: List[Dict[str, Any]],
    final_correct: bool,
) -> List[Optional[float]]:
    """Use gold likelihood if final is correct; otherwise use gold-vs-final margin."""
    key = "gold_answer_likelihood" if final_correct else "gold_vs_final_margin"
    return [safe_float(row.get(key)) for row in results]


def stepwise_gains(values: List[Optional[float]]) -> List[Optional[float]]:
    """values -> [None, value_2 - value_1, value_3 - value_2, ...]."""
    gains: List[Optional[float]] = [None]
    for idx in range(1, len(values)):
        if values[idx] is None or values[idx - 1] is None:
            gains.append(None)
        else:
            gains.append(values[idx] - values[idx - 1])
    return gains


def find_overthinking_steps(
    results: List[Dict[str, Any]],
    final_correct: bool,
    stability_step: Optional[int],
    tail_percentile: float = 0.1,
) -> Dict[str, Any]:
    """
    Mark post-stability steps whose correctness gain is non-positive.

    The global low-tail threshold is still reported as a severity reference.
    """
    values = likelihood_values(results, final_correct)
    gains = stepwise_gains(values)
    valid_gains = [gain for gain in gains if gain is not None]
    threshold = percentile(valid_gains, tail_percentile)

    if stability_step is None:
        return {
            "likelihood_gain_tail_threshold": threshold,
            "overthinking_steps": [],
        }

    overthinking_steps = []
    for idx, gain in enumerate(gains):
        if gain is None:
            continue

        step = results[idx].get("step", idx + 1)
        if step >= stability_step and gain <= 0:
            overthinking_steps.append(step)

    return {
        "likelihood_gain_tail_threshold": threshold,
        "overthinking_steps": overthinking_steps,
    }


def analyze_trajectory(
    example_result: Dict[str, Any],
    stability_tail_percentile: float = 0.1,
    stability_consecutive_required: int = 2,
    likelihood_tail_percentile: float = 0.1,
) -> Dict[str, Any]:
    """
    Diagnose one reasoning trajectory.

    Stability/post-sufficiency is the earliest region where consecutive prefixes
    have low-tail content length and correct induced answers.
    """

    gold_answer = example_result["gold_answer"]
    final_answer = example_result["final_answer"]
    results = example_result["results"]

    final_answer_missing = not final_answer or not str(final_answer).strip()

    stability_info = find_stability_step(
        results=results,
        gold_answer=gold_answer,
        tail_percentile=stability_tail_percentile,
        consecutive_required=stability_consecutive_required,
    )

    if final_answer_missing:
        return {
            "resolved": False,
            "unstable_reason": "no_final_answer_or_truncated_reasoning",
            "final_answer_missing": True,
            "final_correct": None,
            **stability_info,
            "likelihood_gain_tail_threshold": None,
            "overthinking_steps": [],
            "overthinking": False,
            "diagnosis": "unstable_incomplete_reasoning",
        }

    final_correct = answers_match(final_answer, gold_answer)
    likelihood_info = find_overthinking_steps(
        results=results,
        final_correct=final_correct,
        stability_step=stability_info["stability_step"],
        tail_percentile=likelihood_tail_percentile,
    )
    overthinking = bool(likelihood_info["overthinking_steps"])

    if overthinking:
        diagnosis = "overthinking_after_sufficiency"
    elif final_correct:
        diagnosis = "correct_without_overthinking"
    elif stability_info["stable"]:
        diagnosis = "lost_correct_answer_after_sufficiency"
    else:
        diagnosis = "incorrect_without_sufficiency"

    return {
        "resolved": True,
        "unstable_reason": None,
        "final_answer_missing": False,
        "final_correct": final_correct,
        **stability_info,
        **likelihood_info,
        "overthinking": overthinking,
        "diagnosis": diagnosis,
    }
