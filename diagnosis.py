from typing import Any, Dict, List, Optional

from answer_utils import answers_match


def empty_likelihood_post_info() -> Dict[str, Any]:
    """Default post-sufficiency likelihood summary."""
    return {
        "post_raw_gains": [],
        "raw_mean_gain": None,
        "raw_productivity_label": None,
        "mean_abs_gain": None,
        "last_window_mean_gain": None,
        "last_window_mean_abs_gain": None,
        "plateau": False,
        "productivity_label": None,
        "has_post_sufficiency": False,
        "stagnant_or_regressive": None,
    }


def empty_content_post_info() -> Dict[str, Any]:
    """Default post-sufficiency content-length summary."""
    return {
        "content_gains": [],
        "content_mean_gain": None,
        "content_last_window_mean_abs_gain": None,
        "content_length_plateau": False,
        "content_length_label": None,
        "content_truncated_after_sufficiency": False,
    }


def compute_stability(
    induced_answers: List[str],
    gold_answer: Optional[str] = None,
    final_answer: Optional[str] = None,
    window_ratio: float = 0.3,
    min_window: int = 3,
    threshold: float = 0.8,
    reference: str = "final",
) -> Dict[str, Any]:
    """
    Closed-answer stability based on final-window answer agreement.

    reference="final": final-window induced answers should agree with the model final answer.
    reference="gold": final-window induced answers should agree with the gold answer.
    """
    n = len(induced_answers)
    if n == 0:
        return {
            "stable": False,
            "agreement_ratio": 0.0,
            "window_size": 0,
            "reference_answer": None,
        }

    w = max(min_window, int(round(window_ratio * n)))
    w = min(w, n)
    window = induced_answers[-w:]

    ref = gold_answer if reference == "gold" else final_answer
    if not ref:
        return {
            "stable": False,
            "agreement_ratio": 0.0,
            "window_size": w,
            "reference_answer": ref,
        }

    agree = [answers_match(ans, ref) for ans in window]
    ratio = sum(agree) / len(agree)

    return {
        "stable": ratio >= threshold,
        "agreement_ratio": ratio,
        "window_size": w,
        "reference_answer": ref,
    }


def find_earliest_sufficient_step(
    induced_answers: List[str],
    gold_answer: str,
    consecutive_required: int = 3,
) -> Optional[int]:
    """
    Persistent earliest sufficient prefix k*.

    Return the first step where the induced answer matches the gold answer
    for `consecutive_required` consecutive prefixes.
    """
    n = len(induced_answers)

    for i in range(n):
        window = induced_answers[i:i + consecutive_required]

        if len(window) < consecutive_required:
            continue

        if all(answers_match(ans, gold_answer) for ans in window):
            return i + 1  # convert to 1-indexed

    return None


def compute_stepwise_gain(values: List[Optional[float]]) -> List[Optional[float]]:
    """values -> [None, value_2 - value_1, value_3 - value_2, ...]."""
    gains: List[Optional[float]] = [None]
    for i in range(1, len(values)):
        if values[i] is None or values[i - 1] is None:
            gains.append(None)
        else:
            gains.append(values[i] - values[i - 1])
    return gains


def label_scalar_gain(value: Optional[float], epsilon: float = 0.01) -> Optional[str]:
    """
    Label one scalar gain value using tolerance threshold phi.

    productive:  value > epsilon
    stagnant:   abs(value) <= epsilon
    regressive: value < -epsilon
    """
    if value is None:
        return None
    if value > epsilon:
        return "productive"
    if value < -epsilon:
        return "regressive"
    return "stagnant"


def classify_post_sufficiency_gain(
    gains: List[Optional[float]],
    k_star: Optional[int],
    epsilon: float = 0.01,
    stable_window: int = 3,
) -> Dict[str, Any]:
    """
    Classify post-sufficiency reasoning using a plateau-style productivity test.

    Important indexing:
    - k_star is 1-indexed.
    - gains[0] is None.
    - gains[k_star:] are gains AFTER the sufficient step.
    """
    empty = empty_likelihood_post_info()

    if k_star is None:
        return empty

    post_raw_gains = [g for g in gains[k_star:] if g is not None]
    if len(post_raw_gains) == 0:
        return empty

    raw_mean_gain = sum(post_raw_gains) / len(post_raw_gains)
    raw_productivity_label = label_scalar_gain(raw_mean_gain, epsilon=epsilon)

    mean_abs_gain = sum(abs(g) for g in post_raw_gains) / len(post_raw_gains)

    stable_window = max(1, min(stable_window, len(post_raw_gains)))
    last_window = post_raw_gains[-stable_window:]
    last_window_mean_gain = sum(last_window) / len(last_window)
    last_window_mean_abs_gain = sum(abs(g) for g in last_window) / len(last_window)

    plateau = last_window_mean_abs_gain <= epsilon

    if raw_mean_gain < -epsilon:
        productivity_label = "regressive"
    elif plateau:
        productivity_label = "stagnant"
    elif raw_mean_gain > epsilon:
        productivity_label = "productive"
    else:
        productivity_label = "borderline"

    return {
        "has_post_sufficiency": True,
        "post_raw_gains": post_raw_gains,
        "raw_mean_gain": raw_mean_gain,
        "raw_productivity_label": raw_productivity_label,
        "mean_abs_gain": mean_abs_gain,
        "last_window_mean_gain": last_window_mean_gain,
        "last_window_mean_abs_gain": last_window_mean_abs_gain,
        "plateau": plateau,
        "productivity_label": productivity_label,
        "stagnant_or_regressive": productivity_label in {"stagnant", "regressive"},
    }


def classify_post_sufficiency_content_length(
    content_lengths: List[Optional[float]],
    content_truncated: List[bool],
    k_star: Optional[int],
    epsilon: float = 1.0,
    stable_window: int = 3,
) -> Dict[str, Any]:
    """
    Diagnose whether induced content length has stopped changing after sufficiency.

    Interpretation:
    - decreasing: later thinking is replacing answer-side reasoning.
    - plateau: content length is stable, consistent with convergence/redundancy.
    - increasing: forced content is still expanding or generation is unstable.
    Truncated content is excluded from gain estimates because the true length is
    only lower-bounded by max_new_tokens.
    """
    empty = empty_content_post_info()

    if k_star is None or not content_lengths:
        return empty

    gains = compute_stepwise_gain(content_lengths)

    # gains[k_star:] are changes after the sufficient step, matching
    # classify_post_sufficiency_gain indexing.
    post_gains = []
    truncated_after = False
    for idx in range(k_star, len(gains)):
        if idx < len(content_truncated) and (content_truncated[idx] or content_truncated[idx - 1]):
            truncated_after = True
            continue
        if gains[idx] is not None:
            post_gains.append(gains[idx])

    if not post_gains:
        return {
            **empty,
            "content_truncated_after_sufficiency": truncated_after,
        }

    mean_gain = sum(post_gains) / len(post_gains)
    stable_window = max(1, min(stable_window, len(post_gains)))
    last_window = post_gains[-stable_window:]
    last_window_mean_abs_gain = sum(abs(g) for g in last_window) / len(last_window)
    plateau = last_window_mean_abs_gain <= epsilon

    if plateau:
        label = "plateau"
    elif mean_gain < -epsilon:
        label = "decreasing"
    elif mean_gain > epsilon:
        label = "increasing"
    else:
        label = "borderline"

    return {
        "content_gains": post_gains,
        "content_mean_gain": mean_gain,
        "content_last_window_mean_abs_gain": last_window_mean_abs_gain,
        "content_length_plateau": plateau,
        "content_length_label": label,
        "content_truncated_after_sufficiency": truncated_after,
    }


def analyze_trajectory(
    example_result: Dict[str, Any],
    stability_reference: str = "final",
    stability_threshold: float = 0.8,
    gain_epsilon: float = 0.01,
    stable_window: int = 3,
) -> Dict[str, Any]:
    """
    Diagnose one reasoning trajectory.

    Logic:
    1. First filter unresolved / unstable trajectories.
    2. Only stable and resolved trajectories enter correctness / overthinking diagnosis.
    3. For final-correct trajectories, use gold-answer likelihoods.
    4. For final-incorrect trajectories, use gold-vs-final contrastive margins.
    5. Overthinking is diagnosed only after sufficient correct prefixes.
    """

    gold_answer = example_result["gold_answer"]
    final_answer = example_result["final_answer"]
    results = example_result["results"]

    induced_answers = [row["induced_answer"] for row in results]
    gold_lls = [row.get("gold_answer_likelihood") for row in results]
    margins = [row.get("gold_vs_final_margin") for row in results]
    content_lengths = [row.get("content_token_len") for row in results]
    content_truncated = [bool(row.get("content_truncated")) for row in results]

    k_star = find_earliest_sufficient_step(
        induced_answers=induced_answers,
        gold_answer=gold_answer,
    )

    final_answer_missing = not final_answer or not str(final_answer).strip()

    empty_post_info = empty_likelihood_post_info()
    empty_content_info = empty_content_post_info()

    if final_answer_missing:
        gains = compute_stepwise_gain(gold_lls)
        return {
            "resolved": False,
            "unstable_reason": "no_final_answer_or_truncated_reasoning",
            "final_answer_missing": True,
            "final_correct": None,
            "stable": False,
            "stability_agreement_ratio": 0.0,
            "stability_window_size": None,
            "earliest_sufficient_step": k_star,
            "correctness_measure": "gold_answer_likelihood_no_final_answer",
            "stepwise_gains": gains,
            "post_sufficiency_raw_gains": empty_post_info["post_raw_gains"],
            "post_sufficiency_raw_mean_gain": empty_post_info["raw_mean_gain"],
            "post_sufficiency_raw_productivity_label": empty_post_info["raw_productivity_label"],
            "post_sufficiency_mean_abs_gain": empty_post_info["mean_abs_gain"],
            "post_sufficiency_last_window_mean_gain": empty_post_info["last_window_mean_gain"],
            "post_sufficiency_last_window_mean_abs_gain": empty_post_info["last_window_mean_abs_gain"],
            "post_sufficiency_plateau": empty_post_info["plateau"],
            "post_sufficiency_productivity_label": empty_post_info["productivity_label"],
            "has_post_sufficiency": False,
            "post_sufficiency_content_gains": empty_content_info["content_gains"],
            "post_sufficiency_content_mean_gain": empty_content_info["content_mean_gain"],
            "post_sufficiency_content_last_window_mean_abs_gain": empty_content_info["content_last_window_mean_abs_gain"],
            "content_length_plateau": empty_content_info["content_length_plateau"],
            "post_sufficiency_content_length_label": empty_content_info["content_length_label"],
            "content_truncated_after_sufficiency": empty_content_info["content_truncated_after_sufficiency"],
            "overthinking": False,
            "diagnosis": "unstable_incomplete_reasoning",
        }

    final_correct = answers_match(final_answer, gold_answer)

    stability_info = compute_stability(
        induced_answers=induced_answers,
        gold_answer=gold_answer,
        final_answer=final_answer,
        threshold=stability_threshold,
        reference=stability_reference,
    )

    if not stability_info["stable"]:
        if final_correct:
            correctness_values = gold_lls
            correctness_measure = "gold_answer_likelihood"
        else:
            correctness_values = margins
            correctness_measure = "gold_vs_final_margin"

        gains = compute_stepwise_gain(correctness_values)
        return {
            "resolved": False,
            "unstable_reason": "low_final_window_agreement",
            "final_answer_missing": False,
            "final_correct": final_correct,
            "stable": False,
            "stability_agreement_ratio": stability_info["agreement_ratio"],
            "stability_window_size": stability_info["window_size"],
            "earliest_sufficient_step": k_star,
            "correctness_measure": correctness_measure,
            "stepwise_gains": gains,
            "post_sufficiency_raw_gains": empty_post_info["post_raw_gains"],
            "post_sufficiency_raw_mean_gain": empty_post_info["raw_mean_gain"],
            "post_sufficiency_raw_productivity_label": empty_post_info["raw_productivity_label"],
            "post_sufficiency_mean_abs_gain": empty_post_info["mean_abs_gain"],
            "post_sufficiency_last_window_mean_gain": empty_post_info["last_window_mean_gain"],
            "post_sufficiency_last_window_mean_abs_gain": empty_post_info["last_window_mean_abs_gain"],
            "post_sufficiency_plateau": empty_post_info["plateau"],
            "post_sufficiency_productivity_label": empty_post_info["productivity_label"],
            "has_post_sufficiency": False,
            "post_sufficiency_content_gains": empty_content_info["content_gains"],
            "post_sufficiency_content_mean_gain": empty_content_info["content_mean_gain"],
            "post_sufficiency_content_last_window_mean_abs_gain": empty_content_info["content_last_window_mean_abs_gain"],
            "content_length_plateau": empty_content_info["content_length_plateau"],
            "post_sufficiency_content_length_label": empty_content_info["content_length_label"],
            "content_truncated_after_sufficiency": empty_content_info["content_truncated_after_sufficiency"],
            "overthinking": False,
            "diagnosis": "unstable_reasoning",
        }

    if final_correct:
        correctness_values = gold_lls
        correctness_measure = "gold_answer_likelihood"
    else:
        correctness_values = margins
        correctness_measure = "gold_vs_final_margin"

    gains = compute_stepwise_gain(correctness_values)

    post_info = classify_post_sufficiency_gain(
        gains=gains,
        k_star=k_star,
        epsilon=gain_epsilon,
        stable_window=stable_window,
    )
    content_info = classify_post_sufficiency_content_length(
        content_lengths=content_lengths,
        content_truncated=content_truncated,
        k_star=k_star,
        epsilon=1.0,
        stable_window=stable_window,
    )

    overthinking = (
        final_correct
        and stability_info["stable"]
        and k_star is not None
        and post_info["has_post_sufficiency"]
        and post_info["stagnant_or_regressive"]
        and content_info["content_length_label"] in {"plateau", "borderline", "decreasing"}
    )

    if overthinking:
        diagnosis = "overthinking_after_sufficiency"
    elif final_correct and k_star is not None:
        diagnosis = "sufficient_but_productive_or_short"
    elif (not final_correct) and k_star is not None:
        diagnosis = "lost_correct_answer_after_sufficiency"
    elif (not final_correct) and stability_info["stable"] and k_star is None:
        diagnosis = "stable_wrong_convergence"
    elif not final_correct:
        diagnosis = "incorrect_without_sufficiency"
    else:
        diagnosis = "undetermined"

    return {
        "resolved": True,
        "unstable_reason": None,
        "final_answer_missing": False,
        "final_correct": final_correct,
        "stable": stability_info["stable"],
        "stability_agreement_ratio": stability_info["agreement_ratio"],
        "stability_window_size": stability_info["window_size"],
        "earliest_sufficient_step": k_star,
        "correctness_measure": correctness_measure,
        "stepwise_gains": gains,
        "post_sufficiency_raw_gains": post_info["post_raw_gains"],
        "post_sufficiency_raw_mean_gain": post_info["raw_mean_gain"],
        "post_sufficiency_raw_productivity_label": post_info["raw_productivity_label"],
        "post_sufficiency_mean_abs_gain": post_info["mean_abs_gain"],
        "post_sufficiency_last_window_mean_gain": post_info["last_window_mean_gain"],
        "post_sufficiency_last_window_mean_abs_gain": post_info["last_window_mean_abs_gain"],
        "post_sufficiency_plateau": post_info["plateau"],
        "post_sufficiency_productivity_label": post_info["productivity_label"],
        "has_post_sufficiency": post_info["has_post_sufficiency"],
        "post_sufficiency_content_gains": content_info["content_gains"],
        "post_sufficiency_content_mean_gain": content_info["content_mean_gain"],
        "post_sufficiency_content_last_window_mean_abs_gain": content_info["content_last_window_mean_abs_gain"],
        "content_length_plateau": content_info["content_length_plateau"],
        "post_sufficiency_content_length_label": content_info["content_length_label"],
        "content_truncated_after_sufficiency": content_info["content_truncated_after_sufficiency"],
        "overthinking": overthinking,
        "diagnosis": diagnosis,
    }
