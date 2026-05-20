import re
from typing import Any


NUMBER_WORDS = {
    # cardinal numbers
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",

    # ordinal words, useful for answers like "second place"
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
}


def remove_special_tokens(text: Any) -> str:
    """Remove common chat/special tokens from model text."""
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"<\|.*?\|>", "", text)
    for token in ("</s>", "<s>", "<bos>", "<eos>"):
        text = text.replace(token, "")
    return text.strip()


def remove_think_block(text: Any) -> str:
    """
    Remove hidden/reasoning content before answer extraction.

    Qwen-style continuations may contain only `</think>` because the opening
    `<think>` is in the prompt. In that case, keep only text after `</think>`.
    """
    text = remove_special_tokens(text)
    visible_text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    if visible_text == text.strip() and "</think>" in text:
        visible_text = text.split("</think>", 1)[1].strip()

    return visible_text if visible_text else text.strip()


def clean_for_compare(answer: Any) -> str:
    """Lightweight cleaning used for extraction and answer comparison."""
    if answer is None:
        return ""

    answer = remove_special_tokens(answer)
    answer = answer.replace("**", "")
    answer = answer.strip()
    answer = answer.lstrip(":").strip()

    boxed = re.search(r"\\*boxed\s*\{\s*([^{}]+?)\s*\}", answer)
    if boxed:
        answer = boxed.group(1).strip()

    answer = answer.splitlines()[0].strip() if answer else ""
    answer = answer.rstrip(".。")
    return answer.strip()


def clean_extracted_answer(answer: Any) -> str:
    """
    Clean a short answer extracted from model output.

    This is intentionally stricter than `clean_for_compare`: it trims common
    self-correction continuations so an induced answer stays compact.
    """
    answer = clean_for_compare(answer)

    cut_markers = [
        " Wait",
        " wait",
        "? But",
        "? but",
        ". But",
        ". but",
        " But",
        " but",
        " No,",
        " no,",
        " Let me",
        " let me",
        " So,",
        " so,",
        " Therefore",
        " therefore",
        " Step-by-step",
        " step-by-step",
    ]

    for marker in cut_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    return answer.rstrip(".。").strip()


def extract_final_answer(text: Any) -> str:
    """
    Extract a compact final answer from visible model output.

    Prefer explicit answer markers outside `<think>...</think>`. If no marker is
    found, return an empty string instead of guessing from a long explanation.
    """
    search_text = remove_think_block(text)

    patterns = [
        r"####\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"final answer is\s*[:：]?\s*([^\n]+)",
        r"the answer is\s*[:：]?\s*([^\n]+)",
        r"therefore,?\s*the answer is\s*[:：]?\s*([^\n]+)",
        r"so,?\s*the answer is\s*[:：]?\s*([^\n]+)",
        r"\\boxed\{([^{}]+)\}",
        r"boxed\{([^{}]+)\}",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, search_text, flags=re.IGNORECASE))
        if matches:
            return clean_extracted_answer(matches[-1].group(1))

    compact = clean_extracted_answer(search_text)
    if compact and "\n" not in search_text.strip() and len(compact.split()) <= 12:
        return compact

    return ""


def extract_induced_answer(text: Any) -> str:
    """
    Extract a compact answer from forced-prefix induced content.

    This is used for stability/sufficiency checks. It falls back to the first
    generated line because induced content often lacks an explicit answer marker.
    """
    text = remove_special_tokens(text)

    extracted = extract_final_answer(text)
    if extracted:
        return extracted

    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return clean_extracted_answer(boxed.group(1))

    first_line = text.splitlines()[0].strip() if text else ""
    return clean_extracted_answer(first_line)


def surface_candidate_for_likelihood(answer: Any) -> str:
    """
    Prepare an answer candidate for likelihood without canonicalizing semantics.

    Use this when the probability should reflect the surface form that would
    naturally appear in the model's answer context. For example, "$0.05" stays
    "$0.05" instead of becoming "0.05".
    """
    return clean_for_compare(answer)


def canonicalize_answer(answer: Any) -> str:
    """
    Convert surface answer forms into a comparable canonical form.

    Examples:
    - "$0.05" -> "0.05"
    - "5 cents" -> "0.05"
    - "five cents" -> "0.05"
    - "The ball costs $0.05" -> "0.05"
    - "45." -> "45"
    - "two apples" -> "2"
    - "second place" -> "2"
    """
    answer = clean_for_compare(answer).lower()

    if not answer:
        return ""

    # Normalize currency words/symbols.
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")
    answer = answer.replace("usd", "")
    answer = answer.replace("dollars", "")
    answer = answer.replace("dollar", "")
    answer = answer.replace("cents", "cent")
    answer = answer.strip()

    # Convert simple English number words into digits.
    # Word boundaries avoid replacing "one" inside words like "someone".
    for word, digit in NUMBER_WORDS.items():
        answer = re.sub(rf"\b{word}\b", digit, answer)

    # Convert cents into dollars, e.g. "5 cent" -> "0.05".
    cent_match = re.search(r"(-?\d+(?:\.\d+)?)\s*cent\b", answer)
    if cent_match:
        value = float(cent_match.group(1)) / 100
        return f"{value:.2f}"

    # Extract numeric answer if present.
    # We take the last number because model outputs often end with the final answer.
    num_matches = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if num_matches:
        value = float(num_matches[-1])
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return answer


def short_answer_contained(short_answer: str, long_answer: str) -> bool:
    """Conservative fallback for cases like 'Mary' vs 'The answer is Mary'."""
    if not short_answer or not long_answer:
        return False

    short_answer = short_answer.strip().lower()
    long_answer = long_answer.strip().lower()

    # Keep the fallback narrow: it is meant for compact gold answers, not long
    # explanations. Numeric answers use digit boundaries to avoid 1 matching 12.
    if len(short_answer.split()) > 3 or len(short_answer) > 30:
        return False

    if re.fullmatch(r"-?\d+(?:\.\d+)?", short_answer):
        pattern = rf"(?<![\d.]){re.escape(short_answer)}(?![\d.])"
    else:
        pattern = rf"\b{re.escape(short_answer)}\b"

    return re.search(pattern, long_answer) is not None


def answers_match(a: Any, b: Any) -> bool:
    """Return whether two answer strings are equivalent after canonicalization."""
    ca = canonicalize_answer(a)
    cb = canonicalize_answer(b)

    if ca and cb and ca == cb:
        return True

    a_is_numeric = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", ca))
    b_is_numeric = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", cb))
    if a_is_numeric and b_is_numeric:
        return short_answer_contained(ca, cb) or short_answer_contained(cb, ca)

    sa = clean_for_compare(a).lower()
    sb = clean_for_compare(b).lower()
    return short_answer_contained(sa, sb) or short_answer_contained(sb, sa)
