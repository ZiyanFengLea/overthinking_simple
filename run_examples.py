import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from answer_utils import (
    answers_match,
    extract_final_answer,
    extract_induced_answer,
    remove_special_tokens,
    remove_think_block,
    surface_candidate_for_likelihood,
)
from diagnosis_revised import analyze_trajectory

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "Qwen/Qwen3-1.7B"
REASONING_MAX_NEW_TOKENS = 800
CONTENT_MAX_NEW_TOKENS = 512

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto",
)

model.eval()


# import re
#
# import torch
# import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# from answer_utils import (
#     answers_match,
#     extract_final_answer,
#     extract_induced_answer,
#     remove_special_tokens,
#     remove_think_block,
#     surface_candidate_for_likelihood,
# )
# from diagnosis_revised import analyze_trajectory
#
# device = "cpu"
# print("Using device:", device)
#
# model_name = "Qwen/Qwen3-0.6B"
# REASONING_MAX_NEW_TOKENS = 500
# CONTENT_MAX_NEW_TOKENS = 312
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.float32,
#     low_cpu_mem_usage=True,
# )
#
# model.to(device)
# model.eval()


def build_base_messages(question):
    return [
        {
            "role": "system",
            "content": "You are a careful reasoning assistant. Solve the problem step by step.",
        },
        {
            "role": "user",
            "content": f"""Question:
{question}

Please think through the problem and give the final answer.""",
        },
    ]


def generate_reasoning(question, max_new_tokens=REASONING_MAX_NEW_TOKENS):
    messages = build_base_messages(question)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=False)


def extract_thinking(reasoning_text):
    """
    Extract text inside <think>...</think>.
    Qwen3 chat templates may put the opening <think> in the prompt, so the
    decoded continuation can contain only the reasoning text followed by
    </think>.
    If no think tags are found, return the original reasoning as fallback.
    """
    match = re.search(r"<think>(.*?)</think>", reasoning_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in reasoning_text:
        return reasoning_text.split("</think>", 1)[0].strip()
    return reasoning_text.strip()


def split_steps(reasoning_text):
    """
    Split reasoning/content into steps.

    This method counts reasoning steps with sentence segmentation. Use NLTK if it is
    installed; otherwise fall back to a small regex sentence splitter.
    """
    pattern = r"(Step\s+\d+\s*:)"
    parts = re.split(pattern, reasoning_text)

    steps = []
    for i in range(1, len(parts), 2):
        step_title = parts[i]
        step_content = parts[i + 1] if i + 1 < len(parts) else ""
        steps.append((step_title + step_content).strip())

    if steps:
        return steps

    try:
        import nltk

        sentence_parts = nltk.sent_tokenize(reasoning_text)
        return [s.strip() for s in sentence_parts if s.strip()]
    except Exception:
        pass

    sentence_parts = re.split(r"(?<=[.!?。！？])\s+", reasoning_text)
    return [s.strip() for s in sentence_parts if s.strip()]


def build_prefixes(steps):
    prefixes = []
    current = ""
    for step in steps:
        current += step + "\n"
        prefixes.append(current.strip())
    return prefixes


def build_forced_think_end_prompt(
    question,
    prefix,
    answer_prompt=True,
    enable_thinking=True,
):
    messages = build_base_messages(question)

    base_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    stripped = base_text.rstrip()
    suffix = "\n\nFinal answer:" if answer_prompt else "\n"

    if not enable_thinking:
        return stripped + suffix

    if stripped.endswith("<think>"):
        return stripped + f"\n{prefix}\n</think>{suffix}"

    return base_text + f"""<think>
{prefix}
</think>
{suffix.lstrip()}
"""


def token_len(text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def content_length_stats(raw_content):
    """Clean induced content and count its token length."""
    cleaned = raw_content
    cleaned = re.sub(r"^Final answer\s*[:：]\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = remove_special_tokens(cleaned)
    cleaned = remove_think_block(cleaned)

    return {
        "raw_content": cleaned,
        "content_token_len": token_len(cleaned),
    }


def induce_content_forced_think_end(question, prefix, max_new_tokens=512):
    """
    Truncate reasoning at prefix, force-inject </think>,
    and let the same model generate the post-thinking content.
    """
    text = build_forced_think_end_prompt(question, prefix, answer_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_generated_token_len = generated.shape[-1]
    raw_content = tokenizer.decode(generated, skip_special_tokens=False).strip()
    content = tokenizer.decode(generated, skip_special_tokens=True).strip()
    stats = content_length_stats(content)
    content_truncated = raw_generated_token_len >= max_new_tokens

    return {
        "raw_content": raw_content,
        "content": stats["raw_content"],
        "induced_answer": extract_induced_answer(stats["raw_content"]),
        "content_truncated": content_truncated,
        "content_token_len": stats["content_token_len"],
    }


def answer_loglikelihood_forced_think_end(
    question,
    prefix,
    candidate_answer,
    normalize=True,
    enable_thinking=True,
):
    """
    Compute log p(candidate_answer | question, <think>prefix</think>, Final answer:)

    This measures whether the current reasoning prefix makes a target answer
    easier for the model to produce. Normalized log-likelihood is better for
    comparing candidates with different token lengths.
    """
    if not candidate_answer:
        return None

    context_text = build_forced_think_end_prompt(
        question,
        prefix,
        answer_prompt=True,
        enable_thinking=enable_thinking,
    )
    answer_text = " " + candidate_answer.strip()

    context_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(model.device)
    answer_ids = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)

    input_ids = torch.cat([context_ids, answer_ids], dim=1)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

    context_len = context_ids.shape[1]
    answer_token_positions = torch.arange(context_len, input_ids.shape[1], device=model.device)

    target_ids = input_ids[:, answer_token_positions]
    token_log_probs = log_probs[:, answer_token_positions - 1, :].gather(
        2,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    total_logprob = token_log_probs.sum().item()

    if normalize:
        return total_logprob / answer_ids.shape[1]
    return total_logprob


def run_one_example(question, gold_answer):
    reasoning = generate_reasoning(question)

    print("FULL REASONING:")
    print(reasoning)

    thinking_text = extract_thinking(reasoning)
    steps = split_steps(thinking_text)
    prefixes = build_prefixes(steps)

    print("\nPREFIX COUNT:", len(prefixes))
    print("FULL THINKING TOKEN LENGTH:", token_len(thinking_text))
    print("FULL THINKING STEP LENGTH:", len(steps))

    final_answer = extract_final_answer(reasoning)
    final_correct = answers_match(final_answer, gold_answer)
    gold_candidate = surface_candidate_for_likelihood(gold_answer)
    final_candidate = surface_candidate_for_likelihood(final_answer)

    print("\nFINAL ANSWER:")
    print(final_answer)
    print("FINAL CORRECT:", final_correct)

    results = []

    print("\nPREFIX-INDUCED CONTENT:")
    for i, prefix in enumerate(prefixes, 1):
        content_info = induce_content_forced_think_end(
            question,
            prefix,
            max_new_tokens=CONTENT_MAX_NEW_TOKENS,
        )

        print(f"\n--- Prefix {i} content ---")
        print("thinking step length:", i)
        print("content token length:", content_info["content_token_len"])
        print("content truncated:", content_info["content_truncated"])
        print("induced answer:", content_info["induced_answer"])
        if content_info["content_token_len"] == 0 or content_info["content_truncated"]:
            print("raw content:")
            print(content_info["raw_content"])
        print("content:")
        print(content_info["content"])

        gold_ll = answer_loglikelihood_forced_think_end(
            question,
            prefix,
            gold_candidate,
        )

        if final_correct:
            margin = None
        elif final_candidate:
            final_ll = answer_loglikelihood_forced_think_end(
                question,
                prefix,
                final_candidate,
            )
            margin = gold_ll - final_ll if gold_ll is not None and final_ll is not None else None
        else:
            final_ll = None
            margin = None

        print("gold answer likelihood:", gold_ll)
        if not final_correct:
            print("final answer likelihood:", final_ll)
            print("gold - final margin:", margin)

        results.append({
            "step": i,
            "content_token_len": content_info["content_token_len"],
            "induced_answer": content_info["induced_answer"],
            "gold_answer_likelihood": gold_ll,
            "gold_vs_final_margin": margin,
        })

    example_result = {
        "question": question,
        "gold_answer": gold_answer,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "full_thinking_step_len": len(steps),
        "full_thinking_token_len": token_len(thinking_text),
        "results": results,
    }

    diagnosis = analyze_trajectory(example_result)
    example_result["diagnosis"] = diagnosis

    print("\nDIAGNOSIS:")
    print(diagnosis)

    return example_result


examples = [
    # {
    #     "id": "bat_ball",
    #     "question": "If a bat and a ball cost $1.10 in total, and the bat costs $1.00 more than the ball, how much does the ball cost?",
    #     "gold_answer": "$0.05"
    # },
    {
        "id": "sheep",
        "question": "A farmer has 15 sheep, and all but 8 die. How many are left?",
        "gold_answer": "8"
    },
    # {
    #     "id": "addition",
    #     "question": "What is 17 + 28?",
    #     "gold_answer": "45"
    # },
    # {
    #     "id": "days",
    #     "question": "If today is Monday, what day will it be in 3 days?",
    #     "gold_answer": "Thursday"
    # },
    # {
    #     "id": "mary_father",
    #     "question": "Mary's father has five daughters: Nana, Nene, Nini, Nono. What is the fifth daughter's name?",
    #     "gold_answer": "Mary"
    # },
    # {
    #     "id": "race_second",
    #     "question": "You are running a race and you pass the person in second place. What place are you in?",
    #     "gold_answer": "second",
    # },
    # {
    #     "id": "apples_take",
    #     "question": "There are three apples on a table. You take two apples. How many apples do you have?",
    #     "gold_answer": "2",
    # },
    # {
    #     "id": "months_28_days",
    #     "question": "How many months have 28 days?",
    #     "gold_answer": "12",
    # },
    # {
    #     "id": "digit_sum_mod7",
    #     "question": "Let S be the sum of all three-digit positive integers whose digits sum to 15 and which leave a remainder of 2 when divided by 7. Find the remainder when S is divided by 1000.",
    #     "gold_answer": "672",
    # }
]


def run_small_pilot(examples):
    for ex in examples:
        print("\n" + "=" * 80)
        print("RUNNING EXAMPLE:", ex["id"])
        print("=" * 80)

        run_one_example(
            question=ex["question"],
            gold_answer=ex["gold_answer"],
        )


if __name__ == "__main__":
    run_small_pilot(examples)
