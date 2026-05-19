from semantic_matcher import semantic_similarity, answers_equivalent
from answer_utils import canonicalize_answer


test_pairs = [
    # 应该匹配
    ("mother", "the surgeon is his mother"),
    ("match", "the match"),
    ("nowhere", "you do not bury the survivors"),
    ("same", "they weigh the same"),
    ("glass", "a greenhouse is made of glass"),

    # 不应该匹配
    ("first place", "second place"),
    ("yes", "no"),
    ("1", "12"),
    ("1 month has 28 days", "12 months have 28 days"),
]


for a, b in test_pairs:
    ca = canonicalize_answer(a)
    cb = canonicalize_answer(b)

    sim = semantic_similarity(a, b)
    eq = answers_equivalent(a, b, semantic_threshold=0.82)

    print("=" * 60)
    print("A:", a)
    print("B:", b)
    print("canonical A:", ca)
    print("canonical B:", cb)
    print("similarity:", round(sim, 4))
    print("equivalent:", eq)


last_window = [
    "1 month has 28 days",
    "1 month has 28 days",
    "1",
    "1"
]

print("\nPAIRWISE LAST WINDOW SIMILARITY")
for i in range(len(last_window)):
    for j in range(i + 1, len(last_window)):
        sim = semantic_similarity(last_window[i], last_window[j])
        print(i, j, round(sim, 4), "|", last_window[i], "<->", last_window[j])