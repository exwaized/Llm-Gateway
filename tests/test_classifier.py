from llm_gateway.classifier import HeuristicClassifier


def test_simple_query_scores_low():
    clf = HeuristicClassifier()
    result = clf.score("What is the capital of France?")
    assert result.score < 0.35


def test_complex_query_scores_high():
    clf = HeuristicClassifier()
    prompt = (
        "Design a fault-tolerant distributed rate limiter and explain why a "
        "token-bucket approach is preferable to a sliding-window approach here. "
        "Also analyze the trade-offs around clock skew across nodes, and walk "
        "through how you would debug a scenario where the limiter under-throttles "
        "during a thundering herd. Refactor the reference implementation below "
        "to fix the race condition:\n"
        "```python\n"
        "class RateLimiter:\n"
        "    def __init__(self, capacity):\n"
        "        self.capacity = capacity\n"
        "        self.tokens = capacity\n"
        "    def allow(self):\n"
        "        if self.tokens > 0:\n"
        "            self.tokens -= 1\n"
        "            return True\n"
        "        return False\n"
        "```\n"
        "1. consider concurrency 2. consider fairness 3. consider recovery after an outage"
    )
    result = clf.score(prompt)
    assert result.score > 0.6


def test_code_heavy_query_scores_above_trivial():
    clf = HeuristicClassifier()
    trivial = clf.score("hello")
    code = clf.score("```python\ndef f(x):\n    if x:\n        return 1\n```")
    assert code.score > trivial.score


def test_score_is_bounded():
    clf = HeuristicClassifier()
    long_prompt = "and also " * 500
    result = clf.score(long_prompt)
    assert 0.0 <= result.score <= 1.0
