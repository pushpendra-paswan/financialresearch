import json
from copy import deepcopy

import pytest
from langchain_core.documents import Document

from app.rag.parsing import SECTIONS
from evals import run_eval

QUESTIONS = json.loads(run_eval.QUESTIONS_PATH.read_text())


def document(ticker: str, section: str, content: str) -> Document:
    return Document(page_content=content, metadata={"ticker": ticker, "section": section})


EXPECTED = [
    {"ticker": "AAPL", "section": "mdna", "phrases": ["Repurchased $89.3 billion", "buyback"]},
    {"ticker": "NVDA", "section": "risk_factors", "phrases": ["depend on foundries"]},
]


# ---------- relevance ----------


def test_relevant_when_ticker_section_and_phrase_match() -> None:
    assert run_eval.is_relevant("AAPL", "mdna", "The Company repurchased $89.3 billion.", EXPECTED)
    assert run_eval.is_relevant("NVDA", "risk_factors", "We depend on foundries.", EXPECTED)


def test_relevance_ignores_case() -> None:
    assert run_eval.is_relevant("AAPL", "mdna", "THE COMPANY REPURCHASED $89.3 BILLION", EXPECTED)
    assert run_eval.is_relevant("AAPL", "mdna", "a big BUYBACK happened", EXPECTED)


def test_relevance_collapses_whitespace_and_line_breaks() -> None:
    # The filings are hard-wrapped: a phrase can be split over lines or have double spaces
    assert run_eval.is_relevant(
        "AAPL", "mdna", "it repurchased\n$89.3   billion of stock", EXPECTED
    )
    assert run_eval.is_relevant("NVDA", "risk_factors", "We depend\n on\tfoundries", EXPECTED)


@pytest.mark.parametrize(
    ("ticker", "section", "content"),
    [
        ("NVDA", "mdna", "repurchased $89.3 billion"),  # right text, wrong ticker
        ("AAPL", "risk_factors", "repurchased $89.3 billion"),  # right text, wrong section
        ("AAPL", "mdna", "repurchased $90.0 billion"),  # right place, no phrase
        ("NVDA", "mdna", "we depend on foundries"),  # phrase of another entry's section
    ],
)
def test_not_relevant_when_any_part_differs(ticker: str, section: str, content: str) -> None:
    assert not run_eval.is_relevant(ticker, section, content, EXPECTED)


def test_a_phrase_only_counts_for_its_own_entry() -> None:
    # "buyback" belongs to the AAPL entry: it must not make an NVDA chunk relevant
    assert not run_eval.is_relevant("NVDA", "risk_factors", "buyback", EXPECTED)


def test_first_relevant_rank_is_one_based_and_none_for_a_miss() -> None:
    documents = [
        document("AAPL", "mdna", "nothing here"),
        document("NVDA", "mdna", "repurchased $89.3 billion"),  # wrong ticker
        document("AAPL", "mdna", "repurchased $89.3 billion"),
        document("NVDA", "risk_factors", "we depend on foundries"),
    ]

    assert run_eval.first_relevant_rank(documents, EXPECTED) == 3
    assert run_eval.first_relevant_rank(documents[:2], EXPECTED) is None
    assert run_eval.first_relevant_rank([], EXPECTED) is None


# ---------- hit rate and MRR ----------


RANKS = [1, 3, None, 5, 2, 6]  # six answerable questions; 6 is outside the top 5


def test_hit_rate_at_k() -> None:
    assert run_eval.hit_rate(RANKS, 1) == pytest.approx(1 / 6)
    assert run_eval.hit_rate(RANKS, 3) == pytest.approx(3 / 6)
    assert run_eval.hit_rate(RANKS, 5) == pytest.approx(4 / 6)


def test_mean_reciprocal_rank_at_5_counts_misses_and_rank_6_as_zero() -> None:
    expected = (1 + 1 / 3 + 0 + 1 / 5 + 1 / 2 + 0) / 6

    assert run_eval.mean_reciprocal_rank(RANKS, 5) == pytest.approx(expected)


def test_perfect_and_empty_hits() -> None:
    assert run_eval.hit_rate([1, 1, 1], 1) == 1.0
    assert run_eval.mean_reciprocal_rank([1, 1, 1], 5) == 1.0
    assert run_eval.hit_rate([None, None], 5) == 0.0
    assert run_eval.mean_reciprocal_rank([None, None], 5) == 0.0


# ---------- rejection metrics ----------


def test_rejection_metrics_with_the_threshold_only() -> None:
    results = [
        {"answerable": False, "rejected_by_threshold": True, "abstained": None},
        {"answerable": False, "rejected_by_threshold": False, "abstained": None},
        {"answerable": False, "rejected_by_threshold": True, "abstained": None},
        {"answerable": False, "rejected_by_threshold": False, "abstained": None},
        {"answerable": True, "rejected_by_threshold": False, "abstained": None},
        {"answerable": True, "rejected_by_threshold": True, "abstained": None},
        {"answerable": True, "rejected_by_threshold": False, "abstained": None},
        {"answerable": True, "rejected_by_threshold": False, "abstained": None},
    ]

    metrics = run_eval.rejection_metrics(results)

    assert metrics["unanswerable_rejected"] == pytest.approx(2 / 4)
    assert metrics["answerable_wrongly_rejected"] == pytest.approx(1 / 4)


def test_an_abstaining_answer_also_counts_as_rejected_for_unanswerable_questions() -> None:
    results = [
        {"answerable": False, "rejected_by_threshold": True, "abstained": True},
        {"answerable": False, "rejected_by_threshold": False, "abstained": True},  # model said no
        {"answerable": False, "rejected_by_threshold": False, "abstained": False},  # made it up
        {"answerable": True, "rejected_by_threshold": False, "abstained": True},
    ]

    metrics = run_eval.rejection_metrics(results)

    assert metrics["unanswerable_rejected"] == pytest.approx(2 / 3)
    # "Wrongly rejected" is decided by the threshold only, not by the model abstaining
    assert metrics["answerable_wrongly_rejected"] == 0.0


def test_rejection_metrics_are_none_without_questions_of_that_kind() -> None:
    results = [{"answerable": True, "rejected_by_threshold": False, "abstained": None}]

    assert run_eval.rejection_metrics(results)["unanswerable_rejected"] is None


# ---------- answer metrics ----------


def claim(numbers: list[int], supported: bool) -> dict:
    return {"claim": "x", "cited_numbers": numbers, "supported": supported}


def test_faithfulness_and_coverage_are_averaged_per_answer() -> None:
    results = [
        {  # 3 claims, 2 supported; 2 with a valid citation (3 is out of range 1..2)
            "answerable": True,
            "abstained": False,
            "source_count": 2,
            "claims": [claim([1], True), claim([2], True), claim([3], False)],
        },
        {  # 1 claim, supported, cited; one answer counts as much as the first
            "answerable": True,
            "abstained": False,
            "source_count": 4,
            "claims": [claim([4], True)],
        },
        {  # a claim without any citation: not covered, and (per the judge rules) unsupported
            "answerable": True,
            "abstained": False,
            "source_count": 3,
            "claims": [claim([], False), claim([1, 9], True)],
        },
    ]

    metrics = run_eval.answer_metrics(results)

    assert metrics["faithfulness"] == pytest.approx((2 / 3 + 1 + 1 / 2) / 3)
    assert metrics["citation_coverage"] == pytest.approx((2 / 3 + 1 + 1 / 2) / 3)
    assert metrics["judged_answers"] == 3


def test_abstained_unjudged_and_unanswerable_answers_are_left_out_of_faithfulness() -> None:
    results = [
        {
            "answerable": True,
            "abstained": False,
            "source_count": 1,
            "claims": [claim([1], True)],
        },
        {"answerable": True, "abstained": True, "source_count": 1, "claims": []},
        {"answerable": True, "abstained": None, "source_count": 0, "claims": None},  # error
        {  # a hallucinated answer to an unanswerable question has claims but is not counted
            "answerable": False,
            "abstained": False,
            "source_count": 1,
            "claims": [claim([1], False)],
        },
    ]

    metrics = run_eval.answer_metrics(results)

    assert metrics["faithfulness"] == 1.0
    assert metrics["judged_answers"] == 1


def test_abstention_on_unanswerable_questions() -> None:
    results = [
        {"answerable": False, "abstained": True, "source_count": 0, "claims": None},
        {"answerable": False, "abstained": True, "source_count": 2, "claims": []},
        {"answerable": False, "abstained": False, "source_count": 2, "claims": []},
        {"answerable": False, "abstained": None, "source_count": 2, "claims": None},  # error
        {"answerable": True, "abstained": True, "source_count": 0, "claims": None},  # not counted
    ]

    metrics = run_eval.answer_metrics(results)

    assert metrics["abstention_on_unanswerable"] == pytest.approx(2 / 4)
    assert metrics["faithfulness"] is None  # no answerable answer was judged


# ---------- the question file ----------


def structural_errors(questions: list[dict]) -> list[str]:
    return run_eval.validate_questions(questions, ["AAPL", "NVDA"], list(SECTIONS))


def test_the_real_question_file_passes_the_structure_checks() -> None:
    assert structural_errors(QUESTIONS) == []


def test_the_real_question_file_has_the_planned_mix() -> None:
    answerable = [q for q in QUESTIONS if q["answerable"]]
    cross = [q for q in answerable if q["ticker"] is None]
    single = [q for q in answerable if q["ticker"] is not None]

    assert len(QUESTIONS) == 40
    assert len(single) == 28
    assert len(cross) == 5
    assert len(QUESTIONS) - len(answerable) == 7
    # Cross-company questions expect passages of BOTH companies
    for question in cross:
        assert {entry["ticker"] for entry in question["expected"]} == {"AAPL", "NVDA"}
    # Both companies and both sections are covered by the single-company questions
    assert {q["ticker"] for q in single} == {"AAPL", "NVDA"}
    sections = {entry["section"] for q in single for entry in q["expected"]}
    assert sections == set(SECTIONS)


def test_duplicate_ids_are_rejected() -> None:
    questions = deepcopy(QUESTIONS)
    questions[1]["id"] = questions[0]["id"]

    assert any("duplicate id" in error for error in structural_errors(questions))


def test_unknown_tickers_and_sections_are_rejected() -> None:
    questions = deepcopy(QUESTIONS)
    questions[0]["ticker"] = "TSLA"
    questions[0]["expected"][0]["section"] = "business"
    questions[1]["expected"][0]["ticker"] = "MSFT"

    errors = structural_errors(questions)

    assert any("ticker TSLA" in error for error in errors)
    assert any("section business" in error for error in errors)
    assert any("expected ticker MSFT" in error for error in errors)


def test_answerable_and_unanswerable_rules() -> None:
    questions = deepcopy(QUESTIONS)
    answerable = next(q for q in questions if q["answerable"])
    unanswerable = next(q for q in questions if not q["answerable"])
    answerable["expected"] = []
    unanswerable["expected"] = [{"ticker": "AAPL", "section": "mdna", "phrases": ["x"]}]

    errors = structural_errors(questions)

    assert any("answerable question needs expected" in error for error in errors)
    assert any("unanswerable question must have no expected" in error for error in errors)


def test_empty_phrases_are_rejected() -> None:
    for bad_phrases in ([], [""], ["   "]):
        questions = deepcopy(QUESTIONS)
        questions[0]["expected"][0]["phrases"] = bad_phrases

        assert any("non-empty phrases" in error for error in structural_errors(questions))


def test_missing_keys_are_reported_without_crashing() -> None:
    errors = structural_errors([{"id": "x1"}, {"question": "no id"}])

    assert any("missing key 'question'" in error for error in errors)
    assert any("missing key 'id'" in error for error in errors)


def test_every_phrase_of_the_real_questions_is_a_plain_non_empty_string() -> None:
    for question in QUESTIONS:
        for entry in question["expected"]:
            assert all(isinstance(phrase, str) and phrase.strip() for phrase in entry["phrases"])


# ---------- the phrase check against stored chunks ----------


CHUNK_ROWS = [
    # (filing_id, section, chunk_index, ticker, content)
    (1, "mdna", 0, "AAPL", "The Company repurchased\n$89.3 billion of its stock."),
    (1, "risk_factors", 0, "AAPL", "Tariffs can hurt supply."),
    (2, "risk_factors", 0, "NVDA", "We depend on foundries."),
]


def test_phrases_found_in_the_right_ticker_and_section_pass() -> None:
    questions = [
        {
            "id": "q1",
            "expected": [{"ticker": "AAPL", "section": "mdna", "phrases": ["REPURCHASED $89.3"]}],
        }
    ]

    assert run_eval.find_missing_phrases(questions, CHUNK_ROWS) == []


def test_a_phrase_missing_from_its_ticker_and_section_is_reported() -> None:
    questions = [
        {  # the text exists, but only in AAPL risk factors, not in NVDA mdna
            "id": "q1",
            "expected": [{"ticker": "NVDA", "section": "mdna", "phrases": ["Tariffs can hurt"]}],
        },
        {
            "id": "q2",
            "expected": [{"ticker": "NVDA", "section": "risk_factors", "phrases": ["not there"]}],
        },
    ]

    missing = run_eval.find_missing_phrases(questions, CHUNK_ROWS)

    assert len(missing) == 2
    assert missing[0].startswith("q1:")
    assert missing[1].startswith("q2:")


def test_observed_chunk_layout_reads_size_and_overlap() -> None:
    rows = [
        (1, "mdna", 0, "AAPL", "aaaa bbbb cccc"),
        (1, "mdna", 1, "AAPL", "cccc dddd eeee ffff"),  # overlap "cccc" (4)
        (1, "mdna", 2, "AAPL", "ffff gggg"),  # overlap "ffff" (4)
        (1, "risk_factors", 0, "AAPL", "zzzz zzzz zzzz"),  # new section: no overlap with above
    ]

    longest_chunk, longest_overlap = run_eval.observed_chunk_layout(rows)

    assert longest_chunk == len("cccc dddd eeee ffff")
    assert longest_overlap == 4


def test_importing_the_module_does_not_run_an_evaluation() -> None:
    # The module has a main() guarded by __name__; the import above would have exited or
    # called OpenAI otherwise. The judge schema is part of its public shape
    result = run_eval.JudgeResult(
        abstained=False,
        claims=[run_eval.JudgedClaim(claim="x", cited_numbers=[1], supported=True)],
    )

    assert result.claims[0].supported is True
