import json
import sys
from copy import deepcopy
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.alerts import Alert
from app.models.chat import ChatSession
from app.models.users import User
from app.repositories import alerts as alert_repository
from evals import run_agent_eval as agent_eval

TASKS = json.loads(agent_eval.TASKS_PATH.read_text())
RAG_TICKERS = ["AAPL", "NVDA"]


def base_task(**checks: object) -> dict:
    # A minimal valid task; the keyword arguments replace its checks
    return {
        "id": "t1",
        "question": "A question?",
        "ticker": None,
        "role": "analyst",
        "decision": None,
        "checks": {"status": "completed", "writes": "none", **checks},
    }


def base_result(**changes: object) -> dict:
    # A stored result of a task that ran cleanly; the keyword arguments replace fields
    result = {
        "first_status": "completed",
        "final_status": "completed",
        "step_count": 2,
        "tool_calls": [],
        "answer": "An answer.",
        "citation_count": 0,
        "period_ends_stated": 0,
        "invented_numbers": [],
        "alerts_created": [],
        "reports_created": [],
        "owner_user_id": 7,
        "judge": None,
        "judge_error": None,
    }
    return {**result, **changes}


def call(
    tool: str, input: dict | None = None, is_error: bool = False, approval: str = "not_required"
):
    return {
        "tool_name": tool,
        "input": input or {},
        "output": "{}",
        "is_error": is_error,
        "approval_status": approval,
    }


def errors_of(tasks: list[dict]) -> list[str]:
    return agent_eval.validate_tasks(tasks, RAG_TICKERS, 8)


# ---------- the real task file ----------


def test_the_real_task_file_is_valid() -> None:
    assert errors_of(TASKS) == []
    assert len(TASKS) == 15


def test_the_real_task_file_covers_the_designed_scenarios() -> None:
    by_id = {task["id"]: task for task in TASKS}
    assert len(by_id) == 15
    # Exactly one task creates something, and it is approved; nothing else may write
    created = [task for task in TASKS if task["checks"]["writes"] == "created"]
    assert [task["decision"] for task in created] == ["approve"]
    assert {task["checks"]["writes"] for task in TASKS} == {"none", "created"}
    # The abstain tasks, the viewer task and the three write requests exist
    assert sum(1 for task in TASKS if task["checks"].get("must_abstain")) == 3
    assert [task["id"] for task in TASKS if task["role"] == "viewer"] == [
        "14_viewer_report_refused"
    ]
    assert (
        sum(1 for task in TASKS if "create_alert" in task["checks"].get("tools_required", [])) == 2
    )


# ---------- structural validation ----------


def test_a_minimal_task_is_valid() -> None:
    assert errors_of([base_task()]) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": "owner"}, "role must be one of"),
        ({"decision": "maybe"}, "decision must be null"),
        ({"ticker": "MSFT"}, "ticker must be null or one of"),
        ({"question": "  "}, "question must not be empty"),
        ({"id": "bad id!"}, "id must be letters"),
        ({"checks": {}}, "checks must be a non-empty object"),
    ],
)
def test_bad_task_fields_are_reported(change: dict, message: str) -> None:
    task = {**base_task(), **change}
    assert any(message in error for error in errors_of([task]))


def test_duplicate_ids_are_reported() -> None:
    assert any("duplicate id" in error for error in errors_of([base_task(), base_task()]))


def test_a_missing_field_is_reported() -> None:
    task = base_task()
    del task["role"]
    assert any("missing ['role']" in error for error in errors_of([task]))


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ({"status": "finished"}, "status must be"),
        ({"status": ["completed", "failed"]}, "status must be"),
        ({"tools_required": ["web_search"]}, "tools_required must be a list of tool names"),
        ({"tools_forbidden": "search_filings"}, "tools_forbidden must be a list"),
        ({"tool_args": [{"tool": "nope", "args_subset": {"a": 1}}]}, "tool_args entries need"),
        ({"tool_args": [{"tool": "get_financials", "args_subset": {}}]}, "tool_args entries need"),
        ({"writes": "maybe"}, "writes must be none or created"),
        ({"max_steps": 0}, "max_steps must be a positive integer"),
        ({"max_steps": 9}, "above AGENT_MAX_STEPS"),
        ({"min_citations": "one"}, "min_citations must be a positive integer"),
        ({"surprise": True}, "unknown checks"),
        ({"answer_mentions": "role"}, "answer_mentions must be a list"),
        ({"status_after": "completed"}, "status_after needs a decision"),
    ],
)
def test_bad_checks_are_reported(checks: dict, message: str) -> None:
    task = base_task()
    task["checks"].update(checks)
    assert any(message in error for error in errors_of([task]))


def test_a_task_without_status_is_reported() -> None:
    task = base_task()
    del task["checks"]["status"]
    assert any("status must be" in error for error in errors_of([task]))


def test_writes_created_needs_an_approve_decision() -> None:
    task = base_task(writes="created")
    assert any("writes created needs decision approve" in e for e in errors_of([task]))
    task = {**base_task(status="waiting_approval", writes="created"), "decision": "reject"}
    task["checks"]["status"] = "waiting_approval"
    assert any("writes created needs decision approve" in e for e in errors_of([task]))

    approved = base_task(writes="created")
    approved["decision"] = "approve"
    approved["checks"]["status"] = "waiting_approval"
    assert errors_of([approved]) == []


def test_a_decision_needs_a_first_pause() -> None:
    task = {**base_task(), "decision": "approve"}
    assert any("a decision needs status waiting_approval" in e for e in errors_of([task]))


def test_must_abstain_tasks_have_no_min_citations() -> None:
    task = base_task(must_abstain=True, min_citations=1)
    assert any("must_abstain task has no min_citations" in e for e in errors_of([task]))
    assert errors_of([base_task(must_abstain=True)]) == []


def test_an_empty_or_malformed_file_is_reported() -> None:
    assert errors_of([]) != []
    assert any("must be an object" in e for e in errors_of(["not a task"]))  # type: ignore[list-item]


# ---------- argument matching ----------


def test_tickers_match_case_insensitively_and_numbers_by_value() -> None:
    assert agent_eval.args_match({"ticker": "nvda"}, {"ticker": "NVDA", "days": 90})
    assert agent_eval.args_match({"threshold": 300}, {"threshold": 300.0})
    assert agent_eval.args_match({"threshold": 300.0}, {"threshold": 300})
    assert agent_eval.args_match({"tickers": ["aapl", "nvda"]}, {"tickers": ["AAPL", "NVDA"]})


def test_arguments_must_all_match_as_a_subset() -> None:
    assert not agent_eval.args_match({"ticker": "AAPL"}, {"ticker": "NVDA"})
    assert not agent_eval.args_match({"days": 90}, {"ticker": "NVDA"})  # the key is missing
    assert not agent_eval.args_match({"threshold": 300}, {"threshold": 301})
    assert not agent_eval.args_match({"tickers": ["AAPL"]}, {"tickers": ["AAPL", "NVDA"]})
    assert not agent_eval.args_match({"flag": True}, {"flag": 1})  # a bool is not the number 1


# ---------- the deterministic checks ----------


def test_a_clean_result_passes() -> None:
    assert agent_eval.check_run(base_task(), base_result()) == []


def test_status_checks_the_first_stream_and_after_a_decision() -> None:
    task = {**base_task(status="waiting_approval", writes="none"), "decision": "reject"}
    assert agent_eval.check_run(task, base_result(first_status="waiting_approval")) == []

    failed = agent_eval.check_run(task, base_result(first_status="completed"))
    assert failed[0].startswith("status:")
    failed = agent_eval.check_run(
        task, base_result(first_status="waiting_approval", final_status="failed")
    )
    assert failed == ["status_after: failed, expected completed"]

    listed = base_task(status=["waiting_approval", "completed"])
    assert agent_eval.check_run(listed, base_result(first_status="completed")) == []
    assert agent_eval.check_run(listed, base_result(first_status="failed")) != []


def test_tools_required_any_and_forbidden() -> None:
    task = base_task(
        tools_required=["search_filings"],
        tools_required_any=["compare_companies", "compute_metrics"],
        tools_forbidden=["get_price_history"],
    )
    ok = base_result(tool_calls=[call("search_filings"), call("compute_metrics")])
    assert agent_eval.check_run(task, ok) == []

    missing = base_result(tool_calls=[call("compute_metrics")])
    assert agent_eval.check_run(task, missing) == ["tools_required: search_filings was not called"]

    no_any = base_result(tool_calls=[call("search_filings")])
    assert agent_eval.check_run(task, no_any)[0].startswith("tools_required_any:")

    forbidden = base_result(
        tool_calls=[call("search_filings"), call("compare_companies"), call("get_price_history")]
    )
    assert agent_eval.check_run(task, forbidden) == [
        "tools_forbidden: get_price_history was called"
    ]


def test_tool_args_need_a_matching_call_of_that_tool() -> None:
    task = base_task(
        tool_args=[
            {
                "tool": "compute_metrics",
                "args_subset": {"metric": "net_margin_pct", "ticker": "NVDA"},
            }
        ]
    )
    good = base_result(
        tool_calls=[
            call("compute_metrics", {"ticker": "nvda", "metric": "net_margin_pct", "years": 5})
        ]
    )
    assert agent_eval.check_run(task, good) == []

    wrong_metric = base_result(
        tool_calls=[call("compute_metrics", {"ticker": "NVDA", "metric": "gross_margin_pct"})]
    )
    assert agent_eval.check_run(task, wrong_metric)[0].startswith("tool_args:")
    other_tool = base_result(
        tool_calls=[call("compare_companies", {"ticker": "NVDA", "metric": "net_margin_pct"})]
    )
    assert agent_eval.check_run(task, other_tool)[0].startswith("tool_args:")


def test_max_steps_and_min_citations() -> None:
    task = base_task(max_steps=3, min_citations=2)
    assert agent_eval.check_run(task, base_result(step_count=3, citation_count=2)) == []
    failed = agent_eval.check_run(task, base_result(step_count=4, citation_count=1))
    assert [item.split(":")[0] for item in failed] == ["max_steps", "min_citations"]


def test_tools_errored_period_ends_and_answer_mentions() -> None:
    task = base_task(tools_errored=["save_report"], min_period_ends=2, answer_mentions=["role"])
    ok = base_result(
        tool_calls=[call("save_report", is_error=True)],
        period_ends_stated=2,
        answer="Your  ROLE cannot save reports.",
    )
    assert agent_eval.check_run(task, ok) == []

    bad = base_result(
        tool_calls=[call("save_report", is_error=False)], period_ends_stated=1, answer="Done."
    )
    failed = agent_eval.check_run(task, bad)
    assert [item.split(":")[0] for item in failed] == [
        "tools_errored",
        "min_period_ends",
        "answer_mentions",
    ]


def test_writes_created_needs_exactly_one_row_of_the_expected_owner() -> None:
    task = {**base_task(status="waiting_approval", writes="created"), "decision": "approve"}
    row = {"id": 1, "user_id": 7}
    assert (
        agent_eval.check_run(
            task, base_result(first_status="waiting_approval", alerts_created=[row])
        )
        == []
    )

    none = agent_eval.check_run(task, base_result(first_status="waiting_approval"))
    assert none == ["writes: 0 rows created, expected exactly 1"]
    two = agent_eval.check_run(
        task, base_result(first_status="waiting_approval", alerts_created=[row, {**row, "id": 2}])
    )
    assert two == ["writes: 2 rows created, expected exactly 1"]
    other_owner = agent_eval.check_run(
        task, base_result(first_status="waiting_approval", alerts_created=[{"id": 1, "user_id": 9}])
    )
    assert other_owner == ["writes: the created row belongs to another user"]


def test_must_abstain_needs_an_abstaining_answer_without_invented_figures() -> None:
    task = base_task(must_abstain=True)
    abstains = {"abstained": True, "claims": []}
    assert agent_eval.check_run(task, base_result(judge=abstains)) == []
    # Unsupported meta statements ("I can only cover AAPL and NVDA") are not invented figures
    meta = {"abstained": True, "claims": [{"verified": False, "source": None}]}
    assert agent_eval.check_run(task, base_result(judge=meta)) == []

    answers = {"abstained": False, "claims": []}
    assert agent_eval.check_run(task, base_result(judge=answers))[0].startswith("must_abstain:")

    invents = base_result(judge=abstains, invented_numbers=["391.04"])
    failed = agent_eval.check_run(task, invents)
    assert len(failed) == 1 and "figures no tool returned ['391.04']" in failed[0]

    broken = base_result(judge=None, judge_error="ValueError")
    assert "ValueError" in agent_eval.check_run(task, broken)[0]


# ---------- the approval invariant ----------


def alert_row(user_id: int = 7) -> dict:
    return {"id": 1, "user_id": user_id, "ticker": "AAPL"}


def test_nothing_written_holds_the_invariant() -> None:
    assert agent_eval.check_invariant(base_task(), base_result()) == []


@pytest.mark.parametrize("field", ["alerts_created", "reports_created"])
def test_a_row_written_by_a_task_that_must_write_nothing_is_a_violation(field: str) -> None:
    result = base_result(**{field: [alert_row()]})
    violations = agent_eval.check_invariant(base_task(), result)
    assert len(violations) == 1 and "must write nothing" in violations[0]


def test_a_rejected_or_unapproved_write_is_a_violation_even_in_a_created_task() -> None:
    task = {**base_task(status="waiting_approval", writes="created"), "decision": "approve"}
    approved = base_result(
        alerts_created=[alert_row()], tool_calls=[call("create_alert", approval="approved")]
    )
    assert agent_eval.check_invariant(task, approved) == []

    for status in ("pending", "rejected", "not_required"):
        unapproved = base_result(
            alerts_created=[alert_row()], tool_calls=[call("create_alert", approval=status)]
        )
        violations = agent_eval.check_invariant(task, unapproved)
        assert violations and "without an approved call" in violations[0]

    errored = base_result(
        alerts_created=[alert_row()],
        tool_calls=[call("create_alert", is_error=True, approval="approved")],
    )
    assert agent_eval.check_invariant(task, errored)

    # A created task that wrote nothing is a failed check, not an invariant violation
    assert agent_eval.check_invariant(task, base_result()) == []


# ---------- the evidence quote check ----------

OUTPUTS = [
    ("compute_metrics", '{"points": [{"period_end": "2026-01-25", "value": 55.6}]}'),
    (
        "search_filings",
        '{"results": [{"text": "Export controls could disrupt our supply\\nchain and '
        'distribution channels."}]}',
    ),
]


def claim(evidence: str, supported: bool = True, has_citation: bool = False) -> dict:
    return {
        "claim": "c",
        "supported": supported,
        "evidence": evidence,
        "has_citation": has_citation,
    }


def test_a_quote_found_in_a_tool_output_verifies_the_claim_and_names_its_tool() -> None:
    checked = agent_eval.verify_claims(
        [claim('"value": 55.6'), claim("Export controls could disrupt")], OUTPUTS
    )
    assert [(c["verified"], c["source"]) for c in checked] == [
        (True, "compute_metrics"),
        (True, "search_filings"),
    ]


def test_a_made_up_quote_does_not_verify_the_claim() -> None:
    checked = agent_eval.verify_claims(
        [claim('"value": 99.9'), claim("Nvidia will double its revenue")], OUTPUTS
    )
    assert [(c["verified"], c["source"]) for c in checked] == [(False, None), (False, None)]


def test_whitespace_case_and_json_escapes_are_normalized_in_the_quote_check() -> None:
    checked = agent_eval.verify_claims(
        [
            claim("EXPORT   CONTROLS could\ndisrupt our supply chain"),
            claim('"Period_End": "2026-01-25"'),
        ],
        OUTPUTS,
    )
    assert [c["verified"] for c in checked] == [True, True]


def test_an_unsupported_claim_stays_unsupported_even_with_a_real_quote() -> None:
    checked = agent_eval.verify_claims([claim('"value": 55.6', supported=False)], OUTPUTS)
    assert checked[0]["verified"] is False


def test_empty_and_very_short_quotes_never_verify() -> None:
    checked = agent_eval.verify_claims([claim(""), claim("5"), claim("55")], OUTPUTS)
    assert [c["verified"] for c in checked] == [False, False, False]


def test_the_judges_own_quotation_marks_and_punctuation_are_ignored() -> None:
    quotes = [
        '"Export controls could disrupt our supply\nchain and distribution channels."',
        "\u201cexport controls could disrupt our supply chain\u201d,",
        "'Export controls could disrupt our supply chain';",
    ]
    checked = agent_eval.verify_claims([claim(quote) for quote in quotes], OUTPUTS)
    assert [(c["verified"], c["source"]) for c in checked] == [(True, "search_filings")] * 3


def test_a_quote_with_gaps_needs_every_fragment_in_order_in_one_output() -> None:
    in_order = claim("Export controls ... distribution channels")
    out_of_order = claim("distribution channels ... Export controls")
    fake_fragment = claim("Export controls ... double its revenue")
    other_output = claim('"value": 55.6 ... Export controls')  # fragments live in two outputs
    checked = agent_eval.verify_claims(
        [in_order, out_of_order, fake_fragment, other_output], OUTPUTS
    )
    assert [c["verified"] for c in checked] == [True, False, False, False]
    # An ellipsis character works too, and a fragment that is too short voids the quote
    assert agent_eval.verify_claims([claim("Export controls \u2026 channels")], OUTPUTS)[0][
        "verified"
    ]
    assert not agent_eval.verify_claims([claim("Export controls ... ch")], OUTPUTS)[0]["verified"]


def test_typographic_quotes_and_unicode_escapes_in_the_output_match_a_plain_quote() -> None:
    outputs = [("search_filings", '{"text": "NVIDIA\\u2019s \\u201cdesign-out\\u201d risk"}')]
    checked = agent_eval.verify_claims([claim('NVIDIA\'s "design-out" risk')], outputs)
    assert checked[0]["verified"] is True


def test_numbers_that_no_tool_returned_are_found_in_the_answer() -> None:
    calls = [
        {
            "input": {"ticker": "NVDA", "days": 365},
            "output": '{"change_pct": 24.43, "first": "2025-10-06"}',
        },
    ]
    clean = "NVDA rose 24.43% over 365 days, starting 2025-10-06. You asked about 5 days."
    assert agent_eval.find_invented_numbers("What about the last 5 days?", clean, calls) == []

    made_up = "Revenue was 391.04 and might reach 1,200.5 next year, up 24.43%."
    assert agent_eval.find_invented_numbers("A question?", made_up, calls) == ["1,200.5", "391.04"]
    assert agent_eval.find_invented_numbers("A question?", "No figures here.", calls) == []


def test_period_ends_stated_in_the_answer_are_counted_from_the_tool_outputs() -> None:
    outputs = [
        (
            "compare_companies",
            '{"a": {"period_end": "2025-09-27"}, "b": {"period_end": "2026-01-25"}}',
        ),
        ("get_financials", '{"period_end": "2024-09-28"}'),
    ]
    answer = "Apple (period end 2025-09-27) and NVIDIA (2026-01-25), not 2023-01-01."
    assert agent_eval.count_period_ends(answer, outputs) == 2
    assert agent_eval.count_period_ends("No dates.", outputs) == 0


# ---------- the aggregates ----------


def judged(*claims: tuple[bool, str | None, bool], abstained: bool = False) -> dict:
    return {
        "abstained": abstained,
        "claims": [
            {"verified": verified, "source": source, "has_citation": cited, "is_fact": True}
            for verified, source, cited in claims
        ],
    }


def task_result(**changes: object) -> dict:
    result = {
        "checks": {},
        "passed": True,
        "failed_checks": [],
        "invariant_ok": True,
        "judge": None,
        "step_count": 2,
        "seconds": 4.0,
    }
    return {**result, **changes}


def test_the_aggregates_on_fixed_judge_results() -> None:
    results = [
        task_result(  # faithfulness 3/4, coverage 1/2 (two filing claims, one cited)
            checks={"tools_required": ["search_filings"]},
            judge=judged(
                (True, "search_filings", True),
                (True, "search_filings", False),
                (True, "compute_metrics", False),
                (False, None, False),
            ),
        ),
        task_result(  # faithfulness 1.0, no filing claim, so no coverage entry
            checks={"tools_forbidden": ["search_filings"]},
            judge=judged((True, "compute_metrics", False)),
            step_count=4,
            seconds=8.0,
        ),
        task_result(  # a tool-choice failure, a failed task
            checks={"tool_args": [{}]},
            passed=False,
            failed_checks=["tool_args: no get_financials call with {...}"],
        ),
        task_result(  # an abstain task that passed
            checks={"must_abstain": True}, judge=judged(abstained=True)
        ),
        task_result(  # an abstain task that failed
            checks={"must_abstain": True},
            passed=False,
            failed_checks=["must_abstain: the answer does not decline"],
            judge=judged((True, "compute_metrics", False)),
        ),
        task_result(invariant_ok=False, passed=False),
    ]

    metrics = agent_eval.agent_metrics(results)

    assert metrics["tasks"] == 6
    assert metrics["task_pass_rate"] == pytest.approx(3 / 6)
    assert metrics["tool_choice_accuracy"] == pytest.approx(2 / 3)  # 3 tool-check tasks, 1 failed
    assert metrics["approval_safety"] == pytest.approx(5 / 6)
    assert metrics["faithfulness"] == pytest.approx((0.75 + 1.0 + 1.0) / 3)  # empty claims skipped
    assert metrics["judged_answers"] == 3
    assert metrics["citation_coverage"] == pytest.approx(0.5)
    assert metrics["abstention"] == pytest.approx(1 / 2)
    assert metrics["mean_model_calls"] == pytest.approx((2 + 4 + 2 + 2 + 2 + 2) / 6)
    assert metrics["mean_seconds"] == pytest.approx((4 + 8 + 4 + 4 + 4 + 4) / 6)


def test_statements_about_the_assistant_are_not_counted_as_claims() -> None:
    answer = judged((True, "compute_metrics", False))
    answer["claims"].append(
        {"verified": False, "source": None, "has_citation": False, "is_fact": False}
    )
    refusal = judged(abstained=True)
    refusal["claims"].append(
        {"verified": False, "source": None, "has_citation": False, "is_fact": False}
    )

    metrics = agent_eval.agent_metrics([task_result(judge=answer), task_result(judge=refusal)])

    assert metrics["faithfulness"] == 1.0  # the refusal has no fact, so it is not averaged
    assert metrics["judged_answers"] == 1


def test_the_aggregates_with_nothing_to_average_are_none() -> None:
    metrics = agent_eval.agent_metrics([task_result()])
    assert metrics["faithfulness"] is None
    assert metrics["citation_coverage"] is None
    assert metrics["abstention"] is None
    assert metrics["tool_choice_accuracy"] is None
    assert agent_eval.agent_metrics([])["task_pass_rate"] is None


# ---------- main: the hard failure and the cleanup (no real model is called) ----------


class KeepOpen:
    # Hands the test session to main() and ignores its close(), so the rollback isolation stays
    def __init__(self, session: Session) -> None:
        self.session = session

    def __getattr__(self, name: str):
        return getattr(self.session, name)

    def close(self) -> None:
        pass


@pytest.fixture
def eval_main(
    db: Session,
    agent_environment: None,
    companies: dict,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Runs agent_eval.main() against a one-task file, the test session, a temporary runs folder
    # and a fake ask_question that does whatever the test says. No OpenAI call is possible: the
    # chat model is the scripted fake of conftest, and it is never called
    def run(tasks: list[dict], ask, name: str = "unit") -> None:
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))
        monkeypatch.setattr(agent_eval, "TASKS_PATH", tasks_path)
        monkeypatch.setattr(agent_eval, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(agent_eval, "SessionLocal", lambda: KeepOpen(db))
        monkeypatch.setattr(agent_eval.agent_run, "ask_question", ask)
        monkeypatch.setattr(settings, "RERANK_ENABLED", True)  # main() sets it, the test restores
        monkeypatch.setattr(sys, "argv", ["run_agent_eval", "--name", name])
        agent_eval.main()

    return run


def eval_alerts(db: Session) -> int:
    db.expire_all()
    return db.execute(select(func.count()).select_from(Alert)).scalar_one()


def eval_sessions(db: Session) -> int:
    db.expire_all()
    return db.execute(select(func.count()).select_from(ChatSession)).scalar_one()


def test_a_task_that_writes_an_alert_without_approval_is_a_hard_failure(
    db: Session, companies: dict, eval_main, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    def rogue_ask(db, org_id, user_id, session_id, *args):
        # Simulates the worst case: an alert exists although nobody approved anything
        alert_repository.create(
            db, org_id, user_id, companies["NVDA"].id, "price_above", Decimal("500"), date.today()
        )
        db.commit()
        return iter([])

    with pytest.raises(SystemExit) as exit_info:
        eval_main([base_task()], rogue_ask)

    assert exit_info.value.code == 2
    output = capsys.readouterr().out
    assert "HARD FAILURE: the approval invariant was violated" in output
    assert "must write nothing" in output
    # Everything the task created was deleted, and no results were written
    assert eval_alerts(db) == 0
    assert eval_sessions(db) == 0
    assert not (tmp_path / "runs").exists()


def test_the_evaluation_stops_at_the_first_hard_failure(
    db: Session, companies: dict, eval_main, capsys: pytest.CaptureFixture[str]
) -> None:
    asked = []

    def rogue_ask(db, org_id, user_id, session_id, question, *args):
        asked.append(question)
        alert_repository.create(
            db, org_id, user_id, companies["NVDA"].id, "price_above", Decimal("500"), date.today()
        )
        db.commit()
        return iter([])

    second = {**base_task(), "id": "t2", "question": "Second question?"}
    with pytest.raises(SystemExit):
        eval_main([base_task(), second], rogue_ask)

    assert asked == ["A question?"]
    assert "t2" not in capsys.readouterr().out


def test_a_task_that_fails_its_checks_without_a_violation_finishes_and_cleans_up(
    db: Session, eval_main, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    # The fake agent does nothing: no run exists, so the status check fails, which is a normal
    # failed task and not a hard failure
    eval_main([base_task()], lambda *args: iter([]))

    output = capsys.readouterr().out
    assert "FAIL t1" in output
    assert "HARD FAILURE" not in output
    assert "Invariant held" in output
    saved = json.loads((tmp_path / "runs" / "agent_unit.json").read_text())
    assert saved["settings"]["rerank"] is False  # reranking is off unless --rerank
    assert saved["settings"]["langsmith_tracing"] is False
    assert saved["metrics"]["task_pass_rate"] == 0.0
    assert saved["metrics"]["approval_safety"] == 1.0
    assert eval_alerts(db) == 0
    assert eval_sessions(db) == 0


def test_the_run_file_is_named_after_the_run_with_one_agent_prefix(eval_main, tmp_path) -> None:
    eval_main([base_task()], lambda *args: iter([]), name="agent_run_9")
    eval_main([base_task()], lambda *args: iter([]), name="other")

    assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == [
        "agent_other.json",
        "agent_run_9.json",
    ]


def test_the_eval_organization_and_users_are_created_once_and_reused(
    db: Session, eval_main
) -> None:
    eval_main([base_task()], lambda *args: iter([]))
    eval_main([base_task()], lambda *args: iter([]))

    db.expire_all()
    emails = db.execute(select(User.email).where(User.email.like("eval-%"))).scalars().all()
    assert sorted(emails) == sorted(agent_eval.EVAL_EMAILS.values())
    users = db.execute(select(User).where(User.email.like("eval-%"))).scalars().all()
    assert len({user.org_id for user in users}) == 1
    assert {user.role for user in users} == {"admin", "analyst", "viewer"}
    # Nobody can log in: the stored hash is not any password we could know
    assert all(user.hashed_password.startswith("$2") for user in users)


def test_leftover_alerts_of_an_earlier_run_are_removed_before_the_tasks(
    db: Session, companies: dict, eval_main, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_main([base_task()], lambda *args: iter([]))  # creates the organization
    analyst = db.execute(select(User).where(User.email == "eval-analyst@example.com")).scalar_one()
    alert_repository.create(
        db,
        analyst.org_id,
        analyst.id,
        companies["AAPL"].id,
        "price_above",
        Decimal("300"),
        date.today(),
    )
    db.commit()
    capsys.readouterr()

    eval_main([base_task()], lambda *args: iter([]))

    assert "1 leftover alerts/reports removed" in capsys.readouterr().out
    assert eval_alerts(db) == 0


def test_main_refuses_an_invalid_task_file_before_creating_anything(
    db: Session, eval_main, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = deepcopy(base_task())
    bad["role"] = "owner"

    with pytest.raises(SystemExit) as exit_info:
        eval_main([bad], lambda *args: iter([]))

    assert exit_info.value.code == 1
    assert "The task file is not valid" in capsys.readouterr().out
    db.expire_all()
    assert db.execute(select(User).where(User.email.like("eval-%"))).first() is None


def test_main_refuses_to_run_without_an_openai_key(
    db: Session, eval_main, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_key_ask(*args):
        raise AssertionError("nothing may be asked")

    # agent_environment set a fake key; main reads settings, so empty it just for this call
    original_run = eval_main

    def run_without_key(tasks, ask):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
        original_run(tasks, ask)

    with pytest.raises(SystemExit) as exit_info:
        run_without_key([base_task()], no_key_ask)

    assert exit_info.value.code == 1
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().out
