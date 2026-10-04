import json
import logging
from contextlib import nullcontext
from datetime import date, timedelta

import httpx
import openai
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import tools as agent_tools
from app.agent.tools import (
    DERIVED_METRICS,
    PRICE_TOOL_MAX_POINTS,
    READ_ONLY_TOOLS,
    compare_companies,
    compute_metrics,
    get_financials,
    get_price_history,
    search_filings,
)
from app.config import settings
from app.models.alerts import Alert
from app.models.audit import AuditLog
from app.models.chat import ChatSession
from app.models.chunks import DocumentChunk
from app.models.companies import Company
from app.models.filings import Filing
from app.models.financials import FinancialFact
from app.models.ingestion import IngestionRun
from app.models.notifications import Notification
from app.models.prices import PriceBar
from app.rag import llm
from app.repositories import chunks as chunk_repository
from app.repositories import companies as company_repository
from app.schemas.financials import MetricName
from tests.conftest import CHAT_CHUNK_DATA, add_bars, add_facts

CONTEXT = {"org_id": 1, "user_id": 2}
OUT_OF_SCOPE = "Ticker MSFT is not available. Available tickers: AAPL, NVDA"
MAX_OUTPUT_CHARACTERS = 16_000
TODAY = date.today()


@pytest.fixture(autouse=True)
def tool_environment(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each tool opens its own session: give it the test session and never close it, so the
    # rollback isolation of the test database is kept. A (fake) key enables search_filings; the
    # embeddings and the reranker are fakes anyway
    monkeypatch.setattr(agent_tools, "SessionLocal", lambda: nullcontext(db))
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-not-real")
    # Alembic's fileConfig (run when the test database is created) can disable existing loggers
    logging.getLogger("app.agent.tools").disabled = False


def run(tool, args: dict, **context) -> object:
    # Call a tool directly with the context in config["configurable"]
    return tool.invoke(args, config={"configurable": context or CONTEXT})


def run_in_graph(tool, args: dict, configurable: dict | None = None) -> ToolMessage:
    # A real ToolNode needs the runtime of a compiled graph, so it runs inside the smallest
    # possible graph: START -> tools. The AIMessage with the tool call is built by hand (no LLM)
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    message = AIMessage(content="", tool_calls=[{"name": tool.name, "args": args, "id": "call_1"}])
    result = graph.compile().invoke(
        {"messages": [message]},
        config={"configurable": CONTEXT if configurable is None else configurable},
    )
    return result["messages"][-1]


def enum_of(tool, argument: str) -> list[str]:
    # The allowed values of a Literal argument, as the model sees them in the JSON schema
    schema = tool.tool_call_schema.model_json_schema()["properties"][argument]
    if "anyOf" in schema:
        schema = next(option for option in schema["anyOf"] if "enum" in option)
    return schema["enum"]


def size_of(result: dict) -> int:
    return len(json.dumps(result))


# Minimal valid arguments of each tool (used where the data does not matter)
MINIMAL_ARGS = [
    (search_filings, {"query": "export controls"}),
    (get_financials, {"ticker": "AAPL"}),
    (get_price_history, {"ticker": "AAPL"}),
    (compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct"}),
    (compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "revenue"}),
]


# --- Schemas, descriptions and the tool list ---------------------------------------------------


def test_read_only_tools_are_the_five_named_tools() -> None:
    assert [tool.name for tool in READ_ONLY_TOOLS] == [
        "search_filings",
        "get_financials",
        "get_price_history",
        "compute_metrics",
        "compare_companies",
    ]


@pytest.mark.parametrize("tool", READ_ONLY_TOOLS, ids=lambda tool: tool.name)
def test_every_tool_has_a_description_and_hides_the_context(tool) -> None:
    assert len(tool.description) > 100
    visible = set(tool.args) | set(tool.tool_call_schema.model_json_schema()["properties"])
    assert not visible & {"config", "org_id", "user_id", "db", "session"}
    # A ToolException becomes the tool's output (an error message for the model)
    assert tool.handle_tool_error is True


def test_metric_literals_match_their_single_source_of_truth() -> None:
    base_metrics = [metric.value for metric in MetricName]
    assert enum_of(compute_metrics, "metric") == list(DERIVED_METRICS)
    assert enum_of(compare_companies, "metric") == base_metrics + list(DERIVED_METRICS)
    assert enum_of(get_financials, "metric") == base_metrics
    assert enum_of(search_filings, "section") == ["risk_factors", "mdna"]


def test_derived_metrics_table_is_consistent() -> None:
    base_metrics = {metric.value for metric in MetricName}
    for key, spec in DERIVED_METRICS.items():
        assert set(spec) == {"label", "unit", "kind", "inputs", "formula"}, key
        assert spec["kind"] in ("financial", "price")
        if spec["kind"] == "financial":
            assert set(spec["inputs"]) <= base_metrics
    assert not set(DERIVED_METRICS) & base_metrics


# --- Context, scope and errors shared by all tools --------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args"), MINIMAL_ARGS, ids=lambda value: getattr(value, "name", "")
)
@pytest.mark.parametrize(
    "configurable", [{}, {"org_id": 1}, {"user_id": 2}], ids=["none", "no_user", "no_org"]
)
def test_missing_context_is_a_bug_and_raises_value_error(tool, args, configurable) -> None:
    with pytest.raises(ValueError, match="org_id and user_id"):
        tool.invoke(args, config={"configurable": configurable})


def test_missing_context_propagates_through_toolnode() -> None:
    with pytest.raises(ValueError, match="org_id and user_id"):
        run_in_graph(get_financials, {"ticker": "AAPL"}, configurable={})


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (get_financials, {"ticker": "MSFT"}),
        (get_price_history, {"ticker": "MSFT"}),
        (compute_metrics, {"ticker": "MSFT", "metric": "net_margin_pct"}),
        (compare_companies, {"tickers": ["AAPL", "MSFT"], "metric": "revenue"}),
        (search_filings, {"query": "risks", "ticker": "MSFT"}),
    ],
    ids=lambda value: getattr(value, "name", ""),
)
def test_out_of_scope_ticker_lists_the_available_tickers(companies, tool, args) -> None:
    assert run(tool, args) == OUT_OF_SCOPE
    with pytest.raises(ToolException, match="Available tickers: AAPL, NVDA"):
        tool.func(**args, config={"configurable": CONTEXT})


def test_scope_follows_rag_tickers(market, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_TICKERS", " aapl , msft ")
    assert run(get_financials, {"ticker": "NVDA"}) == (
        "Ticker NVDA is not available. Available tickers: AAPL, MSFT"
    )


def test_lowercase_and_padded_tickers_work(market, db: Session) -> None:
    add_bars(db, market["NVDA"], [10, 11])
    assert run(get_financials, {"ticker": " aapl "})["ticker"] == "AAPL"
    assert run(get_price_history, {"ticker": "nvda"})["ticker"] == "NVDA"
    assert run(compute_metrics, {"ticker": "aapl", "metric": "net_margin_pct"})["ticker"] == "AAPL"
    result = run(compare_companies, {"tickers": ["aapl", "nvda"], "metric": "revenue"})
    assert [company["ticker"] for company in result["companies"]] == ["AAPL", "NVDA"]


def test_ticker_without_a_company_row_is_a_tool_error(
    companies, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,TSLA")
    assert run(get_financials, {"ticker": "TSLA"}) == "No data stored for TSLA"
    assert run(get_price_history, {"ticker": "TSLA"}) == "No price data stored for TSLA"
    assert run(compute_metrics, {"ticker": "TSLA", "metric": "max_drawdown_pct"}) == (
        "No data stored for TSLA"
    )


# --- get_financials ---------------------------------------------------------------------------


def test_get_financials_returns_all_nine_metrics_oldest_first(market) -> None:
    result = run(get_financials, {"ticker": "AAPL"})

    assert result["ticker"] == "AAPL"
    assert [series["metric"] for series in result["metrics"]] == [m.value for m in MetricName]
    revenue = result["metrics"][0]
    assert revenue["label"] == "Revenue"
    assert revenue["unit"] == "USD"
    assert [point["fiscal_year"] for point in revenue["points"]] == [2022, 2023, 2024]
    assert [point["value"] for point in revenue["points"]] == [100, 120, 150]
    # Only the fields the model needs, dates as ISO strings
    assert revenue["points"][0] == {
        "fiscal_year": 2022,
        "period_end": "2022-09-28",
        "value": 100.0,
        "concept": "Revenues",
    }
    # A metric without stored data has an empty list
    assert next(s for s in result["metrics"] if s["metric"] == "eps_diluted")["points"] == []
    json.dumps(result)


def test_get_financials_one_metric_and_years_limit(market) -> None:
    result = run(get_financials, {"ticker": "AAPL", "metric": "net_income", "years": 2})

    assert len(result["metrics"]) == 1
    assert [point["value"] for point in result["metrics"][0]["points"]] == [30, 45]


def test_get_financials_without_any_data_is_a_tool_error(
    companies, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,MSFT")
    assert run(get_financials, {"ticker": "MSFT"}) == "No financial data stored for MSFT"


# --- get_price_history ------------------------------------------------------------------------


def test_get_price_history_summary_matches_hand_computed_values(companies, db: Session) -> None:
    # 6 bars on the 6 days before today
    add_bars(db, companies["NVDA"], [10, 12, 8, 11, 15, 9])

    result = run(get_price_history, {"ticker": "NVDA", "days": 30})

    assert result["ticker"] == "NVDA"
    assert result["start_date"] == (TODAY - timedelta(days=6)).isoformat()
    assert result["end_date"] == (TODAY - timedelta(days=1)).isoformat()
    assert result["bars"] == 6
    assert (result["first_close"], result["last_close"]) == (10, 9)
    assert result["change_pct"] == -10.0
    assert result["highest_close"] == {"date": (TODAY - timedelta(days=2)).isoformat(), "close": 15}
    assert result["lowest_close"] == {"date": (TODAY - timedelta(days=4)).isoformat(), "close": 8}
    assert result["latest_volume"] == 1005
    # Fewer bars than the maximum: every bar is a point
    assert [point["close"] for point in result["points"]] == [10, 12, 8, 11, 15, 9]


def test_get_price_history_samples_at_most_the_maximum_and_keeps_first_and_last(
    companies, db: Session
) -> None:
    add_bars(db, companies["AAPL"], [float(100 + i % 50) for i in range(1800)])

    result = run(get_price_history, {"ticker": "AAPL", "days": 1825})

    assert result["bars"] == 1800
    assert len(result["points"]) == PRICE_TOOL_MAX_POINTS
    assert result["points"][0]["date"] == result["start_date"]
    assert result["points"][-1]["date"] == result["end_date"]
    dates = [point["date"] for point in result["points"]]
    assert dates == sorted(set(dates))
    assert size_of(result) <= MAX_OUTPUT_CHARACTERS


def test_get_price_history_exactly_one_more_bar_than_the_maximum(companies, db: Session) -> None:
    add_bars(db, companies["AAPL"], [float(10 + i) for i in range(PRICE_TOOL_MAX_POINTS + 1)])

    result = run(get_price_history, {"ticker": "AAPL", "days": 100})

    assert len(result["points"]) == PRICE_TOOL_MAX_POINTS
    assert result["points"][0]["close"] == 10
    assert result["points"][-1]["close"] == 10 + PRICE_TOOL_MAX_POINTS


def test_get_price_history_without_bars_is_a_tool_error(companies) -> None:
    assert run(get_price_history, {"ticker": "AAPL"}) == "No price data stored for AAPL"


def test_get_price_history_window_excludes_older_bars(companies, db: Session) -> None:
    add_bars(db, companies["NVDA"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    result = run(get_price_history, {"ticker": "NVDA", "days": 3})

    assert result["bars"] == 3
    assert result["first_close"] == 8


# --- compute_metrics: financial metrics -------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "metric", "expected"),
    [
        ("AAPL", "revenue_growth_pct", {2023: 20.0, 2024: 25.0}),
        ("AAPL", "net_margin_pct", {2022: 20.0, 2023: 25.0, 2024: 30.0}),
        ("AAPL", "operating_margin_pct", {2022: 25.0, 2023: 30.0, 2024: 40.0}),
        ("AAPL", "gross_margin_pct", {2022: 40.0, 2023: 50.0, 2024: 50.0}),
        ("AAPL", "return_on_equity_pct", {2022: 10.0, 2023: 20.0, 2024: 15.0}),
        ("AAPL", "liabilities_to_equity", {2022: 0.5, 2023: 1.0, 2024: 2.0}),
        ("NVDA", "revenue_growth_pct", {2024: 100.0, 2025: 200.0}),
        ("NVDA", "net_margin_pct", {2023: 20.0, 2024: 50.0, 2025: 50.0}),
        # Equity is negative in 2024: that year is left out
        ("NVDA", "return_on_equity_pct", {2023: 50.0, 2025: 150.0}),
        ("NVDA", "liabilities_to_equity", {2023: 2.0, 2025: 1.0}),
    ],
)
def test_financial_metrics_match_hand_computed_values(market, ticker, metric, expected) -> None:
    result = run(compute_metrics, {"ticker": ticker, "metric": metric})

    assert {point["fiscal_year"]: point["value"] for point in result["points"]} == expected
    spec = DERIVED_METRICS[metric]
    assert (result["label"], result["unit"], result["formula"]) == (
        spec["label"],
        spec["unit"],
        spec["formula"],
    )
    json.dumps(result)


def test_inputs_equal_the_stored_values_and_period_end_is_shown(market) -> None:
    margin = run(compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct"})
    last = margin["points"][-1]
    assert last["inputs"] == {"net_income": 45, "revenue": 150}
    assert last["period_end"] == "2024-09-28"

    growth = run(compute_metrics, {"ticker": "AAPL", "metric": "revenue_growth_pct"})
    assert growth["points"][-1]["inputs"] == {"revenue": 150, "prior_revenue": 120}

    ratio = run(compute_metrics, {"ticker": "NVDA", "metric": "return_on_equity_pct"})
    assert ratio["points"][0]["inputs"] == {"net_income": 10, "shareholders_equity": 20}


def test_years_limits_the_financial_points(market) -> None:
    result = run(compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct", "years": 2})
    assert [point["fiscal_year"] for point in result["points"]] == [2023, 2024]

    # Growth for 2 years needs revenue of 3 years, and then has no omitted year
    growth = run(compute_metrics, {"ticker": "AAPL", "metric": "revenue_growth_pct", "years": 2})
    assert [point["fiscal_year"] for point in growth["points"]] == [2023, 2024]
    assert growth["notes"] == []


def test_a_year_that_cannot_be_computed_is_left_out_and_explained(market, db: Session) -> None:
    growth = run(compute_metrics, {"ticker": "AAPL", "metric": "revenue_growth_pct"})
    assert growth["notes"] == ["FY2022: left out, no stored revenue for the prior year"]

    negative_equity = run(compute_metrics, {"ticker": "NVDA", "metric": "return_on_equity_pct"})
    assert negative_equity["notes"] == ["FY2024: left out, shareholders_equity is zero or negative"]

    # Gross profit only for 2025: the other years have a missing input
    add_facts(db, market["NVDA"], "gross_profit", {2025: 150}, 1, 26)
    margin = run(compute_metrics, {"ticker": "NVDA", "metric": "gross_margin_pct"})
    assert [(p["fiscal_year"], p["value"]) for p in margin["points"]] == [(2025, 50.0)]
    assert margin["notes"] == [
        "FY2023: left out, no stored value for gross_profit",
        "FY2024: left out, no stored value for gross_profit",
    ]


def test_growth_is_not_computed_across_a_gap_in_the_years(
    market, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Revenue 2022 and 2024 only: 2024 must not be compared with 2022
    company = company_repository.create(db, "TSLA", "0001318605", "Tesla", None)
    add_facts(db, company, "revenue", {2022: 100, 2024: 150}, 12, 31)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,TSLA")

    # Nothing can be computed: both years are left out
    assert run(compute_metrics, {"ticker": "TSLA", "metric": "revenue_growth_pct"}) == (
        "Revenue growth cannot be computed for TSLA: FY2022: left out, no stored revenue for the "
        "prior year; FY2024: left out, no stored revenue for the prior year"
    )


def test_zero_revenue_leaves_the_year_out(
    market, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    company = company_repository.create(db, "AMD", "0000002488", "AMD", None)
    add_facts(db, company, "net_income", {2024: 5}, 12, 31)
    add_facts(db, company, "revenue", {2024: 0}, 12, 31)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,AMD")

    assert run(compute_metrics, {"ticker": "AMD", "metric": "net_margin_pct"}) == (
        "Net margin cannot be computed for AMD: FY2024: left out, revenue is zero or negative"
    )


def test_nothing_computable_is_a_tool_error(market) -> None:
    # NVDA has no gross profit at all
    message = run(compute_metrics, {"ticker": "NVDA", "metric": "gross_margin_pct"})
    assert message.startswith("Gross margin cannot be computed for NVDA: FY2023: left out")


# --- compute_metrics: price metrics -----------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "closes", "expected"),
    [
        ("price_return_pct", [10, 12, 8, 11, 15, 9], -10.0),
        ("max_drawdown_pct", [10, 12, 8, 11, 15, 9], -40.0),
        # The deepest fall is not from the highest close: 20 -> 10 is -50, 30 -> 24 is -20
        ("max_drawdown_pct", [20, 10, 30, 24], -50.0),
        # Rising only: no fall at all
        ("max_drawdown_pct", [1, 2, 3, 4], 0.0),
        # Returns +10% and -10%: stdev 0.1414 x sqrt(252) x 100
        ("annualized_volatility_pct", [100, 110, 99], 224.5),
        # A flat price
        ("annualized_volatility_pct", [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 0.0),
        ("max_drawdown_pct", [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 0.0),
        ("price_return_pct", [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 0.0),
    ],
)
def test_price_metrics_match_hand_computed_values(
    companies, db: Session, metric, closes, expected
) -> None:
    add_bars(db, companies["NVDA"], closes)

    result = run(compute_metrics, {"ticker": "NVDA", "metric": metric})

    assert len(result["points"]) == 1
    point = result["points"][0]
    assert point["value"] == pytest.approx(expected, abs=0.01)
    assert point["period"] == (
        f"{(TODAY - timedelta(days=len(closes))).isoformat()} to "
        f"{(TODAY - timedelta(days=1)).isoformat()}"
    )
    assert point["period_end"] == (TODAY - timedelta(days=1)).isoformat()
    assert point["inputs"]["bars"] == len(closes)
    assert result["notes"] == []


def test_price_metric_inputs_show_the_dates_and_closes_used(companies, db: Session) -> None:
    add_bars(db, companies["NVDA"], [10, 12, 8, 11, 15, 9])

    drawdown = run(compute_metrics, {"ticker": "NVDA", "metric": "max_drawdown_pct"})
    assert drawdown["points"][0]["inputs"] == {
        "peak_date": (TODAY - timedelta(days=2)).isoformat(),
        "peak_close": 15,
        "trough_date": (TODAY - timedelta(days=1)).isoformat(),
        "trough_close": 9,
        "bars": 6,
    }
    price_return = run(compute_metrics, {"ticker": "NVDA", "metric": "price_return_pct"})
    assert price_return["points"][0]["inputs"] == {
        "start_date": (TODAY - timedelta(days=6)).isoformat(),
        "start_close": 10,
        "end_date": (TODAY - timedelta(days=1)).isoformat(),
        "end_close": 9,
        "bars": 6,
    }


def test_price_metrics_use_days_and_financial_metrics_ignore_it(market, db: Session) -> None:
    add_bars(db, market["NVDA"], [10, 20, 30, 40, 50])

    short = run(compute_metrics, {"ticker": "NVDA", "metric": "price_return_pct", "days": 3})
    assert short["points"][0]["inputs"]["bars"] == 3
    assert short["points"][0]["value"] == pytest.approx(66.67, abs=0.01)  # 30 -> 50

    one = run(compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct", "days": 1})
    other = run(compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct", "days": 900})
    assert one == other


@pytest.mark.parametrize(
    ("metric", "bars", "needed"),
    [("price_return_pct", 1, 2), ("max_drawdown_pct", 1, 2), ("annualized_volatility_pct", 2, 3)],
)
def test_too_few_bars_is_a_tool_error(companies, db: Session, metric, bars, needed) -> None:
    add_bars(db, companies["NVDA"], [10.0] * bars)

    message = run(compute_metrics, {"ticker": "NVDA", "metric": metric})

    assert f"needs at least {needed} price bars" in message
    assert f"{bars} stored" in message


# --- compare_companies ------------------------------------------------------------------------


def test_compare_two_companies_side_by_side_with_sorted_latest(market) -> None:
    result = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "revenue"})

    assert result["metric"] == "revenue"
    assert result["unit"] == "USD"
    assert [company["ticker"] for company in result["companies"]] == ["AAPL", "NVDA"]
    aapl = result["companies"][0]["points"]
    assert [(p["fiscal_year"], p["period_end"], p["value"]) for p in aapl] == [
        (2022, "2022-09-28", 100),
        (2023, "2023-09-28", 120),
        (2024, "2024-09-28", 150),
    ]
    # Highest first, each company keeps its own period_end
    assert result["latest"] == [
        {"ticker": "NVDA", "fiscal_year": 2025, "period_end": "2025-01-26", "value": 300},
        {"ticker": "AAPL", "fiscal_year": 2024, "period_end": "2024-09-28", "value": 150},
    ]
    json.dumps(result)


def test_compare_derived_metric(market) -> None:
    result = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "net_margin_pct"})

    assert result["unit"] == "%"
    assert [(item["ticker"], item["value"]) for item in result["latest"]] == [
        ("NVDA", 50.0),
        ("AAPL", 30.0),
    ]


def test_compare_notes_when_the_latest_period_ends_are_far_apart(market) -> None:
    result = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "revenue"})

    assert len(result["notes"]) == 1
    assert "120 days apart (AAPL 2024-09-28, NVDA 2025-01-26)" in result["notes"][0]
    assert "not aligned" in result["notes"][0]


def test_compare_has_no_period_note_when_the_dates_are_close(
    market, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MSFT's year ends 2 days after Apple's
    add_facts(db, market["MSFT"], "revenue", {2024: 200}, 9, 30)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,MSFT")

    result = run(compare_companies, {"tickers": ["AAPL", "MSFT"], "metric": "revenue"})

    assert result["notes"] == []
    assert [item["ticker"] for item in result["latest"]] == ["MSFT", "AAPL"]


def test_compare_gap_is_exactly_sixty_days_without_a_note(
    market, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # More than 60 days is needed for the note: 2024-09-28 plus 60 days is 2024-11-27
    add_facts(db, market["MSFT"], "revenue", {2024: 200}, 11, 27)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,MSFT")
    assert run(compare_companies, {"tickers": ["AAPL", "MSFT"], "metric": "revenue"})["notes"] == []


def test_compare_lists_a_company_without_data_and_returns_the_others(market) -> None:
    # NVDA has no gross profit
    result = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "gross_margin_pct"})

    assert [company["ticker"] for company in result["companies"]] == ["AAPL"]
    assert [item["ticker"] for item in result["latest"]] == ["AAPL"]
    assert len(result["notes"]) == 1
    assert result["notes"][0].startswith("NVDA: Gross margin cannot be computed for NVDA")


def test_compare_notes_of_omitted_years_name_the_company(market) -> None:
    result = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "return_on_equity_pct"})

    assert "NVDA: FY2024: left out, shareholders_equity is zero or negative" in result["notes"]


def test_compare_with_no_working_company_is_a_tool_error(companies) -> None:
    # No price bars at all
    message = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "price_return_pct"})

    assert message.startswith("price_return_pct cannot be computed for any ticker: AAPL: ")


def test_compare_price_metric_has_no_fiscal_year(companies, db: Session) -> None:
    add_bars(db, companies["AAPL"], [10, 12])
    add_bars(db, companies["NVDA"], [10, 15])

    result = run(compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "price_return_pct"})

    assert [(i["ticker"], i["fiscal_year"], i["value"]) for i in result["latest"]] == [
        ("NVDA", None, 50.0),
        ("AAPL", None, 20.0),
    ]
    assert result["notes"] == []


@pytest.mark.parametrize("tickers", [["AAPL"], ["AAPL", "NVDA", "AAPL", "NVDA", "AAPL", "NVDA"]])
def test_compare_rejects_fewer_than_two_or_more_than_five_tickers(market, tickers) -> None:
    with pytest.raises(ValidationError):
        run(compare_companies, {"tickers": tickers, "metric": "revenue"})
    message = run_in_graph(compare_companies, {"tickers": tickers, "metric": "revenue"})
    assert message.status == "error"
    assert "tickers" in message.content


def test_compare_the_same_ticker_twice_is_one_company(market) -> None:
    message = run(compare_companies, {"tickers": ["AAPL", "aapl"], "metric": "revenue"})

    assert message == "Give at least 2 different tickers to compare"


# --- search_filings ---------------------------------------------------------------------------


def test_search_finds_the_expected_chunk_and_drops_chunks_below_the_threshold(chat_chunks) -> None:
    question = CHAT_CHUNK_DATA["nvda_export"][2]

    # top_k 8 retrieves all 3 in-scope chunks; only the one equal to the question passes 0.30
    result = run(search_filings, {"query": question, "top_k": 8})

    assert [item["chunk_id"] for item in result["results"]] == [chat_chunks["nvda_export"].id]
    item = result["results"][0]
    assert item["ticker"] == "NVDA"
    assert item["section"] == "risk_factors"
    assert item["fiscal_year"] == TODAY.year - 1
    assert item["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert item["text"] == question
    assert set(item) == {"chunk_id", "ticker", "fiscal_year", "section", "similarity", "text"}
    assert "message" not in result


def test_search_with_nothing_relevant_is_an_empty_result_not_an_error(chat_chunks) -> None:
    result = run(search_filings, {"query": "banana bread recipe"})

    assert result == {
        "results": [],
        "message": "No passage in the stored filings is relevant to this query.",
    }


def test_search_with_an_empty_scope_returns_the_message_without_searching(
    companies, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("retrieve must not run when the scope is empty")

    monkeypatch.setattr(agent_tools, "retrieve", must_not_run)

    result = run(search_filings, {"query": "export controls"})

    assert result["results"] == []
    assert "No passage" in result["message"]


@pytest.mark.parametrize(
    ("arguments", "found"),
    [
        ({"ticker": "NVDA"}, True),
        ({"ticker": "nvda"}, True),
        ({"ticker": "AAPL"}, False),
        ({"section": "risk_factors"}, True),
        ({"section": "mdna"}, False),
        ({"fiscal_year_from": TODAY.year - 1}, True),
        ({"fiscal_year_from": TODAY.year}, False),
        ({"fiscal_year_to": TODAY.year - 1}, True),
        ({"fiscal_year_to": TODAY.year - 2}, False),
        ({"ticker": "NVDA", "section": "risk_factors", "fiscal_year_from": TODAY.year - 1}, True),
    ],
)
def test_search_filters_by_ticker_section_and_fiscal_year(chat_chunks, arguments, found) -> None:
    question = CHAT_CHUNK_DATA["nvda_export"][2]

    result = run(search_filings, {"query": question, **arguments})

    assert bool(result["results"]) is found


def test_search_never_returns_a_chunk_of_a_filing_outside_the_scope(chat_chunks) -> None:
    # The old AAPL 10-K is outside RAG_LOOKBACK_YEARS: its chunk equals the question (similarity 1)
    # and would pass the threshold, but the scope filter keeps it out
    question = CHAT_CHUNK_DATA["aapl_old"][2]

    result = run(search_filings, {"query": question, "top_k": 8})

    assert chat_chunks["aapl_old"].id not in [item["chunk_id"] for item in result["results"]]
    assert result["results"] == []


def test_search_top_k_limits_the_results(chat_chunks, monkeypatch: pytest.MonkeyPatch) -> None:
    # Everything passes the threshold, so top_k decides
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", -1.0)

    assert len(run(search_filings, {"query": "anything", "top_k": 2})["results"]) == 2
    assert len(run(search_filings, {"query": "anything", "top_k": 1})["results"]) == 1
    # Default: RETRIEVAL_TOP_K (5), but only 3 chunks are in scope
    assert len(run(search_filings, {"query": "anything"})["results"]) == 3


@pytest.mark.parametrize("top_k", [0, 9])
def test_search_top_k_outside_1_to_8_is_invalid(chat_chunks, top_k) -> None:
    with pytest.raises(ValidationError):
        run(search_filings, {"query": "risks", "top_k": top_k})
    message = run_in_graph(search_filings, {"query": "risks", "top_k": top_k})
    assert message.status == "error"
    assert "top_k" in message.content


def test_search_output_stays_within_the_bound_for_full_size_chunks(
    chat_chunks, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 8 chunks of 1,500 characters (the chunk size), all passing the threshold
    anchor = chat_chunks["nvda_export"]
    texts = [(f"Passage {number}. " + "Risk text. " * 150)[:1500] for number in range(8)]
    vectors = llm.get_embeddings().embed_documents(texts)
    chunk_repository.create_many(
        db,
        [
            DocumentChunk(
                filing_id=anchor.filing_id,
                company_id=anchor.company_id,
                section="mdna",
                fiscal_year=anchor.fiscal_year,
                chunk_index=number + 1,
                content=text,
                embedding=vector,
                embedding_model="fake",
            )
            for number, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        ],
    )
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", -1.0)

    result = run(search_filings, {"query": "passage", "top_k": 8})

    assert len(result["results"]) == 8
    assert size_of(result) <= MAX_OUTPUT_CHARACTERS


def test_search_without_an_openai_key_is_a_tool_error(
    chat_chunks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    assert run(search_filings, {"query": "export controls"}) == (
        "Filing search is disabled: OPENAI_API_KEY not set"
    )


def test_search_openai_error_gives_a_generic_message_without_the_exception_text(
    chat_chunks, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FailingEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("not used")

        def embed_query(self, text: str) -> list[float]:
            request = httpx.Request("POST", "https://example.invalid/embeddings")
            raise openai.APIConnectionError(message="secret sk-abc123", request=request)

    monkeypatch.setattr(llm, "get_embeddings", lambda: FailingEmbeddings())

    with caplog.at_level("WARNING"):
        message = run(search_filings, {"query": "export controls"})

    assert message == "Filing search failed. Try again."
    assert "sk-abc123" not in caplog.text
    assert "APIConnectionError" in caplog.text


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"query": "   "}, "The query is empty"),
        (
            {"query": "risks", "fiscal_year_from": 2026, "fiscal_year_to": 2025},
            "fiscal_year_from must not be greater than fiscal_year_to",
        ),
    ],
)
def test_search_bad_input_is_a_tool_error_not_a_bug(chat_chunks, arguments, expected) -> None:
    assert run(search_filings, arguments) == expected


# --- Through a real ToolNode ------------------------------------------------------------------


def test_every_tool_runs_through_toolnode_and_returns_valid_json(
    market, chat_chunks, db: Session
) -> None:
    add_bars(db, market["AAPL"], [10, 11, 12])
    add_bars(db, market["NVDA"], [20, 21, 22])
    calls = [
        (search_filings, {"query": CHAT_CHUNK_DATA["nvda_export"][2]}),
        (get_financials, {"ticker": "AAPL", "metric": "revenue"}),
        (get_price_history, {"ticker": "AAPL", "days": 30}),
        (compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct"}),
        (compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "revenue"}),
    ]
    assert [tool.name for tool, _args in calls] == [tool.name for tool in READ_ONLY_TOOLS]

    for tool, args in calls:
        message = run_in_graph(tool, args)
        assert isinstance(message, ToolMessage)
        assert message.status == "success", message.content
        assert message.name == tool.name
        assert json.loads(message.content) == run(tool, args)


def test_a_tool_exception_becomes_an_error_message_the_run_survives(market) -> None:
    message = run_in_graph(get_financials, {"ticker": "MSFT"})

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert message.content == OUT_OF_SCOPE


def test_invalid_arguments_become_an_error_message(market) -> None:
    for tool, args in (
        (get_price_history, {"ticker": "AAPL", "days": 99999}),
        (compute_metrics, {"ticker": "AAPL", "metric": "not_a_metric"}),
        (get_financials, {}),
    ):
        message = run_in_graph(tool, args)
        assert message.status == "error"
        assert "Error invoking tool" in message.content


def test_a_bug_in_a_tool_propagates_through_toolnode(
    market, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):
        raise ValueError("programming bug")

    monkeypatch.setattr(agent_tools.financial_service, "get_financials", broken)

    with pytest.raises(ValueError, match="programming bug"):
        run_in_graph(get_financials, {"ticker": "AAPL"})


def test_the_context_reaches_the_tool_but_is_not_an_argument(market, caplog) -> None:
    with caplog.at_level("INFO", logger="app.agent.tools"):
        run_in_graph(get_financials, {"ticker": "AAPL"}, configurable={"org_id": 7, "user_id": 8})

    assert "org_id=7 user_id=8" in caplog.text


# --- Output bounds ----------------------------------------------------------------------------


def test_worst_case_financial_outputs_stay_within_the_bound(
    market, db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ten fiscal years of all nine metrics for five companies: the largest possible arguments
    extra = {}
    for ticker, cik in (("TSLA", "0001318605"), ("AMD", "0000002488"), ("INTC", "0000050863")):
        extra[ticker] = company_repository.create(db, ticker, cik, ticker, None)
    for company in (market["MSFT"], *extra.values()):
        for metric in MetricName:
            add_facts(db, company, metric.value, {y: 1e11 + y for y in range(2016, 2026)}, 12, 31)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,MSFT,TSLA,AMD,INTC")

    financials = run(get_financials, {"ticker": "MSFT", "years": 10})
    assert sum(len(series["points"]) for series in financials["metrics"]) == 90
    assert size_of(financials) <= MAX_OUTPUT_CHARACTERS

    computed = run(
        compute_metrics, {"ticker": "MSFT", "metric": "return_on_equity_pct", "years": 9}
    )
    assert len(computed["points"]) == 9
    assert size_of(computed) <= MAX_OUTPUT_CHARACTERS

    compared = run(
        compare_companies,
        {
            "tickers": ["MSFT", "TSLA", "AMD", "INTC", "AAPL"],
            "metric": "return_on_equity_pct",
            "years": 9,
        },
    )
    assert len(compared["companies"]) == 5
    assert size_of(compared) <= MAX_OUTPUT_CHARACTERS


# --- Read-only --------------------------------------------------------------------------------


def test_no_tool_writes_anything(market, chat_chunks, db: Session) -> None:
    add_bars(db, market["AAPL"], [10, 11, 12])
    add_bars(db, market["NVDA"], [20, 21, 22])
    db.flush()
    models = [
        AuditLog,
        IngestionRun,
        Alert,
        Notification,
        ChatSession,
        Company,
        Filing,
        FinancialFact,
        PriceBar,
        DocumentChunk,
    ]

    def counts() -> list[int]:
        return [db.scalar(select(func.count()).select_from(model)) for model in models]

    before = counts()
    for tool, args in (
        (search_filings, {"query": CHAT_CHUNK_DATA["nvda_export"][2]}),
        (get_financials, {"ticker": "AAPL"}),
        (get_price_history, {"ticker": "AAPL"}),
        (compute_metrics, {"ticker": "AAPL", "metric": "net_margin_pct"}),
        (compare_companies, {"tickers": ["AAPL", "NVDA"], "metric": "revenue"}),
    ):
        run(tool, args)
    assert counts() == before
