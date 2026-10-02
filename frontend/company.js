// Company page: ?ticker=AAPL. The header loads first; the other sections load in parallel and
// each one reports its own errors, so one failing request never blanks the page.
import { api, el, renderNav, renderPager, requireLogin, showMessage } from "./common.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const message = document.getElementById("message");
const sections = document.getElementById("company-sections");

const ticker = (new URLSearchParams(window.location.search).get("ticker") || "").trim();
let company = null; // set once the header has loaded
let symbol = ""; // the company's ticker, encoded for use in a URL path
let priceRequestNumber = 0; // lets a slow older price response be ignored

// ---------- Price chart ----------
const rangeButtons = document.querySelectorAll("button[data-days]");
const priceMessage = document.getElementById("price-message");
const priceSummary = document.getElementById("price-summary");
const priceReadout = document.getElementById("price-readout");
const priceChart = document.getElementById("price-chart");

// Loads the bars for the last `days` days and draws the SVG line chart
async function loadPrices(days) {
  for (const button of rangeButtons) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.days) === days));
  }
  priceRequestNumber += 1;
  const thisRequest = priceRequestNumber;
  showMessage(priceMessage, null);

  try {
    const data = await api("GET", "/companies/" + symbol + "/prices?days=" + days);
    if (thisRequest !== priceRequestNumber) {
      return; // a newer range was clicked meanwhile
    }
    const bars = data.bars;
    priceChart.replaceChildren();
    priceSummary.textContent = "";
    priceReadout.textContent = "";
    if (bars.length === 0) {
      priceChart.append(el("p", null, "No price data for this period"));
      return;
    }

    // Summary line: last close and the change over the period (the sign is text, not only color)
    const first = bars[0];
    const last = bars[bars.length - 1];
    priceSummary.append("Last close " + last.close.toFixed(2) + " (" + last.trade_date + ")");
    const direction = last.close >= first.close ? "up" : "down";
    if (bars.length > 1) {
      const change = (last.close / first.close - 1) * 100;
      const sign = change >= 0 ? "+" : "";
      priceSummary.append(" · ");
      priceSummary.append(el("span", direction, sign + change.toFixed(1) + "% over the period"));
    }

    // Chart geometry: viewBox 800 x 320 with margins left 60, right 20, top 20, bottom 30
    const left = 60;
    const top = 20;
    const plotWidth = 720;
    const plotHeight = 270;
    const count = bars.length;
    const closes = bars.map(function (bar) {
      return bar.close;
    });
    const lowest = Math.min(...closes);
    const highest = Math.max(...closes);
    let axisLow = lowest;
    let axisHigh = highest;
    if (highest === lowest) {
      axisLow = lowest - 1; // flat series: avoid dividing by zero
      axisHigh = highest + 1;
    } else {
      const padding = (highest - lowest) * 0.05;
      axisLow = lowest - padding;
      axisHigh = highest + padding;
    }
    // Bars are equally spaced by index (trading days), not by calendar date
    const xFor = function (index) {
      return count === 1 ? left + plotWidth / 2 : left + (index * plotWidth) / (count - 1);
    };
    const yFor = function (value) {
      return top + (plotHeight * (axisHigh - value)) / (axisHigh - axisLow);
    };

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "chart");
    svg.setAttribute("viewBox", "0 0 800 320");
    svg.setAttribute("width", "100%");
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      "Closing price of " + company.ticker + " from " + first.trade_date + " to " +
        last.trade_date + ", from " + first.close.toFixed(2) + " to " + last.close.toFixed(2)
    );

    // Three gridlines (top, middle, bottom) with their price labels
    const levels = [axisHigh, (axisHigh + axisLow) / 2, axisLow];
    for (const level of levels) {
      const y = yFor(level);
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("class", "chart-grid");
      line.setAttribute("x1", String(left));
      line.setAttribute("x2", String(left + plotWidth));
      line.setAttribute("y1", String(y));
      line.setAttribute("y2", String(y));
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("class", "chart-label");
      label.setAttribute("x", String(left - 6));
      label.setAttribute("y", String(y + 4));
      label.setAttribute("text-anchor", "end");
      label.textContent = level.toFixed(2);
      svg.append(line, label);
    }

    // First and last date under the plot (one label when there is a single bar)
    const dateLabels = [[left, "start", first.trade_date]];
    if (count > 1) {
      dateLabels.push([left + plotWidth, "end", last.trade_date]);
    }
    for (const [x, anchor, text] of dateLabels) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("class", "chart-label");
      label.setAttribute("x", String(x));
      label.setAttribute("y", String(top + plotHeight + 20));
      label.setAttribute("text-anchor", anchor);
      label.textContent = text;
      svg.append(label);
    }

    // The price line, or a single dot when there is one bar
    if (count > 1) {
      const points = bars.map(function (bar, index) {
        return xFor(index).toFixed(1) + "," + yFor(bar.close).toFixed(1);
      });
      const polyline = document.createElementNS(SVG_NS, "polyline");
      polyline.setAttribute("class", "chart-line " + direction);
      polyline.setAttribute("points", points.join(" "));
      svg.append(polyline);
    } else {
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("class", "chart-dot " + direction);
      dot.setAttribute("cx", String(xFor(0)));
      dot.setAttribute("cy", String(yFor(first.close)));
      dot.setAttribute("r", "4");
      svg.append(dot);
    }

    // Hover and touch: a vertical line and a dot follow the nearest bar
    if (count > 1) {
      const cursor = document.createElementNS(SVG_NS, "line");
      cursor.setAttribute("class", "chart-cursor");
      cursor.setAttribute("y1", String(top));
      cursor.setAttribute("y2", String(top + plotHeight));
      cursor.setAttribute("visibility", "hidden");
      const marker = document.createElementNS(SVG_NS, "circle");
      marker.setAttribute("class", "chart-dot " + direction);
      marker.setAttribute("r", "4");
      marker.setAttribute("visibility", "hidden");
      const hitArea = document.createElementNS(SVG_NS, "rect");
      hitArea.setAttribute("x", String(left));
      hitArea.setAttribute("y", String(top));
      hitArea.setAttribute("width", String(plotWidth));
      hitArea.setAttribute("height", String(plotHeight));
      hitArea.setAttribute("fill", "transparent");
      svg.append(cursor, marker, hitArea);

      const showNearest = function (event) {
        const box = svg.getBoundingClientRect();
        const svgX = ((event.clientX - box.left) * 800) / box.width;
        let index = Math.round(((svgX - left) / plotWidth) * (count - 1));
        index = Math.max(0, Math.min(count - 1, index));
        const bar = bars[index];
        cursor.setAttribute("x1", String(xFor(index)));
        cursor.setAttribute("x2", String(xFor(index)));
        marker.setAttribute("cx", String(xFor(index)));
        marker.setAttribute("cy", String(yFor(bar.close)));
        cursor.setAttribute("visibility", "visible");
        marker.setAttribute("visibility", "visible");
        priceReadout.textContent =
          bar.trade_date + " · close " + bar.close.toFixed(2) + " · volume " +
          bar.volume.toLocaleString("en-US");
      };
      hitArea.addEventListener("pointermove", showNearest);
      hitArea.addEventListener("pointerdown", showNearest);
      hitArea.addEventListener("pointerleave", function () {
        cursor.setAttribute("visibility", "hidden");
        marker.setAttribute("visibility", "hidden");
        priceReadout.textContent = "";
      });
    }

    priceChart.append(svg);
  } catch (error) {
    if (thisRequest !== priceRequestNumber) {
      return;
    }
    priceChart.replaceChildren();
    showMessage(priceMessage, error.message, "error");
  }
}

for (const button of rangeButtons) {
  button.addEventListener("click", function () {
    loadPrices(Number(button.dataset.days));
  });
}

// ---------- Financials ----------
const financialsMessage = document.getElementById("financials-message");
const financialsTable = document.getElementById("financials-table");

// Loads the nine metrics for the last 5 years: one row per metric, one column per fiscal year
async function loadFinancials() {
  showMessage(financialsMessage, null);
  try {
    const data = await api("GET", "/companies/" + symbol + "/financials?years=5");
    financialsTable.replaceChildren();

    // Columns: every fiscal year that appears in any metric, oldest first
    const yearSet = new Set();
    for (const metric of data.metrics) {
      for (const point of metric.points) {
        yearSet.add(point.fiscal_year);
      }
    }
    const years = Array.from(yearSet).sort(function (a, b) {
      return a - b;
    });
    if (years.length === 0) {
      financialsTable.append(el("p", null, "No financial data available"));
      return;
    }

    const table = el("table", null);
    table.append(el("caption", "sr-only", "Annual financials"));
    const head = el("thead", null);
    const headRow = el("tr", null);
    const metricHeader = el("th", null, "Metric");
    metricHeader.setAttribute("scope", "col");
    headRow.append(metricHeader);
    for (const year of years) {
      const th = el("th", "number", String(year));
      th.setAttribute("scope", "col");
      headRow.append(th);
    }
    head.append(headRow);
    table.append(head);

    const body = el("tbody", null);
    for (const metric of data.metrics) {
      const row = el("tr", null);
      const labelCell = el("th", null, metric.label);
      labelCell.setAttribute("scope", "row");
      row.append(labelCell);

      const valuesByYear = new Map();
      for (const point of metric.points) {
        valuesByYear.set(point.fiscal_year, point.value);
      }
      for (const year of years) {
        if (!valuesByYear.has(year)) {
          row.append(el("td", "number", "–"));
          continue;
        }
        // USD: $391.04B, $12.3M or with separators; USD/shares: $6.08; negatives start with "-"
        const value = valuesByYear.get(year);
        const absolute = Math.abs(value);
        const sign = value < 0 ? "-" : "";
        let text;
        if (metric.unit === "USD/shares") {
          text = sign + "$" + absolute.toFixed(2);
        } else if (metric.unit === "USD" && absolute >= 1e9) {
          text = sign + "$" + (absolute / 1e9).toFixed(2) + "B";
        } else if (metric.unit === "USD" && absolute >= 1e6) {
          text = sign + "$" + (absolute / 1e6).toFixed(1) + "M";
        } else if (metric.unit === "USD") {
          text = sign + "$" + absolute.toLocaleString("en-US", { maximumFractionDigits: 2 });
        } else {
          text = sign + absolute.toLocaleString("en-US", { maximumFractionDigits: 2 });
        }
        row.append(el("td", "number", text));
      }
      body.append(row);
    }
    table.append(body);
    financialsTable.append(table);
  } catch (error) {
    financialsTable.replaceChildren();
    showMessage(financialsMessage, error.message, "error");
  }
}

// ---------- Filings ----------
const filingsMessage = document.getElementById("filings-message");
const filingsTable = document.getElementById("filings-table");
const filingsPager = document.getElementById("filings-pager");
const formTypeSelect = document.getElementById("form-type");

// Loads one page of filings for the selected form type
async function loadFilings(page) {
  showMessage(filingsMessage, null);
  const query = new URLSearchParams();
  if (formTypeSelect.value) {
    query.set("form_type", formTypeSelect.value);
  }
  query.set("page", String(page));
  query.set("page_size", "10");

  try {
    const data = await api("GET", "/companies/" + symbol + "/filings?" + query.toString());
    filingsTable.replaceChildren();
    if (data.items.length === 0) {
      filingsTable.append(el("p", null, "No filings found"));
      renderPager(filingsPager, page, 10, 0, null);
      return;
    }

    const table = el("table", null);
    table.append(el("caption", "sr-only", "Filings"));
    const head = el("thead", null);
    const headRow = el("tr", null);
    for (const title of ["Form type", "Filed on", "Report date", "Fiscal year", "Document"]) {
      const th = el("th", null, title);
      th.setAttribute("scope", "col");
      headRow.append(th);
    }
    head.append(headRow);
    table.append(head);

    // The SEC link is built here from stored fields, never taken from the API as a URL
    const cikWithoutZeros = company.cik.replace(/^0+/, "");
    const body = el("tbody", null);
    for (const filing of data.items) {
      const row = el("tr", null);
      const linkCell = el("td", null);
      const link = el("a", null, "View on SEC.gov");
      link.setAttribute(
        "href",
        "https://www.sec.gov/Archives/edgar/data/" + encodeURIComponent(cikWithoutZeros) + "/" +
          encodeURIComponent(filing.accession_number.replaceAll("-", "")) + "/" +
          encodeURIComponent(filing.primary_document)
      );
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
      linkCell.append(
        link,
        el("span", "sr-only", " (" + filing.form_type + " filed " + filing.filed_on + ", opens in a new tab)")
      );
      row.append(
        el("td", null, filing.form_type),
        el("td", null, filing.filed_on),
        el("td", null, filing.report_date || "–"),
        el("td", null, filing.fiscal_year === null ? "–" : String(filing.fiscal_year)),
        linkCell
      );
      body.append(row);
    }
    table.append(body);
    filingsTable.append(table);

    renderPager(filingsPager, data.page, data.page_size, data.total, loadFilings);
  } catch (error) {
    filingsTable.replaceChildren();
    renderPager(filingsPager, 1, 10, 0, null);
    showMessage(filingsMessage, error.message, "error");
  }
}

formTypeSelect.addEventListener("change", function () {
  loadFilings(1);
});

// ---------- Add to watchlist (admin and analyst only) ----------
const watchlistSection = document.getElementById("watchlist-section");
const watchlistMessage = document.getElementById("watchlist-message");
const watchlistForm = document.getElementById("watchlist-form");
const watchlistSelect = document.getElementById("watchlist-select");
const watchlistEmpty = document.getElementById("watchlist-empty");

// Fills the select with the organization's watchlists
async function loadWatchlistChoices() {
  showMessage(watchlistMessage, null);
  try {
    const watchlists = await api("GET", "/watchlists");
    watchlistSelect.replaceChildren();
    watchlistEmpty.replaceChildren();
    if (watchlists.length === 0) {
      watchlistForm.hidden = true;
      watchlistEmpty.hidden = false;
      const link = el("a", null, "Create a watchlist on the Watchlists page");
      link.setAttribute("href", "watchlists.html");
      watchlistEmpty.append("You have no watchlists yet. ", link, ".");
      return;
    }
    watchlistForm.hidden = false;
    watchlistEmpty.hidden = true;
    for (const watchlist of watchlists) {
      const option = el("option", null, watchlist.name);
      option.value = String(watchlist.id);
      watchlistSelect.append(option);
    }
  } catch (error) {
    showMessage(watchlistMessage, error.message, "error");
  }
}

watchlistForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = watchlistForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(watchlistMessage, null);
  try {
    const watchlistId = watchlistSelect.value;
    const watchlistName = watchlistSelect.selectedOptions[0].textContent;
    await api("POST", "/watchlists/" + encodeURIComponent(watchlistId) + "/companies", {
      ticker: company.ticker,
    });
    showMessage(watchlistMessage, "Added " + company.ticker + " to " + watchlistName + ". ", "success");
    const link = el("a", null, "Open the watchlist");
    link.setAttribute("href", "watchlists.html?id=" + encodeURIComponent(watchlistId));
    watchlistMessage.querySelector(".message").append(link);
  } catch (error) {
    showMessage(watchlistMessage, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

// ---------- Create alert (every role) ----------
const alertMessage = document.getElementById("alert-message");
const alertForm = document.getElementById("alert-form");

alertForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = alertForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(alertMessage, null);
  try {
    // The threshold is sent exactly as typed (a string), never as a JavaScript float
    await api("POST", "/alerts", {
      ticker: company.ticker,
      alert_type: document.getElementById("alert-type").value,
      threshold: document.getElementById("alert-threshold").value.trim(),
    });
    showMessage(alertMessage, "Alert created. ", "success");
    const link = el("a", null, "View your alerts");
    link.setAttribute("href", "alerts.html");
    alertMessage.querySelector(".message").append(link);
    document.getElementById("alert-threshold").value = "";
  } catch (error) {
    showMessage(alertMessage, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

// ---------- Start ----------
try {
  const user = await requireLogin();
  renderNav(user, "companies.html");

  const backLink = el("a", null, "Back to companies");
  backLink.setAttribute("href", "companies.html");
  if (ticker === "") {
    showMessage(message, "No company selected. ", "error");
    message.querySelector(".message").append(backLink);
    document.getElementById("header-section").hidden = true;
  } else {
    // 1) the header; an unknown ticker stops everything else
    try {
      company = await api("GET", "/companies/" + encodeURIComponent(ticker));
    } catch (error) {
      showMessage(message, error.message + " ", "error");
      message.querySelector(".message").append(backLink);
      document.getElementById("header-section").hidden = true;
    }
  }

  if (company !== null) {
    symbol = encodeURIComponent(company.ticker);
    document.title = company.ticker + " - Financial Research Copilot";
    document.getElementById("company-title").textContent = company.ticker + " · " + company.name;
    const details = document.getElementById("company-details");
    details.append(
      el("p", null, "Exchange: " + (company.exchange || "–")),
      el("p", null, "Industry: " + (company.industry || "–")),
      el("p", null, "CIK: " + company.cik)
    );
    sections.hidden = false;

    // 2) everything else in parallel; each loader handles its own errors
    const loaders = [loadPrices(365), loadFinancials(), loadFilings(1)];
    if (user.role === "admin" || user.role === "analyst") {
      watchlistSection.hidden = false;
      loaders.push(loadWatchlistChoices());
    }
    await Promise.all(loaders);
  }
} catch (error) {
  showMessage(message, error.message, "error");
}
