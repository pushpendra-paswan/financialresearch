// Company catalog: search and page state live in the URL (?search=&page=)
import { api, el, renderNav, renderPager, requireLogin, showMessage } from "./common.js";

const PAGE_SIZE = 20;
const message = document.getElementById("message");
const tableArea = document.getElementById("companies-table");
const pagerArea = document.getElementById("companies-pager");

const params = new URLSearchParams(window.location.search);
const search = (params.get("search") || "").trim();
const page = Math.max(1, parseInt(params.get("page"), 10) || 1);
document.getElementById("search").value = search;

// Fetches one page of companies and renders the table and the pager
async function loadCompanies() {
  const query = new URLSearchParams();
  if (search) {
    query.set("search", search);
  }
  query.set("page", String(page));
  query.set("page_size", String(PAGE_SIZE));

  try {
    const data = await api("GET", "/companies?" + query.toString());
    tableArea.replaceChildren();
    if (data.items.length === 0) {
      tableArea.append(el("p", null, "No companies found"));
      renderPager(pagerArea, page, PAGE_SIZE, 0, null);
      return;
    }

    const table = el("table", null);
    table.append(el("caption", "sr-only", "Companies"));
    const headRow = el("tr", null);
    for (const title of ["Ticker", "Name", "Exchange", "Industry"]) {
      const th = el("th", null, title);
      th.setAttribute("scope", "col");
      headRow.append(th);
    }
    const head = el("thead", null);
    head.append(headRow);
    table.append(head);

    const body = el("tbody", null);
    for (const company of data.items) {
      const row = el("tr", null);
      const tickerCell = el("td", null);
      const link = el("a", null, company.ticker);
      link.setAttribute("href", "company.html?ticker=" + encodeURIComponent(company.ticker));
      tickerCell.append(link);
      row.append(
        tickerCell,
        el("td", null, company.name),
        el("td", null, company.exchange || "–"),
        el("td", null, company.industry || "–")
      );
      body.append(row);
    }
    table.append(body);
    tableArea.append(table);

    // Changing the page reloads this page with a new ?page=
    renderPager(pagerArea, data.page, data.page_size, data.total, function (newPage) {
      const next = new URLSearchParams();
      if (search) {
        next.set("search", search);
      }
      next.set("page", String(newPage));
      window.location.href = "companies.html?" + next.toString();
    });
  } catch (error) {
    showMessage(message, error.message, "error");
  }
}

try {
  const user = await requireLogin();
  renderNav(user, "companies.html");
  await loadCompanies();
} catch (error) {
  showMessage(message, error.message, "error");
}
