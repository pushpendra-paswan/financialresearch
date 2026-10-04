// Reports of the organization, newest first. The page lives in the URL (?page=2). Reports are
// created only by the research agent after the user approved the text (chat page), so there is
// no create form here.
import { api, el, renderNav, renderPager, requireLogin, showMessage } from "./common.js";

const PAGE_SIZE = 20;
const message = document.getElementById("message");
const listArea = document.getElementById("reports-list");
const pagerArea = document.getElementById("reports-pager");

const page = Math.max(1, parseInt(new URLSearchParams(window.location.search).get("page"), 10) || 1);

// Loads one page of reports and renders it
async function loadReports() {
  try {
    const data = await api("GET", "/reports?page=" + page + "&page_size=" + PAGE_SIZE);

    listArea.replaceChildren();
    if (data.items.length === 0) {
      listArea.append(
        el(
          "p",
          null,
          data.total === 0
            ? "No reports yet. Reports are created by the research agent after your approval: ask for one in the Research chat."
            : "No reports on this page"
        )
      );
    }
    const list = el("ul", "plain-list");
    for (const report of data.items) {
      const item = el("li", "card");
      const heading = el("p", null);
      const link = el("a", null, report.title);
      link.setAttribute("href", "report.html?id=" + encodeURIComponent(report.id));
      heading.append(link);
      // created_at is shown as received (the date part), never turned into a Date
      item.append(
        heading,
        el("p", "muted", report.tickers.join(", ") + " · " + report.created_by + " · " + report.created_at.slice(0, 10))
      );
      list.append(item);
    }
    listArea.append(list);

    // Changing the page reloads this page with a new ?page=
    renderPager(pagerArea, data.page, data.page_size, data.total, function (newPage) {
      window.location.href = "reports.html?page=" + newPage;
    });
  } catch (error) {
    showMessage(message, error.message, "error");
  }
}

try {
  const user = await requireLogin();
  renderNav(user, "reports.html");
  await loadReports();
} catch (error) {
  showMessage(message, error.message, "error");
}
