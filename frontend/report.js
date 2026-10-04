// One report (?id=). The text is restricted Markdown: #, ## and ### headings, paragraphs, "- "
// bullets, pipe tables and **bold**, plus citation markers like [1] or [1, 2]. It is turned into
// elements here, and every piece of text goes in through textContent. Anything else (links,
// images, HTML, code) stays plain text. [n] markers with a stored citation become buttons that
// open the stored passage.
import { api, el, renderNav, requireLogin, showMessage } from "./common.js";

const message = document.getElementById("message");
const reportBody = document.getElementById("report-body");
const reportTitle = document.getElementById("report-title");
const reportMeta = document.getElementById("report-meta");
const reportContent = document.getElementById("report-content");
const deleteButton = document.getElementById("delete-button");
const sourcesList = document.getElementById("sources-list");
const passagePanel = document.getElementById("passage-panel");
const passageMeta = document.getElementById("passage-meta");
const passageNote = document.getElementById("passage-note");
const passageText = document.getElementById("passage-text");

const reportId = (new URLSearchParams(window.location.search).get("id") || "").trim();
const SECTION_NAMES = { risk_factors: "Risk Factors", mdna: "MD&A" };

// Loads the report, builds its blocks, then fills the inline text of every block
async function loadReport(user) {
  if (reportId === "") {
    showMessage(message, "No report selected", "error");
    return;
  }
  try {
    const report = await api("GET", "/reports/" + encodeURIComponent(reportId));
    document.title = report.title + " - Financial Research Copilot";
    reportTitle.textContent = report.title;
    // created_at is shown as received (the date part), never turned into a Date
    reportMeta.textContent =
      report.companies.map(function (company) {
        return company.ticker;
      }).join(", ") +
      " · " + report.created_by + " · " + report.created_at.slice(0, 10);
    deleteButton.hidden = !(user.role === "admin" || user.role === "analyst");

    const citationsByNumber = new Map();
    for (const citation of report.citations) {
      citationsByNumber.set(citation.number, citation);
    }

    // 1. Blocks. Every element whose text may hold **bold** or [n] is remembered with its raw
    // text in `inlineTargets` and filled in step 2
    const inlineTargets = [];
    const lines = report.content.split("\n");
    let index = 0;
    reportContent.replaceChildren();
    while (index < lines.length) {
      const line = lines[index];
      if (line.trim() === "") {
        index += 1;
        continue;
      }

      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        // The page title is the h1, so # becomes h2
        const block = el("h" + (heading[1].length + 1), null);
        inlineTargets.push([block, heading[2].trim()]);
        reportContent.append(block);
        index += 1;
        continue;
      }

      if (/^\s*- /.test(line)) {
        const list = el("ul", null);
        while (index < lines.length && /^\s*- /.test(lines[index])) {
          const item = el("li", null);
          inlineTargets.push([item, lines[index].replace(/^\s*- /, "").trim()]);
          list.append(item);
          index += 1;
        }
        reportContent.append(list);
        continue;
      }

      if (line.trim().startsWith("|")) {
        const rows = [];
        while (index < lines.length && lines[index].trim().startsWith("|")) {
          const cells = lines[index].trim().replace(/^\|/, "").replace(/\|$/, "").split("|");
          rows.push(cells.map(function (cell) {
            return cell.trim();
          }));
          index += 1;
        }
        // The second row of dashes (|---|---|) marks the first row as the header
        const hasHeader = rows.length > 1 && rows[1].every(function (cell) {
          return /^:?-{2,}:?$/.test(cell);
        });
        const table = el("table", null);
        for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
          if (hasHeader && rowIndex === 1) {
            continue;
          }
          const tableRow = el("tr", null);
          for (const cellText of rows[rowIndex]) {
            const cell = el(hasHeader && rowIndex === 0 ? "th" : "td", null);
            inlineTargets.push([cell, cellText]);
            tableRow.append(cell);
          }
          table.append(tableRow);
        }
        const wrapper = el("div", "scroll");
        wrapper.append(table);
        reportContent.append(wrapper);
        continue;
      }

      // A paragraph runs until a blank line or the start of another block
      const paragraphLines = [];
      while (
        index < lines.length &&
        lines[index].trim() !== "" &&
        !/^#{1,3}\s+\S/.test(lines[index]) &&
        !/^\s*- /.test(lines[index]) &&
        !lines[index].trim().startsWith("|")
      ) {
        paragraphLines.push(lines[index].trim());
        index += 1;
      }
      const paragraph = el("p", null);
      inlineTargets.push([paragraph, paragraphLines.join(" ")]);
      reportContent.append(paragraph);
    }

    // 2. Inline text: **bold** and [n] markers. A marker with a stored citation is a button, any
    // other number stays plain text
    for (const [target, rawText] of inlineTargets) {
      let position = 0;
      for (const match of rawText.matchAll(/\*\*([^*]+)\*\*|\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
        target.append(rawText.slice(position, match.index));
        position = match.index + match[0].length;
        if (match[1] !== undefined) {
          target.append(el("strong", null, match[1]));
          continue;
        }
        const numbers = match[2].split(",").map(function (numberText) {
          return Number(numberText.trim());
        });
        for (const number of numbers) {
          const citation = citationsByNumber.get(number);
          if (!citation) {
            target.append("[" + number + "]");
            continue;
          }
          const button = el("button", "cite secondary", "[" + number + "]");
          button.type = "button";
          button.setAttribute("aria-label", "Show source " + number);
          button.addEventListener("click", function () {
            const sectionName = SECTION_NAMES[citation.section] || citation.section;
            const year = citation.fiscal_year ? " FY" + citation.fiscal_year : "";
            document.getElementById("passage-heading").textContent = "Source passage [" + number + "]";
            passageMeta.textContent =
              citation.ticker + year + " 10-K, " + sectionName + " · similarity " + citation.score.toFixed(3);
            passageNote.hidden = citation.chunk_id !== null;
            passageNote.textContent =
              "Stored copy: this text was saved with the report. The filing has been re-processed since, so the passage may differ from the current index.";
            passageText.textContent = citation.content;
            passagePanel.hidden = false;
            passagePanel.scrollIntoView();
          });
          target.append(button);
        }
      }
      target.append(rawText.slice(position));
    }

    // 3. The data tool calls the agent used
    sourcesList.replaceChildren();
    if (report.data_sources.length === 0) {
      sourcesList.append(el("li", null, "No data tools were used."));
    }
    for (const source of report.data_sources) {
      sourcesList.append(el("li", null, source.tool + " " + JSON.stringify(source.args)));
    }

    reportBody.hidden = false;
  } catch (error) {
    // Includes "Report not found" for a missing id or another organization's report
    reportBody.hidden = true;
    showMessage(message, error.message, "error");
  }
}

deleteButton.addEventListener("click", async function () {
  if (!window.confirm("Delete this report for everyone in your organization?")) {
    return;
  }
  deleteButton.disabled = true;
  showMessage(message, null);
  try {
    await api("DELETE", "/reports/" + encodeURIComponent(reportId));
    window.location.href = "reports.html";
  } catch (error) {
    showMessage(message, error.message, "error");
    deleteButton.disabled = false;
  }
});

document.getElementById("passage-close").addEventListener("click", function () {
  passagePanel.hidden = true;
});

try {
  const user = await requireLogin();
  renderNav(user, "reports.html");
  await loadReport(user);
} catch (error) {
  showMessage(message, error.message, "error");
}
