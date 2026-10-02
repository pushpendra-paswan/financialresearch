// Personal alerts: list, create, switch on and off, edit the threshold, delete
import { api, el, renderNav, requireLogin, showMessage } from "./common.js";

const TYPE_LABELS = {
  price_above: "Price crosses above",
  price_below: "Price crosses below",
  daily_change_pct: "Daily move of at least (%)",
};

const message = document.getElementById("message");
const createMessage = document.getElementById("create-message");
const listMessage = document.getElementById("list-message");
const alertsTable = document.getElementById("alerts-table");
const createForm = document.getElementById("create-form");

// Loads the user's alerts and renders the table with its action buttons
async function loadAlerts() {
  try {
    const alerts = await api("GET", "/alerts");
    alertsTable.replaceChildren();
    if (alerts.length === 0) {
      alertsTable.append(el("p", null, "You have no alerts yet"));
      return;
    }

    const table = el("table", null);
    table.append(el("caption", "sr-only", "Your alerts"));
    const head = el("thead", null);
    const headRow = el("tr", null);
    for (const title of ["Ticker", "Type", "Threshold", "Active", "Created", "Actions"]) {
      const th = el("th", null, title);
      th.setAttribute("scope", "col");
      headRow.append(th);
    }
    head.append(headRow);
    table.append(head);

    const body = el("tbody", null);
    for (const alert of alerts) {
      const row = el("tr", null);

      const tickerCell = el("td", null);
      const link = el("a", null, alert.ticker);
      link.setAttribute("href", "company.html?ticker=" + encodeURIComponent(alert.ticker));
      tickerCell.append(link);

      // Price alerts hold a price level, the daily move holds a percent
      let thresholdText;
      if (alert.alert_type === "daily_change_pct") {
        thresholdText = alert.threshold.toLocaleString("en-US", { maximumFractionDigits: 4 }) + "%";
      } else {
        thresholdText =
          "$" + alert.threshold.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
      }

      // The state is written as text (Yes / No), not only shown by the button
      const activeCell = el("td", null, alert.active ? "Yes " : "No ");
      const toggleButton = el("button", "secondary", alert.active ? "Turn off" : "Turn on");
      toggleButton.type = "button";
      toggleButton.setAttribute("aria-label", (alert.active ? "Turn off" : "Turn on") + " the " + alert.ticker + " alert");
      toggleButton.addEventListener("click", async function () {
        toggleButton.disabled = true;
        showMessage(listMessage, null);
        try {
          await api("PATCH", "/alerts/" + encodeURIComponent(alert.id), { active: !alert.active });
          await loadAlerts();
        } catch (error) {
          showMessage(listMessage, error.message, "error");
        } finally {
          toggleButton.disabled = false;
        }
      });
      activeCell.append(toggleButton);

      const actionsCell = el("td", null);
      const editButton = el("button", "secondary", "Edit threshold");
      editButton.type = "button";
      editButton.setAttribute("aria-label", "Edit threshold of the " + alert.ticker + " alert");
      editButton.addEventListener("click", async function () {
        const answer = window.prompt("New threshold for " + alert.ticker, String(alert.threshold));
        if (answer === null) {
          return; // cancelled
        }
        // Reject bad input here; the typed text itself is what gets sent
        const typed = answer.trim();
        const number = Number(typed);
        if (typed === "" || !Number.isFinite(number) || number <= 0) {
          showMessage(listMessage, "Enter a number greater than 0", "error");
          return;
        }
        editButton.disabled = true;
        showMessage(listMessage, null);
        try {
          await api("PATCH", "/alerts/" + encodeURIComponent(alert.id), { threshold: typed });
          await loadAlerts();
        } catch (error) {
          showMessage(listMessage, error.message, "error");
        } finally {
          editButton.disabled = false;
        }
      });

      const deleteButton = el("button", "danger", "Delete");
      deleteButton.type = "button";
      deleteButton.setAttribute("aria-label", "Delete the " + alert.ticker + " alert");
      deleteButton.addEventListener("click", async function () {
        if (!window.confirm("Delete this " + alert.ticker + " alert and its notifications?")) {
          return;
        }
        deleteButton.disabled = true;
        showMessage(listMessage, null);
        try {
          await api("DELETE", "/alerts/" + encodeURIComponent(alert.id));
          await loadAlerts();
        } catch (error) {
          showMessage(listMessage, error.message, "error");
        } finally {
          deleteButton.disabled = false;
        }
      });
      actionsCell.append(editButton, " ", deleteButton);

      row.append(
        tickerCell,
        el("td", null, TYPE_LABELS[alert.alert_type] || alert.alert_type),
        el("td", "number", thresholdText),
        activeCell,
        // created_at is a timestamp with a time, so a Date is fine here
        el("td", null, new Date(alert.created_at).toLocaleDateString()),
        actionsCell
      );
      body.append(row);
    }
    table.append(body);
    alertsTable.append(table);
  } catch (error) {
    alertsTable.replaceChildren();
    showMessage(listMessage, error.message, "error");
  }
}

createForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = createForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(createMessage, null);
  try {
    const tickerInput = document.getElementById("alert-ticker");
    const thresholdInput = document.getElementById("alert-threshold");
    // The threshold is sent exactly as typed (a string), never as a JavaScript float
    const created = await api("POST", "/alerts", {
      ticker: tickerInput.value.trim(),
      alert_type: document.getElementById("alert-type").value,
      threshold: thresholdInput.value.trim(),
    });
    showMessage(createMessage, "Alert created for " + created.ticker, "success");
    tickerInput.value = "";
    thresholdInput.value = "";
    await loadAlerts();
  } catch (error) {
    showMessage(createMessage, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

try {
  const user = await requireLogin();
  renderNav(user, "alerts.html");
  await loadAlerts();
} catch (error) {
  showMessage(message, error.message, "error");
}
