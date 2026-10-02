// Watchlists: the list on the left, the selected watchlist (?id=) on the right
import { api, el, renderNav, requireLogin, showMessage } from "./common.js";

const message = document.getElementById("message");
const listMessage = document.getElementById("list-message");
const listArea = document.getElementById("watchlist-list");
const createForm = document.getElementById("create-form");
const detailMessage = document.getElementById("detail-message");
const detailEmpty = document.getElementById("detail-empty");
const detailBody = document.getElementById("detail-body");
const detailName = document.getElementById("detail-name");
const detailActions = document.getElementById("detail-actions");
const detailTable = document.getElementById("detail-table");
const addForm = document.getElementById("add-form");

const selectedId = (new URLSearchParams(window.location.search).get("id") || "").trim();
let canEdit = false; // UX only: the server enforces roles
let current = null; // the watchlist shown on the right

// Loads the list of watchlists
async function loadList() {
  showMessage(listMessage, null);
  try {
    const watchlists = await api("GET", "/watchlists");
    listArea.replaceChildren();
    if (watchlists.length === 0) {
      listArea.append(el("li", null, "No watchlists yet"));
      return;
    }
    for (const watchlist of watchlists) {
      const item = el("li", null);
      const link = el("a", null, watchlist.name);
      link.setAttribute("href", "watchlists.html?id=" + encodeURIComponent(watchlist.id));
      if (String(watchlist.id) === selectedId) {
        link.setAttribute("aria-current", "true");
      }
      const count = watchlist.item_count + (watchlist.item_count === 1 ? " company" : " companies");
      item.append(link, el("span", "muted", " (" + count + ")"));
      listArea.append(item);
    }
  } catch (error) {
    listArea.replaceChildren();
    showMessage(listMessage, error.message, "error");
  }
}

// Shows the selected watchlist. Pass a watchlist you already have (the add response) to skip
// the request; otherwise it is fetched.
async function loadDetail(preloaded) {
  if (selectedId === "") {
    return;
  }
  showMessage(detailMessage, null);
  try {
    current = preloaded || (await api("GET", "/watchlists/" + encodeURIComponent(selectedId)));
    detailEmpty.hidden = true;
    detailBody.hidden = false;
    detailName.textContent = current.name;
    detailActions.hidden = !canEdit;
    addForm.hidden = !canEdit;

    detailTable.replaceChildren();
    if (current.companies.length === 0) {
      detailTable.append(el("p", null, "This watchlist has no companies yet"));
      return;
    }
    const table = el("table", null);
    table.append(el("caption", "sr-only", "Companies in " + current.name));
    const head = el("thead", null);
    const headRow = el("tr", null);
    const titles = canEdit ? ["Ticker", "Name", "Remove"] : ["Ticker", "Name"];
    for (const title of titles) {
      const th = el("th", null, title);
      th.setAttribute("scope", "col");
      headRow.append(th);
    }
    head.append(headRow);
    table.append(head);

    const body = el("tbody", null);
    for (const company of current.companies) {
      const row = el("tr", null);
      const tickerCell = el("td", null);
      const link = el("a", null, company.ticker);
      link.setAttribute("href", "company.html?ticker=" + encodeURIComponent(company.ticker));
      tickerCell.append(link);
      row.append(tickerCell, el("td", null, company.name));

      if (canEdit) {
        const removeCell = el("td", null);
        const removeButton = el("button", "danger", "Remove");
        removeButton.type = "button";
        removeButton.setAttribute("aria-label", "Remove " + company.ticker);
        removeButton.addEventListener("click", async function () {
          removeButton.disabled = true;
          showMessage(detailMessage, null);
          try {
            await api(
              "DELETE",
              "/watchlists/" + encodeURIComponent(selectedId) + "/companies/" + encodeURIComponent(company.ticker)
            );
            await Promise.all([loadDetail(), loadList()]);
          } catch (error) {
            showMessage(detailMessage, error.message, "error");
          } finally {
            removeButton.disabled = false;
          }
        });
        removeCell.append(removeButton);
        row.append(removeCell);
      }
      body.append(row);
    }
    table.append(body);
    detailTable.append(table);
  } catch (error) {
    detailBody.hidden = true;
    detailEmpty.hidden = true;
    showMessage(detailMessage, error.message, "error");
  }
}

// Create a watchlist (editors only), then open it
createForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = createForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(listMessage, null);
  try {
    const created = await api("POST", "/watchlists", {
      name: document.getElementById("create-name").value.trim(),
    });
    window.location.href = "watchlists.html?id=" + encodeURIComponent(created.id);
  } catch (error) {
    showMessage(listMessage, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

// Add a company by ticker; the response is the updated watchlist
addForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = addForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(detailMessage, null);
  try {
    const updated = await api("POST", "/watchlists/" + encodeURIComponent(selectedId) + "/companies", {
      ticker: document.getElementById("add-ticker").value.trim(),
    });
    document.getElementById("add-ticker").value = "";
    await Promise.all([loadDetail(updated), loadList()]);
  } catch (error) {
    showMessage(detailMessage, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

document.getElementById("rename-button").addEventListener("click", async function () {
  const answer = window.prompt("New name for this watchlist", current.name);
  if (answer === null) {
    return; // cancelled
  }
  const name = answer.trim();
  if (name === "") {
    showMessage(detailMessage, "The name cannot be empty", "error");
    return;
  }
  showMessage(detailMessage, null);
  try {
    await api("PATCH", "/watchlists/" + encodeURIComponent(selectedId), { name: name });
    await Promise.all([loadDetail(), loadList()]);
  } catch (error) {
    showMessage(detailMessage, error.message, "error");
  }
});

document.getElementById("delete-button").addEventListener("click", async function () {
  if (!window.confirm('Delete the watchlist "' + current.name + '"? Its companies stay in the catalog.')) {
    return;
  }
  showMessage(detailMessage, null);
  try {
    await api("DELETE", "/watchlists/" + encodeURIComponent(selectedId));
    window.location.href = "watchlists.html";
  } catch (error) {
    showMessage(detailMessage, error.message, "error");
  }
});

try {
  const user = await requireLogin();
  renderNav(user, "watchlists.html");
  canEdit = user.role === "admin" || user.role === "analyst";
  createForm.hidden = !canEdit;
  document.getElementById("read-only-note").hidden = canEdit;
  await Promise.all([loadList(), loadDetail()]);
} catch (error) {
  showMessage(message, error.message, "error");
}
