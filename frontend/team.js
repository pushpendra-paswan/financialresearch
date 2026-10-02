// Team page: everyone sees the members; admins can also create users
import { api, el, renderNav, requireLogin, showMessage } from "./common.js";

const message = document.getElementById("message");
const membersMessage = document.getElementById("members-message");
const membersTable = document.getElementById("members-table");
const createSection = document.getElementById("create-section");
const createMessage = document.getElementById("create-message");
const createForm = document.getElementById("create-form");

// Loads the organization's users
async function loadMembers() {
  showMessage(membersMessage, null);
  try {
    const users = await api("GET", "/users");
    membersTable.replaceChildren();

    const table = el("table", null);
    table.append(el("caption", "sr-only", "Team members"));
    const head = el("thead", null);
    const headRow = el("tr", null);
    for (const title of ["Email", "Role", "Created"]) {
      const th = el("th", null, title);
      th.setAttribute("scope", "col");
      headRow.append(th);
    }
    head.append(headRow);
    table.append(head);

    const body = el("tbody", null);
    for (const user of users) {
      const row = el("tr", null);
      // created_at is a timestamp with a time, so a Date is fine here
      row.append(
        el("td", null, user.email),
        el("td", null, user.role),
        el("td", null, new Date(user.created_at).toLocaleDateString())
      );
      body.append(row);
    }
    table.append(body);
    membersTable.append(table);
  } catch (error) {
    membersTable.replaceChildren();
    showMessage(membersMessage, error.message, "error");
  }
}

// Create a user (admins only; the server enforces it)
createForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = createForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(createMessage, null);
  try {
    const created = await api("POST", "/users", {
      email: document.getElementById("user-email").value.trim(),
      password: document.getElementById("user-password").value,
      role: document.getElementById("user-role").value,
    });
    createForm.reset(); // also clears the password
    showMessage(createMessage, "Created " + created.email + " (" + created.role + ")", "success");
    await loadMembers();
  } catch (error) {
    showMessage(createMessage, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

try {
  const user = await requireLogin();
  renderNav(user, "team.html");
  createSection.hidden = user.role !== "admin";
  await loadMembers();
} catch (error) {
  showMessage(message, error.message, "error");
}
