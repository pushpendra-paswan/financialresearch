// Notifications: filter and page live in the URL (?unread_only=true&page=2)
import { api, el, renderNav, renderPager, requireLogin, showMessage } from "./common.js";

const PAGE_SIZE = 20;
const message = document.getElementById("message");
const listArea = document.getElementById("notifications-list");
const pagerArea = document.getElementById("notifications-pager");
const readAllButton = document.getElementById("read-all-button");

const params = new URLSearchParams(window.location.search);
const unreadOnly = params.get("unread_only") === "true";
const page = Math.max(1, parseInt(params.get("page"), 10) || 1);
document.getElementById("unread-only").checked = unreadOnly;

// Loads one page of notifications, renders it and updates the nav badge from unread_count
async function loadNotifications() {
  const query = new URLSearchParams();
  query.set("unread_only", String(unreadOnly));
  query.set("page", String(page));
  query.set("page_size", String(PAGE_SIZE));

  try {
    const data = await api("GET", "/notifications?" + query.toString());

    const badge = document.getElementById("unread-badge");
    if (badge) {
      badge.textContent = String(data.unread_count);
      badge.setAttribute("title", data.unread_count + " unread");
      badge.hidden = data.unread_count === 0;
    }
    readAllButton.disabled = data.unread_count === 0;

    listArea.replaceChildren();
    if (data.items.length === 0) {
      listArea.append(el("p", null, data.total === 0 ? "No notifications" : "No notifications on this page"));
    }
    const list = el("ul", "plain-list");
    for (const notification of data.items) {
      const item = el("li", "card");
      const heading = el("p", null);
      const link = el("a", null, notification.ticker);
      link.setAttribute("href", "company.html?ticker=" + encodeURIComponent(notification.ticker));
      heading.append(link);
      if (!notification.is_read) {
        // Unread is written as text, not only shown by styling
        heading.append(" ", el("span", "tag-new", "NEW"));
      }
      item.append(heading, el("p", null, notification.message));
      item.append(el("p", "muted", "Trade date: " + notification.trade_date));

      if (!notification.is_read) {
        const readButton = el("button", "secondary", "Mark read");
        readButton.type = "button";
        readButton.setAttribute("aria-label", "Mark the " + notification.ticker + " notification from " + notification.trade_date + " as read");
        readButton.addEventListener("click", async function () {
          readButton.disabled = true;
          showMessage(message, null);
          try {
            await api("POST", "/notifications/" + encodeURIComponent(notification.id) + "/read");
            await loadNotifications();
          } catch (error) {
            showMessage(message, error.message, "error");
            readButton.disabled = false;
          }
        });
        item.append(readButton);
      }
      list.append(item);
    }
    listArea.append(list);

    // Changing the page reloads this page with a new ?page=
    renderPager(pagerArea, data.page, data.page_size, data.total, function (newPage) {
      const next = new URLSearchParams();
      if (unreadOnly) {
        next.set("unread_only", "true");
      }
      next.set("page", String(newPage));
      window.location.href = "notifications.html?" + next.toString();
    });
  } catch (error) {
    showMessage(message, error.message, "error");
  }
}

readAllButton.addEventListener("click", async function () {
  readAllButton.disabled = true;
  showMessage(message, null);
  try {
    const result = await api("POST", "/notifications/read-all");
    await loadNotifications();
    showMessage(message, result.updated + " notification(s) marked as read", "success");
  } catch (error) {
    showMessage(message, error.message, "error");
    readAllButton.disabled = false;
  }
});

try {
  const user = await requireLogin();
  renderNav(user, "notifications.html");
  await loadNotifications();
} catch (error) {
  showMessage(message, error.message, "error");
}
