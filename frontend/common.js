// Shared code for every page. This is the only file that talks to the server, the only file
// that touches the login token, and the home of the few widgets all pages use.
// API data is untrusted text: everything is put on the page with textContent, never as HTML.

const TOKEN_KEY = "fincopilot_token";

// Returns the stored token or null. Storage can be blocked by browser settings, so a failure
// means "no token" instead of a blank page.
function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch (error) {
    return null;
  }
}

export function saveToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function hasToken() {
  return readToken() !== null;
}

// Calls the API. Returns the parsed JSON (null for 204) or throws an Error with a message
// that is safe to show to the user.
export async function api(method, path, body) {
  const token = readToken();
  const headers = {};
  if (token) {
    headers["Authorization"] = "Bearer " + token;
  }
  const options = { method: method, headers: headers, signal: AbortSignal.timeout(15000) };
  if (body !== undefined && body !== null) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  let response;
  let data = null;
  try {
    response = await fetch(path, options);
    if (response.status !== 204) {
      const text = await response.text();
      try {
        data = JSON.parse(text);
      } catch (error) {
        data = null; // not JSON: handled below if the status is an error
      }
    }
  } catch (error) {
    if (error.name === "TimeoutError") {
      throw new Error("The request timed out");
    }
    throw new Error("Cannot reach the server");
  }

  // A 401 on a request that carried a token means the session ended. A 401 without a token is
  // just a wrong login and is reported like any other error below.
  if (response.status === 401 && token) {
    localStorage.removeItem(TOKEN_KEY);
    window.location.href = "index.html?expired=1";
    return new Promise(function () {}); // never resolves, so the page code stops here
  }

  if (!response.ok) {
    if (data && typeof data.detail === "string") {
      throw new Error(data.detail);
    }
    throw new Error("Unexpected response (status " + response.status + ")");
  }
  return data;
}

// Protected pages call this first. Returns the current user.
export async function requireLogin() {
  if (!hasToken()) {
    window.location.href = "index.html";
    return new Promise(function () {});
  }
  return api("GET", "/auth/me");
}

// Creates an element. className and text are optional (pass null). Add children with append().
export function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== null && text !== undefined) {
    element.textContent = text;
  }
  return element;
}

// Replaces the banner inside container. kind is "error" or "success"; text null clears it.
export function showMessage(container, text, kind) {
  const old = container.querySelector(":scope > .message");
  if (old) {
    old.remove();
  }
  if (text === null || text === undefined) {
    return;
  }
  const banner = el("div", "message " + kind, text);
  banner.setAttribute("role", kind === "error" ? "alert" : "status");
  container.append(banner);
}

// Fills <header id="nav">. The unread badge is loaded in the background and never blocks.
export async function renderNav(user, activePage) {
  const header = document.getElementById("nav");
  header.replaceChildren();

  const inner = el("div", "nav-inner");
  inner.append(el("span", "nav-brand", "Financial Research Copilot"));

  const nav = el("nav", null);
  nav.setAttribute("aria-label", "Main");
  const list = el("ul", "nav-links");
  const links = [
    ["companies.html", "Companies"],
    ["watchlists.html", "Watchlists"],
    ["alerts.html", "Alerts"],
    ["notifications.html", "Notifications"],
    ["team.html", "Team"],
  ];
  let badge = null;
  for (const [href, label] of links) {
    const item = el("li", null);
    const link = el("a", null, label);
    link.setAttribute("href", href);
    if (href === activePage) {
      link.setAttribute("aria-current", "page");
    }
    item.append(link);
    if (href === "notifications.html") {
      badge = el("span", "badge", "0");
      badge.id = "unread-badge";
      badge.hidden = true;
      item.append(badge);
    }
    list.append(item);
  }
  nav.append(list);
  inner.append(nav);

  const account = el("div", "nav-user");
  account.append(el("span", null, user.email + " (" + user.role + ")"));
  const logout = el("button", "secondary", "Log out");
  logout.type = "button";
  logout.addEventListener("click", function () {
    localStorage.removeItem(TOKEN_KEY);
    window.location.href = "index.html";
  });
  account.append(logout);
  inner.append(account);
  header.append(inner);

  // Unread count: a failure only means no badge
  try {
    const data = await api("GET", "/notifications?page_size=1");
    badge.textContent = String(data.unread_count);
    badge.setAttribute("title", data.unread_count + " unread");
    badge.hidden = data.unread_count === 0;
  } catch (error) {
    badge.hidden = true;
  }
}

// Renders "Page X of Y (N total)" with Previous and Next. Nothing is shown when one page is enough.
export function renderPager(container, page, pageSize, total, onPage) {
  container.replaceChildren();
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  if (total === 0 || pageCount === 1) {
    return;
  }

  const previous = el("button", "secondary", "Previous");
  previous.type = "button";
  previous.disabled = page <= 1;
  previous.addEventListener("click", function () {
    onPage(page - 1);
  });

  const next = el("button", "secondary", "Next");
  next.type = "button";
  next.disabled = page >= pageCount;
  next.addEventListener("click", function () {
    onPage(page + 1);
  });

  container.append(previous, el("span", null, "Page " + page + " of " + pageCount + " (" + total + " total)"), next);
}
