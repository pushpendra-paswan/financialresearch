// Login and registration page
import { api, hasToken, saveToken, showMessage } from "./common.js";

const message = document.getElementById("message");
const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");

// Log in
loginForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = loginForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(message, null);
  try {
    const result = await api("POST", "/auth/login", {
      email: document.getElementById("login-email").value.trim(),
      password: document.getElementById("login-password").value,
    });
    saveToken(result.access_token);
    window.location.href = "companies.html";
  } catch (error) {
    showMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

// Register a new organization; the registering user becomes its admin
registerForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const button = registerForm.querySelector("button[type=submit]");
  button.disabled = true;
  showMessage(message, null);
  try {
    const result = await api("POST", "/auth/register", {
      organization_name: document.getElementById("register-organization").value.trim(),
      email: document.getElementById("register-email").value.trim(),
      password: document.getElementById("register-password").value,
    });
    saveToken(result.access_token);
    window.location.href = "companies.html";
  } catch (error) {
    showMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

// On load: skip this page when a valid session exists
try {
  if (new URLSearchParams(window.location.search).get("expired") === "1") {
    showMessage(message, "Your session expired. Please log in again.", "error");
  }
  if (hasToken()) {
    // A 401 here makes api() clear the token and reload this page with ?expired=1
    await api("GET", "/auth/me");
    window.location.href = "companies.html";
  }
} catch (error) {
  showMessage(message, error.message, "error");
}
