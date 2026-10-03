// Research chat: the session list on the left, the selected session (?id=) on the right.
// Answers stream in as plain text; when the stream ends the conversation is loaded again from the
// server, and every [n] in a stored answer becomes a button that opens the cited passage.
import { api, apiStream, el, renderNav, requireLogin, showMessage } from "./common.js";

const message = document.getElementById("message");
const listMessage = document.getElementById("list-message");
const sessionList = document.getElementById("session-list");
const chatMessage = document.getElementById("chat-message");
const chatEmpty = document.getElementById("chat-empty");
const chatBody = document.getElementById("chat-body");
const chatTitle = document.getElementById("chat-title");
const conversation = document.getElementById("conversation");
const streamStatus = document.getElementById("stream-status");
const askForm = document.getElementById("ask-form");
const askTicker = document.getElementById("ask-ticker");
const askQuestion = document.getElementById("ask-question");
const askButton = document.getElementById("ask-button");
const passagePanel = document.getElementById("passage-panel");
const passageMeta = document.getElementById("passage-meta");
const passageNote = document.getElementById("passage-note");
const passageText = document.getElementById("passage-text");

const selectedId = (new URLSearchParams(window.location.search).get("id") || "").trim();
const SECTION_NAMES = { risk_factors: "Risk Factors", mdna: "MD&A" };

// Loads the list of the user's own chats
async function loadSessions() {
  showMessage(listMessage, null);
  try {
    const sessions = await api("GET", "/chat/sessions");
    sessionList.replaceChildren();
    if (sessions.length === 0) {
      sessionList.append(el("li", null, "No chats yet"));
      return;
    }
    for (const session of sessions) {
      const item = el("li", null);
      const link = el("a", null, session.title || "New chat");
      link.setAttribute("href", "chat.html?id=" + encodeURIComponent(session.id));
      if (String(session.id) === selectedId) {
        link.setAttribute("aria-current", "true");
      }
      // updated_at is a timestamp with a time, so a Date is fine here
      item.append(link, el("span", "muted", " (" + new Date(session.updated_at).toLocaleDateString() + ")"));
      sessionList.append(item);
    }
  } catch (error) {
    sessionList.replaceChildren();
    showMessage(listMessage, error.message, "error");
  }
}

// Shows the selected chat with its stored messages. In an assistant message every [n] that has
// a stored citation becomes a button; other numbers stay plain text.
async function loadConversation() {
  if (selectedId === "") {
    return;
  }
  showMessage(chatMessage, null);
  try {
    const session = await api("GET", "/chat/sessions/" + encodeURIComponent(selectedId));
    chatEmpty.hidden = true;
    chatBody.hidden = false;
    chatTitle.textContent = session.title || "New chat";

    conversation.replaceChildren();
    if (session.messages.length === 0) {
      conversation.append(el("p", "muted", "Ask a question about the AAPL and NVDA 10-K filings."));
    }
    for (const stored of session.messages) {
      const bubble = el("div", "bubble " + stored.role, null);
      bubble.append(el("span", "bubble-label", stored.role === "user" ? "You" : "Answer"));

      if (stored.role === "user") {
        bubble.append(stored.content);
        conversation.append(bubble);
        continue;
      }

      const citationsByNumber = new Map();
      for (const citation of stored.citations) {
        citationsByNumber.set(citation.number, citation);
      }
      // Markers look like [3] or [1, 2]; the text between them is added as it is
      const markerPattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
      let position = 0;
      for (const match of stored.content.matchAll(markerPattern)) {
        bubble.append(stored.content.slice(position, match.index));
        position = match.index + match[0].length;
        for (const numberText of match[1].split(",")) {
          const number = Number(numberText.trim());
          const citation = citationsByNumber.get(number);
          if (!citation) {
            bubble.append("[" + number + "]");
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
              "This text was saved with the answer. The filing has been re-processed since, so the stored passage may differ from the current index.";
            passageText.textContent = citation.content;
            passagePanel.hidden = false;
            passagePanel.scrollIntoView();
          });
          bubble.append(button);
        }
      }
      bubble.append(stored.content.slice(position));
      conversation.append(bubble);
    }
  } catch (error) {
    // Includes "Chat session not found" for a missing id or someone else's chat
    chatBody.hidden = true;
    chatEmpty.hidden = true;
    showMessage(chatMessage, error.message, "error");
  }
}

// Ask a question: show it, stream the answer, then load the stored conversation again
askForm.addEventListener("submit", async function (event) {
  event.preventDefault();
  const question = askQuestion.value.trim();
  if (selectedId === "" || question === "") {
    return;
  }
  // The form stays disabled until the stream has ended
  askQuestion.disabled = true;
  askTicker.disabled = true;
  askButton.disabled = true;
  showMessage(message, null);
  showMessage(chatMessage, null);
  passagePanel.hidden = true;

  const userBubble = el("div", "bubble user", null);
  userBubble.append(el("span", "bubble-label", "You"), question);
  const answerBubble = el("div", "bubble assistant", null);
  const answerLabel = el("span", "bubble-label", "Answer");
  conversation.append(userBubble, answerBubble);
  answerBubble.append(answerLabel);
  const answerText = el("span", null, "");
  answerBubble.append(answerText);
  streamStatus.textContent = "Searching the filings...";

  let text = "";
  let finished = false;
  let failure = null;
  try {
    await apiStream(
      "/chat/sessions/" + encodeURIComponent(selectedId) + "/messages",
      { question: question, ticker: askTicker.value || null },
      function (streamEvent) {
        if (streamEvent.type === "sources") {
          streamStatus.textContent =
            streamEvent.sources.length === 0
              ? "No relevant passages found."
              : "Writing the answer from " + streamEvent.sources.length + " passages...";
        } else if (streamEvent.type === "token") {
          text += streamEvent.text;
          answerText.textContent = text;
        } else if (streamEvent.type === "done") {
          finished = true;
        } else if (streamEvent.type === "error") {
          failure = streamEvent.detail;
        }
      }
    );
    if (!finished && failure === null) {
      failure = "The answer was interrupted. Try again.";
    }
  } catch (error) {
    failure = error.message;
  }

  streamStatus.textContent = "";
  askQuestion.disabled = false;
  askTicker.disabled = false;
  askButton.disabled = false;
  if (failure === null) {
    askQuestion.value = "";
  } else {
    // Nothing was saved for this question: keep it in the box so it can be sent again
    showMessage(message, failure, "error");
  }
  // Show what is really stored (this also turns the [n] markers into buttons)
  await Promise.all([loadConversation(), loadSessions()]);
  askQuestion.focus();
});

document.getElementById("new-chat-button").addEventListener("click", async function () {
  showMessage(listMessage, null);
  try {
    const created = await api("POST", "/chat/sessions");
    window.location.href = "chat.html?id=" + encodeURIComponent(created.id);
  } catch (error) {
    showMessage(listMessage, error.message, "error");
  }
});

document.getElementById("delete-button").addEventListener("click", async function () {
  if (!window.confirm("Delete this chat and its answers?")) {
    return;
  }
  showMessage(chatMessage, null);
  try {
    await api("DELETE", "/chat/sessions/" + encodeURIComponent(selectedId));
    window.location.href = "chat.html";
  } catch (error) {
    showMessage(chatMessage, error.message, "error");
  }
});

document.getElementById("passage-close").addEventListener("click", function () {
  passagePanel.hidden = true;
});

try {
  const user = await requireLogin();
  renderNav(user, "chat.html");
  await Promise.all([loadSessions(), loadConversation()]);
} catch (error) {
  showMessage(message, error.message, "error");
}
