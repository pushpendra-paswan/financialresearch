// Research chat: the session list on the left, the selected session (?id=) on the right.
// Answers stream in as plain text; when the stream ends the conversation is loaded again from the
// server, and every [n] in a stored answer becomes a button that opens the cited passage. Answers
// of the research agent also get a "Steps" button that shows the stored tool calls of the run, and
// an answer whose run waits for the user's decision (a write action such as creating an alert)
// shows an approval card with Approve and Reject. A saved report gets an "Open report" link.
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
const askMode = document.getElementById("ask-mode");
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
    // The research agent's answers have a run: its status is shown next to the answer, so the
    // runs are loaded together with the conversation (one request per agent answer)
    const runs = new Map();
    for (const stored of session.messages) {
      if (stored.run_id === null || stored.run_id === undefined) {
        continue;
      }
      try {
        runs.set(stored.run_id, await api("GET", "/agent/runs/" + encodeURIComponent(stored.run_id)));
      } catch (error) {
        // The answer is still shown, only without its steps
      }
    }

    for (const stored of session.messages) {
      const bubble = el("div", "bubble " + stored.role, null);
      const label = el("span", "bubble-label", stored.role === "user" ? "You" : "Answer");
      const run = runs.get(stored.run_id);
      if (run && run.status !== "completed") {
        label.append(" · run " + run.status);
      }
      bubble.append(label);

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

      // The approval card of a run that waits for the user's decision. The text is the summary the
      // server generated from the arguments; it is never rebuilt here. The buttons carry the ids
      // the decision must name, and one click handler below sends it
      if (run && run.status === "waiting_approval" && run.pending_approval) {
        const pending = run.pending_approval;
        const card = el("div", "approval", null);
        // Short arguments are one JSON line. A long text (a report) goes into a collapsible block
        // that keeps its line breaks, open so the user reads what they approve
        const shortArguments = {};
        const longArguments = [];
        for (const [name, value] of Object.entries(pending.args)) {
          if (typeof value === "string" && value.length > 300) {
            longArguments.push([name, value]);
          } else {
            shortArguments[name] = value;
          }
        }
        card.append(
          el("strong", null, pending.expired ? "Expired" : "Approval needed"),
          el("p", null, pending.summary),
          el("pre", null, JSON.stringify(shortArguments))
        );
        for (const [name, value] of longArguments) {
          const longDetails = el("details", "long-argument");
          longDetails.open = true;
          longDetails.append(el("summary", null, name + " (" + value.length + " characters)"), el("pre", null, value));
          card.append(longDetails);
        }
        card.append(el("p", "muted", (pending.expired ? "Expired at " : "Expires at ") + pending.expires_at));
        if (pending.expired) {
          card.append(
            el("p", "muted", "This action was not approved in time and can no longer be approved. Ask again if you still want it.")
          );
        } else {
          const actions = el("div", "approval-actions", null);
          for (const [label, decision, className] of [
            ["Approve", "approve", null],
            ["Reject", "reject", "secondary"],
          ]) {
            const decisionButton = el("button", className, label);
            decisionButton.type = "button";
            decisionButton.setAttribute("data-decision", decision);
            decisionButton.setAttribute("data-run-id", String(run.id));
            decisionButton.setAttribute("data-tool-call-id", String(pending.tool_call_id));
            actions.append(decisionButton);
          }
          card.append(actions, el("p", "approval-live muted", ""));
        }
        conversation.append(card);
      }

      // The trace of the agent run: one entry per tool call, the output folded away
      if (run) {
        // A report that was approved and saved: link to it (the stored output of save_report)
        for (const call of run.tool_calls) {
          if (call.tool_name !== "save_report" || call.is_error) {
            continue;
          }
          let saved = null;
          try {
            saved = JSON.parse(call.output);
          } catch (error) {
            saved = null;
          }
          if (saved && saved.status === "created") {
            const reportLink = el("a", null, "Open report #" + saved.report_id);
            reportLink.setAttribute("href", "report.html?id=" + encodeURIComponent(saved.report_id));
            conversation.append(el("p", null), reportLink);
          }
        }

        const stepsButton = el("button", "secondary steps-toggle", "Steps (" + run.tool_calls.length + ")");
        stepsButton.type = "button";
        stepsButton.setAttribute("aria-expanded", "false");
        const stepsPanel = el("div", "steps", null);
        stepsPanel.hidden = true;
        if (run.tool_calls.length === 0) {
          stepsPanel.append(el("p", "muted", "The agent answered without calling a tool."));
        }
        const stepsList = el("ol", null, null);
        for (const call of run.tool_calls) {
          const entry = el("li", null, null);
          const outcome = call.is_error ? "error" : "ok";
          // pending, approved, rejected or expired; calls that needed no approval show nothing
          const approval = call.approval_status === "not_required" ? "" : " \u00b7 approval " + call.approval_status;
          // Same split as the approval card: long text arguments in a collapsed block
          const shortInput = {};
          const longInput = [];
          for (const [name, value] of Object.entries(call.input)) {
            if (typeof value === "string" && value.length > 300) {
              longInput.push([name, value]);
            } else {
              shortInput[name] = value;
            }
          }
          entry.append(
            el("strong", null, "Step " + call.step + ": " + call.tool_name),
            " \u00b7 " + outcome + approval + " \u00b7 " + call.duration_ms + " ms",
            el("pre", null, JSON.stringify(shortInput))
          );
          for (const [name, value] of longInput) {
            const longDetails = el("details", "long-argument");
            longDetails.append(el("summary", null, name + " (" + value.length + " characters)"), el("pre", null, value));
            entry.append(longDetails);
          }
          const output = el("details", null, null);
          output.append(
            el("summary", null, "Output (" + call.output.length + " characters)"),
            el("pre", "step-output", call.output)
          );
          entry.append(output);
          stepsList.append(entry);
        }
        stepsPanel.append(stepsList);
        stepsButton.addEventListener("click", function () {
          stepsPanel.hidden = !stepsPanel.hidden;
          stepsButton.setAttribute("aria-expanded", String(!stepsPanel.hidden));
        });
        conversation.append(stepsButton, stepsPanel);
      }
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
  askMode.disabled = true;
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
  streamStatus.textContent = askMode.value === "rag" ? "Searching the filings..." : "Thinking...";

  let text = "";
  let finished = false;
  let failure = null;
  try {
    await apiStream(
      "/chat/sessions/" + encodeURIComponent(selectedId) + "/messages",
      { question: question, ticker: askTicker.value || null, mode: askMode.value },
      function (streamEvent) {
        // Event types this page does not know are ignored
        if (streamEvent.type === "route") {
          streamStatus.textContent = "Routing: " + streamEvent.route;
        } else if (streamEvent.type === "step") {
          const args = streamEvent.args || {};
          const target = args.ticker || (Array.isArray(args.tickers) ? args.tickers.join(", ") : "");
          streamStatus.textContent =
            "Step " + streamEvent.step + ": " + streamEvent.tool + (target ? " (" + target + ")" : "");
        } else if (streamEvent.type === "step_result") {
          streamStatus.textContent =
            "Step " + streamEvent.step + " finished: " + streamEvent.tool + (streamEvent.ok ? "" : " (error)");
        } else if (streamEvent.type === "approval_required") {
          // The card itself comes from the reload after the stream
          streamStatus.textContent = "Waiting for your approval: " + streamEvent.tool;
        } else if (streamEvent.type === "sources") {
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
  askMode.disabled = false;
  askButton.disabled = false;
  if (failure === null) {
    askQuestion.value = "";
  } else {
    // The filings chat saves nothing when it fails (an agent run saves the question and its
    // trace). Either way the question stays in the box so it can be sent again
    showMessage(message, failure, "error");
  }
  // Show what is really stored (this also turns the [n] markers into buttons)
  await Promise.all([loadConversation(), loadSessions()]);
  askQuestion.focus();
});

// Approve or reject a paused run: the click on a card button sends the decision, which names the
// pending call, and streams the rest of the run like a question. A 409 (already decided, expired,
// another run active) or a 503 is shown as the server wrote it. Afterwards the conversation is
// loaded again, so the placeholder becomes the final answer
conversation.addEventListener("click", async function (event) {
  const button = event.target.closest("button[data-decision]");
  if (!button) {
    return;
  }
  const card = button.closest(".approval");
  for (const decisionButton of card.querySelectorAll("button[data-decision]")) {
    decisionButton.disabled = true;
  }
  askQuestion.disabled = true;
  askTicker.disabled = true;
  askMode.disabled = true;
  askButton.disabled = true;
  showMessage(message, null);
  streamStatus.textContent = "Sending your decision...";
  const liveText = card.querySelector(".approval-live");

  let text = "";
  let finished = false;
  let failure = null;
  try {
    await apiStream(
      "/agent/runs/" + encodeURIComponent(button.getAttribute("data-run-id")) + "/decision",
      {
        tool_call_id: Number(button.getAttribute("data-tool-call-id")),
        decision: button.getAttribute("data-decision"),
      },
      function (streamEvent) {
        // Event types this page does not know are ignored
        if (streamEvent.type === "decision") {
          streamStatus.textContent = "Decision recorded: " + streamEvent.decision;
        } else if (streamEvent.type === "step") {
          streamStatus.textContent = "Step " + streamEvent.step + ": " + streamEvent.tool;
        } else if (streamEvent.type === "step_result") {
          streamStatus.textContent =
            "Step " + streamEvent.step + " finished: " + streamEvent.tool + (streamEvent.ok ? "" : " (error)");
        } else if (streamEvent.type === "approval_required") {
          streamStatus.textContent = "Waiting for your approval: " + streamEvent.tool;
        } else if (streamEvent.type === "token") {
          text += streamEvent.text;
          liveText.textContent = text;
        } else if (streamEvent.type === "done") {
          finished = true;
        } else if (streamEvent.type === "error") {
          failure = streamEvent.detail;
        }
      }
    );
    if (!finished && failure === null) {
      failure = "The run was interrupted. Reload the page to see where it stands.";
    }
  } catch (error) {
    failure = error.message;
  }

  streamStatus.textContent = "";
  askQuestion.disabled = false;
  askTicker.disabled = false;
  askMode.disabled = false;
  askButton.disabled = false;
  if (failure !== null) {
    showMessage(message, failure, "error");
  }
  await Promise.all([loadConversation(), loadSessions()]);
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
