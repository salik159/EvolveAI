import os
import sys
import requests

from dotenv import load_dotenv
from mem0 import Memory
from ollama import chat


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

OLLAMA_URL = "http://localhost:11434"

OLLAMA_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

COLLECTION_NAME = "evolveai_memory"

USER_ID = "salik"


# ============================================================
# CHECK OLLAMA
# ============================================================

print("🤖 Checking Ollama...")

try:

    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=3
    )

    if response.status_code != 200:

        print("❌ Ollama is not running.")
        sys.exit(1)

    print("✅ Ollama is running!")

except Exception as error:

    print(f"❌ Ollama connection failed: {error}")
    sys.exit(1)


# ============================================================
# CHECK MODELS
# ============================================================

try:

    models = response.json().get("models", [])

    installed_models = [
        model.get("name", "")
        for model in models
    ]

    if not any(
        name.startswith(OLLAMA_MODEL)
        for name in installed_models
    ):

        print(
            f"❌ {OLLAMA_MODEL} is not installed."
        )

        sys.exit(1)


    if not any(
        name.startswith(EMBED_MODEL)
        for name in installed_models
    ):

        print(
            f"❌ {EMBED_MODEL} is not installed."
        )

        sys.exit(1)


    print("✅ Required Ollama models found!")

except Exception as error:

    print(
        f"❌ Model check failed: {error}"
    )

    sys.exit(1)


# ============================================================
# QDRANT
# ============================================================

if not QDRANT_URL or not QDRANT_API_KEY:

    print(
        "❌ QDRANT_URL or QDRANT_API_KEY "
        "is missing from .env"
    )

    sys.exit(1)


print("☁️ Connecting to Qdrant Cloud...")


# ============================================================
# MEM0 CONFIG
# ============================================================

print("🧠 Starting memory system...")


memory_config = {

    "version": "v1.1",

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    "embedder": {

        "provider": "ollama",

        "config": {

            "model": EMBED_MODEL,

            "ollama_base_url": OLLAMA_URL

        }

    },

    # --------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------

    "vector_store": {

        "provider": "qdrant",

        "config": {

            "url": QDRANT_URL,

            "api_key": QDRANT_API_KEY,

            "collection_name": COLLECTION_NAME,

            "embedding_model_dims": 768

        }

    },

    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    "llm": {

        "provider": "ollama",

        "config": {

            "model": OLLAMA_MODEL,

            "temperature": 0.1,

            "max_tokens": 150,

            "ollama_base_url": OLLAMA_URL

        }

    }

}


try:

    memory = Memory.from_config(
        memory_config
    )

    print(
        "☁️ Connected to Qdrant Cloud!"
    )

except Exception as error:

    print(
        "❌ Memory initialization failed:"
    )

    print(error)

    sys.exit(1)


# ============================================================
# SEARCH CLOUD MEMORY
# ============================================================

def search_memory(user_input):

    try:

        print(
            "☁️ Searching cloud memory...",
            end=" ",
            flush=True
        )


        result = memory.search(

            query=user_input,

            filters={
                "user_id": USER_ID
            },

            limit=3

        )


        # ----------------------------------------------------
        # MEM0 RESPONSE
        # ----------------------------------------------------

        if isinstance(result, dict):

            results = result.get(
                "results",
                []
            )

        elif isinstance(result, list):

            results = result

        else:

            results = []


        memories = []


        # ----------------------------------------------------
        # EXTRACT ACTUAL MEMORY
        # ----------------------------------------------------

        for item in results:

            if not isinstance(item, dict):

                continue


            text = item.get(
                "memory"
            )


            if not text:

                continue


            memories.append(
                str(text).strip()
            )


        # Only use the best 2
        memories = memories[:2]


        if memories:

            print(
                "✅",
                flush=True
            )

        else:

            print(
                "ℹ️ no relevant memory",
                flush=True
            )


        return memories


    except Exception as error:

        print()

        print(
            f"⚠️ Memory search failed: {error}"
        )

        return []


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_ollama(
    user_input,
    memories,
    history=None
):


    # ========================================================
    # MEMORY FOUND
    # ========================================================

    if memories:

        memory_context = "\n".join(

            f"- {memory}"

            for memory in memories

        )


        system_prompt = f"""
You are EvolveAI.

You have access to verified information
retrieved from the user's long-term memory.

LONG-TERM MEMORY:

{memory_context}

Use the memory only when it answers
or helps answer the user's question.

Rules:

- Do not invent personal information.
- Do not change names.
- Do not invent relationships.
- Do not invent events.
- Do not add details that aren't in memory.
- If the memory directly answers the question,
  answer directly.
- Keep the response concise.
"""


    # ========================================================
    # NO MEMORY
    # ========================================================

    else:

        system_prompt = """
You are EvolveAI, a helpful AI assistant.

No relevant personal memory was found
in the user's cloud memory.

Answer the question normally using
your general knowledge.

Do not invent personal information.

Keep the answer concise.
"""


    # ========================================================
    # OLLAMA STREAMING
    # ========================================================

    try:

        print(
            "🦙 EvolveAI:",
            flush=True
        )


        stream = chat(

            model=OLLAMA_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                # Earlier turns of this conversation (optional).
                *(history or []),

                {
                    "role": "user",
                    "content": user_input
                }

            ],

            options={

                "temperature": 0.1,

                "num_predict": 150

            },

            stream=True

        )


        answer = ""


        for chunk in stream:

            text = (
                chunk[
                    "message"
                ][
                    "content"
                ]
            )


            print(
                text,
                end="",
                flush=True
            )


            answer += text


        print()

        return answer.strip()


    except Exception as error:

        print(
            f"❌ Ollama error: {error}"
        )

        return ""


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(
    user_input
):

    try:

        print(
            "☁️ Saving to cloud...",
            end=" ",
            flush=True
        )


        # ====================================================
        # IMPORTANT
        #
        # infer=False means:
        #
        # SAVE EXACTLY WHAT THE USER SAID.
        #
        # No second Ollama call.
        # No memory extraction LLM.
        #
        # This makes saving MUCH faster.
        # ====================================================

        memory.add(

            [
                {
                    "role": "user",
                    "content": user_input
                }
            ],

            user_id=USER_ID,

            infer=False

        )


        print(
            "✅",
            flush=True
        )


    except Exception as error:

        print()

        print(
            f"⚠️ Cloud save failed: {error}"
        )


# ============================================================
# SHOULD SAVE?
# ============================================================

def should_save(user_input):

    text = user_input.strip()

    if not text:

        return False


    lower = text.lower()


    # Don't save obvious questions.
    question_starters = (

        "who ",
        "what ",
        "where ",
        "when ",
        "why ",
        "how ",
        "can ",
        "could ",
        "would ",
        "is ",
        "are ",
        "do ",
        "does ",
        "did ",
        "tell me ",
        "explain "

    )


    if lower.startswith(
        question_starters
    ):

        return False


    return True


# ============================================================
# COMMAND-LINE CHAT   (python main.py --cli)
# ============================================================

def run_cli():

    # ============================================================
    # START
    # ============================================================

    print()

    print(
        "🤖 EvolveAI is ready!"
    )

    print(
        "🧠 Long-term cloud memory is active!"
    )

    print(
        "🛠️ Tools are active!"
    )

    print(
        "🦙 Running locally with Ollama!"
    )

    print(
        "💰 No AI API costs!"
    )

    print(
        "⚡ Fast mode enabled!"
    )

    print(
        "☁️ Memory source: Qdrant Cloud"
    )

    print(
        "Type 'exit' to stop."
    )

    print()


    # ============================================================
    # MAIN LOOP
    # ============================================================

    while True:

        try:

            user_input = input(
                "You: "
            ).strip()


            if not user_input:

                continue


            if user_input.lower() in [
                "exit",
                "quit"
            ]:

                print(
                    "\n👋 Goodbye!"
                )

                break


            # ====================================================
            # 1. SEARCH CLOUD
            # ====================================================

            memories = search_memory(
                user_input
            )


            # ====================================================
            # 2. ASK OLLAMA
            # ====================================================

            answer = ask_ollama(

                user_input,

                memories

            )


            # ====================================================
            # 3. SAVE USER INFORMATION
            #
            # IMPORTANT:
            # This is synchronous.
            #
            # We don't start the next question until the cloud
            # save is finished.
            #
            # This guarantees that:
            #
            # "my full name is Salik Kamal"
            #
            # is actually in Qdrant before:
            #
            # "what is my full name?"
            #
            # ====================================================

            if should_save(
                user_input
            ):

                save_memory(
                    user_input
                )


            print()


        except KeyboardInterrupt:

            print(
                "\n\n👋 EvolveAI stopped."
            )

            break


        except Exception as error:

            print(
                f"\n❌ Unexpected error: {error}"
            )


# ============================================================
# WEB CHAT   (python main.py)
#
# Needs Flask:  pip install flask
# ============================================================

CHAT_HOST = "127.0.0.1"
CHAT_PORT = int(os.getenv("CHAT_PORT", "8000"))

MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_MESSAGES = 10

CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>EvolveAI</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #eef1ef; --surface: #fff; --ink: #17211d; --muted: #5f6b66; --line: #dfe4e1;
  --accent: #17695a; --accent-ink: #fff; --ai: #f0f3f1; --danger: #a4281c; --danger-bg: #fdf0ee;
  --code-bg: #14201c; --code-ink: #e3ece7; --ok: #1f9d55; --off: #b42318;
  --sans: "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "Cascadia Code", Consolas, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b100e; --surface: #121916; --ink: #e6ece8; --muted: #92a199; --line: #25302b;
    --accent: #2a8f7b; --ai: #1a231f; --danger: #f2a49b; --danger-bg: #2a1714;
    --code-bg: #0a0f0d; --ok: #3ccf7e; --off: #f27a6e;
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--ink); font: 16px/1.55 var(--sans); }
button, textarea { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.app { display: flex; flex-direction: column; height: 100dvh; max-width: 820px; margin: 0 auto;
  background: var(--surface); border-inline: 1px solid var(--line); }
header { display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: calc(12px + env(safe-area-inset-top, 0px)) 20px 12px; border-bottom: 1px solid var(--line); }
h1 { margin: 0; font-size: 1.05rem; font-weight: 650; letter-spacing: -0.01em; }
.status { display: flex; align-items: center; gap: 6px; font-size: .8rem; color: var(--muted); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.status[data-state="online"] .dot { background: var(--ok); }
.status[data-state="offline"] .dot { background: var(--off); }
.model { margin-inline-start: 6px; opacity: .8; }
.ghost { background: none; border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px;
  font-size: .85rem; cursor: pointer; color: var(--ink); }
.ghost:hover:not(:disabled) { background: var(--ai); }
.ghost:disabled { opacity: .45; cursor: default; }

#messages { flex: 1; overflow-y: auto; padding: 20px; scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { #messages { scroll-behavior: auto; } }
.empty { max-width: 460px; margin: 12vh auto 0; text-align: center; }
.empty h2 { margin: 0 0 6px; font-size: 1.4rem; font-weight: 650; letter-spacing: -0.015em; }
.empty p { margin: 0 0 20px; color: var(--muted); }
.chips { display: flex; flex-direction: column; gap: 8px; }
.chip { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px;
  text-align: start; cursor: pointer; }
.chip:hover:not(:disabled) { border-color: var(--accent); }

.row { display: flex; margin: 0 0 14px; }
.row.user { justify-content: flex-end; }
.col { max-width: min(88%, 640px); min-width: 0; }
.row.user .col { display: flex; flex-direction: column; align-items: flex-end; }
.bubble { padding: 10px 14px; border-radius: 16px; overflow-wrap: anywhere; }
.user .bubble { background: var(--accent); color: var(--accent-ink); border-end-end-radius: 4px; white-space: pre-wrap; }
.ai .bubble { background: var(--ai); border-end-start-radius: 4px; }
.bubble p { margin: 0 0 .6em; } .bubble > :last-child { margin-bottom: 0; }
.bubble ul, .bubble ol { margin: 0 0 .6em; padding-inline-start: 1.3em; }
.bubble code { font: .88em var(--mono); background: rgba(127,127,127,.18); padding: .1em .35em; border-radius: 4px; }
.bubble pre { margin: .2em 0 .7em; padding: 12px 14px; background: var(--code-bg); color: var(--code-ink);
  border-radius: 10px; overflow-x: auto; max-width: 100%; }
.bubble pre code { background: none; padding: 0; font-size: .85rem; white-space: pre; color: inherit; }
.copy { margin-top: 4px; padding: 2px 6px; background: none; border: 0; border-radius: 6px; font-size: .78rem;
  color: var(--muted); cursor: pointer; }
.copy:hover { color: var(--ink); background: var(--ai); }

.typing { display: inline-flex; gap: 5px; align-items: center; padding: 6px 2px; }
.typing i { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); animation: blink 1.2s infinite ease-in-out; }
.typing i:nth-child(2) { animation-delay: .15s; } .typing i:nth-child(3) { animation-delay: .3s; }
@keyframes blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }
.slow { display: none; margin: 4px 0 0; font-size: .8rem; color: var(--muted); }
.slow.show { display: block; }
.vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }

.error { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 14px; border-radius: 12px;
  background: var(--danger-bg); color: var(--danger); font-size: .92rem; }
.error button { background: none; border: 1px solid currentColor; border-radius: 8px; padding: 3px 10px;
  cursor: pointer; font-size: .85rem; color: inherit; }

.composer { padding: 10px 20px calc(14px + env(safe-area-inset-bottom, 0px)); border-top: 1px solid var(--line); }
.field { display: flex; align-items: flex-end; gap: 8px; border: 1px solid var(--line); border-radius: 14px;
  padding: 6px 6px 6px 14px; background: var(--surface); }
.field:focus-within { border-color: var(--accent); }
textarea { flex: 1; resize: none; border: 0; outline: 0; background: none; padding: 7px 0; max-height: 160px; line-height: 1.45; }
#send { flex: none; width: 38px; height: 38px; border: 0; border-radius: 10px; background: var(--accent);
  color: var(--accent-ink); cursor: pointer; display: grid; place-items: center; }
#send:disabled { opacity: .4; cursor: default; }
#send svg { width: 18px; height: 18px; }
#send .spin { display: none; width: 16px; height: 16px; border: 2px solid currentColor; border-top-color: transparent;
  border-radius: 50%; animation: spin .8s linear infinite; }
#send.busy svg { display: none; } #send.busy .spin { display: block; }
@keyframes spin { to { transform: rotate(360deg); } }
.hint { margin: 6px 2px 0; font-size: .75rem; color: var(--muted); }

@media (max-width: 640px) {
  .app { border-inline: 0; } header, #messages { padding-inline: 14px; } .composer { padding-inline: 12px; }
  .col { max-width: 92%; } .hint, .model { display: none; } .empty { margin-top: 6vh; }
}
@media (prefers-reduced-motion: reduce) { .typing i, #send .spin { animation-duration: 3s; } }
</style>
</head>
<body>
<div class="app">
  <header>
    <div>
      <h1>EvolveAI</h1>
      <div class="status" id="status" data-state="checking" title="">
        <span class="dot"></span><span id="status-text">Checking…</span><span class="model" id="model"></span>
      </div>
    </div>
    <button class="ghost" id="new-chat" type="button" disabled>New chat</button>
  </header>

  <main id="messages" role="log" aria-live="polite" aria-label="Conversation">
    <section class="empty" id="empty">
      <h2>What can I help with?</h2>
      <p>Ask a question, or tell me something to remember. I run on your computer and keep long-term memory.</p>
      <div class="chips">
        <button class="chip" type="button">Explain what an API is in simple terms</button>
        <button class="chip" type="button">Write a Python function that reverses a string</button>
        <button class="chip" type="button">What do you remember about me?</button>
      </div>
    </section>
  </main>

  <div class="composer">
    <div class="field">
      <textarea id="input" rows="1" maxlength="4000" placeholder="Message EvolveAI" aria-label="Message"></textarea>
      <button id="send" type="button" aria-label="Send message" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
        <span class="spin" aria-hidden="true"></span>
      </button>
    </div>
    <p class="hint">Enter to send, Shift+Enter for a new line.</p>
  </div>
</div>

<script>
const MAX_HISTORY = 10, TIMEOUT_MS = 180000, SLOW_MS = 10000;
const $ = (id) => document.getElementById(id);
const box = $("messages"), input = $("input"), sendBtn = $("send"), newBtn = $("new-chat"), empty = $("empty");
let history = [], busy = false;

/* ---------- safe mini-markdown: escape first, then add a few tags ---------- */
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const inline = (s) => s.split(/(`[^`\n]+`)/g).map((p, i) => i % 2 ? `<code>${p.slice(1, -1)}</code>`
  : p.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")).join("");

function renderMarkdown(src) {
  const blocks = [];
  let text = src.replace(/\u0000/g, "").replace(/```[\w+-]*\n?([\s\S]*?)(?:```|$)/g, (m, code) => {
    blocks.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\n\u0000${blocks.length - 1}\u0000\n`;
  });
  let html = "", list = null, para = [];
  const flush = () => { if (para.length) html += `<p>${inline(para.join("<br>"))}</p>`; para = []; };
  const close = () => { if (list) html += `</${list}>`; list = null; };
  for (const line of esc(text).split("\n")) {
    const ph = line.match(/^\u0000(\d+)\u0000$/), li = line.match(/^\s*(?:([-*])|\d+[.)])\s+(.*)/), h = line.match(/^#{1,6}\s+(.*)/);
    if (ph) { flush(); close(); html += blocks[+ph[1]] || ""; }
    else if (li) { flush(); const tag = li[1] ? "ul" : "ol"; if (list !== tag) { close(); html += `<${tag}>`; list = tag; } html += `<li>${inline(li[2])}</li>`; }
    else if (h) { flush(); close(); html += `<p><strong>${inline(h[1])}</strong></p>`; }
    else if (!line.trim()) { flush(); close(); }
    else { close(); para.push(line); }
  }
  flush(); close();
  return html;
}

/* ---------- UI helpers ---------- */
function scrollDown() { box.scrollTo({ top: box.scrollHeight }); }
function addRow(kind, colHtml) {
  empty.hidden = true;
  const row = document.createElement("div");
  row.className = `row ${kind}`;
  row.innerHTML = `<div class="col">${colHtml}</div>`;
  box.appendChild(row); scrollDown();
  return row;
}
function addUser(text) {
  const row = addRow("user", `<div class="bubble"></div>`);
  row.querySelector(".bubble").textContent = text;
}
function addAssistant(text) {
  const row = addRow("ai", `<div class="bubble">${renderMarkdown(text)}</div><button class="copy" type="button">Copy</button>`);
  const btn = row.querySelector(".copy");
  btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(text); }
    catch { const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove(); }
    btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy"), 1500);
  });
}
function addTyping() {
  const row = addRow("ai", `<div class="bubble"><span class="typing" role="status"><i></i><i></i><i></i><span class="vh">EvolveAI is typing</span></span>
    <p class="slow">Still working. The first reply can take a while if the model is loading.</p></div>`);
  const timer = setTimeout(() => row.querySelector(".slow").classList.add("show"), SLOW_MS);
  return { remove() { clearTimeout(timer); row.remove(); } };
}
function addError(message, text) {
  const row = addRow("ai", `<div class="error"><span></span><button type="button">Retry</button></div>`);
  row.querySelector("span").textContent = message;
  row.querySelector("button").addEventListener("click", () => { row.remove(); send(text, true); });
}
function setStatus(state, model) {
  $("status").dataset.state = state;
  $("status-text").textContent = { online: "Online", offline: "Offline", checking: "Checking…" }[state];
  $("model").textContent = state === "online" && model ? model : "";
}
function refreshControls() {
  sendBtn.disabled = busy || !input.value.trim();
  sendBtn.classList.toggle("busy", busy);
  newBtn.disabled = busy || (!history.length && empty.hidden === false);
  document.querySelectorAll(".chip").forEach((c) => (c.disabled = busy));
}
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; }

/* ---------- talk to the backend ---------- */
async function send(raw, isRetry = false) {
  const text = raw.trim();
  if (busy || !text) return;
  busy = true;
  if (!isRetry) { addUser(text); input.value = ""; autosize(); }
  refreshControls();
  const typing = addTyping();
  const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: ctrl.signal,
      body: JSON.stringify({ message: text, history: history.slice(-MAX_HISTORY) }),
    });
    let data = null;
    try { data = await res.json(); } catch {}
    if (!res.ok || !data || typeof data.reply !== "string") throw new Error((data && data.error) || `The server returned an error (${res.status}).`);
    typing.remove();
    addAssistant(data.reply);
    history.push({ role: "user", content: text }, { role: "assistant", content: data.reply });
    if ($("status").dataset.state !== "online") checkHealth();
  } catch (err) {
    typing.remove();
    const msg = err.name === "AbortError" ? "The model took too long to respond."
      : err.name === "TypeError" ? "Can't reach the server. Check that it is still running." : err.message;
    addError(msg, text);
    checkHealth();
  } finally {
    clearTimeout(timer); busy = false; refreshControls(); input.focus(); scrollDown();
  }
}

async function checkHealth() {
  try {
    const d = await (await fetch("/api/health")).json();
    setStatus(d.ok ? "online" : "offline", d.model);
  } catch { setStatus("offline"); }
}

function newChat() {
  if (busy) return;
  history = [];
  box.querySelectorAll(".row").forEach((r) => r.remove());
  empty.hidden = false; refreshControls(); input.focus();
}

/* ---------- events ---------- */
input.addEventListener("input", () => { autosize(); refreshControls(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); }
});
sendBtn.addEventListener("click", () => send(input.value));
newBtn.addEventListener("click", newChat);
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => send(c.textContent)));
document.addEventListener("visibilitychange", () => { if (!document.hidden) checkHealth(); });

checkHealth(); setInterval(() => { if (!document.hidden) checkHealth(); }, 30000);
refreshControls(); input.focus();
</script>
</body>
</html>
"""


def clean_history(raw_history):

    """Keep only valid user/assistant turns, newest MAX_HISTORY_MESSAGES."""

    if not isinstance(raw_history, list):

        return []

    cleaned = []

    for item in raw_history[-MAX_HISTORY_MESSAGES:]:

        if not isinstance(item, dict):

            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):

            continue

        if not isinstance(content, str) or not content.strip():

            continue

        cleaned.append(
            {
                "role": role,
                "content": content.strip()[:MAX_MESSAGE_CHARS]
            }
        )

    return cleaned


def run_web():

    try:

        from flask import Flask, Response, jsonify, request

    except ImportError:

        print("❌ Flask is not installed. Run: pip install flask")
        sys.exit(1)

    import threading

    app = Flask(__name__, static_folder=None)

    # One chat request at a time, like the command-line chat.
    chat_lock = threading.Lock()

    def error_response(message, status):

        return jsonify(error=message), status

    @app.get("/")
    def index():

        return Response(CHAT_HTML, mimetype="text/html")

    @app.get("/api/health")
    def health():

        try:

            r = requests.get(
                f"{OLLAMA_URL}/api/tags",
                timeout=2
            )

            ok = r.status_code == 200

        except requests.RequestException:

            ok = False

        return jsonify(ok=ok, model=OLLAMA_MODEL)

    @app.post("/api/chat")
    def chat_endpoint():

        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return error_response("Request body must be JSON.", 400)

        message = data.get("message")

        if not isinstance(message, str) or not message.strip():

            return error_response("Message cannot be empty.", 400)

        message = message.strip()

        if len(message) > MAX_MESSAGE_CHARS:

            return error_response(
                f"Message is too long (max {MAX_MESSAGE_CHARS} characters).",
                400
            )

        history = clean_history(data.get("history"))

        with chat_lock:

            try:

                memories = search_memory(message)

                answer = ask_ollama(message, memories, history)

                if not answer:

                    return error_response(
                        "The model did not return a response. "
                        "Check that Ollama is running.",
                        502
                    )

                if should_save(message):

                    save_memory(message)

                return jsonify(reply=answer)

            except Exception as error:

                print(f"❌ Chat request failed: {error}")

                return error_response(
                    "Something went wrong while generating a response.",
                    500
                )

    print()
    print(f"💬 EvolveAI chat is running at http://{CHAT_HOST}:{CHAT_PORT}")
    print("Press Ctrl+C to stop.")
    print()

    app.run(host=CHAT_HOST, port=CHAT_PORT, debug=False, threaded=True)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if "--cli" in sys.argv:

        run_cli()

    else:

        run_web()
