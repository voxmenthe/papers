### Mini-Project Plan: “Chat-With-Your-PDFs” (CLI-Only, No Concurrency)

#### 1. Scope & Goals
* Import a handful of PDFs.  
* Auto-summarize each document and store page-level chunks + embeddings.  
* Interactive CLI lets the user:
  1. List PDFs and their summaries (`/list` or `/show <id>`).  
  2. Start a chat session against selected PDFs (`/chat --docs 1,2`).  
  3. Load / switch system prompts (`/prompt use research_assistant`).  
  4. Recall past sessions or show conversation summaries (`/history`, `/history --summarize`).  

Target: usable prototype in 1–2 days.

---

#### 2. Minimal Architecture

```
CLI (click)
   │
   ▼
MemoryService  ← SQLite + FTS5 + .npy embeddings
   │
   ├─ pdf_import()   # parses + stores
   ├─ summary()      # one-shot LLM call per PDF
   └─ query_embed()  # semantic k-NN (vector cosine)
LLMClient (GeminiAPI or OpenAI or local)
```

No async / threading; everything runs sequentially.

---

#### 3. Data Model (SQLite)

| Table | Columns |
|-------|---------|
| pdfs | id PK, filename, title, added_ts, summary_text |
| chunks | id PK, pdf_id FK, page_no, text, emb_path |
| chats | id PK, started_ts, prompt_name, doc_ids (comma list) |
| messages | id PK, chat_id FK, role (‘user’/‘assistant’), ts, text |
| msg_summaries | chat_id FK, summary_text (rolling) |

Embeddings live in `.memory_db/embeddings/{chunk_id}.npy`.

---

#### 4. Key CLI Commands

```
pdf import <path>           # ingest & summarise
pdf list                    # show imported docs
pdf show <id>               # print summary

prompt list                 # canned prompts in prompts/*.txt
prompt use <name>           # sets current system prompt

chat start --docs 1,2       # new session
chat ask "question ?"       # uses current context
chat history [--summarize]  # list or summarise past sessions
```

All commands live under a single `chatpdf` executable (Click group).

---

#### 5. Implementation Steps

Week-Day | Task | Core Code | Status
---------|------|-----------|--------
D0.5 | `MemoryService` skeleton + schema bootstrap | `memory.py` | 
D0.5 | PDF importer (pdfplumber) → chunks (text only) | `importer.py` | 
D0.5 | CLI scaffolding (`click`) + `pdf`, `prompt`, `chat` sub-commands | `cli.py` | 
D1 | Interactive REPL implementation | `cli.py` | 
D1 | History commands, nice printing | `cli.py` | 
D1 | Prompt loader + switching (`prompts/*.txt`) | `cli.py` | (Basic)
 D1 | Chat loop: Basic structure for `ask` (no retrieval/LLM yet) | `chat.py` | (Basic)
 D1.5 | **LLM wrapper**: `google-generativeai` (Gemini) | `llm.py` (new) | Done
 D1.5 | **Integrate LLM**: Basic Q&A integration in `ask_question` | `chat.py`, `llm.py`, `cli.py` | Done
 D2 | **Implement Embeddings**: `sentence-transformers` -> `MemoryService` | `importer.py`, `memory.py` | **Next**
 D2 | **Implement Chat Retrieval**: Cosine sim query in `MemoryService` | `memory.py`, `chat.py` | To Do
 D2.5 | Integrate Retrieval + LLM in `chat_ask` | `chat.py`, `llm.py` | To Do
 D2.5 | Summarizer: LLM call per PDF on import | `importer.py`, `llm.py` | To Do
 D3 | Rolling conversation summary | `chat.py` | To Do
 D3 | README, demo GIF, `requirements.txt` finalization | — | To Do

Total ≈ 3 days focused work.

---

#### 6. Representative Code Snippets

h4. MemoryService (excerpt)
```python
class MemoryService:
    def __init__(self, db="memory.db"):
        self.db = sqlite3.connect(db, check_same_thread=False)
        self._setup()

    def add_chunk(self, pdf_id, page_no, text):
        cid = str(uuid4())
        emb = self.embedder.encode(text)
        np.save(f".memory_db/embeddings/{cid}.npy", emb)
        with self.db:
            self.db.execute(
              "INSERT INTO chunks VALUES (?,?,?,?,?)",
              (cid, pdf_id, page_no, text, f"embeddings/{cid}.npy"))
        return cid

    def query(self, query_text, top_k=5, filter_pdf_ids=None):
        q_emb = self.embedder.encode(query_text)
        cur = self.db.execute(
            "SELECT id, text, emb_path FROM chunks"
            + (" WHERE pdf_id IN ({})".format(",".join("?"*len(filter_pdf_ids)))
               if filter_pdf_ids else ""),
            filter_pdf_ids or [])
        rows = cur.fetchall()
        scored = []
        for cid, txt, path in rows:
            emb = np.load(f".memory_db/{path}")
            score = 1 - spatial.distance.cosine(q_emb, emb)
            scored.append((score, txt))
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_k]]
```

h4. Chat loop
```python
def chat_ask(chat_id, question):
    ctx_chunks = mem.query(question, top_k=4, filter_pdf_ids=_doc_ids(chat_id))
    prompt = _build_prompt(ctx_chunks, question, _system_prompt(chat_id))
    answer = llm(prompt)
    mem.save_message(chat_id, "user", question)
    mem.save_message(chat_id, "assistant", answer)
    return answer
```

---

#### 7. Files & Layout

```
chatpdf/
 ├─ cli.py          # entry-point
 ├─ importer.py
 ├─ chat.py
 ├─ memory.py
 ├─ llm.py
 ├─ prompt.py
 └─ prompts/
     ├─ default.txt
     └─ research_assistant.txt
.memory_db/  (created at runtime, git-ignored)
```

---

#### 8. Dependencies (`requirements.txt`)

```
click
pdfplumber
pymupdf
sentence-transformers
numpy
scipy        # for cosine
genai        # for Gemini API
openai       # or litellm/ollama optional
tabulate     # nice CLI tables
```

---

#### 9. Next Actions

_Current Status: Basic CLI structure with PDF import (text only), chat session management, and REPL is functional. Direct Q&A with the Gemini LLM is integrated and working, including chat history saving. The next major goal is to make the chat context-aware by using document content._

1.  **Implement Embeddings:**
    *   Add `sentence-transformers` to `pyproject.toml` and install.
    *   Initialize the `SentenceTransformer` model within `MemoryService` (`memory.py`).
    *   Modify `importer.py::import_pdf` (and potentially `memory.py::add_chunk`) to generate embeddings using the model when PDFs are imported/chunked.
    *   Ensure embeddings are saved correctly (e.g., as `.npy` files referenced in the DB).
2.  **Implement Chat Retrieval:**
    *   Update `memory.py::MemoryService::query` to load saved embeddings and perform cosine similarity search between the query embedding and chunk embeddings.
    *   Implement filtering based on `doc_ids` associated with the chat session.
3.  **Integrate Retrieval with LLM:**
    *   Update `chat.py::ask_question` to:
        *   Call `memory.query` to fetch relevant text chunks based on the user's question and the chat's associated `doc_ids`.
        *   Construct a prompt that includes these retrieved chunks as context along with the user's question (requires creating a `_build_prompt` helper function).
        *   Send this context-rich prompt to `llm.call_llm`.
4.  **Implement PDF Summarization:**
    *   Create a `summarize_pdf` function (likely in `importer.py` or `llm.py`) that uses the LLM to generate a summary for a PDF upon import.
    *   Update `pdf import` command and `MemoryService` to store and retrieve these summaries.
5.  **Refine and Test:** Thoroughly test the context retrieval and summarization features.

---
