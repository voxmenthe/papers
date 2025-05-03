# memory.py content based on the plan's skeleton
import sqlite3
import uuid
import numpy as np
from scipy import spatial
import os
from sentence_transformers import SentenceTransformer
from typing import List

class MemoryService:
    """
    Handles storage and retrieval of PDF text chunks, embeddings,
    and chat history using SQLite and numpy files.
    """
    def __init__(self, db_path="memory.db", embeddings_dir=".memory_db/embeddings"):
        self.db_path = db_path
        self.embeddings_dir = embeddings_dir
        self.db = None # Initialized in _setup
        # Initialize the Sentence Transformer model
        # Using a common, lightweight model. Consider making this configurable.
        self.embedder = None # Defer loading
        self._embedder_loaded = False # Flag to track loading
        self._setup()

    def _setup(self):
        """Initializes the database connection and schema if needed."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        with self.db:
            # PDFs Table
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS pdfs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    title TEXT,
                    added_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    summary_text TEXT
                )
            """)
            # Chunks Table
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    pdf_id INTEGER NOT NULL,
                    page_no INTEGER,
                    text TEXT NOT NULL,
                    emb_path TEXT NOT NULL,
                    FOREIGN KEY (pdf_id) REFERENCES pdfs(id) ON DELETE CASCADE
                )
            """)
            # Chats Table
            self.db.execute("""
                 CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prompt_name TEXT,
                    doc_ids TEXT -- Comma-separated list of pdf_ids
                 )
            """)
            # Messages Table
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    text TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)
            # Message Summaries Table (Optional, for rolling summaries)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS msg_summaries (
                    chat_id INTEGER PRIMARY KEY,
                    summary_text TEXT,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)
            # Consider adding FTS5 for text search later if needed
            # self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content=chunks, content_rowid=id)")


    def _get_embedder(self) -> SentenceTransformer | None:
        """Loads the SentenceTransformer model on first access."""
        if not self._embedder_loaded:
            print("Loading SentenceTransformer model (this may take a moment)...")
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("SentenceTransformer model loaded successfully.")
            except Exception as e:
                print(f"[bold red]Error loading SentenceTransformer model: {e}[/]")
                self.embedder = None # Explicitly set to None on error
            finally:
                self._embedder_loaded = True # Mark as attempted load
        return self.embedder

    def add_pdf(self, filename, title=None):
        """Adds a new PDF record."""
        with self.db:
            cur = self.db.execute(
                "INSERT INTO pdfs (filename, title) VALUES (?, ?) ON CONFLICT(filename) DO NOTHING RETURNING id",
                (filename, title or os.path.basename(filename))
            )
            result = cur.fetchone()
            return result[0] if result else None # Return the new or existing ID

    def add_chunk(self, pdf_id, page_no, text):
        """Adds a text chunk, generates embedding, and saves it."""
        if not text or text.isspace():
            # print(f"Skipping empty chunk for pdf {pdf_id}, page {page_no}")
            return None

        chunk_id = str(uuid.uuid4())
        # Get the embedder (loads if not already loaded)
        embedder = self._get_embedder()

        # Generate actual embedding if embedder is available
        emb = None
        if embedder:
            try:
                emb = embedder.encode(text)
            except Exception as e:
                print(f"Error generating embedding for chunk {chunk_id}: {e}")
                # Decide how to handle - skip chunk? save without embedding?
                return None # Skip this chunk if embedding fails
        else:
            print("Embedder not available, skipping embedding generation.")
            # If no embedder, we can't really do semantic search. 
            # Maybe store placeholder or skip? For now, skip chunk.
            return None

        emb_filename = f"{chunk_id}.npy"
        emb_filepath = os.path.join(self.embeddings_dir, emb_filename)
        np.save(emb_filepath, emb)

        with self.db:
            self.db.execute(
              "INSERT INTO chunks (id, pdf_id, page_no, text, emb_path) VALUES (?, ?, ?, ?, ?)",
              (chunk_id, pdf_id, page_no, text, emb_filename) # Store only filename
            )
        return chunk_id

    def query(self, query_text, top_k=5, filter_pdf_ids=None):
        """Finds top_k relevant chunks based on semantic similarity."""
        # Get the embedder (loads if not already loaded)
        embedder = self._get_embedder()

        # Generate query embedding if embedder is available
        q_emb = None
        if embedder:
            try:
                q_emb = embedder.encode(query_text)
            except Exception as e:
                print(f"Error generating query embedding: {e}")
                return [] # Cannot perform query without embedding
        else:
            print("Embedder not available, cannot perform semantic query.")
            return [] # Cannot perform query

        placeholders = ""
        params = []
        if filter_pdf_ids:
            placeholders = " WHERE pdf_id IN ({})".format(",".join("?"*len(filter_pdf_ids)))
            params.extend(filter_pdf_ids)

        # Fetch chunks matching the filter
        cur = self.db.execute(
            f"SELECT id, text, emb_path FROM chunks{placeholders}",
            params
        )
        rows = cur.fetchall()

        scored = []
        if not rows:
            print("No relevant chunks found in the database for the given filter.")
            return []

        # Calculate similarity scores
        for chunk_id, txt, emb_filename in rows:
            emb_filepath = os.path.join(self.embeddings_dir, emb_filename)
            if not os.path.exists(emb_filepath):
                print(f"Warning: Embedding file not found, skipping chunk: {emb_filepath}")
                continue
            try:
                emb = np.load(emb_filepath)
                # Ensure consistent embedding dimensions if needed (though less likely with same model)
                if emb.shape != q_emb.shape:
                    print(f"Warning: Embedding shape mismatch for chunk {chunk_id}. Expected {q_emb.shape}, got {emb.shape}. Skipping.")
                    continue
                # Cosine Similarity = 1 - Cosine Distance
                score = 1 - spatial.distance.cosine(q_emb, emb)
                scored.append((score, txt))
            except ValueError as ve:
                 print(f"Error calculating similarity for {emb_filepath} (possibly empty/invalid embedding): {ve}")
                 continue # Skip chunk if similarity calc fails
            except Exception as e:
                print(f"Error loading or comparing embedding {emb_filepath}: {e}")
                continue # Skip chunk on other errors

        # Sort by score (descending) and return top_k texts
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:top_k]]

    def get_pdf_summary(self, pdf_id):
        """Retrieves the stored summary for a PDF."""
        cur = self.db.execute("SELECT summary_text FROM pdfs WHERE id = ?", (pdf_id,))
        result = cur.fetchone()
        return result[0] if result else None

    def update_pdf_summary(self, pdf_id, summary):
        """Updates the summary for a given PDF."""
        with self.db:
            self.db.execute("UPDATE pdfs SET summary_text = ? WHERE id = ?", (summary, pdf_id))

    def list_pdfs(self):
        """Lists all imported PDFs with their IDs and summaries."""
        cur = self.db.execute("SELECT id, filename, title, summary_text FROM pdfs ORDER BY added_ts DESC")
        return cur.fetchall()

    def get_pdf_id_by_filename(self, filename: str) -> int | None:
        """Retrieves the ID of a PDF by its filename, returns None if not found."""
        with self.db:
            cur = self.db.execute("SELECT id FROM pdfs WHERE filename = ?", (filename,))
            result = cur.fetchone()
            return result[0] if result else None

    def delete_embeddings_for_pdf(self, pdf_id: int):
        """Deletes embedding files associated with a given pdf_id."""
        with self.db:
            cur = self.db.execute("SELECT emb_path FROM chunks WHERE pdf_id = ?", (pdf_id,))
            emb_paths = cur.fetchall()
            deleted_count = 0
            for (emb_filename,) in emb_paths:
                if emb_filename: # Check if emb_path is not None
                    emb_filepath = os.path.join(self.embeddings_dir, emb_filename)
                    try:
                        if os.path.exists(emb_filepath):
                            os.remove(emb_filepath)
                            deleted_count += 1
                    except OSError as e:
                        print(f"Error deleting embedding file {emb_filepath}: {e}")
            print(f"Deleted {deleted_count} embedding files for pdf_id {pdf_id}.")

    def delete_chunks_by_pdf_id(self, pdf_id: int):
        """Deletes chunk records from the database for a given pdf_id."""
        with self.db:
            # First, delete associated embeddings
            self.delete_embeddings_for_pdf(pdf_id)
            # Then, delete the chunk entries
            cur = self.db.execute("DELETE FROM chunks WHERE pdf_id = ?", (pdf_id,))
            print(f"Deleted {cur.rowcount} chunks from DB for pdf_id {pdf_id}.")

    def delete_pdf_by_id(self, pdf_id: int):
        """Deletes the PDF record itself from the pdfs table."""
        with self.db:
            # Ensure associated chunks are deleted first
            self.delete_chunks_by_pdf_id(pdf_id)
            # Now delete the PDF record
            cur = self.db.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
            if cur.rowcount > 0:
                print(f"Deleted PDF record with id {pdf_id}.")
            else:
                print(f"Warning: No PDF record found with id {pdf_id} to delete.")

    def get_latest_pdf_id(self) -> int | None:
        """Returns the ID of the most recently added PDF, or None if no PDFs exist."""
        with self.db:
            cur = self.db.execute("SELECT id FROM pdfs ORDER BY added_ts DESC LIMIT 1")
            result = cur.fetchone()
            return result[0] if result else None

    def list_pdfs(self):
        """Lists all imported PDFs with their IDs and summaries."""
        cur = self.db.execute("SELECT id, filename, title, summary_text FROM pdfs ORDER BY added_ts DESC")
        return cur.fetchall()

    # --- Chat related methods ---

    def start_chat(self, doc_ids=None, prompt_name="default"):
        """Starts a new chat session."""
        doc_ids_str = ",".join(map(str, doc_ids)) if doc_ids else None
        with self.db:
            cur = self.db.execute(
                "INSERT INTO chats (doc_ids, prompt_name) VALUES (?, ?) RETURNING id",
                (doc_ids_str, prompt_name)
            )
            return cur.fetchone()[0]

    def save_message(self, chat_id, role, text):
        """Saves a message to the chat history."""
        with self.db:
            self.db.execute(
                "INSERT INTO messages (chat_id, role, text) VALUES (?, ?, ?)",
                (chat_id, role, text)
            )

    def get_chat_history(self, chat_id):
        """Retrieves all messages for a given chat session."""
        cur = self.db.execute(
            "SELECT role, text, ts FROM messages WHERE chat_id = ? ORDER BY ts ASC",
            (chat_id,)
        )
        return cur.fetchall()

    def get_chat_details(self, chat_id):
         """Retrieves details about a specific chat session."""
         cur = self.db.execute(
             "SELECT id, started_ts, prompt_name, doc_ids FROM chats WHERE id = ?",
             (chat_id,)
         )
         return cur.fetchone()

    def get_chat_doc_ids(self, chat_id: int) -> List[int] | None:
        """Retrieves the list of associated PDF IDs for a given chat."""
        cur = self.db.execute("SELECT doc_ids FROM chats WHERE id = ?", (chat_id,))
        result = cur.fetchone()
        if result and result[0]:
            try:
                # Split comma-separated string and convert to integers
                return [int(doc_id) for doc_id in result[0].split(',')]
            except ValueError:
                print(f"Warning: Invalid doc_ids format found for chat {chat_id}: {result[0]}")
                return None # Return None if format is bad
        return None # Return None if no doc_ids associated or chat not found

    def list_chats(self):
        """Lists all past chat sessions."""
        cur = self.db.execute(
            "SELECT c.id, c.started_ts, c.prompt_name, c.doc_ids, COUNT(m.id) as msg_count "
            "FROM chats c LEFT JOIN messages m ON c.id = m.chat_id "
            "GROUP BY c.id ORDER BY c.started_ts DESC"
        )
        return cur.fetchall()

    # --- TODO: Add methods for conversation summarization ---
    # def update_conversation_summary(self, chat_id, summary): ...
    # def get_conversation_summary(self, chat_id): ...

    def close(self):
        """Closes the database connection."""
        if self.db:
            self.db.close()
            self.db = None

# Example Usage (for testing)
if __name__ == '__main__':
    mem = MemoryService(db_path=".memory_db/test_memory.db")
    print("Database and tables set up.")

    # Example: Add PDF and chunks
    # pdf_id = mem.add_pdf("example.pdf", "Example Document")
    # if pdf_id:
    #     print(f"Added PDF: ID {pdf_id}")
    #     chunk_id1 = mem.add_chunk(pdf_id, 1, "This is the first chunk of text.")
    #     chunk_id2 = mem.add_chunk(pdf_id, 1, "This is another piece of text from page 1.")
    #     chunk_id3 = mem.add_chunk(pdf_id, 2, "Text from the second page.")
    #     print(f"Added chunks: {chunk_id1}, {chunk_id2}, {chunk_id3}")

    #     # Example: Query
    #     query = "Tell me about the first page"
    #     results = mem.query(query, top_k=2, filter_pdf_ids=[pdf_id])
    #     print(f"\nQuery: '{query}'")
    #     print("Results:")
    #     for res in results:
    #         print(f"- {res}")

    #     # Example: List PDFs
    #     print("\nListing PDFs:")
    #     for pid, fname, title, summary in mem.list_pdfs():
    #         print(f"ID: {pid}, File: {fname}, Title: {title}, Summary: {summary or 'N/A'}")

    # else:
    #     print("PDF 'example.pdf' might already exist.")


    # Example: Chat
    # chat_id = mem.start_chat(doc_ids=[pdf_id] if pdf_id else None)
    # print(f"\nStarted Chat ID: {chat_id}")
    # mem.save_message(chat_id, "user", "What is on page 1?")
    # mem.save_message(chat_id, "assistant", "Page 1 contains the first chunk and another piece of text.")
    # print("Chat History:")
    # for role, text, ts in mem.get_chat_history(chat_id):
    #     print(f"[{ts}] {role.capitalize()}: {text}")

    # print("\nListing Chats:")
    # for cid, start_ts, p_name, d_ids, m_count in mem.list_chats():
    #     print(f"ID: {cid}, Started: {start_ts}, Prompt: {p_name}, Docs: {d_ids}, Msgs: {m_count}")


    mem.close()
    print("\nDatabase connection closed.")
    # Clean up test db
    # if os.path.exists(".memory_db/test_memory.db"):
    #    os.remove(".memory_db/test_memory.db")
    #    # Potentially remove embeddings too if needed for clean test runs