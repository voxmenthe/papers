# chatpdf/chat.py
"""Very thin wrapper around MemoryService chat helpers."""
from __future__ import annotations
from typing import List, Tuple, Optional
from rich.console import Console
from .memory import MemoryService
from .llm import call_llm


console = Console()


def _build_prompt(question: str, context_chunks: List[str]) -> str:
    """Builds a prompt including retrieved context."""
    if not context_chunks:
        return question # No context, return original question

    context = "\n\n".join(context_chunks)
    # Simple prompt template, can be refined
    prompt = f"""Based on the following context, please answer the question.

Context:
---
{context}
---

Question: {question}

Answer:"""
    return prompt


def start_chat(mem: MemoryService, doc_ids: List[int] | None = None) -> int:
    """Starts a new chat session, returns the chat_id."""
    # Ensure doc_ids are integers if provided
    safe_doc_ids = [int(did) for did in doc_ids if str(did).isdigit()] if doc_ids else None
    return mem.start_chat(doc_ids=safe_doc_ids)


def ask_question(mem: MemoryService, question: str, chat_id: int | None = None) -> str:
    """Posts a question to a chat, retrieves answer from LLM, stores both."""
    if chat_id is None:
        chat_id = mem.get_latest_chat_id()
        if chat_id is None:
            return "[red]No active chat. Use 'chat start' first.[/]"

    console.log(f"Asking question to chat_id={chat_id}: {question}")

    # --- Context Retrieval --- #
    relevant_chunks = []
    # 1. Get associated doc_ids for the chat
    doc_ids = mem.get_chat_doc_ids(chat_id)
    if doc_ids:
        console.log(f"Retrieving context from doc IDs: {doc_ids} for chat {chat_id}")
        # 2. Query MemoryService for relevant chunks
        try:
            relevant_chunks = mem.query(question, top_k=3, filter_pdf_ids=doc_ids) # Get top 3 chunks
            if relevant_chunks:
                console.log(f"Retrieved {len(relevant_chunks)} context chunks.")
            else:
                console.log("No relevant context chunks found for this question.")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to retrieve context chunks: {e}[/]")
    else:
        console.log(f"No specific documents associated with chat {chat_id}, querying all documents.")
        # Optionally query without doc_id filter if no docs are linked? Or require docs?
        # For now, let's query all if no specific docs are linked
        try:
            relevant_chunks = mem.query(question, top_k=3)
            if relevant_chunks:
                console.log(f"Retrieved {len(relevant_chunks)} context chunks from all documents.")
            else:
                console.log("No relevant context chunks found in any document.")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to retrieve context chunks: {e}[/]")

    # 3. Build the prompt
    prompt = _build_prompt(question, relevant_chunks)
    # Optional: Log the prompt (can be long)
    # console.log(f"Generated prompt (first 100 chars): {prompt[:100]}...")

    # Call the LLM
    answer = call_llm(prompt)

    # Store question and answer
    try:
        mem.save_message(chat_id, "user", question)
        mem.save_message(chat_id, "assistant", answer)
        console.log(f"Stored question and answer for chat_id={chat_id}")
    except Exception as e:
        console.print(f"[bold red]Error saving message to database: {e}[/]")
        # Decide if we should still return the answer or an error message
        return f"[yellow]Received answer, but failed to save to history: {e}[/]\n\n{answer}"

    return answer # Return the raw answer for now


def show_history(mem: MemoryService, chat_id: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """Returns the message history for a chat (defaults to latest)."""
    if chat_id is None:
        chats = mem.list_chats()
        if not chats:
             raise click.ClickException("No chats found.")
        chat_id = chats[0]["id"]

    history = mem.get_chat_history(chat_id)
    if not history and not mem.get_chat_details(chat_id): # Check if chat ID itself is valid
         raise click.ClickException(f"Chat with ID {chat_id} not found or has no messages.")

    return history # Returns list of (role, text, ts) tuples
