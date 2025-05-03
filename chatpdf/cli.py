"""
chatpdf.cli
-----------
Command-line entry-point for the Chat-With-Your-PDFs tool.

Usage examples
$ chatpdf pdf import paper.pdf
$ chatpdf pdf list
$ chatpdf chat start --docs 1,2
$ chatpdf chat ask "What is the main finding?"
"""

from __future__ import annotations
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
import shlex
from dotenv import load_dotenv

from .memory import MemoryService
from .importer import import_pdf
from .chat import start_chat, ask_question, show_history

console = Console()
HISTORY_FILE = Path("~/.chatpdf_history").expanduser()

# Load environment variables from .env file, if it exists
# Specify the path to the user's home directory .env file
dotenv_path = Path('~/.env').expanduser()
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    console.log(f"Loaded environment variables from {dotenv_path}")
else:
    console.log("~/.env file not found, relying on system environment variables.")

# --- Global Memory Service --- #
# Initialize MemoryService instance once
# TODO: Make DB path configurable?
mem_service = MemoryService(db_path=".memory_db/memory.db")

# --- State for REPL --- #
active_chat_id: int | None = None

# --- Interactive REPL helpers --------------------------------------------- #
_COMMAND_WORDS = [
    "/exit",
    "/help",
    "show",
]

# Setup keybindings for meta+enter execution
kb = KeyBindings()

@kb.add("escape", "enter")
def _(event):
    event.cli.current_buffer.validate_and_handle()


def _run_repl_command(ctx: click.Context, command_line: str):
    """Parses and executes a command line string within the REPL."""
    global active_chat_id, mem_service # Access globals

    if not command_line.strip():
        return

    # Check for special REPL commands first
    if command_line.lower() == "/exit":
        raise EOFError()
    if command_line.lower() == "/help":
        console.print(main.get_help(ctx))
        # Also show subcommand help for convenience?
        # for cmd_name, cmd_obj in main.commands.items():
        #     console.print(f"\n--- {cmd_name.upper()} --- ")
        #     console.print(cmd_obj.get_help(click.Context(cmd_obj, parent=ctx)))
        return

    # Default to '/chat ask' if input doesn't start with '/' or '/chat'
    if not command_line.startswith('/'):
        if active_chat_id is None:
            # Start a chat implicitly first
            try:
                # Get the latest PDF ID
                latest_pdf_id = mem_service.get_latest_pdf_id()

                if latest_pdf_id is not None:
                    console.print(f"[dim]No active chat. Starting new chat linked to latest PDF (ID: {latest_pdf_id})...[/]")
                    # Pass the global mem_service and latest doc ID
                    chat_id = start_chat(mem=mem_service, doc_ids=[latest_pdf_id])
                    if chat_id:
                        active_chat_id = chat_id
                        # console.print(f"[dim]Started new chat (ID: {active_chat_id}) linked to PDF {latest_pdf_id}[/]")
                    else:
                        console.print("[bold red]Error starting implicit chat linked to PDF.[/]")
                        return
                else:
                    # No PDFs imported yet
                    console.print("[yellow]No PDFs imported yet. Cannot start implicit chat. Use '/pdf import <path>' first.[/]")
                    return # Don't proceed to ask question

            except Exception as e:
                console.print(f"[bold red]Error starting implicit chat: {e}[/]")
                return
        # If we successfully started a chat (or one was already active), format as /chat ask
        command_line = f"/chat ask \"{command_line}\"" # Wrap in quotes for shlex

    # Remove leading slash for Click processing
    if command_line.startswith('/'):
        command_line = command_line[1:]

    # Parse arguments using shlex
    try:
        args = shlex.split(command_line)
    except ValueError as e:
        console.print(f"[bold red]Error parsing command: {e}[/]")
        return

    if not args:
        return

    cmd_name = args[0]
    cmd_args = args[1:]

    # Find the Click command object
    if cmd_name in main.commands:
        cmd_obj = main.commands[cmd_name]
        try:
            # Invoke the command with parsed args
            # Create a sub-context for the command
            sub_ctx = cmd_obj.make_context(cmd_name, cmd_args, parent=ctx)
            # *** Explicitly pass the global mem_service to the sub-context ***
            sub_ctx.obj = mem_service
            with sub_ctx:
                result = cmd_obj.invoke(sub_ctx)
                # Special handling for chat commands to update active_chat_id
                if cmd_name == 'chat':
                    # If start command was successful, update active_chat_id
                    if args[1] == 'start' and isinstance(result, int):
                         active_chat_id = result
                    # If ask command ran, potentially update based on its return?
                    # (ask_question currently prints directly)

        except click.exceptions.UsageError as e:
            console.print(f"[bold red]Usage Error ({cmd_name}): {e}[/]")
            # console.print(cmd_obj.get_help(click.Context(cmd_obj, parent=ctx)))
        except click.exceptions.ClickException as e:
            console.print(f"[bold red]Error ({cmd_name}): {e}[/]")
        except Exception as e:
            console.print(f"[bold red]Unexpected error running command '{command_line}': {e}[/]")
            # import traceback
            # console.print(traceback.format_exc())
    else:
        console.print(f"Unknown command: {cmd_name}")
        console.print("Type /help for available commands.")


def _run_repl(ctx: click.Context):
    """Runs the interactive Read-Eval-Print Loop (REPL)."""
    # Setup prompt session with history
    session = PromptSession(history=FileHistory(HISTORY_FILE))
    completer = WordCompleter(_COMMAND_WORDS, ignore_case=True)

    console.print("chatpdf REPL — /exit to quit, /help for commands")

    while True:
        try:
            with patch_stdout(): # Ensure rich output works with prompt_toolkit
                command_line = session.prompt(
                    "chatpdf> ",
                    completer=completer,
                    key_bindings=kb,
                    refresh_interval=0.5 # Check for async updates (if any)
                )
            _run_repl_command(ctx, command_line)
        except KeyboardInterrupt:
            continue # Handle Ctrl+C gracefully
        except EOFError:
            console.print("Exiting...")
            break # Handle Ctrl+D or /exit
        except Exception as e:
            console.print(f"[bold red]Unexpected REPL error: {e}[/]")
            # Optionally log traceback here
            # import traceback
            # console.print(traceback.format_exc())

@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", package_name="chatpdf")
@click.pass_context
def main(ctx: click.Context):
    """Chat with your PDFs from the command line."""
    # If no subcommand is given, run the REPL
    if ctx.invoked_subcommand is None:
        _run_repl(ctx)

# --- PDF commands --- #
@main.group()
def pdf() -> None:
    """Import or display PDFs."""


@pdf.command("import")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_obj
def pdf_import(mem: MemoryService, path: Path) -> None:
    """Import a PDF file, auto-chunks & stores embeddings."""
    pdf_id = import_pdf(path, mem)
    console.print(f"[green]Imported[/] {path.name} -> id={pdf_id}")


@pdf.command("list")
@click.pass_obj
def pdf_list(mem: MemoryService) -> None:
    """List all imported PDFs."""
    rows = mem.list_pdfs()
    if not rows:
        console.print("[grey50]No PDFs imported yet.[/]")
        return
    table = Table(title="Imported PDFs")
    table.add_column("ID"), table.add_column("Filename"), table.add_column("Title"), table.add_column("Summary")
    for pid, fname, title, summary in rows:
        table.add_row(str(pid), Path(fname).name, title or "", (summary or "")[:80])
    console.print(table)


@pdf.command("show")
@click.argument("pdf_id", type=int)
@click.pass_obj
def pdf_show(mem: MemoryService, pdf_id: int) -> None:
    """Show stored summary for a PDF."""
    summary = mem.get_pdf_summary(pdf_id)
    if summary:
        console.print(textwrap.fill(summary, 100))
    else:
        console.print("[yellow]No summary stored.[/]")


# --- Chat commands --- #

@main.group()
def chat():
    """Start chats or ask questions."""
    pass

@chat.command()
@click.pass_obj
def start(mem: MemoryService):
    """Start a new chat session."""
    global active_chat_id # Allow modification of global state
    chat_id = start_chat(mem=mem)
    if chat_id:
        active_chat_id = chat_id
        console.print(f"Started new chat (ID: {active_chat_id}). Set as active session.")
        return chat_id # Return ID for _run_repl_command to potentially capture
    else:
        console.print("[bold red]Failed to start chat session.[/]")
        return None

@chat.command()
@click.argument('question', type=str)
@click.pass_obj
def ask(mem: MemoryService, question: str):
    """Ask a question within the active chat session."""
    global active_chat_id # Access global state
    if active_chat_id is None:
        console.print("[bold yellow]No active chat session. Use '/chat start' first or just type your question.[/]")
        # Or implicitly start one here?
        # console.print("[dim]Starting new chat implicitly...[/]")
        # chat_id = start_chat()
        # if chat_id:
        #     active_chat_id = chat_id
        # else:
        #     console.print("[bold red]Failed to start implicit chat session.[/]")
        #     return
    else:
        console.print(f"[dim]Asking in chat {active_chat_id}...[/]")

    if active_chat_id:
        # Capture the returned answer and print it
        answer = ask_question(mem=mem, chat_id=active_chat_id, question=question)
        if answer:
            console.print(answer)
        else:
            # Handle cases where ask_question might return None or empty
            console.print("[yellow]Received no answer.[/]")

@chat.command()
@click.option('--chat-id', type=int, default=None, help='Specific chat ID to show history for (default: active chat).')
@click.pass_obj
def history(mem: MemoryService, chat_id: int | None):
    """Show the history of the active or a specific chat session."""
    global active_chat_id # Access global state
    target_chat_id = chat_id if chat_id is not None else active_chat_id

    if target_chat_id is None:
        console.print("[bold yellow]No active chat session and no --chat-id specified.[/]")
        return
    # Pass the global mem_service
    history_data = show_history(mem=mem, chat_id=target_chat_id)
    # (Assuming show_history now returns data to be printed here)
    if history_data:
        for role, text, ts in history_data:
            color = "cyan" if role == "user" else "magenta"
            console.print(f"[grey50]{ts}[/] [{color}]{role}[/]: {text}")
    else:
        console.print(f"[grey50]No history found for chat {target_chat_id}.[/]")


# --- Prompt commands (Example - to be implemented) --- #

@main.group()
def prompt():
    """List or switch canned system prompts."""
    pass

@prompt.command()
def list():
    """List available system prompts."""
    console.print("Available system prompts: (Not implemented yet)")
    # TODO: Implement listing prompts from memory/config

@prompt.command()
@click.argument('prompt_name', type=str)
def use(prompt_name: str):
    """Set the system prompt for the active chat session."""
    console.print(f"Setting system prompt to '{prompt_name}'... (Not implemented yet)")
    # TODO: Implement setting prompt in memory for active_chat_id

if __name__ == '__main__':
    main()