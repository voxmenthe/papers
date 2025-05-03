"""
Classifier that decides whether a user line is:
  * /meta command
  * Python statement
  * SQL block (ends with ;)
Very small on purpose – keeps cli.py readable.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .manager import TableManager

META_PATTERN = re.compile(r"^/\w+")


@dataclass
class MetaCommand:
    raw: str

    def execute(self, mgr: "TableManager"):              # noqa: F821
        parts = self.raw.strip().split()
        cmd, *args = parts
        match cmd:
            case "/create":
                if not args:
                    raise ValueError("Usage: /create <table>")
                mgr.create(args[0])
            case "/load":
                if not args:
                    raise ValueError("Usage: /load <table> [<table2> ...]")
                loaded = []
                for table_name in args:
                    mgr.load(table_name)
                    loaded.append(table_name)
                if len(loaded) == 1:
                    return f"Table '{loaded[0]}' loaded."
                return f'Loaded tables: {", ".join(loaded)}.'
            case "/list":
                return mgr.list()
            case "/clear":
                if not args:
                    raise ValueError("Usage: /clear <table> [<table2> ...]")
                cleared = []
                for table_name in args:
                    mgr.clear(table_name)
                    cleared.append(table_name)
                if len(cleared) == 1:
                    return f"Table '{cleared[0]}' cleared."
                return f'Cleared tables: {", ".join(cleared)}.'
            case "/clear_all":
                """Clear all tables from the manager and database."""
                mgr.clear_all()
                return "All tables cleared."
            case "/save":
                if len(args) < 1:
                    raise ValueError("Usage: /save <table> [file.pkl]")
                name, *file = args
                filename = file[0] if file else f"{name}.pkl"
                if not filename.endswith(".pkl"):
                    filename = f"{Path(filename).stem}.pkl"
                mgr.save(name, filename)
                return f"Table '{name}' saved to {filename}"
            case "/save_all":
                mgr.save_all_to_temp(mgr.temp_dir)
                return f"Saved {len(mgr.tables)} table(s) to {mgr.temp_dir}"
            case "/export":
                if len(args) < 1:
                    raise ValueError("Usage: /export <table> [file.csv]")
                name, *file = args
                filename = file[0] if file else f"{name}.csv"
                if not filename.endswith(".csv"):
                     filename = f"{Path(filename).stem}.csv"
                mgr.export(name, filename)
                return f"Table '{name}' exported to {filename}"
            case "/schema":
                if not args:
                     raise ValueError("Usage: /schema <table>")
                return mgr.schema(args[0])
            case "/help":
                return (
                    "Meta Commands:\n"
                    "  /create <tbl>         : Create a new empty table\n"
                    "  /load <tbl> [<tbl>...] : Load table(s) from auto-save directory\n"
                    "  /clear <tbl> [<tbl>...] : Remove table(s) from memory and database\n"
                    "  /clear_all            : Remove all tables from memory and database\n"
                    "  /save <tbl> [f.pkl]   : Save table to .pkl file (default: <tbl>.pkl)\n"
                    "  /save_all             : Save all current tables to default directory\n"
                    "  /export <tbl> [f.csv] : Export table to .csv file (default: <tbl>.csv)\n"
                    "  /list                 : List current tables\n"
                    "  /schema <tbl>         : Show table schema (columns and types)\n"
                    "  /help                 : Show this help message\n"
                    "  /exit                 : Quit the playground\n\n"
                    "Enter Python code directly, or end with ';' for SQL."
                 )
            case "/exit":
                raise SystemExit
            case _:
                raise ValueError(f"Unknown meta command {cmd}")


@dataclass
class SqlBlock:
    sql: str


@dataclass
class PythonStmt:
    code: str


def classify(text: str) -> MetaCommand | SqlBlock | PythonStmt:
    stripped_text = text.strip()
    if not stripped_text:
        return PythonStmt("")     # no-op

    # Treat as MetaCommand only if it starts with / and contains no newlines
    if META_PATTERN.match(stripped_text) and '\n' not in stripped_text:
        return MetaCommand(stripped_text)

    # Treat as SQL if the stripped text ends with ;
    if stripped_text.endswith(";"):
        return SqlBlock(stripped_text) # Use stripped text for SQL check
    
    # Also treat as SQL block if it starts with common SQL keywords (no semicolon needed)
    if stripped_text.upper().startswith(("SELECT ", "WITH ", "PRAGMA ", "EXPLAIN ")):
        return SqlBlock(stripped_text)

    # Otherwise, it's Python. Use the original text for exec().
    return PythonStmt(text) 