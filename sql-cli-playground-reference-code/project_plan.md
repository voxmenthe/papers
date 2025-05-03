# Project Plan

## 1 .  High-level architecture

| Layer | Responsibility | Key libs |
|-------|----------------|----------|
| **REPL front-end** | Prompt, history, multiline edit, tab completion | `prompt_toolkit` |
| **Command router** | Decide whether a line is *meta*, *Python*, or *SQL* and dispatch | `re` / simple FSM |
| **Execution sandbox** | `exec`/`eval` for Python; `sqlite3` for SQL | stdlib |
| **Table manager** | Owns an in-memory dict `{name: pandas.DataFrame}` and keeps DB in sync | `pandas`, `numpy`, `sqlite3` |
| **Persistence helpers** | `/load`, `/save`, `/export` | `pandas.read_*/to_*`, `pickle` |
| **Pretty printer** | Render `DataFrame` and result sets | `rich` (optional) |

The CLI is therefore just a thin skin over:

```text
┌──────────────┐
│ prompt_toolkit│  ← interactive shell
└─────▲────────┘
      │  text
┌─────┴────────┐
│ Command router│
└─▲───────▲────┘
  │       │
Python  SQL
exec()   sqlite3
  │       │
┌─┴───────┴───┐
│ TableManager │  ← keeps `df` and SQLite table in lock-step
└──────────────┘
```

---

## 2 .  Command grammar

* **Meta-commands** (always start with `/`):

```
/create <table_name>            → new empty DataFrame + empty SQLite table
/load   <table_name> [file]     → pandas.read_(csv/parquet/pkl) etc.
/list                           → show loaded tables
/schema <table_name>            → PRAGMA table_info()
/save   <table_name> [file]     → df.to_pickle / to_parquet
/export <table_name>.csv        → df.to_csv
/help                           → show this help
/exit                           → quit
```

* **Python mode** – anything else **not** ending in a semicolon is handed to `exec` with a shared namespace containing:

```python
globals = {
    "np": numpy,
    "pd": pandas,
    **table_manager.tables      # each table is a DataFrame
}
```

Assignments like `table1["col1"] = np.random.randint(...)` therefore mutate the DataFrame in-place.  The TableManager watches for mutations (monkey-patching `DataFrame.__setitem__` is overkill – just run `to_sql()` after every successful exec).

* **SQL mode** – a block that ends with **`;`** (possibly multiline).  Feed the raw string to `conn.executescript()`.  Results are fetched into a pandas DataFrame with `pd.read_sql_query()` for pretty printing and for further Python use.

---

## 3 .  Table synchronisation logic

```python
class TableManager:
    def __init__(self, conn):
        self.conn = conn          # sqlite3.Connection
        self.tables = {}          # name → DataFrame

    def _push(self, name):
        df = self.tables[name]
        df.to_sql(name, self.conn, if_exists="replace", index=False)

    def create(self, name, df=None):
        self.tables[name] = pd.DataFrame() if df is None else df
        self._push(name)

    def load(self, name, filepath):
        df = pd.read_pickle(filepath) if filepath.endswith(".pkl") else ...
        self.tables[name] = df
        self._push(name)

    def save(self, name, filepath):
        self.tables[name].to_pickle(filepath)

    def export(self, name, filepath):
        # Placeholder for export logic, e.g., CSV
        self.tables[name].to_csv(filepath, index=False)

    def schema(self, name):
        # Placeholder for schema retrieval
        return pd.read_sql_query(f"PRAGMA table_info({name})", self.conn)

    def refresh_all(self):
        """When SQL might have modified tables, pull fresh copies into Python."""
        current_tables = self.tables.keys()
        # Potentially check sqlite_master for new tables created by SQL?
        for name in list(current_tables): # Iterate over copy in case SQL drops a table
            try:
                self.tables[name] = pd.read_sql_query(f"SELECT * FROM {name}", self.conn)
            except pd.io.sql.DatabaseError: # Handle case where table might have been dropped
                del self.tables[name]
```

*When the user runs arbitrary SQL that mutates a table (INSERT/UPDATE/etc.), call `refresh_all()` before returning control to the Python prompt so Python and SQLite stay consistent.*

---

## 4 .  REPL with `prompt_toolkit`

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

session = PromptSession(
    "(tables: {tables}) >> ",
    multiline=True,
    history=FileHistory("~/.sqlplayground_history"),
)

while True:
    with patch_stdout():                       # keeps prints responsive in multiline
        try:
            raw = session.prompt(
                default="",
                bottom_toolbar="Enter a Python stmt or end with ';' for SQL",
            )
        except (KeyboardInterrupt, EOFError):
            break

    router.dispatch(raw)
```

* Tab completion can hook `WordCompleter(tables + python_builtins + sql_keywords)`.
* Multiline editing is built-in – pressing **Enter** inserts a newline unless you're at an empty line after a semicolon.

---

## 5 .  Datatype inference rules

Mapping NumPy → SQLite:

| NumPy dtype kind | SQLite type |
|------------------|-------------|
| `i` / `u`        | INTEGER |
| `f`              | REAL     |
| `b`              | INTEGER (0/1) |
| `M` (datetime64) | TEXT (ISO) or REAL (unix ts) |
| other            | BLOB via `df.astype("object")` |

pandas handles 99 % of this automatically when you call `to_sql`.  Fail fast by catching `ValueError` on dtype conversions.

---

## 6 .  Error policy

* **Python exec errors** – let them bubble; traceback shown in minimal form (stdlib's default).
* **SQL errors** – catch `sqlite3.DatabaseError` and display full SQL + error message (`print(e)`).
* **Schema mismatch** – your wrapper around `to_sql` can compare `len(df)` with existing table rows and abort if unequal, unless the op is "replace".

---

## 7 .  Packaging (Poetry)

```
[tool.poetry]
name = "sql-cli-playground"
version = "0.1.0"
description = "A tiny CLI to experiment with SQL over ad-hoc NumPy/Pandas tables"
authors = ["You <you@example.com>"]
packages = [{ include = "sql-cli-playground" }]

[tool.poetry.dependencies]
python = "^3.12"
pandas = "^2.2"
numpy = "^2.0"
prompt_toolkit = "^3.0"
rich = "^13.0"

[tool.poetry.scripts]
sql-cli-playground = "sql-cli-playground.cli:main"
```

Users install with

```bash
git clone ...
cd sql-cli-playground
poetry install
poetry run sql-cli-playground
```

---

## 8 .  Minimal "happy-path" demo

```text
(tables: []) >> /create table1
(tables: ["table1"]) >> table1["col1"] = np.random.randint(1, 50, 100)
(tables: ["table1"]) >> 
SELECT col1 FROM table1
LIMIT 5;
┌───┬─────┐
│   │ col1│
├───┼─────┤
│0  │  13 │
│1  │  47 │
│2  │   4 │
│3  │  19 │
│4  │  25 │
└───┴─────┘
(tables: ["table1"]) >>
```

After the SQL executes, the result is also stored as the implicit variable `_` (the last DataFrame) so you can:

```python
_.hist()   # classic pandas plotting
```

---

## 9 .  Implementation milestones

1. **Bootstrapping**
   * `cli.py` with prompt_toolkit REPL.
   * Hard-code in-memory SQLite; `/create` and `/list`.

2. **Python ⇄ SQLite bridge**
   * Inject DataFrames into globals.
   * After every Python exec, push dirty tables to DB.
   * After every SQL exec, pull modified tables.

3. **Meta-commands**
   * `/load`, `/save`, `/export`, `/schema`.

4. **Polish**
   * Tab completion, nice colours via `rich`.
   * Unit-tests for router and TableManager (pytest, no big perf suite).

Total code footprint is ~500 LoC.

---

### What you *don't* have to solve now

* Fine-grained dirty-flag tracking – brute-force `to_sql(replace)` is OK for small/medium toy data.
* Advanced SQL parsing – you simply pass strings to SQLite.
* Security – local `exec` is acceptable by your spec.


Below is a self-contained starter repo you can drop straight into `poetry new sql-cli-playground` (if you prefer, just copy the tree verbatim into an empty directory and run `poetry install`).

## Project Structure
```
sql-cli-playground/
├── pyproject.toml          ← Poetry metadata & dependencies
├── README.md               ← Quick-start & usage examples
├── src/          ← Package
│   ├── __init__.py
│   ├── cli.py              ← Interactive REPL entry-point
│   ├── manager.py          ← TableManager abstraction
│   └── router.py           ← Tiny dispatcher for /meta, Python, SQL
└── tests/
    └── test_router.py      ← Example pytest scaffold
```

---

## 1 · `pyproject.toml`

```toml
[tool.poetry]
name = "sql-cli-playground"
version = "0.1.0"
description = "CLI playground for mixing NumPy/Pandas with SQLite"
authors = ["<your-name>"]
packages = [{ include = "src" }]

[tool.poetry.dependencies]
python          = "^3.12"
numpy           = "^2.0"
pandas          = "^2.2"
prompt_toolkit  = "^3.0"
rich            = "^13.0"

[tool.poetry.scripts]
sql-cli-playground = "sql-cli-playground.cli:main"
```

---

## 2 · `sql-cli-playground/cli.py`

```python
"""
sql-cli-playground.cli
=================
REPL front-end with multiline editing that auto-switches
between Python and SQL, plus `/meta` commands.
"""

from __future__ import annotations

import sqlite3
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.table import Table

from .manager import TableManager
from .router import classify, MetaCommand, SqlBlock, PythonStmt

console = Console()
HISTORY = FileHistory("~/.sqlplayground_history")


def _build_completer(tbl_mgr: TableManager) -> WordCompleter:
    words = set(tbl_mgr.tables.keys()) | {
        "SELECT", "FROM", "WHERE", "LIMIT", "INSERT", "UPDATE",
        "CREATE", "DROP", "DELETE"}
    words |= set(dir(np)) | set(dir(pd))
    return WordCompleter(list(words), ignore_case=True)


def _render_df(df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        console.print("[grey50]No results.[/]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.head(50).iterrows():          # cap preview
        table.add_row(*map(lambda x: str(x)[:40], row))
    console.print(table)
    if len(df) > 50:
        console.print(f"[grey62]... {len(df) - 50} more rows[/]")


def main() -> None:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    tables = TableManager(conn)
    psession = PromptSession(history=HISTORY, completer=_build_completer(tables),
                             multiline=True,
                             prompt_continuation="... ")

    banner = "[italic cyan]SQL-playground — mix Python & SQL.  /help for commands[/]"
    console.print(banner)

    # Start with core modules + empty tables dict, update as tables are added/removed
    globals_ns: Dict[str, Any] = {"np": np, "pd": pd, **tables.tables}

    while True:
        prompt = f"(tables: {list(tables.tables)}) >> "
        try:
            with patch_stdout():
                text = psession.prompt(prompt)
        except KeyboardInterrupt:
            console.print("\n[bold]Interrupted. Use /exit or Ctrl-D to quit.[/]")
            continue # Go back to prompt
        except EOFError:
            console.print("\n[bold]Bye![/]")
            sys.exit(0)

        block = classify(text)

        # -- META ----------------------------------------------------------------
        if isinstance(block, MetaCommand):
            try:
                out = block.execute(tables)
                if isinstance(out, pd.DataFrame):
                    _render_df(out)
                elif out is not None: # Could be list of tables or help string
                    console.print(out)
                # Update globals and completer *only if* tables changed
                globals_ns.update(tables.tables) # Cheaper than checking, just update
                psession.completer = _build_completer(tables)
            except (ValueError, FileNotFoundError, sqlite3.Error) as e:
                console.print(f"[red]Error: {e}[/]")
            except SystemExit:
                console.print("\n[bold]Bye![/]")
                sys.exit(0)
            except Exception as e: # Catch unexpected errors
                 console.print(f"[bold red]Unexpected Error:[/]\n[red]{type(e).__name__}: {e}[/]")
                 console.print_exception(max_frames=1)
            continue

        # -- PYTHON --------------------------------------------------------------
        if isinstance(block, PythonStmt):
            try:
                exec(block.code, globals_ns)     # noqa: S102 exec is intended
            except Exception: # Catch any exec error
                console.print_exception(max_frames=4)
            finally:
                # Always push changes after Python execution attempt
                try:
                    tables.push_all()
                except sqlite3.Error as e:
                     console.print(f"[red]DB sync error after Python: {e}[/]")
            continue

        # -- SQL -----------------------------------------------------------------
        if isinstance(block, SqlBlock):
            df_result = None
            try:
                # Try reading as a query first (SELECT, PRAGMA)
                df_result = pd.read_sql_query(block.sql, conn)
                globals_ns["_"] = df_result # Store last result
            except (pd.io.sql.DatabaseError, sqlite3.OperationalError) as e:
                # If read_sql fails, it might be DML/DDL or an error
                if "cannot operate on a closed database" in str(e).lower() or \
                   "no statement executed" in str(e).lower() :
                    # Likely DML/DDL, try executescript
                    try:
                        conn.executescript(block.sql)
                        conn.commit() # Commit changes for DML/DDL
                        console.print("[green]OK.[/]")
                    except sqlite3.Error as exec_e:
                         console.print(f"[red]SQL Error: {exec_e.__class__.__name__}: {exec_e}[/]")
                else:
                    # It was likely a genuine error in a SELECT-like query
                    console.print(f"[red]SQL Error: {e.__class__.__name__}: {e}[/]")
            except Exception as e: # Catch other potential pandas/SQL errors
                 console.print(f"[bold red]Unexpected SQL Error:[/]\n[red]{type(e).__name__}: {e}[/]")
            finally:
                 # Refresh Python view of tables if SQL might have changed them
                 try:
                     tables.refresh_all()
                     globals_ns.update(tables.tables) # Ensure globals reflect refreshed state
                     psession.completer = _build_completer(tables) # Update completer
                 except sqlite3.Error as e:
                      console.print(f"[red]DB sync error after SQL: {e}[/]")

            # Render result *after* refresh_all, only if read_sql succeeded
            if df_result is not None:
                _render_df(df_result)

            continue
```

### Multiline UX

`PromptSession(multiline=True, prompt_continuation='... ')` lets a user hit **Enter** for a new line; the prompt only submits when they press **Ctrl-D** (EOF) or **Enter** on an empty line.  It feels exactly like psql or SQLite's shell.

---

## 3 · `sql-cli-playground/manager.py`

```python
"""
Minimal TableManager that mirrors DataFrames into SQLite after Python
mutations and pulls fresh copies after SQL execution.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import sqlite3
from typing import List


class TableManager:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.tables: dict[str, pd.DataFrame] = {}

    # ---------- public API for the router / meta-commands ------------------
    def create(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"Invalid table name: '{name}'. Must be valid Python identifier.")
        if name in self.tables:
             raise ValueError(f"Table '{name}' already exists.")
        self.tables[name] = pd.DataFrame()
        self._push(name)

    def load(self, name: str, path: str | Path) -> None:
        if not name.isidentifier():
             raise ValueError(f"Invalid table name: '{name}'.")
        path = Path(path)
        if not path.exists():
             raise FileNotFoundError(f"Cannot load file: {path}")

        try:
            if path.suffix == ".pkl":
                df = pd.read_pickle(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                 df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}. Use .pkl, .parquet, or .csv")

            if not isinstance(df, pd.DataFrame):
                 raise TypeError(f"Loaded object from {path} is not a pandas DataFrame.")

            self.tables[name] = df
            self._push(name) # Push to DB immediately after loading
        except Exception as e:
            raise RuntimeError(f"Failed to load table '{name}' from {path}: {e}")


    def save(self, name: str, path: str | Path) -> None:
        if name not in self.tables:
             raise ValueError(f"Table '{name}' not found.")
        path = Path(path)
        if path.suffix != ".pkl":
             # For simplicity, only supporting pickle for save initially
             # Could add .parquet, .csv with more options later
             raise ValueError("Can only save to '.pkl' files currently.")
        try:
             self.tables[name].to_pickle(path)
        except Exception as e:
            raise RuntimeError(f"Failed to save table '{name}' to {path}: {e}")


    def export(self, name: str, path: str | Path) -> None:
        if name not in self.tables:
            raise ValueError(f"Table '{name}' not found.")
        path = Path(path)
        if path.suffix != ".csv":
             raise ValueError("Can only export to '.csv' files currently.")
        try:
            self.tables[name].to_csv(path, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to export table '{name}' to {path}: {e}")


    def schema(self, name: str) -> pd.DataFrame:
         if name not in self.tables:
             # Check DB directly in case it was created purely via SQL
             cursor = self.conn.cursor()
             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
             if cursor.fetchone() is None:
                 raise ValueError(f"Table '{name}' not found in Python or database.")
             # If found in DB but not Python dict, pull it? Or just show schema?
             # For now, just show schema from DB if it exists there.
         try:
             # PRAGMA returns table info even if table is empty
             return pd.read_sql_query(f"PRAGMA table_info('{name}')", self.conn)
         except Exception as e:
             raise RuntimeError(f"Failed to get schema for table '{name}': {e}")


    def list(self) -> List[str]:
        # Potentially sync with DB's sqlite_master table?
        # For now, just list tables known to the manager.
        return sorted(list(self.tables))

    # ---------- sync helpers -----------------------------------------------
    def _push(self, name: str) -> None:
        if name not in self.tables:
             # Should not happen if called internally, but good practice
             return
        df = self.tables[name]
        try:
            # Use 'multi' method for potentially better performance with many rows
            df.to_sql(name, self.conn, if_exists="replace",
                      index=False, method="multi")
            self.conn.commit() # Commit changes
        except Exception as e:
            # Wrap driver errors in a more informative exception
            raise sqlite3.Error(f"Failed to push table '{name}' to SQLite: {e}")


    def push_all(self) -> None:
        for name in self.tables:
            self._push(name) # _push now includes error handling

    def _pull(self, name: str) -> None:
        try:
            self.tables[name] = pd.read_sql_query(f"SELECT * FROM {name}", self.conn)
        except Exception as e:
             # If table doesn't exist in DB (e.g., dropped via SQL)
             # Should we remove from self.tables? Yes.
             if name in self.tables:
                 del self.tables[name]
             # Don't raise error here, just means table is gone from DB
             # print(f"Debug: Table '{name}' not found in DB during pull.") # Optional debug


    def refresh_all(self) -> None:
        # Get tables currently known to the manager
        managed_tables = set(self.tables.keys())

        # Get tables actually in the database
        try:
            db_tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", self.conn)
            db_tables = set(db_tables_df['name'])
        except Exception as e:
             raise sqlite3.Error(f"Failed to query database master table: {e}")

        # Tables to remove from manager (exist in manager but not DB)
        to_remove = managed_tables - db_tables
        for name in to_remove:
            del self.tables[name]
            # print(f"Debug: Removed table '{name}' from manager (not in DB).") # Optional debug

        # Tables to pull/refresh (exist in DB)
        # This includes tables newly created by SQL or existing ones
        for name in db_tables:
             self._pull(name) # _pull handles adding/updating self.tables
             # print(f"Debug: Pulled/Refreshed table '{name}' from DB.") # Optional debug

```

---

## 4 · `sql-cli-playground/router.py`

```python
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
                if len(args) < 1:
                    raise ValueError("Usage: /load <table> [file]")
                name, *file = args
                mgr.load(name, file[0] if file else f"{name}.pkl")
            case "/list":
                return mgr.list()
            case "/save":
                if len(args) < 1:
                    raise ValueError("Usage: /save <table> [file.pkl]")
                name, *file = args
                # Default to .pkl if no extension provided or different one used
                filename = file[0] if file else f"{name}.pkl"
                if not filename.endswith(".pkl"):
                    filename = f"{Path(filename).stem}.pkl"
                mgr.save(name, filename)
                return f"Table '{name}' saved to {filename}" # Confirmation message
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
                return mgr.schema(args[0]) # Returns a DataFrame
            case "/help":
                return (
                    "Meta Commands:\n"
                    "  /create <tbl>         : Create a new empty table\n"
                    "  /load <tbl> [f.ext]   : Load table from .pkl, .parquet, .csv (default: <tbl>.pkl)\n"
                    "  /save <tbl> [f.pkl]   : Save table to .pkl file (default: <tbl>.pkl)\n"
                    "  /export <tbl> [f.csv] : Export table to .csv file (default: <tbl>.csv)\n"
                    "  /list                 : List current tables\n"
                    "  /schema <tbl>         : Show table schema (columns and types)\n"
                    "  /help                 : Show this help message\n"
                    "  /exit                 : Quit the playground\n\n"
                    "Enter Python code directly, or end with ';' for SQL."
                 )
            case "/exit":
                raise SystemExit # Let cli.py handle the exit message
            case _:
                raise ValueError(f"Unknown meta command {cmd}")


@dataclass
class SqlBlock:
    sql: str


@dataclass
class PythonStmt:
    code: str


def classify(text: str) -> MetaCommand | SqlBlock | PythonStmt:
    text = text.strip()
    if not text:
        return PythonStmt("")     # no-op

    if META_PATTERN.match(text):
        return MetaCommand(text)

    if text.endswith(";"):
        return SqlBlock(text)

    return PythonStmt(text)
```

---

## 5 · `README.md` (excerpt)

```md
### Quick demo

```bash
poetry run sql-cli-playground
(tables: []) >> /create table1
(tables: ['table1']) >> import numpy as np; table1['col1'] = np.random.randint(1, 50, 5)
(tables: ['table1']) >> table1['col2'] = np.random.rand(5) * 10
(tables: ['table1']) >>
SELECT col1, col2 FROM table1 LIMIT 3;
# → nicely formatted table via rich
┌──────┬───────────┐
│ col1 │ col2      │
├──────┼───────────┤
│ 34   │ 4.823...  │
│ 12   │ 9.155...  │
│ 4    │ 0.567...  │
└──────┴───────────┘
[grey62]... 2 more rows[/]
(tables: ['table1']) >> _ # Access last SQL result DataFrame
# → Shows DataFrame representation
(tables: ['table1']) >> /schema table1
# → Shows PRAGMA table_info() output via rich
┌─────┬───────┬─────────┬─────────┬────────┬─────────┐
│ cid │ name  │ type    │ notnull │ dflt_value │ pk │
├─────┼───────┼─────────┼─────────┼────────┼─────────┤
│ 0   │ col1  │ INTEGER │ 0       │ <NA>       │ 0  │
│ 1   │ col2  │ REAL    │ 0       │ <NA>       │ 0  │
└─────┴───────┴─────────┴─────────┴────────┴─────────┘
(tables: ['table1']) >> /exit
```

---

## 6 · A smoke-test (`tests/test_router.py`)

```python
from src.router import classify, MetaCommand, SqlBlock, PythonStmt
from src.manager import TableManager # Needed if testing meta commands that use it
import sqlite3 # Needed if testing meta commands that use it
import pandas as pd # Needed if testing meta commands that return DataFrames
import pytest # For fixtures or raising exceptions

# Minimal fixture for tests needing a manager
@pytest.fixture
def table_manager():
    conn = sqlite3.connect(":memory:")
    yield TableManager(conn)
    conn.close()


def test_classify():
    assert isinstance(classify("/list"), MetaCommand)
    assert isinstance(classify("  SELECT 1;  "), SqlBlock)
    assert classify("  SELECT 1;  ").sql == "SELECT 1;" # Check stripping
    assert isinstance(classify("x = 2"), PythonStmt)
    assert classify("x = 2").code == "x = 2" # Check stripping
    assert isinstance(classify(""), PythonStmt) # Empty input is Python no-op


def test_meta_help(table_manager):
    cmd = classify("/help")
    assert isinstance(cmd, MetaCommand)
    output = cmd.execute(table_manager)
    assert isinstance(output, str)
    assert "/create" in output
    assert "/load" in output
    assert "/save" in output
    assert "/export" in output
    assert "/list" in output
    assert "/schema" in output
    assert "/exit" in output


def test_meta_create_list(table_manager):
    # Create
    cmd_create = classify("/create my_table")
    assert isinstance(cmd_create, MetaCommand)
    result = cmd_create.execute(table_manager)
    assert result is None # Create returns nothing
    assert "my_table" in table_manager.tables
    assert isinstance(table_manager.tables["my_table"], pd.DataFrame)

    # List
    cmd_list = classify("/list")
    assert isinstance(cmd_list, MetaCommand)
    result = cmd_list.execute(table_manager)
    assert result == ["my_table"]

    # Test create error (already exists)
    with pytest.raises(ValueError, match="already exists"):
         classify("/create my_table").execute(table_manager)

    # Test create error (invalid name)
    with pytest.raises(ValueError, match="Invalid table name"):
         classify("/create 1invalid").execute(table_manager)


def test_meta_schema(table_manager):
     # Test schema on non-existent table
     with pytest.raises(ValueError, match="not found"):
         classify("/schema no_such_table").execute(table_manager)

     # Create a table first
     classify("/create tbl_for_schema").execute(table_manager)
     table_manager.tables["tbl_for_schema"]["colA"] = [1, 2]
     table_manager.tables["tbl_for_schema"]["colB"] = ["a", "b"]
     table_manager.push_all() # Push to DB

     cmd_schema = classify("/schema tbl_for_schema")
     assert isinstance(cmd_schema, MetaCommand)
     df_schema = cmd_schema.execute(table_manager)

     assert isinstance(df_schema, pd.DataFrame)
     assert list(df_schema.columns) == ['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']
     assert df_schema['name'].tolist() == ['colA', 'colB']
     # SQLite types might vary slightly based on pandas version / data
     assert df_schema['type'].tolist() == ['INTEGER', 'TEXT']


# Add more tests for /load, /save, /export, /exit as needed
# Remember to handle file operations appropriately (e.g., using tmp_path fixture)

```

Run with `pytest -q`.