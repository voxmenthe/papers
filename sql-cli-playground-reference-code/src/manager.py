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
    def __init__(self, conn: sqlite3.Connection, temp_dir: Path):
        self.conn = conn
        self.tables: dict[str, pd.DataFrame] = {}
        self.temp_dir = temp_dir # Store the path to the temp directory

    # ---------- public API for the router / meta-commands ------------------
    def create(self, name: str, df: pd.DataFrame | None = None) -> None:
        """Creates a table in the manager and DB.
        If df is provided, initializes with its data, otherwise creates empty.
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid table name: '{name}'. Must be valid Python identifier.")
        if name in self.tables:
             raise ValueError(f"Table '{name}' already exists.")

        # Use provided df or create an empty one
        self.tables[name] = df if df is not None else pd.DataFrame()
        self._push(name)

    def load(self, name: str) -> None:
        """Loads a table by name, assuming it exists as <name>.pkl in the temp dir."""
        if not name.isidentifier():
             raise ValueError(f"Invalid table name: '{name}'.")

        # Construct the expected path in the temp directory
        path = self.temp_dir / f"{name}.pkl"

        if not path.exists():
             # Provide a more specific error message
             raise FileNotFoundError(f"Table file not found at expected location: {path}")

        try:
            # Only need to handle pickle now, as that's what we save
            df = pd.read_pickle(path)

            if not isinstance(df, pd.DataFrame):
                 raise TypeError(f"Loaded object from {path} is not a pandas DataFrame.")

            # Use create to handle adding/replacing in manager and pushing to DB
            # If table exists, create will raise ValueError, which is okay for load.
            # We might want a different behavior? Overwrite on load?
            # For now, let's just update the in-memory dict and push.
            # This avoids the Table already exists error from self.create()
            self.tables[name] = df
            self._push(name) # Push to DB immediately after loading

        except Exception as e:
            # Catch potential pickle errors, etc.
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
             # For now, just show schema from DB if it exists there.
         try:
             return pd.read_sql_query(f"PRAGMA table_info('{name}')", self.conn)
         except Exception as e:
             raise RuntimeError(f"Failed to get schema for table '{name}': {e}")


    def list(self) -> List[str]:
        return sorted(list(self.tables))

    def clear(self, name: str) -> None:
        """Remove a table from the manager and drop it from the database."""
        if name not in self.tables:
            raise ValueError(f"Table '{name}' not found.")
        # Remove from in-memory dict
        del self.tables[name]
        # Drop table in SQLite
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {name}")
            self.conn.commit()
        except Exception as e:
            raise sqlite3.Error(f"Failed to drop table '{name}': {e}")

    def clear_all(self) -> None:
        """Remove all tables from the manager and drop them from the database."""
        for name in list(self.tables.keys()):
            self.clear(name)

    # ---------- sync helpers -----------------------------------------------
    def _push(self, name: str) -> None:
        if name not in self.tables:
             return
        df = self.tables[name]
        # Skip push for DataFrames with no columns to avoid SQL CREATE TABLE syntax error
        if len(df.columns) == 0:
            return
        try:
            df.to_sql(name, self.conn, if_exists="replace",
                      index=False, method="multi")
            self.conn.commit() # Commit changes
        except Exception as e:
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

        # Tables to pull/refresh (exist in DB)
        for name in db_tables:
             self._pull(name) # _pull handles adding/updating self.tables 

    def save_all_to_temp(self, directory: Path) -> None:
        """Saves all current tables to the specified temp directory as .pkl files."""
        # Use self.temp_dir consistently
        self.temp_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        for name, df in self.tables.items():
            filepath = self.temp_dir / f"{name}.pkl"
            try:
                df.to_pickle(filepath)
                # print(f"DEBUG: Saved {name} to {filepath}") # Optional debug
            except Exception as e:
                # Use console.print for errors if available, or just print
                print(f"[Warning] Failed to auto-save table '{name}' to {filepath}: {e}") 