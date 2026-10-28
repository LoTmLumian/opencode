"""opencode CLI package（对应原 packages/cli）。"""

from .cli import app, default_db_path, interactive, main

__all__ = ["app", "main", "default_db_path", "interactive"]
