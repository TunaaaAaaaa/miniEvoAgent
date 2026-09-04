"""Initialize PostgreSQL tables for miniEvoAgent experiment storage."""

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evo_repro import PostgreSQLStorage  # noqa: E402


def main() -> None:
    load_dotenv()
    storage = PostgreSQLStorage()
    storage.initialize()
    print("PostgreSQL storage initialized.")


if __name__ == "__main__":
    main()
