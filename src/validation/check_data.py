"""Simple data validation script: Verifies that key paths exist."""

from src.config import summarize_paths


def main() -> None:
    print(summarize_paths())


if __name__ == "__main__":
    main()
