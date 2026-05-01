"""Backward-compatible wrapper for ``python src/main.py``."""
from hermecho import cli


def main() -> None:
    """Delegate to the packaged CLI entrypoint."""
    cli.main()


if __name__ == "__main__":
    main()
