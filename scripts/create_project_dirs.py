"""Create directories required by the data and dashboard pipeline."""

from scripts.config import (
    ANALYSIS_DIR,
    FIGURE_DIR,
    RUNTIME_DIR,
    TABLE_DIR,
    WORK_DIR,
)


def main() -> None:
    directories = [
        WORK_DIR,
        RUNTIME_DIR,
        TABLE_DIR,
        ANALYSIS_DIR,
        FIGURE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {directory}")


if __name__ == "__main__":
    main()
