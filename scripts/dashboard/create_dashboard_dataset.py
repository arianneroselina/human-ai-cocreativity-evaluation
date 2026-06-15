import pandas as pd

from scripts.config import MASTER_DATASET_PATH, DASHBOARD_DATASET_PATH

# Columns needed by src/lib/research-dashboard/stats.ts at runtime.
# Keep this file deploy-safe: no raw poem text, no raw comments, no session IDs.
DASHBOARD_COLUMNS = [
    "roundId",
    "poemId",
    "participantId",
    "roundIndex",
    "workflow",
    "timeMs",
    "passed",
    "constraintScore",
    "condition",
    "effectiveTimeMinutes",
    "meanFluency",
    "meanThemeAlignment",
    "meanMeaningfulness",
    "meanPoeticness",
    "meanOverallQuality",
    "qualityComposite",
    "satisfactionResult",
    "frustration",
    "effort",
    "performance",
    "aiPerformanceOverall",
    "aiUnderstanding",
    "aiCollaboration",
    "aiCreativitySupport",
]


def main():
    if not MASTER_DATASET_PATH.exists():
        raise RuntimeError(
            f"Missing {MASTER_DATASET_PATH}. Run `make create-master` first."
        )

    df = pd.read_csv(MASTER_DATASET_PATH)
    available_columns = [column for column in DASHBOARD_COLUMNS if column in df.columns]
    missing_columns = [
        column for column in DASHBOARD_COLUMNS if column not in df.columns
    ]

    dashboard_df = df[available_columns].copy()
    dashboard_df.to_csv(DASHBOARD_DATASET_PATH, index=False)

    print(f"Created {DASHBOARD_DATASET_PATH}")
    print(f"Rows: {len(dashboard_df)}")
    print(f"Columns: {len(available_columns)}")

    if missing_columns:
        print("Warning: missing columns:")
        for column in missing_columns:
            print(f"- {column}")


if __name__ == "__main__":
    main()
