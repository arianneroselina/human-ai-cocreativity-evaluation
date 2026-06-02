import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

PRISMA_DATABASE_URL = os.getenv("PRISMA_DATABASE_URL")

if not PRISMA_DATABASE_URL:
    raise RuntimeError("Missing PRISMA_DATABASE_URL in .env")

EXPECTED_EVALUATORS = 3
ERROR_ROUND_INDEX = 5

INPUTS_DIR = Path("inputs")

WORK_DIR = Path("data/work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DATASET_PATH = WORK_DIR / "master_round_dataset.csv"
POEM_SCORES_PATH = WORK_DIR / "poem_scores.csv"
RATINGS_EXPORT_PATH = WORK_DIR / "ratings_export.csv"
TABLE_DIR = WORK_DIR / "dashboard_tables"

RUNTIME_DIR = Path("data/runtime")
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_DATASET_PATH = RUNTIME_DIR / "dashboard_dataset.csv"
DASHBOARD_FINAL_RANKING_PATH = RUNTIME_DIR / "dashboard_final_ranking.csv"
DASHBOARD_COMMENT_THEMES_PATH = RUNTIME_DIR / "dashboard_comment_themes.csv"
WORKFLOW_FEEDBACK_SUMMARY_PATH = RUNTIME_DIR / "workflow_feedback_summaries.csv"

FIGURE_DIR = Path("public/research-dashboard/figures")

WORKFLOW_ORDER = ["human", "ai", "human_ai", "ai_human"]

WORKFLOW_LABELS = {
    "human": "Human only",
    "ai": "AI only",
    "human_ai": "Human → AI",
    "ai_human": "AI → Human",
}

AI_SUPPORTED_WORKFLOWS = ["ai", "human_ai", "ai_human"]

PARTICIPANT_COLUMN_ALIASES = {
    "participantId": ["whatisyourparticipantid", "participantid"],
    "age": ["age"],
    "gender": ["gender"],
    "genderOther": ["genderother"],
    "education": ["highesteducation"],
    "educationOther": ["highesteducationother"],
    "nativeLanguage": ["nativelanguage"],
    "englishLevel": ["englishlevel"],
    "englishLevelOther": ["englishlevelother"],
    "writingConfidence": ["ifeelconfidentwritingshorttextsundertimepressure"],
    "aiLife": ["aiwillimprovemylife"],
    "aiWork": ["aiwillimprovemywork"],
    "aiFutureUse": ["iwilluseaitechnologyinthefuture"],
    "aiHumanity": ["aitechnologyispositiveforhumanity"],
}

PARTICIPANT_CATEGORY_COLUMNS = {
    "gender": "Gender",
    "education": "Highest education",
    "nativeLanguage": "Native language",
    "englishLevel": "English level",
}

PARTICIPANT_LIKERT_COLUMNS = {
    "writingConfidence": "Confidence writing under time pressure",
    "aiLife": "AI will improve my life",
    "aiWork": "AI will improve my work",
    "aiFutureUse": "I will use AI in the future",
    "aiHumanity": "AI is positive for humanity",
}

EXPOSURE_LABELS = {
    "error_exposed": "Error-exposed",
    "not_exposed": "Not exposed",
}
