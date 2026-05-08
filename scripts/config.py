import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

PRISMA_DATABASE_URL = os.getenv("PRISMA_DATABASE_URL")

if not PRISMA_DATABASE_URL:
    raise RuntimeError("Missing PRISMA_DATABASE_URL in .env")

EXPECTED_EVALUATORS = 3

MASTER_DATASET_PATH = Path("data/processed/master_round_dataset.csv")
INPUTS_DIR = Path("inputs")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

POEM_SCORES_PATH = PROCESSED_DIR / "poem_scores.csv"

FIGURE_DIR = Path("public/research-dashboard/figures")
TABLE_DIR = Path("data/processed/dashboard_tables")

WORKFLOW_ORDER = ["human", "ai", "human_ai", "ai_human"]

WORKFLOW_LABELS = {
    "human": "Human only",
    "ai": "AI only",
    "human_ai": "Human → AI",
    "ai_human": "AI → Human",
}

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
