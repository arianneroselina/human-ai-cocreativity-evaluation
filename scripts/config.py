import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Project paths and environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

PRISMA_DATABASE_URL = os.getenv("PRISMA_DATABASE_URL")
if not PRISMA_DATABASE_URL:
    raise RuntimeError("Missing PRISMA_DATABASE_URL in .env")

INPUTS_DIR = PROJECT_ROOT / "inputs"
INTERVIEW_NOTES_PATH = INPUTS_DIR / "interview_error_notes.csv"

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "public" / "research-dashboard"
FIGURE_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

WORK_DIR = DATA_DIR / "work"
MASTER_DATASET_PATH = WORK_DIR / "master_round_dataset.csv"
POEM_SCORES_PATH = WORK_DIR / "poem_scores.csv"
RATINGS_EXPORT_PATH = WORK_DIR / "ratings_export.csv"
TABLE_DIR = WORK_DIR / "dashboard_tables"

RUNTIME_DIR = DATA_DIR / "runtime"
DASHBOARD_DATASET_PATH = RUNTIME_DIR / "dashboard_dataset.csv"
WORKFLOW_FEEDBACK_SUMMARY_PATH = RUNTIME_DIR / "workflow_feedback_summaries.csv"


# ---------------------------------------------------------------------------
# Study design: workflows and rounds
# ---------------------------------------------------------------------------

WORKFLOW_ORDER = [
    "human",
    "ai",
    "human_ai",
    "ai_human",
]

WORKFLOW_LABELS = {
    "human": "Human-only",
    "ai": "AI-only",
    "human_ai": "Human → AI",
    "ai_human": "AI → Human",
}

WORKFLOW_COLORS = {
    "human": "#4C78A8",
    "ai": "#F58518",
    "human_ai": "#54A24B",
    "ai_human": "#B279A2",
}

AI_SUPPORTED_WORKFLOWS = [
    "ai",
    "human_ai",
    "ai_human",
]

MIXED_WORKFLOWS = [
    "human_ai",
    "ai_human",
]

PHASES = ["practice", "main"]

PRACTICE_ROUND_INDICES = [1, 2, 3, 4]
MAIN_ROUND_INDICES = [5, 6, 7]
ERROR_ROUND_INDEX = 5

ROUND_LABELS = {
    1: "Practice 1",
    2: "Practice 2",
    3: "Practice 3",
    4: "Practice 4",
    5: "Main 1",
    6: "Main 2",
    7: "Main 3",
}


# ---------------------------------------------------------------------------
# Output quality configuration
# ---------------------------------------------------------------------------

QUALITY_PRIMARY_METRIC = "meanOverallQuality"
QUALITY_SCALE_MIN = 1
QUALITY_SCALE_MAX = 5

QUALITY_DIMENSION_LABELS = {
    "meanFluency": "Fluency",
    "meanThemeAlignment": "Theme alignment",
    "meanMeaningfulness": "Meaningfulness",
    "meanPoeticness": "Poeticness",
    "meanOverallQuality": "Overall quality",
}


# ---------------------------------------------------------------------------
# Evaluator ratings configuration
# ---------------------------------------------------------------------------

RATING_DIMENSIONS = [
    "fluency",
    "themeAlignment",
    "meaningfulness",
    "poeticness",
    "overallQuality",
]

RATING_DIMENSION_LABELS = {
    "fluency": "Fluency",
    "themeAlignment": "Theme alignment",
    "meaningfulness": "Meaningfulness",
    "poeticness": "Poeticness",
    "overallQuality": "Overall quality",
}

OVERALL_QUALITY_COLUMN = "overallQuality"
RATING_SCALE = [1, 2, 3, 4, 5]

EVALUATOR_LABELS = {
    "1": "Human 1",
    "2": "Human 2",
    "ai-evaluator-gpt-4o-mini": "AI (GPT-4o-mini)",
}

EVALUATOR_ORDER = [
    "1",
    "2",
    "ai-evaluator-gpt-4o-mini",
]

EVALUATOR_COLORS = {
    "1": "#7F7F7F",
    "2": "#8C564B",
    "ai-evaluator-gpt-4o-mini": "#17BECF",
}


# ---------------------------------------------------------------------------
# Feedback metrics
# ---------------------------------------------------------------------------

SATISFACTION_COLUMN = "satisfactionResult"

AI_EXPERIENCE_METRICS = {
    "aiUnderstanding": "AI understanding",
    "aiCollaboration": "AI collaboration",
    "aiCreativitySupport": "AI creativity support",
    "aiPerformanceOverall": "Overall AI performance",
}

TLX_METRICS = {
    "mentalDemand": "Mental demand",
    "physicalDemand": "Physical demand",
    "temporalDemand": "Temporal demand",
    "effort": "Effort",
    "frustration": "Frustration",
    "performance": "Perceived performance\n(lower = better)",
}


# ---------------------------------------------------------------------------
# Error exposure and evaluation
# ---------------------------------------------------------------------------

EXPOSURE_LABELS = {
    True: "Error-exposed",
    False: "Not exposed",
}

EXPECTED_EVALUATORS = 3

AWARENESS_LABELS = {
    "noticed": "Noticed injected error",
    "not_noticed": "Did not notice injected error",
}

OTHER_AI_ERROR_LABELS = {
    "counting_constraints": "Counting constraints",
    "requirement_following": "Requirement following",
    "formatting": "Formatting",
    "time_conversion": "Time conversion",
    "required_words": "Required words",
    "case_sensitivity": "Case sensitivity",
    "quality_degradation": "Quality degradation",
}


# ---------------------------------------------------------------------------
# Participant questionnaire configuration
# ---------------------------------------------------------------------------

PARTICIPANT_LIKERT_COLUMNS = {
    "writingConfidence": "Confidence writing under time pressure",
    "aiLife": "AI will improve my life",
    "aiWork": "AI will improve my work",
    "aiFutureUse": "I will use AI in the future",
    "aiHumanity": "AI is positive for humanity",
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
