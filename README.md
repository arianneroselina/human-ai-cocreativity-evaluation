# Human-AI Co-Creativity Evaluation 🧠📊

A web-based evaluation and research-dashboard application for an already completed human-AI co-creativity study.

This repository does **not** conduct the original writing study. It receives exported study results as private local input, stores poems and ratings in Prisma/PostgreSQL, and generates deployable dashboard outputs for analysis.

---

## Application Routes

After starting the development server, the app has two main routes:

| Route                                                                                           | Purpose                                                                                                              |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [`/evaluation`](https://human-ai-cocreativity-evaluation.vercel.app/evaluation)                 | Blind poem evaluation interface. Evaluators rate one poem at a time without seeing authorship or workflow condition. |
| [`/research-dashboard`](https://human-ai-cocreativity-evaluation.vercel.app/research-dashboard) | Research dashboard for evaluator progress, processed results, generated figures, and feedback summaries.             |

The root route `/` redirects to `/evaluation`.

---

## Project Scope

### Input

Private local study exports in `inputs/`, for example:

- `Session.csv`
- `Round.csv`
- `RoundFeedback.csv`
- `Feedback.csv`
- survey export CSVs

These files are required locally for seeding and regenerating analysis outputs, but they must **not** be committed.

### Evaluation

- Poems are seeded into Prisma/PostgreSQL.
- Human evaluators rate poems through `/evaluation`.
- Optional AI-based ratings can be generated with the AI evaluator.
- Ratings are stored in the database.

### Output

The processing pipeline creates:

- local full analysis files
- deploy-safe dashboard dataset files
- generated figures for `/research-dashboard`

---

## Evaluation Design

The evaluation is based on a controlled dataset where:

- 24 participants each wrote poems across 7 topics
- each poem belongs to one workflow condition:
  - **Human-only**
  - **AI-only**
  - **Human → AI**
  - **AI → Human**

The original dataset contains 168 poems. Since 2 poems are empty, the final evaluation dataset consists of 166 valid poems.

---

## Evaluation Method

The platform uses a blind single-poem rating approach:

- one poem is shown at a time
- the topic is shown for context
- authorship and workflow condition are hidden
- evaluators rate multiple quality dimensions
- ratings use a 1–5 scale

This reduces authorship/workflow bias and allows each poem to be evaluated independently.

---

## Repository Structure

```text
src/app/                         Next.js routes and API endpoints
src/lib/                         Shared application and dashboard logic
src/data/                        Study-data loading helpers and task definitions
src/components/                  Shared global UI components
prisma/                          Prisma schema, migrations, and seed script
scripts/                         Data-processing scripts
scripts/dashboard_figures/       Figure-generation scripts
inputs/                          Private raw study exports; do not commit
data/runtime/                    Generated dashboard data
data/work/                       Generated local processing data
public/research-dashboard/       Generated dashboard figures for deployment
```

---

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

Create `.env` from `.env.example` and configure the database connection:

```env
PRISMA_DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DATABASE"
```

### 3. Seed the database

Make sure the private raw exports are available locally in `inputs/`, then run:

```bash
make setup-db
```

This runs Prisma migrations and seeds the poem data.

### 4. Start development

```bash
make run
```

Open:

```text
http://localhost:3000/evaluation
http://localhost:3000/research-dashboard
```

---

## Data Processing Pipeline

Run the full pipeline:

```bash
make process-data
```

This runs:

```text
eval-check
export-ratings
aggregate-scores
create-master
create-dashboard-dataset
create-dashboard-tables
generate-figures
```

The important deploy-safe outputs are:

```text
data/runtime/dashboard_dataset.csv
data/runtime/dashboard_final_ranking.csv
data/runtime/dashboard_comment_themes.csv
data/runtime/workflow_feedback_summaries.csv
public/research-dashboard/
```

Do **not** commit `inputs/` or full raw/local processing files.

---

## AI Evaluator

Run the optional AI evaluator:

```bash
make run-ai-evaluator
```

This creates AI-based poem ratings and stores them in the database.

`OPENAI_API_KEY` is required for AI-based functionality.

---

## Useful Commands

```bash
make run                    # Start local dev server
make setup-db               # Run migrations and seed poems
make reset-seed             # Reset local database and seed again
make process-data           # Run the full processing pipeline
make generate-figures       # Regenerate dashboard figures only
make studio                 # Open Prisma Studio
make run-ai-evaluator       # Run optional AI evaluator
```

---

## Acknowledgements

This project is part of a Master's Thesis exploring human-AI co-creativity and the evaluation of generative outputs.
