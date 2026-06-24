.DEFAULT_GOAL := help

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

PYTHON ?= python


# ------------------------------------------------------------
# App
# ------------------------------------------------------------

.PHONY: run
run:
	npm run dev

.PHONY: format
format:
	npm run format


# ------------------------------------------------------------
# Prisma / Database
# ------------------------------------------------------------

.PHONY: gen
gen:
	npx prisma generate

.PHONY: migrate
migrate:
	npx prisma migrate dev

.PHONY: migrate-new
migrate-new:
	npx prisma migrate dev -n $(name)

.PHONY: seed-db
seed-db:
	npm run db:seed

.PHONY: setup-db
setup-db: migrate seed-db
	@echo "Database migrated and seeded."

.PHONY: reset
reset:
	npx prisma migrate reset

.PHONY: reset-seed
reset-seed:
	npx prisma migrate reset --force
	npm run db:seed

.PHONY: deploy
deploy:
	npx prisma migrate deploy

.PHONY: studio
studio:
	npx prisma studio

.PHONY: clean
clean:
	rm -rf node_modules/.prisma generated/prisma

.PHONY: rebuild
rebuild: clean gen migrate seed-db


# ------------------------------------------------------------
# Evaluation data processing
# ------------------------------------------------------------

.PHONY: create-dirs
create-dirs:
	$(PYTHON) -m scripts.create_project_dirs

.PHONY: eval-check
eval-check:
	$(PYTHON) -m scripts.evaluation.check_evaluation_data

.PHONY: export-ratings
export-ratings:
	$(PYTHON) -m scripts.evaluation.export_ratings

.PHONY: aggregate-scores
aggregate-scores:
	$(PYTHON) -m scripts.evaluation.aggregate_poem_scores

.PHONY: create-master
create-master:
	$(PYTHON) -m scripts.create_master_dataset

.PHONY: create-dashboard-dataset
create-dashboard-dataset:
	$(PYTHON) -m scripts.create_dashboard_dataset

.PHONY: generate-figures
generate-figures:
	$(PYTHON) -m scripts.dashboard_figures.generate

.PHONY: analyze-quality
analyze-quality:
	$(PYTHON) -m scripts.evaluation.analyze_quality_by_round_workflow

.PHONY: process-data
process-data: create-dirs eval-check export-ratings aggregate-scores create-master create-dashboard-dataset generate-figures analyze-quality
	@echo "Evaluation data exported, dashboard dataset created, and figures generated."

.PHONY: clean-data
clean-data:
	rm -rf data/runtime/* data/work/*
	rm -rf public/research-dashboard/figures/* public/research-dashboard/analysis/*

# ------------------------------------------------------------
# AI Evaluator
# ------------------------------------------------------------

.PHONY: run-ai-evaluator
run-ai-evaluator:
	npm run rate:ai

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

.PHONY: help
help:
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "App:"
	@echo "  make run                         Run dev server"
	@echo "  make format                      Format files using Prettier"
	@echo ""
	@echo "Prisma / Database:"
	@echo "  make gen                         Generate Prisma Client"
	@echo "  make migrate                     Apply local migrations"
	@echo "  make migrate-new name=x          Create a new named migration"
	@echo "  make seed-db                     Seed database with poem data"
	@echo "  make setup-db                    Run migrations and seed database"
	@echo "  make reset                       Reset local database"
	@echo "  make reset-seed                  Reset local database and seed it"
	@echo "  make deploy                      Deploy migrations to production"
	@echo "  make studio                      Open Prisma Studio"
	@echo "  make clean                       Clean generated Prisma client only"
	@echo "  make rebuild                     Clean, generate client, migrate, and seed"
	@echo ""
	@echo "Evaluation data processing:"
	@echo "  make eval-check                  Check poems, evaluators, and rating completeness"
	@echo "  make export-ratings              Export raw evaluator ratings to CSV"
	@echo "  make aggregate-scores            Create poem-level mean rating scores"
	@echo "  make create-master               Create final master_round_dataset.csv"
	@echo "  make create-dashboard-dataset    Create deployable dashboard runtime dataset"
	@echo "  make generate-figures            Generate the visualization figures"
	@echo "  make analyze-quality             Run mixed-effects statistical analysis"
	@echo "  make process-data                Run full evaluation data pipeline"
	@echo ""
