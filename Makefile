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

.PHONY: eval-check
eval-check:
	$(PYTHON) -m scripts.check_evaluation_data

.PHONY: export-ratings
export-ratings:
	$(PYTHON) -m scripts.export_ratings

.PHONY: aggregate-scores
aggregate-scores:
	$(PYTHON) -m scripts.aggregate_poem_scores

.PHONY: create-master
create-master:
	$(PYTHON) -m scripts.create_master_dataset

.PHONY: generate-figures
generate-figures:
	$(PYTHON) -m scripts.dashboard_figures.generate

.PHONY: process-data
process-data: eval-check export-ratings aggregate-scores create-master generate-figures
	@echo "Evaluation data exported, master dataset created, and figures generated."


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
	@echo "  make process-data                Run full evaluation data pipeline"
	@echo ""
