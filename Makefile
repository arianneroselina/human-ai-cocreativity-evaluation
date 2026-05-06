.DEFAULT_GOAL := help

.PHONY: run
run:
	npm run dev

.PHONY: gen
gen:
	npx prisma generate

.PHONY: migrate
migrate:
	npx prisma migrate dev -n init

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
	rm -rf prisma/migrations node_modules/.prisma generated/prisma

.PHONY: rebuild
rebuild: clean reset gen migrate seed-db

.PHONY: format
format:
	npm run format

.PHONY: help
help:
	@echo "Available commands:"
	@echo "gen        Generate Prisma Client"
	@echo "migrate    Apply local migrations"
	@echo "seed-db    Seed database with initial poem data"
	@echo "setup-db   Run migrations and seed database"
	@echo "reset      Reset local database"
	@echo "reset-seed Reset local database and seed it"
	@echo "deploy     Deploy migrations to production"
	@echo "studio     Open Prisma Studio"
	@echo "clean      Clean Prisma directories (migrations, generated)"
	@echo "rebuild    Clean, generate Prisma Client, apply migrations, and seed"
	@echo "format	  Format files using Prettier"
