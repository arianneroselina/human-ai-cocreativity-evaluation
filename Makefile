.DEFAULT_GOAL := help

.PHONY: run
run:
	npm run dev

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

.PHONY: format
format:
	npm run format

.PHONY: help
help:
	@echo "Available commands:"
	@echo "run                 Run dev server"
	@echo "gen                 Generate Prisma Client"
	@echo "migrate             Apply local migrations"
	@echo "migrate-new name=x  Create a new named migration"
	@echo "seed-db             Seed database with initial poem data"
	@echo "setup-db            Run migrations and seed database"
	@echo "reset               Reset local database"
	@echo "reset-seed          Reset local database and seed it"
	@echo "deploy              Deploy migrations to production"
	@echo "studio              Open Prisma Studio"
	@echo "clean               Clean generated Prisma client only"
	@echo "rebuild             Regenerate client, migrate local DB, and seed"
	@echo "format              Format files using Prettier"
