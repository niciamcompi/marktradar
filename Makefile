.PHONY: install lint test pipeline update serve docker

install:          ## Abhängigkeiten installieren
	uv sync

lint:             ## Formatierung, Linting, Typprüfung
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

test:             ## Tests mit Coverage
	uv run pytest

pipeline:         ## Komplette ML-Pipeline auf echten BAG-Daten
	uv run marktradar download
	uv run marktradar build
	uv run marktradar train

update:           ## Neue BAG-Daten suchen, Live-Güte messen, neu trainieren
	uv run marktradar update

serve:            ## Oberfläche + API lokal starten (http://localhost:8000)
	uv run marktradar serve

docker:           ## Image bauen (trainiert im Build) und starten
	docker build -t marktradar .
	docker run --rm -p 8000:8000 marktradar
