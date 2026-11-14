.PHONY: install lint test pipeline update serve docker

install:          ## Install dependencies
	uv sync

lint:             ## Formatting, linting, type checks
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

test:             ## Tests with coverage
	uv run pytest

pipeline:         ## Full ML pipeline on real BAG data
	uv run marktradar download
	uv run marktradar build
	uv run marktradar train

update:           ## Look for new BAG data, live check, retrain
	uv run marktradar update

serve:            ## Start web UI + API (http://localhost:8000)
	uv run marktradar serve

docker:           ## Build the image (trains during the build) and run it
	docker build -t marktradar .
	docker run --rm -p 8000:8000 marktradar
