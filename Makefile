.PHONY: install test run offline
install:
	pip install -e ".[dev,real]"
test:
	pytest -q
run:
	model-distillation --json results.json
offline:
	model-distillation --offline
