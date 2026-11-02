# Independent acceptance entry points. The repository's Taiwan Makefile is unchanged.
REPO := $(abspath ..)
DATA_ROOT ?= $(REPO)/.cache/us-market-v1/acceptance
PORT ?= 8001
PYTHON ?= python
.PHONY: test pipeline api
test:
	cd "$(REPO)" && $(PYTHON) -m pytest -q tests/test_us_market.py
pipeline:
	cd "$(REPO)" && $(PYTHON) -m us_market.cli fixture --root "$(DATA_ROOT)"
api:
	cd "$(REPO)" && $(PYTHON) -m us_market.cli serve --root "$(DATA_ROOT)" --port "$(PORT)"
