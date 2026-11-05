# Independent acceptance entry points. The repository's Taiwan Makefile is unchanged.
REPO := $(abspath ..)
PROVIDER ?= massive
DATA_ROOT ?= $(REPO)/.cache/us-market-v1/$(if $(filter alpaca,$(PROVIDER)),alpaca-acceptance,acceptance)
PORT ?= 8001
PYTHON ?= python
.PHONY: test pipeline api
test:
	cd "$(REPO)" && $(PYTHON) -m pytest -q tests/test_us_market.py tests/test_alpaca_market.py tests/test_us_research.py
pipeline:
	cd "$(REPO)" && $(PYTHON) -m us_market.cli fixture --provider "$(PROVIDER)" --root "$(DATA_ROOT)"
api:
	cd "$(REPO)" && $(PYTHON) -m us_market.cli serve --provider "$(PROVIDER)" --root "$(DATA_ROOT)" --port "$(PORT)"
