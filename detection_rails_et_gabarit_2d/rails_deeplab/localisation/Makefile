PYTHON ?= .venv/bin/python
CONFIG ?= configurations/pipeline.local.json

.PHONY: help check test run all network
help:
	@printf '%s\n' 'make check CONFIG=... : vérifier les chemins' 'make test : tests du programme' 'make run CONFIG=... : GPS, jointures, vidéo et ZIP' 'make all CONFIG=... : tests puis traitement' 'make network : récupérer la carte africaine publique'

check:
	"$(PYTHON)" run_pipeline.py --config "$(CONFIG)" --check

test:
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m unittest discover -s tests -v

run:
	"$(PYTHON)" run_pipeline.py --config "$(CONFIG)"

all: test
	"$(PYTHON)" run_pipeline.py --config "$(CONFIG)"

network:
	"$(PYTHON)" download_network.py
