PYTHON ?= python3
CONFIG ?= configurations/projet.example.json

.PHONY: help check predictions presentation all test
help:
	@echo "make check CONFIG=chemin.json        : vérifier les sources"
	@echo "make predictions CONFIG=chemin.json  : calculer rails et objets"
	@echo "make presentation CONFIG=chemin.json : créer vidéos et ZIP simplifiés"
	@echo "make all CONFIG=chemin.json          : exécuter les deux étapes"
	@echo "make test                           : vérifier les contrats de présentation"
check:
	$(PYTHON) lancer.py --config "$(CONFIG)" --stage check
predictions:
	$(PYTHON) lancer.py --config "$(CONFIG)" --stage predictions
presentation:
	$(PYTHON) lancer.py --config "$(CONFIG)" --stage presentation
all:
	$(PYTHON) lancer.py --config "$(CONFIG)" --stage all
test:
	$(PYTHON) -m unittest discover -s tests -v
