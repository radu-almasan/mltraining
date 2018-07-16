SHELL := /bin/bash

all: venv init resources test

clean:
	rm -rf .pytest_cache
	rm -rf bin
	rm -rf include
	rm -rf lib
	rm -f pip-selfcheck.json
	rm -f pyvenv.cfg
	rm -f FuelConsumption.csv

venv:
	test -f pyvenv.cfg || python3 -m venv .

init:
	source bin/activate && \
	pip install -r requirements.txt

resources: venv
	wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

test: venv
	source bin/activate && \
	py.test tests


