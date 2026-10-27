PY ?= .venv/bin/python

.PHONY: setup simulate ingest train demo serve test

setup:
	uv venv --python 3.12 .venv && uv pip install -r requirements.txt

simulate:  ## generate the seeded plant: ground truth + metadata in data/sim
	$(PY) -m simulator generate

ingest:  ## offline ingestion (no brokers): decode, validate, Parquet, resample
	$(PY) -m ingestion offline

train:  ## features, anomaly detector, classifier, metrics + figures
	$(PY) -m models.train

demo: simulate ingest train  ## end-to-end, offline

serve:  ## live simulator on Modbus TCP :5020 (add MQTT with: make serve ARGS="--mqtt-host localhost")
	$(PY) -m simulator serve $(ARGS)

test:
	$(PY) -m pytest -q
