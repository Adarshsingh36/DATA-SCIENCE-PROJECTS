PY ?= python3
export PYTHONPATH := src

.PHONY: data prepare eda forecast representative inventory all test dashboard figures
data:            ; $(PY) scripts/download_data.py
prepare:         ; $(PY) -m walmartfc.pipeline prepare
eda:             ; $(PY) -m walmartfc.pipeline eda
forecast:        ; $(PY) -m walmartfc.pipeline forecast
representative:  ; $(PY) -m walmartfc.pipeline representative
inventory:       ; $(PY) -m walmartfc.pipeline inventory
all:             ; $(PY) -m walmartfc.pipeline all && $(PY) scripts/make_report.py
test:            ; $(PY) -m pytest tests -q
dashboard:       ; $(PY) -m walmartfc.dashboard.app
figures:         ; $(PY) scripts/make_report.py
