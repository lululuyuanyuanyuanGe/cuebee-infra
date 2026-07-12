PYTHON ?= python3

.PHONY: test demo benchmark-speaker benchmark-stt serve

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

demo:
	PYTHONPATH=src $(PYTHON) -m cuebee.demo

benchmark-speaker:
	PYTHONPATH=src $(PYTHON) benchmarks/speaker_loadgen.py

benchmark-stt:
	PYTHONPATH=src $(PYTHON) benchmarks/stt_trace_replay.py

serve:
	PYTHONPATH=src $(PYTHON) -m cuebee.api.server

