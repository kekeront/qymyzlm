.PHONY: data data-local tokenizer tokenizer-local pack pack-local \
        train train-debug train-500m train-nano eval sft install lint test

install:
	uv sync --all-extras

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

test:
	PYTHONPATH=src pytest tests/ -v

# --- Full pipeline (GCP / large disk) ---
data:
	PYTHONPATH=src python scripts/download_data.py
	PYTHONPATH=src python scripts/clean_data.py

tokenizer:
	PYTHONPATH=src python scripts/train_tokenizer.py tokenizer=unigram_50k

pack:
	PYTHONPATH=src python scripts/pack_data.py

# --- Local validation pipeline (Wikipedia + multidomain only, fast) ---
data-local:
	PYTHONPATH=src python scripts/download_data.py data=local_validate
	PYTHONPATH=src python scripts/clean_data.py data=local_validate

tokenizer-local:
	PYTHONPATH=src python scripts/train_tokenizer.py data=local_validate tokenizer=unigram_50k

pack-local:
	PYTHONPATH=src python scripts/pack_data.py data=local_validate training=pretrain_debug

# --- Training ---
train:
	accelerate launch --config_file configs/accelerate_fsdp.yaml scripts/train.py

train-debug:
	PYTHONPATH=src python scripts/train.py training=pretrain_debug model=kaz50m_debug \
		data=local_validate

train-nano:
	PYTHONPATH=src python scripts/train.py model=kaz_nano training=pretrain_nano \
		data=local_validate

train-500m:
	PYTHONPATH=src python scripts/train.py model=kaz500m training=pretrain_500m

# --- Eval & SFT ---
eval:
	PYTHONPATH=src python scripts/evaluate.py

sft:
	PYTHONPATH=src python scripts/train_sft.py training=sft_lora
