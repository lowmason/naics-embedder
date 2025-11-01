
# NAICS Contrastive Learning

This project implements a production-ready training shell for learning NAICS code embeddings using contrastive learning.

It features a multi-channel architecture (title, description, examples), LoRA-based fine-tuning, cross-attention, and a Mixture-of-Experts (MoE) fusion layer. Training is performed using a hierarchical triplet loss and a curriculum-driven streaming dataset.

## Setup (using uv)

This project uses `uv` for package management, which is extremely fast.

1.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

2.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    If you are on a new machine, you can generate a new lock file and sync from it:
    ```bash
    # Compile the pyproject.toml into a new uv.lock
    uv pip compile pyproject.toml -o uv.lock
    
    # Sync the environment with the lock file
    uv pip sync uv.lock
    ```
    
    Alternatively, for development, install the project in editable mode:
    ```bash
    uv pip install --editable .
    ```

## Data Preparation

The project requires three core data files. You can generate them using the CLI (powered by `typer`):

```bash
# This will run all data generation steps in order
uv run naics-cl data --all

# You can also run individual steps:
uv run naics-cl data --descriptions
uv run naics-cl data --distances
uv run naics-cl data --triplets
```

This will create the following files in the `./data/` directory:
* `naics_descriptions.parquet`
* `naics_distances.parquet`
* `naics_training_pairs.parquet`

## Training

Once the data is prepared, you can start the training process:

```bash
# This will load the default config and start training
uv run naics-cl train

# You can also specify a different config file
uv run naics-cl train --config-path conf/my_custom_config.yaml
```

The trainer uses settings defined in the `conf/` directory, including the model architecture and curriculum stages.
