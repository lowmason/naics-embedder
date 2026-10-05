'''Five tiny trained seeds of the reference fixture, with every file under pytest's temp root.'''

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pytest
import pytorch_lightning as pyl

from naics_embedder.cli.commands import training
from naics_embedder.panels import window_summaries
from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, load_validated_bundle
from naics_embedder.text_model import shared_encoder
from naics_embedder.text_model.export import code_token_config
from naics_embedder.utils.config import Config, TokenizationConfig
from naics_embedder.utils.training import HardwareInfo, create_trainer, run_settings
from tests.fixtures.shared_encoder import tiny_bert
from tests.fixtures.supervision import build_reference_bundle

SEEDS = (1, 2, 3, 4, 5)
REVISION = 'fixture-backbone-revision'

def cached_tiny_backbone(name: str):
    model = tiny_bert(name)
    model.config._commit_hash = REVISION
    return model

@dataclass(frozen=True)
class TrainedSeeds:
    root: Path
    cfg: Config
    bundle: ValidatedSupervisionBundle
    text_only: Path
    seeds: Tuple[int, ...] = SEEDS

    def directory(self, seed: int) -> Path:
        return self.root / f'seed-{seed}'

@pytest.fixture(scope='session')
def trained_seeds(tmp_path_factory, minilm_tokenizer) -> TrainedSeeds:
    root = tmp_path_factory.mktemp('checkpoint-seeds')
    manifest = build_reference_bundle(root / 'reference')
    bundle = load_validated_bundle(manifest)
    descriptions = bundle.manifest.generation_parameters['descriptions_parquet']
    cfg = Config().override(
        {
            'supervision.manifest_path': str(manifest),
            'data_loader.streaming.descriptions_parquet': descriptions,
            'data_loader.queries_per_step': 4,
            'model.dimension': 8,
            'training.trainer.max_epochs': 3,
            'training.trainer.log_every_n_steps': 1,
            'dirs.output_dir': str(root / 'outputs'),
        }
    )
    token_path = root / 'tokens' / 'cache.pt'

    def moved_tokens(config: Config) -> TokenizationConfig:
        return code_token_config(config).model_copy(update={'output_path': str(token_path)})

    pin = window_summaries.SummariesPin(
        path='/nonexistent/window_summaries.csv', sha256='5' * 64, window=128
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shared_encoder, 'load_base_model', cached_tiny_backbone)
        patch.setitem(window_summaries.WINDOW_SUMMARIES, cfg.model.base_model_name, pin)
        patch.setattr(training, 'code_token_config', moved_tokens)
        for seed in SEEDS:
            seeded = cfg.override({'seed': seed})
            directory = root / f'seed-{seed}'
            hardware = HardwareInfo(accelerator='cpu', precision='32-true', num_devices=1)
            settings = run_settings(seeded, accelerator='cpu', precision='32-true')
            pyl.seed_everything(seed, verbose=False)
            monitor = training.build_monitor_from_config(
                seeded, bundle, directory, selection_log=root / f'training-{seed}.jsonl'
            )
            model = training.build_model_from_config(
                seeded,
                training.runtime_contract_for(seeded, bundle),
                bundle,
                run_settings=settings,
                monitor=monitor
            )
            datamodule = training.build_datamodule_from_config(seeded, bundle)
            trainer, _, _ = create_trainer(seeded, hardware, directory)
            trainer.fit(model, datamodule)
        table = root / 'text_only.parquet'
        build_text_only_table(
            descriptions,
            table,
            backbone=cfg.model.base_model_name,
            max_length=cfg.data_loader.streaming.max_length,
            model=cached_tiny_backbone(cfg.model.base_model_name),
            tokenizer=minilm_tokenizer,
            revision=REVISION
        )
    return TrainedSeeds(root, cfg, bundle, table)
