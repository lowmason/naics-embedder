'''
Build the window-fitting summaries (roadmap Stage 6b).

``naics-embedder data summaries`` runs this once per backbone. Every channel text whose marked
form is over the backbone's trained window is cut into units (``panels.window_summaries``), and
the summary keeps the units whose pooled vectors best approximate the whole text's (centrality):

1. Of the units whose marked summary fits the window on its own, keep the one whose weighted
   vector has the highest cosine with the target, the weighted sum of all the text's units.
2. Repeatedly add the unit that most raises that cosine, among the units that keep the marked
   summary, joined in source order, within the window.
3. Stop when no remaining unit fits, or none raises the cosine strictly.

The frozen backbone reads each unit unmarked and alone, on the CPU in float32; its last hidden
state is mean-pooled over the attention mask and L2-normalized, and a unit's weight is its token
count without special tokens. Exact ties go to the earlier unit. The kept units are emitted in
source order.

The artifact is written to a temporary file and checked by ``resolve_channel_texts`` under a
temporary pin before it is moved into place, so a failing invariant leaves no artifact. It is
committed and pinned in ``WINDOW_SUMMARIES``; neither ``data preprocess`` nor ``data all`` runs
this.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch

from naics_embedder.panels.leakage import EXAMPLES_SEPARATOR
from naics_embedder.panels.text_only import load_backbone, provenance_path
from naics_embedder.panels.window_summaries import (
    NO_BREAK_PATTERN,
    SUMMARIES_SCHEMA,
    SUMMARY_CHANNELS,
    UNIT_RULE,
    SummariesPin,
    over_window,
    resolve_channel_texts,
    summary_budget,
    text_sha256,
    text_units,
    token_counter,
    write_window_summaries,
)
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.fields import CHANNELS, marked_text
from naics_embedder.utils.input_window import trained_window

logger = logging.getLogger(__name__)

SELECTION_RULE = 'centrality-v1'

# Texts to unit vectors, one row per text
UnitEmbedder = Callable[[Sequence[str]], np.ndarray]

# -------------------------------------------------------------------------------------------------
# Selection
# -------------------------------------------------------------------------------------------------

def backbone_embedder(model: Any, tokenizer: Any, *, batch_size: int = 64) -> UnitEmbedder:
    '''
    Unit vectors from a frozen backbone: each unit read unmarked and alone, its last hidden state
    mean-pooled over the attention mask.

    Args:
        model: A Hugging Face encoder returning ``last_hidden_state``.
        tokenizer: Its tokenizer.
        batch_size: Units per forward pass.

    Returns:
        A function from units to their vectors (float64).
    '''

    def embed(units: Sequence[str]) -> np.ndarray:
        parts = []
        for start in range(0, len(units), batch_size):
            tokens = tokenizer(
                list(units[start:start + batch_size]),
                padding=True,
                truncation=False,
                return_tensors='pt',
            )
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            parts.append(pooled.cpu().to(torch.float64).numpy())
        return np.concatenate(parts)

    return embed

@dataclass(frozen=True)
class Selection:
    '''
    The units a summary keeps.

    Attributes:
        kept: Indices of the kept units, in source order.
        plateau: True when the selection stopped because no unit that still fit raised the cosine.
    '''

    kept: List[int]
    plateau: bool

def _cosine(total: np.ndarray, target: np.ndarray) -> float:
    return float(total @ target / (np.linalg.norm(total) * np.linalg.norm(target)))

def select_units(
    vectors: np.ndarray,
    weights: np.ndarray,
    fits: Callable[[List[List[int]]], List[bool]],
) -> Selection:
    '''
    Greedy centrality selection over one text's units.

    Args:
        vectors: One row per unit, in source order.
        weights: Each unit's token count, without special tokens.
        fits: For each candidate (unit indices in source order), whether its marked summary fits
            the window.

    Returns:
        The kept units.

    Raises:
        ValueError: If no unit fits the window on its own.
    '''

    weighted = weights[:, None] * (vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    target = weighted.sum(axis=0)
    kept: List[int] = []
    best = -np.inf
    while True:
        remaining = [index for index in range(len(vectors)) if index not in kept]
        candidates = [sorted(kept + [index]) for index in remaining]
        fitting = [index for index, ok in zip(remaining, fits(candidates)) if ok]
        if not fitting:
            if not kept:
                raise ValueError('no unit fits the window on its own')
            return Selection(kept=sorted(kept), plateau=False)
        top, top_score = fitting[0], -np.inf
        for index in fitting:
            score = _cosine(weighted[kept + [index]].sum(axis=0), target)
            # Strictly greater: an exact tie goes to the earlier unit
            if score > top_score:
                top, top_score = index, score
        if kept and not top_score > best:
            return Selection(kept=sorted(kept), plateau=True)
        kept.append(top)
        best = top_score

def joiner(channel: str) -> str:
    '''What a summary's units are joined by: the examples separator, or one space.'''

    return EXAMPLES_SEPARATOR if channel == 'examples' else ' '

def summary_rows(
    descriptions: pl.DataFrame,
    tokenizer: Any,
    embed: UnitEmbedder,
    *,
    window: int,
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    '''
    One summary row per over-window channel text.

    Identical source texts get identical summaries, one row per code. Each distinct unit is
    embedded once.

    Args:
        descriptions: Codes and their four channel texts.
        tokenizer: The backbone's tokenizer.
        embed: Unit vectors (``backbone_embedder``).
        window: The trained window.

    Returns:
        The rows, with the columns of ``SUMMARIES_SCHEMA``, and per channel the summaries that
        ended at the plateau stop.

    Raises:
        ValueError: If a title is over the window, or a unit is over its channel's budget.
    '''

    count_marked = token_counter(tokenizer)
    count_plain = token_counter(tokenizer, special_tokens=False)
    over = over_window(descriptions, count_marked, window)
    texts = {
        (code, channel): text
        for channel in CHANNELS
        for code, text in descriptions.select('code', channel).iter_rows()
    }
    units_of = {
        pair: text_units(
            pair[1], texts[pair], count_plain, summary_budget(count_marked, pair[1], window)
        )
        for pair in over
    }
    unique = list(dict.fromkeys(unit for units in units_of.values() for unit in units))
    logger.info(f'Embedding {len(unique):,} distinct units of {len(over):,} over-window texts')
    vectors = dict(zip(unique, embed(unique))) if unique else {}
    weights = dict(zip(unique, count_plain(unique)))

    rows: List[Dict[str, Any]] = []
    plateau = {channel: 0 for channel in SUMMARY_CHANNELS}
    chosen: Dict[Tuple[str, str], Tuple[str, int, bool]] = {}
    for code, channel in over:
        source, units = texts[(code, channel)], units_of[(code, channel)]
        join = joiner(channel)
        if (channel, source) not in chosen:

            def fits(candidates: List[List[int]]) -> List[bool]:
                summaries = [join.join(units[index] for index in kept) for kept in candidates]
                marked = [marked_text(channel, summary) for summary in summaries]
                return [tokens <= window for tokens in count_marked(marked)]

            selection = select_units(
                np.stack([vectors[unit] for unit in units]),
                np.array([weights[unit] for unit in units], dtype=np.float64),
                fits,
            )
            summary = join.join(units[index] for index in selection.kept)
            chosen[(channel, source)] = (summary, len(selection.kept), selection.plateau)
        summary, kept, stopped = chosen[(channel, source)]
        plateau[channel] += int(stopped)
        rows.append(
            {
                'code': code,
                'channel': channel,
                'source_sha256': text_sha256(source),
                'window': window,
                'summary': summary,
                'source_tokens': count_marked([marked_text(channel, source)])[0],
                'summary_tokens': count_marked([marked_text(channel, summary)])[0],
                'units_kept': kept,
                'units_total': len(units),
            }
        )
    return pl.DataFrame(rows, schema=SUMMARIES_SCHEMA), plateau

# -------------------------------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------------------------------

def generate_window_summaries(
    descriptions_path: Path,
    output_path: Path,
    *,
    backbone: str,
    force: bool = False,
    model: Optional[Any] = None,
    tokenizer: Any = None,
    revision: Optional[str] = None,
) -> SummariesPin:
    '''
    Summarize every over-window channel text, check the artifact, and write it and its provenance.

    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``, from the local Hugging
    Face cache; tests pass small ones.

    Args:
        descriptions_path: The descriptions parquet.
        output_path: The artifact (CSV); its provenance is written beside it.
        backbone: The backbone, whose trained window the summaries fit.
        force: Overwrite an existing artifact.
        model: The backbone's model.
        tokenizer: The backbone's tokenizer.
        revision: The backbone's snapshot revision.

    Returns:
        The pin to commit in ``WINDOW_SUMMARIES``.

    Raises:
        FileExistsError: If the artifact exists and ``force`` is False.
        ValueError: If a text cannot be summarized, or the artifact fails a resolver check.
    '''

    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f'{output_path} exists: the summaries are built once and committed, and a new '
            'artifact needs a new pin. Pass --force only to rebuild it deliberately.'
        )
    window = trained_window(backbone)
    descriptions_path = Path(descriptions_path)
    descriptions = pl.read_parquet(descriptions_path)
    if model is None or tokenizer is None:
        model, tokenizer, revision = load_backbone(backbone)
    rows, plateau = summary_rows(
        descriptions, tokenizer, backbone_embedder(model, tokenizer), window=window
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + '.tmp')
    try:
        sha256 = write_window_summaries(rows, temporary)
        # Every invariant of the resolver, against the bytes about to be committed
        resolve_channel_texts(
            descriptions,
            tokenizer,
            backbone,
            window,
            pin=SummariesPin(path=str(temporary), sha256=sha256, window=window),
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    provenance = _provenance(
        descriptions,
        descriptions_path,
        rows,
        plateau,
        tokenizer,
        backbone=backbone,
        revision=revision,
        window=window,
        sha256=sha256,
    )
    provenance_path(output_path).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(f'Window summaries ({rows.height:,} texts, sha256 {sha256}): {output_path}')
    return SummariesPin(path=str(output_path), sha256=sha256, window=window)

def _provenance(
    descriptions: pl.DataFrame,
    descriptions_path: Path,
    rows: pl.DataFrame,
    plateau: Dict[str, int],
    tokenizer: Any,
    *,
    backbone: str,
    revision: Optional[str],
    window: int,
    sha256: str,
) -> Dict[str, Any]:
    count_marked = token_counter(tokenizer)
    over = over_window(descriptions, count_marked, window)
    channels: Dict[str, Dict[str, Any]] = {}
    for channel in CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        summarized = rows.filter(pl.col('channel') == channel)
        tokens = summarized.get_column('summary_tokens')
        share = summarized.get_column('summary_tokens') / summarized.get_column('source_tokens')
        channels[channel] = {
            'present': sum(text is not None and bool(text.strip()) for text in texts),
            'over_window': sum(pair[1] == channel for pair in over),
            'summarized': summarized.height,
            'mean_kept_share': share.mean() if summarized.height else None,
            'min_summary_tokens': tokens.min() if summarized.height else None,
            'p10_summary_tokens': (
                int(tokens.quantile(0.1, interpolation='lower')) if summarized.height else None
            ),
            'plateau_stops': plateau.get(channel, 0),
        }
    return {
        'descriptions': {
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'backbone': backbone,
        'revision': revision,
        # The tokenizer and the model load from one snapshot
        'tokenizer': backbone,
        'tokenizer_revision': revision,
        'window': window,
        'budget': {
            channel: summary_budget(count_marked, channel, window)
            for channel in CHANNELS
        },
        'units': {
            'rule': UNIT_RULE,
            'no_break_pattern': NO_BREAK_PATTERN
        },
        'selection': {
            'rule': SELECTION_RULE,
            'pooling': 'attention-masked mean of the last hidden state, each unit unmarked',
            'weights': 'token count without special tokens',
            'device': 'cpu',
            'dtype': 'float32',
        },
        'channels': channels,
        'artifact_sha256': sha256,
        'library_versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'polars', 'numpy')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
