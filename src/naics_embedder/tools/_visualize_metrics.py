'''Plot the durable monitor and epoch health values (P20); structural diagnostics stay separate.'''

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from naics_embedder.text_model.epoch_summary import read_epoch_summary

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

PLOT_FILENAME = 'epoch_metrics.png'
LOSS_KEYS = ('loss/task', 'loss/code_code', 'loss/radial', 'loss/total', 'loss/load_balancing')
SCALE_KEYS = ('logit_scale/task', 'logit_scale/code_code')

def _plot_fields(axis: Any, rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> None:
    '''Plot only the samples actually recorded for each field.'''

    for key in keys:
        samples = [row for row in rows if row.get(key) is not None]
        if samples:
            axis.plot(
                [row['epoch'] for row in samples], [row[key] for row in samples],
                marker='o',
                label=key
            )
    axis.set_xlabel('Epoch')
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend()

def create_visualizations(metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    '''
    Plot MRR, each loss term, both scales and every level's radius mean with its SD band.

    Values come from ``epoch_summary.jsonl``. Missing samples are omitted, and no structural
    statistic or heuristic recommendation is computed (Req 6). The output is ``epoch_metrics.png``.

    Raises:
        ValueError: If there are no epochs to plot.
        ImportError: If Matplotlib is unavailable.
    '''

    if not metrics:
        raise ValueError('No metrics found to visualize')
    if not HAS_MATPLOTLIB:
        raise ImportError('Matplotlib is not available')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    try:
        mrr, losses, scales, radii = axes.flat
        _plot_fields(mrr, metrics, ('mrr', ))
        mrr.set_title('Outcome monitor MRR')
        mrr.set_ylabel('MRR')
        _plot_fields(losses, metrics, LOSS_KEYS)
        losses.set_title('Epoch loss means')
        losses.set_ylabel('Loss')
        _plot_fields(scales, metrics, SCALE_KEYS)
        scales.set_title('Logit scales')
        scales.set_ylabel('Scale')
        radius_keys = sorted(
            {key
             for row in metrics
             for key in row if key.startswith('radius/mean/')}
        )
        _plot_fields(radii, metrics, radius_keys)
        radius_colors = {line.get_label(): line.get_color() for line in radii.lines}
        for key in radius_keys:
            sd_key = key.replace('/mean/', '/sd/')
            samples = [
                row for row in metrics if row.get(key) is not None and row.get(sd_key) is not None
            ]
            if samples:
                radii.fill_between(
                    [row['epoch']
                     for row in samples], [row[key] - row[sd_key] for row in samples], [
                         row[key] + row[sd_key] for row in samples
                     ],
                    color=radius_colors[key],
                    alpha=0.2
                )
        radii.set_title('Radius per level (mean ± SD)')
        radii.set_ylabel('Radius')
        figure.tight_layout()
        path = output_dir / PLOT_FILENAME
        figure.savefig(path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(figure)
    return path

def main() -> None:
    '''Plot an epoch summary from the command line.'''

    import argparse

    parser = argparse.ArgumentParser(description='Plot a durable epoch summary')
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    arguments = parser.parse_args()
    try:
        directory = arguments.output_dir or arguments.summary.parent / 'visualizations'
        path = create_visualizations(read_epoch_summary(arguments.summary), directory)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f'Visualization failed: {exc}\n')
    logger.info('Epoch metrics: %s', path)

if __name__ == '__main__':
    main()
