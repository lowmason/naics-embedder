'''
Metrics visualization tools.

Provides a function to visualize training metrics. Req 6's structural statistics come from
``tools diagnostics`` alone: ``tools investigate`` is retired (spec 4.4).
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Optional

try:
    import matplotlib

    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from naics_embedder.tools._visualize_metrics import (
        create_visualizations,
        parse_log_file,
        print_analysis,
    )

    HAS_VISUALIZE = True
except ImportError:
    HAS_VISUALIZE = False

# -------------------------------------------------------------------------------------------------
# Visualize metrics
# -------------------------------------------------------------------------------------------------

def visualize_metrics(
    stage: str = '02_text',
    log_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Dict:
    '''
    Visualize training metrics from log files.

    Args:
        stage: Stage name to filter (e.g., '02_text')
        log_file: Path to log file (default: logs/train_sequential.log)
        output_dir: Output directory for plots (default: outputs/visualizations/)
        project_root: Project root directory (default: current working directory)

    Returns:
        Dictionary with metrics and output file path
    '''

    if project_root is None:
        project_root = Path.cwd()

    if log_file is None:
        log_file = project_root / 'logs' / 'train_sequential.log'

    if output_dir is None:
        output_dir = project_root / 'outputs' / 'visualizations'

    if not log_file.exists():
        raise FileNotFoundError(f'Log file not found: {log_file}')

    if not HAS_VISUALIZE:
        raise ImportError('Visualization tools not available. Missing dependencies.')

    # Parse metrics
    metrics = parse_log_file(log_file, stage=stage)

    if not metrics:
        raise ValueError(f"No metrics found for stage '{stage}' in log file!")

    # Create visualizations
    if HAS_MATPLOTLIB:
        create_visualizations(metrics, output_dir, stage)
        output_file = output_dir / f'{stage}_metrics.png'
    else:
        output_file = None
        print('⚠️  Matplotlib not available. Skipping visualization creation.')

    # Print analysis
    print_analysis(metrics, stage)

    # Print summary table
    print('\n' + '=' * 90)
    print('METRICS SUMMARY TABLE')
    print('=' * 90)
    print(f"{'Epoch':<8} {'Radius':<15} {'Dist CV':<10} {'Collapse':<10}")
    print('-' * 90)
    for m in metrics:
        epoch = m.get('epoch', 'N/A')
        radius = f"{m.get('radius_mean', 0):.2f}±{m.get('radius_std', 0):.2f}"
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(f'{epoch:<8} {radius:<15} {dist_cv:<10} {collapse:<10}')
    print()

    return {
        'metrics': metrics,
        'output_file': output_file,
        'stage': stage,
        'num_epochs': len(metrics),
    }
