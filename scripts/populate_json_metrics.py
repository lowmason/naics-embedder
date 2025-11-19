"""
One-off script to populate JSON metrics files from existing logs and TensorBoard data
for the three completed stages: 01_text, 02_text, 03_text
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from naics_embedder.tools._visualize_metrics import parse_log_file

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available. Will only extract from log files.")


def extract_tensorboard_metrics(tb_dir: Path, stage: str) -> Dict[int, Dict[str, float]]:
    """
    Extract metrics from TensorBoard event files.
    
    Returns:
        Dictionary mapping epoch -> {metric_name: value}
    """
    if not HAS_TENSORBOARD:
        return {}
    
    metrics = {}
    
    # Find the latest version directory for this stage
    stage_dir = tb_dir / stage
    if not stage_dir.exists():
        print(f"Warning: TensorBoard directory not found: {stage_dir}")
        return metrics
    
    # Find all version directories
    version_dirs = sorted([d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith('version_')])
    if not version_dirs:
        print(f"Warning: No version directories found in {stage_dir}")
        return metrics
    
    # Use the latest version
    latest_version = version_dirs[-1]
    event_files = list(latest_version.glob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"Warning: No event files found in {latest_version}")
        return metrics
    
    # Use the first event file (usually there's only one)
    event_file = event_files[0]
    
    try:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        
        # Get scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        # Extract epoch-end values for train and val loss
        train_loss_tag = None
        val_loss_tag = None
        
        for tag in scalar_tags:
            if 'train' in tag.lower() and 'loss' in tag.lower() and 'total' in tag.lower():
                train_loss_tag = tag
            elif 'val' in tag.lower() and 'loss' in tag.lower() and 'contrastive' in tag.lower():
                val_loss_tag = tag
        
        # Extract values - need to map steps to epochs
        # PyTorch Lightning logs validation at end of each epoch
        if train_loss_tag:
            train_loss_scalars = ea.Scalars(train_loss_tag)
            # Group by epoch (assuming steps are logged per epoch)
            for scalar in train_loss_scalars:
                step = scalar.step
                # Try to infer epoch from step (this may need adjustment based on logging frequency)
                epoch = int(step)
                if epoch not in metrics:
                    metrics[epoch] = {}
                # Keep the latest value for this epoch
                if 'train_loss' not in metrics[epoch] or scalar.wall_time > metrics[epoch].get('_train_time', 0):
                    metrics[epoch]['train_loss'] = scalar.value
                    metrics[epoch]['_train_time'] = scalar.wall_time
        
        if val_loss_tag:
            val_loss_scalars = ea.Scalars(val_loss_tag)
            for scalar in val_loss_scalars:
                step = scalar.step
                epoch = int(step)
                if epoch not in metrics:
                    metrics[epoch] = {}
                # Keep the latest value for this epoch
                if 'val_loss' not in metrics[epoch] or scalar.wall_time > metrics[epoch].get('_val_time', 0):
                    metrics[epoch]['val_loss'] = scalar.value
                    metrics[epoch]['_val_time'] = scalar.wall_time
        
        # Clean up temporary time fields
        for epoch in metrics:
            metrics[epoch].pop('_train_time', None)
            metrics[epoch].pop('_val_time', None)
                
    except Exception as e:
        print(f"Error reading TensorBoard file {event_file}: {e}")
    
    return metrics


def convert_log_metrics_to_json_format(log_metrics: List[Dict], tb_metrics: Dict[int, Dict[str, float]]) -> List[Dict]:
    """
    Convert log-parsed metrics to the JSON format that the model would save.
    
    Maps the log file format to the structured JSON format.
    """
    json_metrics = []
    
    for log_metric in log_metrics:
        epoch = log_metric.get('epoch')
        if epoch is None:
            continue
        
        # Start with base structure
        json_metric = {
            'epoch': epoch,
            # Training/validation loss from TensorBoard
            'train_loss': None,
            'train_contrastive_loss': None,
            'val_loss': None,
            # Hyperbolic metrics from log
            'hyperbolic_radius_mean': log_metric.get('radius_mean'),
            'hyperbolic_radius_std': log_metric.get('radius_std'),
            'lorentz_norm_mean': None,  # Not in log file
            'lorentz_norm_std': None,   # Not in log file
            'lorentz_norm_violation_max': None,  # Not in log file
            'manifold_valid': None,  # Not in log file
            # Embedding statistics (not in log file, but we can infer some)
            'mean_norm': None,
            'std_norm': None,
            'mean_pairwise_distance': None,
            'std_pairwise_distance': None,
            # Collapse detection from log
            'norm_cv': log_metric.get('norm_cv'),
            'distance_cv': log_metric.get('dist_cv'),
            'collapse_detected': log_metric.get('collapse', False),
            # Hierarchy preservation from log
            'cophenetic_correlation': log_metric.get('cophenetic'),
            'cophenetic_n_pairs': log_metric.get('n_pairs'),
            'spearman_correlation': None,  # Not tracked anymore
            'spearman_n_pairs': None,
            # Ranking metrics (not in log file)
            'ndcg@5': None,
            'ndcg@10': None,
            'ndcg@20': None,
            'ndcg@5_n_queries': None,
            'ndcg@10_n_queries': None,
            'ndcg@20_n_queries': None,
            # Distortion metrics (not in log file)
            'mean_distortion': None,
            'std_distortion': None,
            'median_distortion': None,
            # Sample size
            'num_samples': log_metric.get('n_pairs'),  # Approximate
        }
        
        # Merge TensorBoard metrics if available
        if epoch in tb_metrics:
            tb_data = tb_metrics[epoch]
            if 'train_loss' in tb_data:
                json_metric['train_loss'] = tb_data['train_loss']
            if 'val_loss' in tb_data:
                json_metric['val_loss'] = tb_data['val_loss']
        
        json_metrics.append(json_metric)
    
    return json_metrics


def populate_stage_json(stage: str, log_file: Path, tb_dir: Path, output_base: Path):
    """Populate JSON metrics file for a single stage."""
    print(f"\n{'='*90}")
    print(f"Processing stage: {stage}")
    print(f"{'='*90}\n")
    
    # Find the output directory for this stage
    stage_dir = output_base / stage
    if not stage_dir.exists():
        print(f"Error: Stage directory not found: {stage_dir}")
        return False
    
    # Find the latest version directory
    version_dirs = sorted([d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith('version_')])
    if not version_dirs:
        print(f"Error: No version directories found in {stage_dir}")
        return False
    
    latest_version = version_dirs[-1]
    json_file = latest_version / 'evaluation_metrics.json'
    
    # Check if JSON already exists
    if json_file.exists():
        print(f"Warning: JSON file already exists: {json_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping...")
            return False
    
    # Parse log file
    print(f"Parsing log file: {log_file}")
    log_metrics = parse_log_file(log_file, stage=stage)
    
    if not log_metrics:
        print(f"Warning: No metrics found for stage {stage} in log file!")
        return False
    
    print(f"Found {len(log_metrics)} evaluation epochs in log file")
    
    # Extract TensorBoard metrics
    tb_metrics = {}
    if HAS_TENSORBOARD:
        print(f"Extracting TensorBoard metrics from: {tb_dir / stage}")
        tb_metrics = extract_tensorboard_metrics(tb_dir, stage)
        if tb_metrics:
            print(f"Found TensorBoard metrics for {len(tb_metrics)} epochs")
        else:
            print("No TensorBoard metrics found")
    else:
        print("TensorBoard not available, skipping TensorBoard extraction")
    
    # Convert to JSON format
    print("Converting to JSON format...")
    json_metrics = convert_log_metrics_to_json_format(log_metrics, tb_metrics)
    
    # Save to JSON file
    print(f"Saving to: {json_file}")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"✓ Successfully saved {len(json_metrics)} metrics to {json_file}")
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Populate JSON metrics from logs and TensorBoard')
    parser.add_argument('--stages', nargs='+', default=['01_text', '02_text', '03_text'],
                       help='Stages to process')
    parser.add_argument('--log-file', type=Path,
                       default=project_root / 'logs' / 'logs' / 'train.log',
                       help='Path to log file')
    parser.add_argument('--tb-dir', type=Path,
                       default=project_root / 'outputs',
                       help='TensorBoard output directory')
    parser.add_argument('--output-dir', type=Path,
                       default=project_root / 'outputs',
                       help='Output directory (where JSON files will be saved)')
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    print("="*90)
    print("Populating JSON Metrics Files from Logs and TensorBoard")
    print("="*90)
    print(f"Log file: {args.log_file}")
    print(f"TensorBoard dir: {args.tb_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Stages: {', '.join(args.stages)}")
    print("="*90)
    
    success_count = 0
    for stage in args.stages:
        if populate_stage_json(stage, args.log_file, args.tb_dir, args.output_dir):
            success_count += 1
    
    print(f"\n{'='*90}")
    print(f"Complete! Successfully processed {success_count}/{len(args.stages)} stages")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    main()

