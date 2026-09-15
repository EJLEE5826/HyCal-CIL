"""
Experiment Runner - Main Entry Point

This module contains the main entry point and experiment setup:
- Command line argument parsing
- Configuration loading
- Device and wandb initialization
- Experiment execution coordination
"""

import os
import logging
from datetime import datetime

from hycal.experiment_setup import parse_args, load_configs, setup_output_dir, setup_logging
logger = logging.getLogger(__name__)


def setup_experiment(args, cfg):
    """Setup device, logging, and wandb for experiment.

    Args:
        args: Command line arguments
        cfg: Configuration object

    Returns:
        device: PyTorch device (cuda/cpu)
    """
    import torch
    from task_datasets.dataset import set_seed

    # GPU configuration
    if torch.cuda.is_available():
        try:
            gpu_id = int(args.gpu)
            if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
                torch.cuda.set_device(gpu_id)
                print(f"Using GPU device {gpu_id}")
            else:
                print(f"Invalid GPU ID {gpu_id}, using device 0 instead")
                torch.cuda.set_device(0)
        except ValueError:
            print(f"Invalid GPU ID format {args.gpu}, using device 0 instead")
            torch.cuda.set_device(0)

    # Set random seeds for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
            
    # Log command line arguments
    logger.info('Command line arguments:')
    for arg in vars(args):
        logger.info(f'  {arg}: {getattr(args, arg)}')
    
    # Log device info
    logger.info(f'Using device: {device}')
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(args.gpu)}')
    
    # Initialize wandb for experiment tracking
    if args.use_wandb:
        _setup_wandb(args, cfg)
    
    logger.info(f"Experiment setup complete. Output directory: {args.output_dir}")
    
    return device


def _setup_wandb(args, cfg):
    """Initialize wandb with experiment configuration.

    Args:
        args: Command line arguments
        cfg: Configuration object
    """
    import wandb

    # Extract experiment info for wandb run name
    if args.wandb_run_name is None:
        run_name = args.output_dir
    else:
        run_name = args.wandb_run_name + args.output_dir
    
    wandb_config = {}
    
    # Add args attributes to wandb_config (exclude None values)
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            wandb_config[arg] = value
    
    # Add cfg attributes to wandb_config (exclude None values)
    for attr in vars(cfg):
        value = getattr(cfg, attr)
        if value is not None and attr not in wandb_config:
            wandb_config[attr] = value

    wandb_config["num_classes"] = len(cfg.datasets)

    # Initialize wandb with retry logic
    try:
        # Close any existing wandb session
        if wandb.run is not None:
            wandb.finish()
            
        # Initialize wandb with timeout setting
        init_kwargs = {
            "project": args.wandb_project,
            "name": run_name,
            "config": wandb_config,
            "dir": args.output_dir,
            "reinit": True,
            "settings": wandb.Settings(
                start_method="thread",
                _disable_stats=True
            ),
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)
        
        logger.info(f"Initialized wandb tracking with run name: {run_name}")
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        logger.warning("Continuing experiment without wandb tracking")
        args.use_wandb = False


def main(args, cfg, few_shot_cfg):
    """Execute main experiment workflow.

    Args:
        args: Command line arguments
        cfg: Configuration object
        few_shot_cfg: Few-shot configuration
    """
    from hycal.experiment_workflow import run_experiment

    # Setup experiment
    device = setup_experiment(args, cfg)
    
    # Validate test-only mode
    if args.mode in ['testonly', 'testonly_any', 'testonly_fs'] and not args.checkpoint:
        raise ValueError("Checkpoint path must be provided for test-only mode")
    
    # Run experiment
    results = run_experiment(args, device, cfg, few_shot_cfg, checkpoint_path=args.checkpoint)
    
    logger.info(f"Experiment completed. Results saved at: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    if args.fusion == 'None':
        args.fusion = None

    # Load configuration files
    cfg, few_shot_cfg = load_configs(args)

    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare output directory
    args, cfg = setup_output_dir(args, cfg, timestamp)

    # Setup logging
    logger = setup_logging(log_file=os.path.join(args.output_dir, "experiment_log.log"))

    # Run main experiment
    main(args, cfg, few_shot_cfg)
