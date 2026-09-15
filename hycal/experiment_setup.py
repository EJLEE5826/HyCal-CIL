import argparse
import os
import logging


def parse_args():
    """Parse command line arguments for HyCal experiments."""
    parser = argparse.ArgumentParser(description='HyCal Continual Learning Experiment')

    parser.add_argument('--real', action='store_true',
                        help='lambda check')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='scale')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='default',
                        choices=['default', 'eval', 'testonly', 'testonly_any', 'testonly_fs'],
                        help='Operation mode (default, eval, testonly, etc.)')
    
    # Data arguments
    parser.add_argument('--cfg', type=str, default='config/data_config.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--few_shot_cfg', type=str, default='config/few_shot_config_highimbal.yaml',
                        help='Path to few-shot configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (default: 0)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override the dataset root from the YAML config')
    parser.add_argument('--eval_setting', '--eval-setting', dest='eval_setting', type=str, default='cil',
                        choices=['cil', 'mtil', 'xtail'],
                        help='Evaluation setting (cil, mtil, xtail)')
    parser.add_argument('--sim_metric', type=str, default='weighted',
                        choices=['cosine', 'md', 'avg', 'weighted'],
                        help='Similarity metric for CLIP features (cosine or mahalanobis or avg or weighted)')

    parser.add_argument('--num_shots', type=int, default=None,
                        help='Number of shots for few-shot learning (default: None)')
    parser.add_argument('--num_shots_each_range', nargs="+", default=None, type=int, #[5,50]
                        help='Range of any shots for few-shot learning (default: None)')    
    # Model selection
    parser.add_argument('--model_provider', type=str, choices=['openai', 'openclip', 'laion'], 
                        default='openai', help='CLIP model provider (openai or openclip)')
    parser.add_argument('--fusion', type=str, choices=['sum', 'concat', 'None'],
                        default='sum', help='Fusion method for CLIP features (sum, concat, or None)')
    parser.add_argument("--bg-ref-class", type=str, default="first", choices=["first", "average"], 
                        help="Background reference class selection for fusion mode")
    
    # Checkpoint resuming
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--resume_with_eval', type=int, default=0,  # action='store_true',
                        help='Resume from first checkpoint, evaluate, and continue training')
    
    # Model related arguments
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                        help='CLIP model type')
    parser.add_argument('--pretrained', type=str, default=None, # 'laion400m_e32', 'webli'(for siglip)
                        help='Path to pretrained model weights (optional)')
    parser.add_argument('--n_min_p', type=int, default=5,
                        help='Minimum number of samples for precision matrix calculation')
    parser.add_argument('--lambda_reg', type=float, default=1e-4,
                        help='Precision matrix regularization coefficient')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Dynamic weight scale')
    # parser.add_argument('--k_maha', type=float, default=0.5,
    #                     help='Mahalanobis distance scale')
    # parser.add_argument('--zeroshot_ratio', type=float, default=0.1,
    #                     help='Ratio of text')
    
    # RMD specific arguments
    parser.add_argument('--lambda_bg', type=float, default=1e-4,
                        help='Background precision matrix regularization coefficient (RMD only)')
    # parser.add_argument('--scale_rmd', type=float, default=1.0,
                        # help='Scaling factor for RMD (RMD only)')
    # parser.add_argument('--offset_rmd', type=float, default=0.0,
                        # help='Offset for RMD before sigmoid (RMD only)')
    
    # Execution related arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for saving results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--seed_val', type=int, default=None,
                        help='Seed value for selecting crossimbal config (optional, for tracking purposes)')
    parser.add_argument('--save_emb', action='store_true', default=False,
                        help='Save extracted embeddings')
                        
    # Wandb related arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for tracking experiments')
    parser.add_argument('--wandb_project', type=str, default='HyCal',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity name (optional username or team name)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name (optional)')
                            
    return parser.parse_args()


def _convert_str_none(obj):
    """Recursively convert string 'None'/'null' to Python None in configs."""
    if isinstance(obj, dict):
        return {k: _convert_str_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_str_none(v) for v in obj]
    elif isinstance(obj, str):
        if obj.strip().lower() in {"none", "null", "nan", ""}:
            return None
        return obj
    else:
        return obj
    

def load_configs(args, logger=None):
    """Load dataset and few-shot configuration files.

    Args:
        args: Command line arguments
        logger: Logger instance (optional)

    Returns:
        Tuple of (cfg, few_shot_cfg)
    """
    from easydict import EasyDict
    import yaml

    # Load dataset config
    with open(args.cfg, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    # Load few-shot config if provided
    few_shot_cfg = None
    if args.few_shot_cfg and os.path.exists(args.few_shot_cfg):
        with open(args.few_shot_cfg, 'r') as f:
                few_shot_cfg = EasyDict(yaml.safe_load(f))
    
    if logger:
        logger.info(f'Successfully loaded config from {args.cfg}')
        if few_shot_cfg:
            logger.info(f'Successfully loaded few-shot config from {args.few_shot_cfg}')
    
    # convert string 'None' → Python None
    cfg = EasyDict(_convert_str_none(cfg))
    if few_shot_cfg:
        few_shot_cfg = EasyDict(_convert_str_none(few_shot_cfg))

    if args.data_root is not None:
        cfg.data_root = args.data_root

    return cfg, few_shot_cfg
    

def setup_output_dir(args, cfg, timestamp):
    """Setup output directory and integrate command line arguments into config.

    Args:
        args: Command line arguments
        cfg: Configuration object
        timestamp: Timestamp for output directory naming

    Returns:
        Tuple of (args, cfg) with updated output directory
    """
    # Apply command line args to config
    if args.num_shots is not None:
        cfg.num_shots = args.num_shots  

    if args.num_shots_each_range is not None:
        cfg.num_shots_each_range = args.num_shots_each_range
    else:
        cfg.num_shots_each_range = None
    assert not (cfg.num_shots_each_range is not None and cfg.num_shots is not None), \
        "'num_shots_each_range' and 'num_shots' cannot be specified at the same time."

    # 체크포인트가 존재하는 경우 해당 위치를 output_dir로 설정
    if args.checkpoint is not None:
        if '.pkl' in args.checkpoint:
            args.output_dir = os.path.dirname(os.path.dirname(args.checkpoint))
            print(f"Using checkpoint directory as output directory: {args.output_dir}")
        else:
            args.output_dir = args.checkpoint
            print(f"Using checkpoint directory as output directory: {args.output_dir}")
        
        if args.resume_with_eval == 1:
            args.output_dir += f"_eval_{timestamp}"
            print(f"Resuming with evaluation, new output directory: {args.output_dir}")

    else:
        # 체크포인트가 없는 경우 일반적인 경로 설정

        if 'any' not in args.cfg:
            if '_highimbal' in args.few_shot_cfg:
                args.output_dir = os.path.join(args.output_dir, f"XD_v2")
            elif '_bal' in args.few_shot_cfg:
                args.output_dir = os.path.join(args.output_dir, f"XD_v1")
            elif '_crossimbal' in args.few_shot_cfg:
                args.output_dir = os.path.join(args.output_dir, f"XD_v3")
            elif cfg.num_shots is not None:
                args.output_dir = os.path.join(args.output_dir, f"fs_{cfg.num_shots}")
        else:
            args.output_dir = os.path.join(args.output_dir, f"any")
            if '_bal' in args.few_shot_cfg:
                args.output_dir += "_v1"
            elif '_imbal' in args.few_shot_cfg:
                args.output_dir += "_v2"
            elif cfg.num_shots is not None:
                args.output_dir += f"fs_{cfg.num_shots}"
            
        # 새로운 실험을 위한 timestamp 생성
        args.output_dir = os.path.join(args.output_dir, f"{args.eval_setting}_{len(cfg.datasets)}_{args.fusion}_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    return args, cfg


def setup_logging(log_file='experiment.log'):
    """Initialize logging configuration with console and file handlers.

    Args:
        log_file: Path to log file

    Returns:
        Logger instance
    """
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    while root_logger.handlers:
        handler = root_logger.handlers[0]
        handler.close()
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger("HyCal_Experiment")
    
    # logger 핸들러도 제거
    while logger.handlers:
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)
        
    return logger