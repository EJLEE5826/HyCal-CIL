"""
Experiment Core - Training and Evaluation Logic

This module contains the core experiment logic for HyCal:
- Training loop for sequential tasks
- Evaluation loop for all learned tasks
- Zero-shot evaluation
- Results collection and saving
- Metrics calculation
"""

import os
import pandas as pd
import json
import wandb
import torch
import logging

from .hycal_model import HyCalModel
from .hycal_evaluator import HyCalEvaluator
from .dataloader_utils import get_data_stats, save_data_stats
from task_datasets.dataset import CustomDataloader
from .plotting import generate_plots
from .metrics import calculate_CDE_score, calculate_forgetting_scores, calculate_transfer_accuracy, calculate_average_accuracy

logger = logging.getLogger(__name__)


def run_experiment(args, device, cfg, few_shot_cfg=None, checkpoint_path=None):
    """
    Run FS-CDCCL experiment
    
    Args:
        args: Command line arguments
        device: Device to use (cuda/cpu)
        cfg: Configuration object
        few_shot_cfg: Few-shot configuration (optional)
        checkpoint_path: Path to checkpoint for resuming (optional)
        
    Returns:
        Dictionary containing all experiment results
    """
    logger.info(f"Starting experiment: {len(cfg.datasets)} tasks")
    
    # Initialize model and evaluator
    model = HyCalModel(
        clip_model_name=args.clip_model,
        model_provider=args.model_provider,
        pretrained=args.pretrained,
        device=device,
        n_min_p=args.n_min_p,
        lambda_reg=args.lambda_reg,
        lambda_bg=args.lambda_bg,
        scale=args.scale,
        fusion=args.fusion,
        bg_ref_class=args.bg_ref_class,
        sim_metric=args.sim_metric,
        real=args.real,
        gamma=args.gamma,
    )
    evaluator = HyCalEvaluator(model)

    # Dictionary for storing results
    results = {
        'task_accuracies': [],
        'class_accuracies': [],
        'zero_shot_task_accuracies': [],
        'zero_shot_class_accuracies': [],
        'task_order': list(cfg.datasets),
        'learning_accuracies': [],
        'forgetting_scores': []
    }

    # Handle checkpoint loading
    start_task = _load_checkpoint_if_needed(args, model, checkpoint_path, results)

    # Setup transforms
    # _setup_transforms(args, cfg, model)
    cfg.train_transform = model.val_preprocess
    cfg.val_transform = model.val_preprocess


    # Create dataloaders
    logger.info("Creating all dataloaders at once...")
    dataloaders, total_classes, dataset_shots = CustomDataloader.create_dataloaders(
        cfg=cfg,
        few_shot_cfg=few_shot_cfg,
    )
    
    if dataset_shots:
        results['dataset_shots'] = dataset_shots
        _log_dataset_shots(dataset_shots)
    
    # Save dataset statistics if starting from beginning
    if start_task == 0:
        dataset_stats = get_data_stats(dataloaders)
        save_data_stats(dataset_stats, os.path.join(args.output_dir, "dataset_stats.json"))
    
    # Process tasks sequentially
    task_names = cfg.datasets
    
    for task_idx, task_name in enumerate(task_names[start_task:], start=start_task):
        logger.info(f"\n===== Task {task_idx+1}/{len(task_names)}: {task_name} =====")
        
        # Handle resume with evaluation mode
        if args.resume_with_eval == 1:
            _load_task_checkpoint(args, model, checkpoint_path, task_idx)
        
        # Train current task
        if args.resume_with_eval == 0:
            _train_task(args, model, dataloaders, task_name, task_idx)
        
        # Evaluate on all learned tasks
        task_accuracies, class_accuracies = _evaluate_learned_tasks(
            args, evaluator, dataloaders, task_names, total_classes, 
            task_idx, args.output_dir
        )
        
        # Calculate and log average accuracy
        avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
        logger.info(f"Average accuracy after task {task_idx+1}: {avg_acc:.4f}")
        
        # Save results
        results['task_accuracies'].append(task_accuracies)
        results['class_accuracies'].append(class_accuracies)
        
        # Log to wandb
        if args.use_wandb:
            _log_task_results_to_wandb(results, task_idx, avg_acc)

        # Zero-shot evaluation (only after last task)
        if task_idx == len(task_names) - 1:
            zero_shot_task_accuracies, zero_shot_class_accuracies = _evaluate_zeroshot(
                args, evaluator, dataloaders, task_names, total_classes
            )
            results['zero_shot_task_accuracies'].append(zero_shot_task_accuracies)
            results['zero_shot_class_accuracies'].append(zero_shot_class_accuracies)

        # Save intermediate results
        _save_results(args.output_dir, results)
    
    # Post-experiment processing
    _finalize_experiment(args, results, task_names)
    
    return results


def _load_checkpoint_if_needed(args, model, checkpoint_path, results):
    """Load checkpoint if specified and return start_task index"""
    start_task = 0
    
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            if 'models' not in checkpoint_path:
                checkpoint_path = os.path.join(checkpoint_path, "models")
            logger.info(f"Loading checkpoint from {checkpoint_path}")

            if args.resume_with_eval == 0:
                if '.pkl' not in checkpoint_path:
                    all_checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pkl')]
                    if not all_checkpoints:
                        raise FileNotFoundError(f"No valid checkpoints found in directory: {checkpoint_path}")
                    all_checkpoints.sort(key=lambda x: int(x.split('task_')[-1].split('.')[0]))
                    checkpoint_path = os.path.join(checkpoint_path, all_checkpoints[-1])
                
                model.load(checkpoint_path)
                start_task = int(checkpoint_path.split('task_')[-1].split('.')[0])
                
                # Load previous results
                results_path = os.path.join(args.output_dir, "results.json")
                if os.path.exists(results_path):
                    saved_results = json.load(open(results_path, 'r'))
                    results.update(saved_results)
                
                logger.info(f"Resuming from task {start_task}, starting from task {start_task+1}")
            else:
                start_task = 0
                logger.info(f"Resuming with evaluation from checkpoint, starting from task {start_task+1}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return start_task


# def _setup_transforms(args, cfg, model):
    # """Setup data transforms based on configuration"""
    # if args.wandb_run_name is not None and 'testtrf' in args.wandb_run_name:
    # cfg.train_transform = model.val_preprocess
    # cfg.val_transform = model.val_preprocess
    # else:
    #     cfg.train_transform = model.train_preprocess
    #     cfg.val_transform = model.val_preprocess


def _log_dataset_shots(dataset_shots):
    """Log random shots per dataset"""
    logger.info("===== Random Shots per Dataset =====")
    max_dataset_len = max(len(ds) for ds in dataset_shots.keys())
    for dataset_name, shots in dataset_shots.items():
        logger.info(f"{dataset_name.ljust(max_dataset_len)}: {shots} shots")


def _load_task_checkpoint(args, model, checkpoint_path, task_idx):
    """Load checkpoint for specific task in resume_with_eval mode"""
    if '.pkl' in checkpoint_path:
        checkpoint_path = checkpoint_path[:-5] + f'{task_idx+1}.pkl'
    else:
        checkpoint_path = os.path.join(checkpoint_path, f'after_task_{task_idx+1}.pkl')
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading model from {checkpoint_path}")
        model.load(checkpoint_path)
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}. Progress training.")
        args.resume_with_eval = 0


def _train_task(args, model, dataloaders, task_name, task_idx):
    """Train on current task and save checkpoint"""
    train_loader = dataloaders[task_name]['train']
    model.train_task(train_loader)
    
    # Save model checkpoint
    model_path = os.path.join(args.output_dir, "models", f"after_task_{task_idx+1}.pkl")
    model.save(model_path)

def _evaluate_learned_tasks(args, evaluator, dataloaders, task_names, total_classes, 
                            current_task_idx, output_dir):
    """
    Evaluate on all learned tasks up to current task
    
    Returns:
        task_accuracies: Dictionary of task-wise accuracies
        class_accuracies: Dictionary of class-wise accuracies
    """
    task_accuracies = {}
    class_accuracies = {}
    tested_cls_num = 0
    
    for eval_idx, eval_task_name in enumerate(task_names[:current_task_idx+1]):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Evaluating task {eval_task_name}...")
        val_loader = dataloaders[eval_task_name]['test']
        
        # Get class names for current evaluation task
        if eval_task_name == 'mnist':
            class_to_idx = dataloaders[eval_task_name]['test'].dataset.class_to_idx
            eval_classnames = ['number: "{}"'.format(label) for label in class_to_idx.keys()]
        else:
            eval_classnames = dataloaders[eval_task_name]['test'].dataset.classnames
        
        class_range_min = tested_cls_num
        class_range_max = tested_cls_num + len(eval_classnames)
        
        # Set target classes based on evaluation setting
        if args.eval_setting == 'cil':
            target_classes = None
        elif args.eval_setting == 'mtil':
            target_classes = eval_classnames
        else:  # X-TAIL
            target_classes = total_classes
        
        # Save embeddings only on last task
        save_embed = (current_task_idx == len(task_names) - 1) and args.save_emb
        
        accuracy, cls_accs = evaluator.evaluate(
            dataloader=val_loader,
            target_classes=target_classes,
            task_name=eval_task_name,
            save_embed=save_embed,
            path=output_dir if save_embed else None,
        )

        task_accuracies[eval_task_name] = accuracy
        class_accuracies[eval_task_name] = cls_accs
        
        logger.info(f"Accuracy for task {eval_task_name}: {accuracy:.4f}")
        tested_cls_num += len(eval_classnames)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                f"task_acc/{eval_task_name}": accuracy,
                "current_task": current_task_idx + 1,
                "eval_task": eval_idx + 1
            })
    
    return task_accuracies, class_accuracies


def _evaluate_zeroshot(args, evaluator, dataloaders, task_names, total_classes):
    """
    Perform zero-shot evaluation on all tasks

    Returns:
        zero_shot_task_accuracies: Dictionary of zero-shot task accuracies
        zero_shot_class_accuracies: Dictionary of zero-shot class accuracies
    """
    zero_shot_task_accuracies = {}
    zero_shot_class_accuracies = {}
    zero_tested_cls_num = 0
    
    for eval_idx, eval_task_name in enumerate(task_names):
        logger.info(f"Zero-shot evaluation for task {eval_task_name}...")
        val_loader = dataloaders[eval_task_name]['test']
        
        # Get class names
        if eval_task_name == 'mnist':
            class_to_idx = dataloaders[eval_task_name]['test'].dataset.class_to_idx
            eval_classnames = ['number: "{}"'.format(label) for label in class_to_idx.keys()]
        else:
            eval_classnames = dataloaders[eval_task_name]['test'].dataset.classnames
        
        class_range_min = zero_tested_cls_num
        class_range_max = zero_tested_cls_num + len(eval_classnames)
        
        # Set target classes based on evaluation setting
        if args.eval_setting == 'cil':
            target_classes = total_classes
        elif args.eval_setting == 'mtil':
            target_classes = total_classes[class_range_min:class_range_max]
        else:  # X-TAIL
            target_classes = total_classes
        
        accuracy, cls_accs = evaluator.evaluate_zeroshot(val_loader, target_classes)
        zero_shot_task_accuracies[eval_task_name] = accuracy
        zero_shot_class_accuracies[eval_task_name] = cls_accs

        logger.info(f"Zero-shot accuracy for task {eval_task_name}: {accuracy:.4f}")
        zero_tested_cls_num += len(eval_classnames)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                f"zero_shot_acc/{eval_task_name}": accuracy,
                "eval_task": eval_idx + 1
            })

    # Log zero-shot results table to wandb
    if args.use_wandb:
        zero_shot_table = wandb.Table(
            columns=["task_name", "accuracy"],
            data=[[name, acc] for name, acc in zero_shot_task_accuracies.items()]
        )
        wandb.log({"zero_shot_results": zero_shot_table})
    
    return zero_shot_task_accuracies, zero_shot_class_accuracies


def _log_task_results_to_wandb(results, task_idx, avg_acc):
    """Log task results to wandb"""
    wandb.log({
        "task_avg_acc": avg_acc,
        "current_task": task_idx + 1,
    })
    
    # Calculate overall average accuracy up to current task
    avg_accuracies = []
    for i in range(task_idx + 1):
        if i < len(results['task_accuracies']):
            task_avg = sum(results['task_accuracies'][i].values()) / len(results['task_accuracies'][i])
            avg_accuracies.append(task_avg)
    
    if avg_accuracies:
        current_overall_avg_acc = sum(avg_accuracies) / len(avg_accuracies)
        wandb.log({
            "overall_avg_acc": current_overall_avg_acc,
            "current_task": task_idx + 1,
        })


def _save_results(output_dir, results):
    """Save intermediate results to JSON"""
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)


def _finalize_experiment(args, results, task_names):
    """
    Finalize experiment by calculating metrics and saving all results
    """
    # Calculate forgetting scores
    forgetting_scores, avg_forgetting = calculate_forgetting_scores(results)
    results['forgetting_scores'] = forgetting_scores
    results['average_forgetting'] = avg_forgetting

    # Calculate transfer accuracy (X-TAIL benchmark metric)
    transfer_per_domain, avg_transfer = calculate_transfer_accuracy(results)
    results['transfer_accuracy'] = transfer_per_domain
    results['average_transfer_accuracy'] = avg_transfer

    # Calculate average accuracy (X-TAIL style)
    task_avg_accuracies, overall_avg_acc = calculate_average_accuracy(results)
    results['task_avg_accuracies'] = task_avg_accuracies
    results['overall_avg_accuracy'] = overall_avg_acc

    # Save final results
    _save_results(args.output_dir, results)
    
    # Log forgetting scores to wandb
    if args.use_wandb:
        forgetting_logs = {"average_forgetting": avg_forgetting}
        for task_name, score in forgetting_scores.items():
            forgetting_logs[f"forgetting_score/{task_name}"] = score
        wandb.log(forgetting_logs)
    
    # Create and save accuracy dataframe
    accuracy_df = _create_accuracy_dataframe(results)
    accuracy_df.to_csv(os.path.join(args.output_dir, "task_accuracies.csv"))
    
    # Save edited version with zero-shot row at top
    if 'zero-shot' in accuracy_df.index:
        accuracy_df_edit = _reorder_accuracy_dataframe(accuracy_df, args.eval_setting)
        accuracy_df_edit.to_csv(os.path.join(args.output_dir, "task_accuracies_edit.csv"))
    
    # Save forgetting scores
    _save_forgetting_scores(results, args.output_dir)
    
    # Calculate CDE metric
    CDE_result = calculate_CDE_score(args.output_dir)
    logger.info(f"CDE result: {CDE_result}")

    # Log final summary (X-TAIL benchmark format)
    _log_final_summary(results, CDE_result)

    # Generate plots
    generate_plots(results, args.output_dir)
    
    # Log final results to wandb
    if args.use_wandb:
        _log_final_results_to_wandb(args, results, task_names, accuracy_df, CDE_result)


def _create_accuracy_dataframe(results):
    """Create accuracy dataframe from results"""
    accuracy_df = pd.DataFrame()
    
    # Add task-wise accuracies
    for task_idx, task_acc_dict in enumerate(results['task_accuracies']):
        for task_name, acc in task_acc_dict.items():
            accuracy_df.loc[f'task {task_idx+1}', task_name] = acc
    
    # Add zero-shot accuracies
    if results['zero_shot_task_accuracies']:
        zero_shot_acc_dict = results['zero_shot_task_accuracies'][-1]
        for task_name, acc in zero_shot_acc_dict.items():
            accuracy_df.loc['zero-shot', task_name] = acc
    
    return accuracy_df


def _reorder_accuracy_dataframe(accuracy_df, eval_setting):
    """Reorder accuracy dataframe with zero-shot row at top"""
    zero_shot_row = accuracy_df.loc['zero-shot']
    if eval_setting != 'cil':
        accuracy_df = accuracy_df.fillna(zero_shot_row)
    accuracy_df = pd.concat([accuracy_df.loc[['zero-shot']], accuracy_df.drop('zero-shot')])
    return accuracy_df
 

def _save_forgetting_scores(results, output_dir):
    """Save forgetting scores to CSV"""
    if 'forgetting_scores' in results and results['forgetting_scores']:
        fs_df = pd.DataFrame()
        for task_name, score in results['forgetting_scores'].items():
            fs_df.loc[task_name, 'Forgetting_Score'] = score
        fs_df.loc['Average', 'Forgetting_Score'] = results.get('average_forgetting', 0.0)
        fs_df.to_csv(os.path.join(output_dir, "forgetting_scores.csv"))

def _log_final_summary(results, CDE_result):
    """Log final summary in X-TAIL benchmark format"""
    # Final average accuracy (Last Accuracy)
    final_task_accs = results['task_accuracies'][-1]
    last_acc = sum(final_task_accs.values()) / len(final_task_accs) if final_task_accs else 0.0

    # X-TAIL style metrics
    avg_acc = results.get('overall_avg_accuracy', 0.0)
    transfer_acc = results.get('average_transfer_accuracy')
    avg_forgetting = results.get('average_forgetting', 0.0)
    cde_score = CDE_result.get('S_CDE', 0.0) if CDE_result else 0.0

    logger.info("=" * 50)
    logger.info("           Final Results (X-TAIL Benchmark)")
    logger.info("=" * 50)
    logger.info(f"Last Accuracy:     {last_acc:.2f}%")
    logger.info(f"Average Accuracy:  {avg_acc:.2f}%")
    if transfer_acc is None:
        logger.info("Transfer Accuracy: N/A")
    else:
        logger.info(f"Transfer Accuracy: {transfer_acc:.2f}%")
    logger.info(f"CDE Score:         {cde_score:.4f}")
    logger.info(f"Avg Forgetting:    {avg_forgetting:.2f}%")
    logger.info("=" * 50)

    # Per-domain summary
    task_order = results.get('task_order', [])
    transfer_per_domain = results.get('transfer_accuracy') or {}
    forgetting_scores = results.get('forgetting_scores', {})

    if task_order:
        logger.info("\nPer-domain metrics:")
        for domain_name in task_order:
            last_domain_acc = final_task_accs.get(domain_name, 0.0)
            transfer_domain = transfer_per_domain.get(domain_name)
            forgetting_domain = forgetting_scores.get(domain_name, 0.0)
            transfer_text = "N/A" if transfer_domain is None else f"{transfer_domain:.1f}%"
            logger.info(f"  {domain_name}: Last={last_domain_acc:.1f}%, "
                       f"Transfer={transfer_text}, "
                       f"Forgetting={forgetting_domain:.1f}%")


def _log_final_results_to_wandb(args, results, task_names, accuracy_df, CDE_result):
    """Log all final results to wandb"""
    # Final average accuracy (Last Accuracy)
    final_task_accs = results['task_accuracies'][-1]
    final_avg_acc = sum(final_task_accs.values()) / len(final_task_accs)

    # X-TAIL style average accuracy
    overall_avg_acc = results.get('overall_avg_accuracy', 0.0)

    # Transfer accuracy
    avg_transfer = results.get('average_transfer_accuracy')

    # Overall accuracy per task (column-wise, excluding zero-shot)
    overall_acc_logs = {}
    for task_name in accuracy_df.columns:
        task_column = accuracy_df[task_name]
        if 'zero-shot' in accuracy_df.index:
            task_column = task_column.drop('zero-shot')
        task_column_clean = task_column.dropna()
        if len(task_column_clean) > 0:
            overall_acc = task_column_clean.mean()
            overall_acc_logs[f"overall_acc/{task_name}"] = overall_acc

    # Transfer accuracy per domain
    transfer_acc_logs = {}
    transfer_per_domain = results.get('transfer_accuracy') or {}
    for domain_name, transfer_acc in transfer_per_domain.items():
        if transfer_acc is not None:
            transfer_acc_logs[f"transfer_acc/{domain_name}"] = transfer_acc

    # Log final metrics
    final_metrics = {
        "final_avg_accuracy": final_avg_acc,          # Last Accuracy
        "overall_avg_accuracy": overall_avg_acc,      # Average Accuracy (X-TAIL style)
        "completed_tasks": len(task_names),
        "dataset_shots": results.get('dataset_shots', {}),
        **overall_acc_logs,
        **transfer_acc_logs
    }
    if avg_transfer is not None:
        final_metrics["average_transfer_accuracy"] = avg_transfer
    wandb.log(final_metrics)

    wandb.log({"CDE_metrics": CDE_result})
    
    # Log accuracy table
    task_acc_table = wandb.Table(
        dataframe=pd.read_csv(os.path.join(args.output_dir, "task_accuracies_edit.csv"))
    )
    wandb.log({"final_task_accuracies": task_acc_table})
    
    # Log dataset shots table
    if 'dataset_shots' in results:
        shots_data = [[dataset, shots] for dataset, shots in results['dataset_shots'].items()]
        shots_table = wandb.Table(columns=["Dataset", "Num Shots"], data=shots_data)
        wandb.log({"dataset_shots_table": shots_table})
    
    # Log forgetting scores table
    if 'forgetting_scores' in results and results['forgetting_scores']:
        fs_table = wandb.Table(
            columns=["Task", "Forgetting_Score"],
            data=[
                [task_name, score] 
                for task_name, score in results['forgetting_scores'].items()
            ] + [["Average", results.get("average_forgetting", 0.0)]]
        )
        wandb.log({"forgetting_scores_table": fs_table, "completed": True})
    
    # Upload result files
    wandb.save(os.path.join(args.output_dir, "results.json"))
    wandb.save(os.path.join(args.output_dir, "task_accuracies.csv"))
    if os.path.exists(os.path.join(args.output_dir, "task_accuracies_edit.csv")):
        wandb.save(os.path.join(args.output_dir, "task_accuracies_edit.csv"))
    if os.path.exists(os.path.join(args.output_dir, "CDE_metrics.json")):
        wandb.save(os.path.join(args.output_dir, "CDE_metrics.json"))

    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
