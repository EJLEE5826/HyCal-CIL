import os
import json
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("HyCal_Experiment")


def calculate_CDE_score(results_root_path, file_name=None, stats_name=None, output_filename=None):
    """Calculate and save S_CDE score.

    Args:
        results_root_path: Path to results directory
        file_name: CSV file name (default: 'task_accuracies_edit.csv')
        stats_name: Stats JSON file name (default: 'dataset_stats.json')
        output_filename: Output JSON file name (default: 'CDE_metrics.json')

    Returns:
        Dictionary with CDE scores or None if failed
    """
    if file_name is not None:
        file_name = file_name if file_name.endswith('.csv') else file_name + '.csv'
    else:
        file_name = 'task_accuracies_edit.csv'
    csv_file = os.path.join(results_root_path, file_name)
    stats_name = stats_name if stats_name is not None else 'dataset_stats.json'
    stats_json_path = os.path.join(results_root_path, stats_name)
    output_filename = output_filename if output_filename is not None else "CDE_metrics.json"

    try:
        scores_CDEs, Kt_values_used = calculate_CDE(csv_file, stats_json_path)
        result = save_results_to_json(csv_file, scores_CDEs, Kt_values_used, output_filename)

        logger.info(f"[OK] Saved: {os.path.join(results_root_path, 'CDE_metrics.json')}")
        return result

    except Exception as e:
        logger.info(f"[FAIL] {csv_file}: {str(e)}")
        return None

def calculate_CDE(csv_filepath, stats_json_path):
    """Calculate CDE score from accuracy CSV and dataset statistics.

    Args:
        csv_filepath: Path to task accuracies CSV
        stats_json_path: Path to dataset statistics JSON

    Returns:
        Tuple of ((S_cde, S_adapt, S_last), Kt_values)
    """

    # Load data
    df_accuracies = pd.read_csv(csv_filepath, index_col=0)
    task_names = df_accuracies.columns.tolist()
    T = len(task_names)
    
    with open(stats_json_path, 'r') as f:
        dataset_stats = json.load(f)

    # K_t: number of training samples per domain
    Kt_values = np.array(
        [dataset_stats[task]["train"]["total_samples"] for task in task_names],
        dtype=float,
    )

    # Zero-shot accuracy per domain
    Z_t = [float(df_accuracies.loc['zero-shot', task]) for task in task_names]

    # Accuracy on domain t right after training step t
    A_t_t = [float(df_accuracies.loc[f'task {i+1}', task]) for i, task in enumerate(task_names)]

    # Accuracy on domain t at final step T
    A_T_t = [float(df_accuracies.loc[f'task {T}', task]) for task in task_names]

    # Replace NaNs with 0.0
    Z_t = np.nan_to_num(np.array(Z_t, dtype=float), nan=0.0)
    A_t_t = np.nan_to_num(np.array(A_t_t, dtype=float), nan=0.0)
    A_T_t = np.nan_to_num(np.array(A_T_t, dtype=float), nan=0.0)

    # Calculate normalized weight
    s_t = Kt_values**(-0.5)
    weight_sum = s_t.sum()
    w_t = s_t / weight_sum

    # Calculate Scores
    S_adapt = np.sum(w_t * (Z_t + A_t_t) / 2.0)
    S_last = np.sum(w_t * A_T_t)

    denominator = S_adapt + S_last
    S_cde = 0.0 if denominator == 0 else (2 * S_adapt * S_last) / denominator

    total_scores = (S_cde, S_adapt, S_last)

    return total_scores, Kt_values


def save_results_to_json(csv_filepath, scores_CDEs, Kt_values_used, output_filename="CDE_metrics.json"):
    """Save CDE metrics to JSON file.

    Args:
        csv_filepath: Path to source CSV
        scores_CDEs: Tuple of (S_cde, S_adapt, S_last)
        Kt_values_used: Array of training sample counts
        output_filename: Output JSON file name

    Returns:
        Dictionary with CDE results
    """
    S_cde, S_adapt, S_last = scores_CDEs
    output_dir = os.path.dirname(csv_filepath)
    output_filename = os.path.join(output_dir, output_filename)

    try:
        temp_df = pd.read_csv(csv_filepath, index_col=0, nrows=0)
        task_names = temp_df.columns.tolist()
    except:
        task_names = [f"task_{i+1}" for i in range(len(Kt_values_used))]

    results_data = {
        'source_csv_filename': os.path.basename(csv_filepath),
        'S_CDE': round(S_cde, 6),
        'S_adapt': round(S_adapt, 6),
        'S_last': round(S_last, 6),
        'Kt_values_used': {
            task_names[i] if i < len(task_names) else f"task_{i+1}": Kt_values_used[i]
            for i in range(len(Kt_values_used))
        }
    }

    with open(output_filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    return results_data



def calculate_forgetting_scores(results):
    """Calculate catastrophic forgetting scores for each task.

    Args:
        results: Dictionary with experiment results including task_accuracies and task_order

    Returns:
        Tuple of (forgetting_scores, avg_forgetting) where scores are per task
    """
    task_count = len(results['task_accuracies'])
    if task_count <= 1:
        return {}, 0.0  # No forgetting if only one task
    
    forgetting_scores = {}

    # Accuracies at final step T
    final_accuracies = results['task_accuracies'][task_count - 1]

    # Task order
    task_order = results['task_order']

    forgetting_sum = 0.0
    count = 0
    logger.info(f"===== Forgetting Scores =====")

    for task_idx, task_name in enumerate(task_order[:-1]):
        task_first_learned_idx = task_idx

        try:
            initial_acc = results['task_accuracies'][task_first_learned_idx][task_name]
            final_acc = final_accuracies[task_name]

            # Forgetting = initial accuracy - final accuracy
            forgetting = initial_acc - final_acc
            forgetting_scores[task_name] = forgetting

            forgetting_sum += forgetting
            count += 1

            logger.info(f"Task {task_name}: Init Acc {initial_acc:.6f}, final Acc {final_acc:.6f}, Forgetting {forgetting:.6f}")
        except (KeyError, IndexError) as e:
            logger.warning(f"Unable to calculate forgetting for {task_name}: {e}")

    # Average forgetting score
    avg_forgetting = forgetting_sum / count if count > 0 else 0.0
    logger.info(f"Average Forgetting: {avg_forgetting:.6f}")

    return forgetting_scores, avg_forgetting


def calculate_transfer_accuracy(results):
    """Calculate forward transfer accuracy (upper triangle excluding diagonal).

    Transfer accuracy measures performance on unseen domains before they are learned.
    This is computed from the upper triangle of the accuracy matrix (excluding diagonal).

    Args:
        results: Dictionary with experiment results including task_accuracies and task_order

    Returns:
        Tuple of (transfer_per_domain, avg_transfer) where:
            - transfer_per_domain: Dict mapping domain name to its transfer accuracy
            - avg_transfer: Overall average transfer accuracy, or None when no
              upper-triangle observations are available
    """
    task_count = len(results['task_accuracies'])
    task_order = results['task_order']
    domain_count = len(task_order)

    # Build accuracy matrix [task_id, domain_id]
    acc_matrix = np.ma.masked_all((task_count, domain_count), dtype=float)
    observed_count = 0
    for task_idx, task_acc_dict in enumerate(results['task_accuracies']):
        for domain_idx in range(task_idx + 1, domain_count):
            domain_name = task_order[domain_idx]
            if domain_name not in task_acc_dict:
                continue

            value = task_acc_dict[domain_name]
            if (isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))
                    and np.isfinite(value)):
                acc_matrix[task_idx, domain_idx] = float(value)
                observed_count += 1

    possible_count = sum(
        max(domain_count - task_idx - 1, 0)
        for task_idx in range(task_count)
    )
    if 0 < observed_count < possible_count:
        logger.warning(
            "Transfer accuracy coverage is partial: %d/%d upper-triangle "
            "cells observed; missing or non-finite values are excluded.",
            observed_count,
            possible_count,
        )

    logger.info(f"===== Transfer Accuracy =====")
    if observed_count == 0:
        logger.info("Average Transfer Accuracy: N/A")
        return {}, None

    # Upper triangle (excluding diagonal) = forward transfer
    # These are accuracies on domains BEFORE they are learned
    # Keep RAIL's original zero-excluding aggregation for observed values.
    masked = np.ma.masked_equal(acc_matrix, 0)

    # Per-domain transfer accuracy (column-wise mean of upper triangle)
    transfer_per_domain = {}
    for domain_idx, domain_name in enumerate(task_order):
        col_values = masked[:, domain_idx].compressed()
        if len(col_values) > 0:
            transfer_per_domain[domain_name] = float(np.mean(col_values))

    # Overall transfer accuracy - two-step average matching primal_RAIL.py
    # First compute column-wise mean, then average those (weights each domain equally)
    transfer_per_col = np.ma.mean(masked, axis=0)  # Column-wise mean first
    avg_transfer = float(np.mean(transfer_per_col.compressed())) if transfer_per_col.count() > 0 else 0.0

    for domain_name, transfer_acc in transfer_per_domain.items():
        logger.info(f"{domain_name}: {transfer_acc:.6f}")
    logger.info(f"Average Transfer Accuracy: {avg_transfer:.6f}")

    return transfer_per_domain, avg_transfer


def calculate_average_accuracy(results):
    """Calculate X-TAIL style average accuracy.

    For each task t, compute mean accuracy over domains 0..t (learned domains),
    then average across all tasks.

    Args:
        results: Dictionary with experiment results including task_accuracies and task_order

    Returns:
        Tuple of (task_avg_accuracies, overall_avg_acc) where:
            - task_avg_accuracies: List of average accuracies per task
            - overall_avg_acc: Overall average accuracy (mean of task averages)
    """
    task_count = len(results['task_accuracies'])
    task_order = results['task_order']

    if task_count == 0:
        return [], 0.0

    task_avg_accuracies = []

    logger.info(f"===== Average Accuracy (X-TAIL style) =====")

    for task_idx in range(task_count):
        task_acc_dict = results['task_accuracies'][task_idx]
        # Get accuracies for domains 0..task_idx (learned domains up to this point)
        accs = [task_acc_dict.get(task_order[i], 0.0) for i in range(task_idx + 1)]
        if accs:
            task_avg = sum(accs) / len(accs)
            task_avg_accuracies.append(task_avg)
            logger.info(f"Task {task_idx + 1}: Avg over {task_idx + 1} domains = {task_avg:.6f}")

    overall_avg_acc = sum(task_avg_accuracies) / len(task_avg_accuracies) if task_avg_accuracies else 0.0
    logger.info(f"Overall Average Accuracy: {overall_avg_acc:.6f}")

    return task_avg_accuracies, overall_avg_acc
