import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_plots(results, output_dir):
    """Visualize experiment results"""
    # 1. Task-wise accuracy matrix (Confusion Matrix style)
    plot_accuracy_matrix(results, output_dir)

    # 2. Performance changes over time (Forgetting analysis)
    plot_forgetting_curve(results, output_dir)

    # 3. Class-wise accuracy distribution
    plot_class_accuracy_distribution(results, output_dir)


def plot_accuracy_matrix(results, output_dir):
    """Visualize task-wise accuracy matrix"""
    task_names = results['task_order']
    num_tasks = len(task_names)

    # Prepare accuracy matrix
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    for train_idx in range(num_tasks):
        for test_idx in range(train_idx + 1):
            test_task = task_names[test_idx]
            accuracy_matrix[train_idx, test_idx] = results['task_accuracies'][train_idx][test_task]

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=task_names, yticklabels=[f"After Task {i+1}" for i in range(num_tasks)])

    plt.xlabel("Evaluated On")
    plt.ylabel("Trained After")
    plt.title("Task-wise Accuracy Matrix")
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(output_dir, "plots", "accuracy_matrix.png"), dpi=300)
    plt.close()


def plot_forgetting_curve(results, output_dir):
    """Visualize performance changes over time (Forgetting analysis)"""
    task_names = results['task_order']
    num_tasks = len(task_names)

    # Track accuracy changes for each task
    plt.figure(figsize=(12, 6))

    for test_idx, test_task in enumerate(task_names):
        accuracies = []

        for train_idx in range(test_idx, num_tasks):
            accuracies.append(results['task_accuracies'][train_idx][test_task])

        # Visualize accuracy changes after learning the task
        x = list(range(test_idx + 1, num_tasks + 1))
        plt.plot(x, accuracies, marker='o', label=f"Task {test_idx+1}: {test_task}")

    plt.xlabel("After Learning Task")
    plt.ylabel("Accuracy")
    plt.title("Forgetting Curve per Task")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(list(range(1, num_tasks+1)))
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(output_dir, "plots", "forgetting_curve.png"), dpi=300)
    plt.close()


def plot_class_accuracy_distribution(results, output_dir):
    """Visualize class-wise accuracy distribution"""
    task_names = results['task_order']

    # Class accuracies after learning all tasks
    final_class_accuracies = {}

    for task_name in task_names:
        final_class_accuracies.update(results['class_accuracies'][-1][task_name])

    # Sort by accuracy
    sorted_accuracies = sorted(final_class_accuracies.items(), key=lambda x: x[1])
    class_names = [item[0] for item in sorted_accuracies]
    accuracy_values = [item[1] for item in sorted_accuracies]

    # Color mapping for each task's classes
    class_to_task = {}
    for i, task_name in enumerate(task_names):
        for task in results['class_accuracies'][-1]:
            if task == task_name:
                for class_name in results['class_accuracies'][-1][task]:
                    class_to_task[class_name] = i

    colors = plt.cm.tab10(np.array([class_to_task.get(c, 0) for c in class_names]))

    # Visualization
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(class_names)), accuracy_values, color=colors)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Class-wise Accuracy After Learning All Tasks")
    plt.xticks([])  # Skip class names if too many

    # Add legend (per task)
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(i)) for i in range(len(task_names))]
    plt.legend(handles, [f"Task {i+1}: {name}" for i, name in enumerate(task_names)])

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(output_dir, "plots", "class_accuracy_distribution.png"), dpi=300)
    plt.close()
