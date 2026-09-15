"""
HyCal Evaluator - Inference and Evaluation

This module contains the evaluation logic for HyCal model:
- Distance metric computation (Cosine, Mahalanobis)
- Score fusion and normalization
- Model evaluation (standard and zero-shot)
- Embedding saving
"""

import os
import numpy as np
import torch
from typing import Dict, Tuple
import logging
from torch.utils.data import DataLoader
from collections import defaultdict
import math

logger = logging.getLogger("HyCal")


class HyCalEvaluator:
    """HyCal evaluator for inference and evaluation using distance metrics."""

    def __init__(self, model):
        """Initialize evaluator with trained HyCalModel.

        Args:
            model: HyCalModel instance
        """
        self.model = model
        self.device = model.device
    
    def compute_mahalanobis_distance(self,
                                    emb: torch.Tensor,
                                    mu: torch.Tensor,
                                    P: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance from embedding to distribution.

        Args:
            emb: Embedding vector [D]
            mu: Mean vector [D]
            P: Precision matrix [D, D]

        Returns:
            Mahalanobis distance (scalar)
        """
        diff = (emb - mu).float()
        return torch.matmul(torch.matmul(diff.unsqueeze(0), P), diff.unsqueeze(1)).squeeze()
    
    def _gaussian_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian CDF for standardized values.

        Args:
            x: Standardized values

        Returns:
            Phi(x) values in [0, 1]
        """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    def _normalize_rmd_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """Normalize RMD scores using Gaussian CDF per sample.

        Args:
            raw_scores: Raw RMD scores [B, C]

        Returns:
            Normalized scores in [0, 1] [B, C]
        """
        # Calculate mean and std for each sample (across all classes)
        mean_scores = raw_scores.mean(dim=1, keepdim=True)  # (batch_size, 1)
        std_scores = raw_scores.std(dim=1, keepdim=True)    # (batch_size, 1)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-6
        std_scores = torch.clamp(std_scores, min=epsilon)
        
        # Standardize scores
        z_scores = (raw_scores - mean_scores) / std_scores
        
        # Apply Gaussian CDF to get final scores in [0, 1]
        normalized_scores = self._gaussian_cdf(z_scores)
        
        return normalized_scores
    
    def evaluate(self, dataloader: DataLoader, target_classes: list = None,
                task_name: str = None, save_embed: bool = False,
                path=None):
        """Evaluate classification performance on dataset.

        Args:
            dataloader: DataLoader yielding (image, label) pairs
            target_classes: Target class names (None for all learned classes)
            task_name: Task name for embedding save
            save_embed: Whether to save embeddings
            path: Embedding save path

        Returns:
            Tuple of (accuracy, class_accuracies)
        """
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        if save_embed:
            all_embeddings = []
            all_predictions = {
                'cosine': [],
                'md': [],
                'avg': [],
                'weighted': []
            }
            all_labels = []

        # Get background statistics for RMD
        mu_bg = self.model.bg_stats['mu_bg']
        P_bg = self.model.bg_stats['P_bg']
        
        # Flag for using RMD
        use_rmd = mu_bg is not None and P_bg is not None
        if use_rmd:
            mu_bg = mu_bg.to(self.device)
            P_bg = P_bg.to(self.device)
        
        with torch.no_grad():
            # Prepare class prototypes and metadata
            if target_classes is not None:
                class_names = target_classes
            else: 
                class_names = self.model.learned_classes

            if self.model.sim_metric == 'md':
                missing_precisions = [
                    class_name for class_name in class_names
                    if (class_name not in self.model.class_stats or
                        self.model.class_stats[class_name][1] is None)
                ]
                if missing_precisions:
                    names = ', '.join(map(str, missing_precisions))
                    raise ValueError(
                        "sim_metric='md' requires a stored precision matrix for "
                        f"every candidate class; missing for {len(missing_precisions)} "
                        f"class(es): {names}"
                    )
            
            prototypes = []  # Class prototype embeddings
            precisions = []  # Precision matrices
            sample_counts = []  # Sample counts
            weights = []     # Dynamic weights
            
            for class_name in class_names:
                if class_name in self.model.class_stats:
                    # Learned class: use prototype from training
                    mu_c, P_c, N_c = self.model.class_stats[class_name]
                    mu_c = mu_c.to(self.device)
                else:
                    # Unlearned class: use zero-shot text embedding as prototype
                    text_input = self.model.template[0].format(class_name.replace('_', ' '))
                    text_token = self.model.tokenizer([text_input], context_length=self.model.context_length).to(self.device)
                    mu_c = self.model.model.encode_text(text_token)
                    mu_c = self.model._normalize_embedding(mu_c).squeeze(0)  # [D]
                    P_c = None
                    N_c = 0

                prototypes.append(mu_c)
                sample_counts.append(N_c)

                # Calculate dynamic weight
                w_c = 0.0
                if P_c is not None:
                    P_c = P_c.to(self.device)
                    w_c = 1.0 / (1.0 + torch.exp(-torch.tensor(
                        (N_c - self.model.n_min_p) / self.model.scale))).item()
                    precisions.append(P_c)
                else:
                    precisions.append(None)
                weights.append(w_c)
            
            prototypes = torch.stack(prototypes)  # [C, D]
            weights = torch.tensor(weights, device=self.device)  # [C]

            # Prepare text embeddings if fusion mode
            if self.model.fusion is not None:
                text_inputs = [t.format(class_name.replace('_', ' '))
                             for t in self.model.template for class_name in class_names]
                text_tokens = self.model.tokenizer(text_inputs, context_length=self.model.context_length).to(self.device)
                text_embeddings = self.model.model.encode_text(text_tokens)
                text_embeddings = self.model._normalize_embedding(text_embeddings)
            else:
                text_embeddings = None
            
            # Process batches
            for images, labels in dataloader:
                if isinstance(labels, torch.Tensor):
                    # MNIST
                    idx_to_class = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
                    labels = ['number: "{}"'.format(idx_to_class[label.item()]) for label in labels]
                    
                if isinstance(images, list):
                    images = torch.stack(images)
                images = images.to(self.device)
                
                # Extract image embeddings
                image_embeddings = self.model.model.encode_image(images)
                image_embeddings = self.model._normalize_embedding(image_embeddings)  # [B, D]
                batch_size = image_embeddings.shape[0]
                
                # Apply fusion if specified
                if self.model.fusion == 'sum':
                    expanded_images = image_embeddings.unsqueeze(1)  # [B, 1, D]
                    expanded_texts = text_embeddings.unsqueeze(0)    # [1, C, D]
                    combined = expanded_images + expanded_texts      # [B, C, D]
                    test_embeddings = self.model._normalize_embedding(
                        combined.reshape(batch_size * len(class_names), -1))
                    test_embeddings = test_embeddings.view(batch_size, len(class_names), -1)  # [B, C, D]
                    
                elif self.model.fusion == 'concat':
                    expanded_images = image_embeddings.unsqueeze(1).expand(
                        batch_size, len(class_names), -1)  # [B, C, D]
                    expanded_texts = text_embeddings.unsqueeze(0).expand(
                        batch_size, len(class_names), -1)  # [B, C, D]
                    combined = torch.cat([expanded_images, expanded_texts], dim=-1)  # [B, C, 2*D]
                    test_embeddings = self.model._normalize_embedding(
                        combined.reshape(batch_size * len(class_names), -1))
                    test_embeddings = test_embeddings.view(batch_size, len(class_names), -1)  # [B, C, 2*D]
                else:
                    test_embeddings = image_embeddings  # [B, D]
                
                # Calculate cosine similarities
                if self.model.fusion in ['sum', 'concat']:
                    cos_sims = torch.einsum('bcd,cd->bc', test_embeddings, prototypes)
                else:
                    cos_sims = torch.matmul(test_embeddings, prototypes.T)  # [B, C]

                cos_sims = self._normalize_rmd_scores(cos_sims)  # Normalize to [0, 1]
                # Compute Mahalanobis distances if available
                if use_rmd:
                    bg_maha_dists = torch.zeros(batch_size, device=self.device)
                    
                    if self.model.fusion in ['sum', 'concat']:
                        # Select reference embeddings based on bg_ref_class setting
                        if self.model.bg_ref_class == 'average':
                            ref_embs = torch.mean(test_embeddings, dim=1)  # [B, D]
                        else:
                            ref_embs = test_embeddings[:, 0, :]  # [B, D]

                        # Compute background Mahalanobis distances
                        diff_bg = (ref_embs - mu_bg.unsqueeze(0)).float()  # [B, D]
                        bg_maha_dists = torch.sum(torch.matmul(diff_bg, P_bg) * diff_bg, dim=1)  # [B]

                        # Compute class-specific Mahalanobis distances
                        raw_scores = torch.zeros((batch_size, len(class_names)), device=self.device)
                        prefer_idx = []

                        for c, P_c in enumerate(precisions):
                            if P_c is None:
                                continue

                            cls_embs = test_embeddings[:, c, :]
                            diffs = (cls_embs - prototypes[c].unsqueeze(0)).float()  # [B, D]
                            class_maha_dists = torch.sum(torch.matmul(diffs, P_c) * diffs, dim=1)  # [B]

                            # RMD score: bg_dist - class_dist
                            raw_scores[:, c] = bg_maha_dists - class_maha_dists

                            if sample_counts[c] >= self.model.n_min_p and P_c is not None:
                                prefer_idx.append(c)

                    else:
                        # Standard mode (image embeddings only)
                        diffs_bg = (test_embeddings - mu_bg.unsqueeze(0)).float()  # [B, D]
                        bg_maha_dists = torch.sum(torch.matmul(diffs_bg, P_bg) * diffs_bg, dim=1)  # [B]

                        raw_scores = torch.zeros((batch_size, len(class_names)), device=self.device)
                        prefer_idx = []

                        for c, P_c in enumerate(precisions):
                            if P_c is None:
                                continue

                            diffs = (test_embeddings - prototypes[c].unsqueeze(0)).float()  # [B, D]
                            class_maha_dists = torch.sum(torch.matmul(diffs, P_c) * diffs, dim=1)  # [B]
                            raw_scores[:, c] = bg_maha_dists - class_maha_dists

                            if sample_counts[c] >= self.model.n_min_p and P_c is not None:
                                prefer_idx.append(c)
                    
                    # Compute final scores based on similarity metric
                    if self.model.sim_metric == 'cosine':
                        scores = cos_sims
                    else:
                        # Normalize Mahalanobis scores
                        maha_sims = self._normalize_rmd_scores(raw_scores)  # [B, C]

                        if self.model.sim_metric == 'md':
                            scores = maha_sims
                        elif self.model.sim_metric == 'avg':
                            scores = cos_sims.clone()  # [B, C]
                            if prefer_idx:
                                scores[:, prefer_idx] = 0.5 * (cos_sims[:, prefer_idx] + 
                                                              maha_sims[:, prefer_idx])
                        else:  # weighted
                            scores = cos_sims.clone()  # [B, C]
                            if prefer_idx:
                                scores[:, prefer_idx] = ((1 - weights[prefer_idx]) * cos_sims[:, prefer_idx] + 
                                                        weights[prefer_idx] * maha_sims[:, prefer_idx])
                else:
                    # Use only cosine similarity when Mahalanobis isn't available
                    scores = cos_sims
                
                # Get predictions
                pred_indices = torch.argmax(scores, dim=1)  # [B]
                predictions = [class_names[i] for i in pred_indices]

                # Save embeddings and predictions if requested
                if save_embed:
                    metrics = ['cosine', 'md', 'avg', 'weighted']
                    metrics.remove(self.model.sim_metric)

                    if self.model.sim_metric == 'cosine' and use_rmd:
                        maha_sims = self._normalize_rmd_scores(raw_scores)

                    for metric in metrics:
                        if metric == 'cosine':
                            tmp_scores = cos_sims
                        elif metric == 'md' and use_rmd:
                            tmp_scores = maha_sims
                        elif metric == 'avg' and use_rmd:
                            tmp_scores = cos_sims.clone()
                            if prefer_idx:
                                tmp_scores[:, prefer_idx] = 0.5 * (cos_sims[:, prefer_idx] + 
                                                                  maha_sims[:, prefer_idx])
                        elif metric == 'weighted' and use_rmd:
                            tmp_scores = cos_sims.clone()
                            if prefer_idx:
                                tmp_scores[:, prefer_idx] = ((1 - weights[prefer_idx]) * cos_sims[:, prefer_idx] + 
                                                            weights[prefer_idx] * maha_sims[:, prefer_idx])
                        else:
                            continue
                    
                        tmp_pred_indices = torch.argmax(tmp_scores, dim=1)
                        tmp_predictions = [class_names[i] for i in tmp_pred_indices]
                        all_predictions[metric].extend(tmp_predictions)

                    all_embeddings.append(test_embeddings.cpu())
                    all_predictions[self.model.sim_metric].extend(predictions)
                    all_labels.extend(labels)

                # Update accuracy statistics
                for pred, true_class in zip(predictions, labels):
                    total += 1
                    class_total[true_class] += 1
                    if pred == true_class:
                        correct += 1
                        class_correct[true_class] += 1
        
        # Calculate overall accuracy (in percentage)
        accuracy = (correct / total if total > 0 else 0.0) * 100.0
        
        # Calculate class-wise accuracies
        class_accuracies = {}
        logger.info("\n" + "="*50)
        logger.info("Evaluation Results")
        logger.info("-"*50)
        logger.info(f"{'Class':<30} {'Accuracy':<15} {'Correct/Total':<15}")
        logger.info("-"*50)
        
        for class_name in sorted(class_total.keys()):
            acc = (class_correct[class_name] / class_total[class_name]) * 100.0
            class_accuracies[class_name] = acc

        logger.info("-"*50)
        logger.info(f"{'Overall':<30} {accuracy:>10.4f}% {correct:>4d}/{total:<4d}")
        logger.info("="*50 + "\n")

        # Save embeddings if requested
        if save_embed:
            embeddings_tensor = torch.cat(all_embeddings, dim=0)
            embeddings_array = embeddings_tensor.numpy()

            save_path = f'embed_{task_name}.npz'
            if path is not None:
                save_path = os.path.join(path, save_path)

            save_dict = {
                'embeddings': embeddings_array,
                'labels': np.array(all_labels)
            }

            for metric, preds in all_predictions.items():
                if preds:
                    save_dict[f'predictions_{metric}'] = np.array(preds)

            np.savez(save_path, **save_dict)

            saved_metrics = [m for m in all_predictions.keys() if all_predictions[m]]
            logger.info(f"Embeddings saved to {save_path}")
            logger.info(f"Prediction metrics saved: {', '.join(saved_metrics)}")

        return accuracy, class_accuracies

    def evaluate_zeroshot(self, dataloader: DataLoader,
                         total_classes: list = None) -> Tuple[float, Dict[str, float]]:
        """Evaluate using standard zero-shot CLIP inference.

        Args:
            dataloader: DataLoader yielding (image, label) pairs
            total_classes: List of all class names

        Returns:
            Tuple of (accuracy, class_accuracies)
        """
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        # Collect unique class names
        if total_classes is not None:
            unique_classes = set(total_classes)
        else:
            unique_classes = set()
            for _, labels in dataloader:
                if isinstance(labels, torch.Tensor):
                    idx_to_class = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
                    labels = ['number: "{}"'.format(idx_to_class[label.item()]) for label in labels]
                unique_classes.update(labels)
        
        # Create text prompts and embeddings
        text_inputs = [f"a photo of a {class_name.replace('_', ' ')}."
                      for class_name in unique_classes]
        text_tokens = self.model.tokenizer(text_inputs, context_length=self.model.context_length).to(self.device)
        
        with torch.no_grad():
            # Calculate text embeddings
            text_features = self.model.model.encode_text(text_tokens)
            text_features = self.model._normalize_embedding(text_features)
            
            # Process batches
            for images, labels in dataloader:
                if isinstance(labels, torch.Tensor):
                    idx_to_class = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
                    labels = ['number: "{}"'.format(idx_to_class[label.item()]) for label in labels]
                    
                if isinstance(images, list):
                    images = torch.stack(images)
                images = images.to(self.device)
                
                # Extract image embeddings
                image_features = self.model.model.encode_image(images)
                image_features = self.model._normalize_embedding(image_features)
                
                # Calculate image-text similarities
                similarity = torch.matmul(image_features, text_features.T)
                
                # Get predictions
                pred_indices = similarity.argmax(dim=1)
                predictions = [list(unique_classes)[i] for i in pred_indices]
                
                # Update accuracy statistics
                for pred, true_class in zip(predictions, labels):
                    total += 1
                    class_total[true_class] += 1
                    if pred == true_class:
                        correct += 1
                        class_correct[true_class] += 1
        
        # Calculate overall accuracy
        accuracy = (correct / total if total > 0 else 0.0) * 100.0
        
        # Calculate class-wise accuracies
        class_accuracies = {}
        logger.info("\n" + "="*50)
        logger.info("Zero-shot Evaluation Results")
        logger.info("-"*50)
        logger.info(f"{'Class':<30} {'Accuracy':<15} {'Correct/Total':<15}")
        logger.info("-"*50)
        
        for class_name in sorted(class_total.keys()):
            acc = (class_correct[class_name] / class_total[class_name]) * 100.0
            class_accuracies[class_name] = acc
        
        logger.info("-"*50)
        logger.info(f"{'Overall':<30} {accuracy:>10.4f}% {correct:>4d}/{total:<4d}")
        logger.info("="*50 + "\n")
        
        return accuracy, class_accuracies
