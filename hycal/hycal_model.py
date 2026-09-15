"""
HyCal Model - Training and State Management

This module contains the core HyCal model class responsible for:
- Initialization and CLIP model loading
- Embedding extraction and fusion
- Class and background statistics computation
- Training task and state updates
- Model saving and loading
"""

import os
import pickle
import numpy as np
import torch
import clip
import open_clip
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import DataLoader
from collections import defaultdict

# Setup logging
logger = logging.getLogger("HyCal")


class HyCalModel:
    """HyCal model for few-shot class-incremental learning using frozen CLIP embeddings."""
    
    def __init__(self,
                 clip_model_name: str = "ViT-B/32",
                 model_provider: str = "openai",
                 pretrained: str = "laion400m_e32", # 'laion400m_e32', 'webli'(for siglip)
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 n_min_p: int = 5,
                 lambda_reg: float = 1e-3,
                 lambda_bg: float = 1e-3,
                 scale: float = 2.0,
                 template: str = ["a photo of a {}."],
                 fusion: Optional[str] = None,
                 bg_ref_class: str = "first",
                 sim_metric: str = 'weighted',
                 real: bool = False,
                 gamma: float = 1.0):
        """Initialize HyCal model with CLIP backbone and hyperparameters.

        Args:
            clip_model_name: CLIP model variant (ViT-B/32, ViT-B/16, etc.)
            model_provider: 'openai' or 'openclip'
            pretrained: Pretrained weights name of OpenCLIP model
            device: 'cuda' or 'cpu'
            n_min_p: Minimum samples for precision matrix
            lambda_reg: Class precision regularization
            lambda_bg: Background precision regularization
            scale: Dynamic weight scale
            template: Text prompt templates
            fusion: None, 'sum', or 'concat'
            bg_ref_class: 'first' or 'average'
            sim_metric: 'cosine', 'md', 'avg', or 'weighted'
        """
        self.device = device
        self.n_min_p = n_min_p
        self.lambda_reg = lambda_reg
        self.lambda_bg = lambda_bg
        self.scale = scale
        self.sim_metric = sim_metric
        self.template = template
        self.fusion = fusion
        self.bg_ref_class = bg_ref_class
        self.real = real
        self.gamma = gamma
        
        # Load CLIP model
        if model_provider == 'openclip' or model_provider == 'laion':
            logger.info(f"Loading LAION-trained CLIP model '{clip_model_name}'...")
            self.model, train_transform, val_transform = open_clip.create_model_and_transforms(
                clip_model_name, pretrained=pretrained) # laion400m_e32
            self.train_preprocess = train_transform
            self.val_preprocess = val_transform
        else:  # openai
            logger.info(f"Loading OpenAI CLIP model '{clip_model_name}'...")
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                clip_model_name, device=device, jit=False)
        
        self.model = self.model.to(device)

        # Get model's context length for proper tokenization
        if hasattr(self.model, 'context_length'):
            self.context_length = self.model.context_length
        elif hasattr(self.model, 'text') and hasattr(self.model.text, 'context_length'):
            self.context_length = self.model.text.context_length
        else:
            self.context_length = 77  # Default CLIP context length

        # Use SigLIP tokenizer for SigLIP models (context_length=64), otherwise use default
        if 'siglip' in clip_model_name.lower() or self.context_length == 64:
            logger.info("Detected SigLIP model - using SigLipTokenizer with c4-en sentencepiece vocabulary")
            from open_clip.tokenizer import SigLipTokenizer
            self._siglip_tokenizer = SigLipTokenizer('c4-en', context_length=64)
            self.tokenizer = self._siglip_tokenize
        else:
            self.tokenizer = open_clip.tokenize if model_provider == 'openclip' or model_provider == 'laion' else clip.tokenize

        # Freeze model (no training needed)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        
        # Class statistics storage
        self.class_stats = {}  # class_name -> (mean embedding, precision matrix, sample count)
        self.learned_classes = []  # All learned class names
        
        # Background statistics for RMD
        self.bg_stats = {
            'mu_bg': None,   # Global mean embedding
            'P_bg': None,    # Global precision matrix
            'N_total': 0     # Total number of samples used
        }
        
        logger.info(f"HyCal Model initialization complete (device: {device})")

    def _siglip_tokenize(self, texts, context_length=None):
        """Tokenize texts using SigLIP tokenizer with c4-en sentencepiece vocabulary.

        Args:
            texts: List of text strings or single text string
            context_length: Maximum sequence length (default: self.context_length)

        Returns:
            torch.Tensor of token IDs with shape [batch_size, context_length]
        """
        # SigLipTokenizer handles both single strings and lists
        # It returns tokens with shape [batch_size, context_length]
        return self._siglip_tokenizer(texts)

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """L2 normalize embeddings"""
        norm = embedding.norm(dim=-1, keepdim=True)
        return embedding / (norm + 1e-10)
    
    def extract_embeddings(self, dataloader: DataLoader) -> Tuple[torch.Tensor, List[str]]:
        """Extract and fuse image-text embeddings from dataloader.

        Args:
            dataloader: DataLoader yielding (image, label) pairs

        Returns:
            Tuple of (embeddings [B, D], labels [B])
        """
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                if isinstance(images, list):
                    images = torch.stack(images)
                images = images.to(self.device)
                
                # Extract embeddings through CLIP image encoder
                image_embeddings = self.model.encode_image(images)
                image_embeddings = self._normalize_embedding(image_embeddings)
                
                # Handle text labels
                if isinstance(labels, torch.Tensor):
                    # MNIST case
                    idx_to_class = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
                    labels = ['number: "{}"'.format(idx_to_class[label.item()]) for label in labels]
                    
                labels = [str(label.item()) if isinstance(label, torch.Tensor) else label
                         for label in labels]

                # Only encode text if fusion mode is enabled
                if self.fusion is None:
                    all_embeddings.append(image_embeddings)
                else:
                    text = [self.template[0].format(label.replace('_', ' ')) for label in labels]
                    text = self.tokenizer(text, context_length=self.context_length).to(self.device)
                    text_embeddings = self.model.encode_text(text)
                    text_embeddings = self._normalize_embedding(text_embeddings)

                    # Fusion image and text embeddings
                    if self.fusion == 'sum':
                        all_embeddings.append(self._normalize_embedding(image_embeddings + text_embeddings))
                    elif self.fusion == 'concat':
                        all_embeddings.append(self._normalize_embedding(
                            torch.cat((image_embeddings, text_embeddings), dim=-1)))
                
                all_labels.extend(labels)
        
        # Stack all embeddings into a single tensor
        all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings_tensor, all_labels
    
    def compute_class_statistics(self,
                                embeddings: torch.Tensor,
                                labels: List[str]) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], int]]:
        """Compute class statistics: mean embedding, precision matrix, sample count.

        Args:
            embeddings: Tensor of embeddings [B, D]
            labels: Class labels [B]

        Returns:
            Dict: class_name -> (mean, precision_matrix, sample_count)
        """
        class_stats = {}
        
        # Group embeddings by class
        class_embeddings = defaultdict(list)
        labels_array = np.array(labels)
        unique_labels = np.unique(labels_array)
        
        for label in unique_labels:
            mask = labels_array == label
            class_embs = embeddings[torch.tensor(mask, device=self.device)]
            class_embeddings[str(label)] = class_embs
        
        # Calculate statistics for each class
        for class_name, embs_tensor in class_embeddings.items():
            N_c = embs_tensor.size(0)
            
            # 1. Calculate mean embedding (prototype)
            mu_c = embs_tensor.mean(dim=0)
            mu_c = self._normalize_embedding(mu_c)
            
            # 2. Calculate precision matrix (if sufficient samples)
            P_c = None
            if N_c > 1:
                # Calculate covariance matrix
                centered = embs_tensor - mu_c.unsqueeze(0)
                cov = torch.matmul(centered.T, centered) / (N_c - 1)
            else:
                # Use small diagonal matrix if only one sample
                d = mu_c.shape[0]
                cov = torch.eye(d, device=self.device) * 1e-3
            
            # Calculate precision matrix (inverse covariance)
            P_c = self._compute_regularized_precision(cov, real=self.real)
            
            # Store class statistics
            class_stats[class_name] = (mu_c, P_c, N_c)
        
        return class_stats
    
    def _compute_regularized_precision(self, cov: torch.Tensor,
                                      lambda_val: Optional[float] = None,
                                      real: bool = False) -> torch.Tensor:
        """Compute regularized precision matrix from covariance.

        Args:
            cov: Covariance matrix [D, D]
            lambda_val: Regularization coefficient (uses self.lambda_reg if None)

        Returns:
            Precision matrix [D, D]
        """
        if lambda_val is None:
            lambda_val = self.lambda_reg
            
        # Add regularization term (numerical stability)
        d = cov.shape[0]
        if real == False:
            regularized_cov = cov + lambda_val * torch.eye(d, device=self.device)
        else:
            regularized_cov = (1-lambda_val) * cov + lambda_val * self.gamma * torch.eye(d, device=self.device)
        
        try:
            # Calculate inverse (precision matrix)
            precision = torch.pinverse(regularized_cov)
            return precision
        except RuntimeError as e:
            logger.warning(f"Error calculating precision matrix: {str(e)}")
            # Return diagonal precision matrix on failure
            return torch.diag(1.0 / (torch.diag(regularized_cov)))
    
    def update_stats(self, new_class_stats: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], int]]):
        """Update learned class statistics.

        Args:
            new_class_stats: Dict of class_name -> (mean, precision, count)
        """
        # Integrate new class statistics
        for class_name, (mu_c, P_c, N_c) in new_class_stats.items():
            self.class_stats[class_name] = (mu_c, P_c, N_c)
            if class_name not in self.learned_classes:
                self.learned_classes.append(class_name)
        
        logger.info(f"Class statistics updated: {len(self.learned_classes)} classes")
    
    def update_background_stats(self, embeddings: torch.Tensor):
        """Update global background statistics for RMD calculation.

        Args:
            embeddings: Tensor of embeddings [B, D]
        """
        N_new = embeddings.shape[0]
        
        # Skip if no embeddings
        if N_new == 0:
            return
        
        # First task initialization
        if self.bg_stats['mu_bg'] is None:
            # Initialize background statistics
            mu_bg = embeddings.mean(dim=0)
            
            # Calculate covariance matrix if we have sufficient samples
            P_bg = None
            if N_new > 1:
                centered = embeddings - mu_bg.unsqueeze(0)
                cov_bg = torch.matmul(centered.T, centered) / (N_new - 1)
                P_bg = self._compute_regularized_precision(cov_bg, self.lambda_bg, real=self.real)
            
            self.bg_stats['mu_bg'] = mu_bg
            self.bg_stats['P_bg'] = P_bg
            self.bg_stats['N_total'] = N_new
        
        else:
            # Recursive update of global mean and precision matrix
            N_prev = self.bg_stats['N_total']
            mu_prev = self.bg_stats['mu_bg']
            
            # Step 1: Update mean embedding
            mu_new = embeddings.mean(dim=0)
            mu_updated = (N_prev * mu_prev + N_new * mu_new) / (N_prev + N_new)
            
            # Step 2: Update precision matrix using recursive formula
            if N_new > 1 and N_prev > 0:
                # First, get previous covariance (if exists)
                P_prev = self.bg_stats['P_bg']
                if P_prev is not None:
                    # Convert precision back to covariance (approximate)
                    cov_prev = torch.inverse(P_prev)
                else:
                    # If P_prev doesn't exist, initialize cov_prev
                    d = mu_updated.shape[0]
                    cov_prev = torch.eye(d, device=self.device) * self.lambda_bg
                
                # Calculate covariance of new data
                centered_new = embeddings - mu_new.unsqueeze(0)
                cov_new = torch.matmul(centered_new.T, centered_new) / (N_new - 1)
                
                # Cross-covariance term for mean difference
                mean_diff = (mu_prev - mu_new).unsqueeze(1)
                cross_term = (N_prev * N_new) / (N_prev + N_new) * torch.matmul(mean_diff, mean_diff.T)
                
                # Weighted combination of covariances + cross-term
                cov_updated = (N_prev - 1) / (N_prev + N_new - 1) * cov_prev + \
                             (N_new - 1) / (N_prev + N_new - 1) * cov_new + \
                             cross_term / (N_prev + N_new - 1)
                
                # Calculate updated precision matrix
                P_updated = self._compute_regularized_precision(cov_updated, self.lambda_bg, real=self.real)
                
            else:
                # Not enough samples for a reliable update, keep previous P if it exists
                P_updated = self.bg_stats['P_bg']
            
            # Update background statistics
            self.bg_stats['mu_bg'] = mu_updated
            self.bg_stats['P_bg'] = P_updated
            self.bg_stats['N_total'] += N_new
        
        logger.info(f"Background statistics updated: N_total = {self.bg_stats['N_total']}")
    
    def train_task(self, dataloader: DataLoader):
        """Train on single task using prototype learning.

        Args:
            dataloader: Current task's dataloader
        """
        logger.info(f"Extracting embeddings from dataset...")
        embeddings, labels = self.extract_embeddings(dataloader)
        
        logger.info(f"Computing class-wise statistics...")
        class_stats = self.compute_class_statistics(embeddings, labels)
        
        # Check for new classes
        new_classes = set(class_stats.keys()) - set(self.learned_classes)
        logger.info(f"Detected {len(new_classes)} new classes: {', '.join(new_classes)}")
        
        # Update existing class statistics
        self.update_stats(class_stats)
        
        # Update background statistics for RMD
        logger.info(f"Updating background statistics for RMD...")
        self.update_background_stats(embeddings)
    
    def save(self, path: str):
        """Save model state to checkpoint.

        Args:
            path: Checkpoint save path
        """
        save_dict = {
            'class_stats': {k: (v[0].cpu(), None if v[1] is None else v[1].cpu(), v[2]) 
                          for k, v in self.class_stats.items()},
            'learned_classes': self.learned_classes,
            'bg_stats': {
                'mu_bg': None if self.bg_stats['mu_bg'] is None else self.bg_stats['mu_bg'].cpu(),
                'P_bg': None if self.bg_stats['P_bg'] is None else self.bg_stats['P_bg'].cpu(),
                'N_total': self.bg_stats['N_total']
            },
            'config': {
                'n_min_p': self.n_min_p,
                'lambda_reg': self.lambda_reg,
                'lambda_bg': self.lambda_bg,
                'scale': self.scale,
                'fusion': self.fusion,
                'bg_ref_class': self.bg_ref_class,
                'sim_metric': self.sim_metric
            }
        }
        
        # Create directory
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model state saved: {path}")
    
    def load(self, path: str):
        """Load model state from checkpoint.

        Args:
            path: Checkpoint load path
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Load class statistics
        self.class_stats = {}
        for class_name, (mu_c, P_c, N_c) in save_dict['class_stats'].items():
            self.class_stats[class_name] = (
                mu_c.to(self.device) if mu_c is not None else None,
                P_c.to(self.device) if P_c is not None else None,
                N_c
            )
            
        self.learned_classes = save_dict['learned_classes']
        
        # Load background statistics
        bg_stats = save_dict.get('bg_stats', {})
        self.bg_stats = {
            'mu_bg': bg_stats.get('mu_bg').to(self.device) if bg_stats.get('mu_bg') is not None else None,
            'P_bg': bg_stats.get('P_bg').to(self.device) if bg_stats.get('P_bg') is not None else None,
            'N_total': bg_stats.get('N_total', 0)
        }
        
        # Restore configuration
        config = save_dict.get('config', {})
        self.n_min_p = config.get('n_min_p', self.n_min_p)
        self.lambda_reg = config.get('lambda_reg', self.lambda_reg)
        self.lambda_bg = config.get('lambda_bg', self.lambda_bg)
        self.scale = config.get('scale', self.scale)
        self.fusion = config.get('fusion', self.fusion)
        self.bg_ref_class = config.get('bg_ref_class', self.bg_ref_class)
        self.sim_metric = config.get('sim_metric', self.sim_metric)
        
        logger.info(f"Model state loaded: {len(self.learned_classes)} classes")