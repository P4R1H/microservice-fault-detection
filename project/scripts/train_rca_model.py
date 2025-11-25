#!/usr/bin/env python3
"""
Training Pipeline for RCA Model

Trains the complete multimodal RCA system end-to-end.

Features:
- Multi-epoch training with early stopping
- Learning rate scheduling
- Checkpoint saving
- TensorBoard logging
- Evaluation on validation set

Usage:
    python scripts/train_rca_model.py --config config/model_config.yaml --epochs 50
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import RCAEvalDataLoader
from src.data.preprocessing import MetricsPreprocessor, TracesPreprocessor
from src.encoders.metrics_encoder import TCNEncoder
from src.encoders.traces_encoder import GCNEncoder
from src.models.rca_model import RCAModel
from src.evaluation.metrics import RCAEvaluator


class RCATrainer:
    """Trainer for RCA model"""

    def __init__(
        self,
        model: RCAModel,
        train_loader,
        val_loader,
        config: Dict,
        output_dir: str,
        device: str = 'cpu'
    ):
        """
        Initialize trainer

        Args:
            model: RCA model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Directory for checkpoints and logs
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize optimizer
        self.optimizer = self._init_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._init_scheduler()

        # Initialize loss function
        self.criterion = self._init_criterion()

        # Evaluator
        self.evaluator = RCAEvaluator()

        # TensorBoard writer
        log_dir = self.output_dir / 'logs'
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Training state
        self.best_val_ac1 = 0.0
        self.patience_counter = 0
        self.global_step = 0

    def _init_optimizer(self):
        """Initialize optimizer"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam').lower()
        lr = opt_config.get('lr', 0.001)
        weight_decay = opt_config.get('weight_decay', 1e-4)

        if opt_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'sgd':
            momentum = opt_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr,
                           momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'step').lower()

        if sched_type == 'step':
            step_size = sched_config.get('step_size', 10)
            gamma = sched_config.get('gamma', 0.5)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        elif sched_type == 'cosine':
            t_max = sched_config.get('t_max', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

        else:
            return None

    def _init_criterion(self):
        """Initialize loss function"""
        # For ranking task, use margin ranking loss or cross-entropy
        # This is simplified - real implementation would use proper ranking loss
        return torch.nn.CrossEntropyLoss()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch}/{self.config['epochs']}")
        print("-" * 70)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                metrics_data=batch.get('metrics'),
                logs_data=batch.get('logs'),
                traces_data=batch.get('traces'),
                service_graph=batch.get('service_graph')
            )

            # Compute loss (simplified - actual implementation would use ranking loss)
            # For now, treat as classification over services
            service_scores = outputs['service_scores']
            ground_truth_indices = batch['ground_truth_index']

            loss = self.criterion(service_scores, ground_truth_indices)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}: Loss = {avg_loss:.4f}")

        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        all_results = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    metrics_data=batch.get('metrics'),
                    logs_data=batch.get('logs'),
                    traces_data=batch.get('traces'),
                    service_graph=batch.get('service_graph')
                )

                # Get rankings
                service_scores = outputs['service_scores']
                service_names = batch['service_names']

                # Evaluate each sample in batch
                for i in range(len(service_scores)):
                    scores = service_scores[i].cpu().numpy()
                    names = service_names[i]
                    ground_truth = batch['ground_truth'][i]

                    # Create ranking
                    ranking = [(name, score) for name, score in zip(names, scores)]
                    ranking.sort(key=lambda x: x[1], reverse=True)

                    # Evaluate
                    result = self.evaluator.evaluate_single_case(
                        predicted_ranking=ranking,
                        ground_truth=ground_truth
                    )

                    all_results.append(result)

        # Aggregate results
        metrics = self.evaluator.aggregate_results(all_results)

        # Log to TensorBoard
        self.writer.add_scalar('val/AC@1', metrics['AC@1'], epoch)
        self.writer.add_scalar('val/AC@3', metrics['AC@3'], epoch)
        self.writer.add_scalar('val/AC@5', metrics['AC@5'], epoch)
        self.writer.add_scalar('val/MRR', metrics['MRR'], epoch)

        print(f"\n  Validation Results:")
        print(f"    AC@1: {metrics['AC@1']:.3f}")
        print(f"    AC@3: {metrics['AC@3']:.3f}")
        print(f"    AC@5: {metrics['AC@5']:.3f}")
        print(f"    MRR:  {metrics['MRR']:.3f}")

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model: {best_path}")

        print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    def train(self):
        """Main training loop"""
        print("="*80)
        print("TRAINING RCA MODEL")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print()

        epochs = self.config.get('epochs', 50)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/lr', current_lr, epoch)

            # Check for improvement
            current_ac1 = val_metrics['AC@1']
            is_best = current_ac1 > self.best_val_ac1

            if is_best:
                self.best_val_ac1 = current_ac1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"  No improvement for {early_stopping_patience} epochs")
                break

            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
            print(f"Best AC@1 so far: {self.best_val_ac1:.3f}")

        # Close writer
        self.writer.close()

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\nBest validation AC@1: {self.best_val_ac1:.3f}")
        print(f"Best model saved to: {self.output_dir / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description='Train RCA model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, default='data/RCAEval',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/training',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"⚠ Config not found: {config_path}")
        print("Using default configuration")
        config = {
            'epochs': args.epochs or 50,
            'batch_size': 16,
            'optimizer': {'type': 'adam', 'lr': 0.001, 'weight_decay': 1e-4},
            'scheduler': {'type': 'step', 'step_size': 10, 'gamma': 0.5},
            'early_stopping_patience': 10
        }

    if args.epochs is not None:
        config['epochs'] = args.epochs

    # NOTE: This is a simplified training script
    # The actual data loading and batching would need proper implementation
    # For now, this serves as a template

    print("="*80)
    print("RCA MODEL TRAINING - TEMPLATE")
    print("="*80)
    print("\n⚠ NOTE: This is a training pipeline template")
    print("For actual training, you need to implement:")
    print("  1. Proper data loader with batching")
    print("  2. Data collation for variable-length sequences")
    print("  3. Loss function for ranking (e.g., ListNet, LambdaRank)")
    print("  4. Integrate with your trained RCA model")
    print("\nConfiguration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    print("To proceed, implement the data loader and integrate with RCAModel.")


if __name__ == '__main__':
    main()
