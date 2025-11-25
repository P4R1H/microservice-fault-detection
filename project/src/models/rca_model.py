"""
End-to-end RCA model.

Combines multimodal fusion with service ranking for root cause analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class RCAModel(nn.Module):
    """
    Complete RCA pipeline.

    Architecture:
    1. Multimodal fusion → Combined representation
    2. RCA ranking head → Service probabilities
    3. Top-k prediction with ground truth comparison

    Evaluation:
    - AC@1, AC@3, AC@5: Accuracy at top-k
    - Avg@k: Position-weighted accuracy
    - MRR: Mean reciprocal rank
    """

    def __init__(
        self,
        fusion_model: nn.Module,
        num_services: int,
        fusion_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_service_embedding: bool = True
    ):
        super().__init__()

        self.fusion_model = fusion_model
        self.num_services = num_services
        self.use_service_embedding = use_service_embedding

        # Optional: Service embeddings for better representation
        if use_service_embedding:
            self.service_embedding = nn.Embedding(num_services, 64)
            # When using service embeddings, we score each service individually
            # Input: (batch, fusion_dim + 64), Output: (batch, 1)
            self.ranking_head = nn.Sequential(
                nn.Linear(fusion_dim + 64, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)  # Single score per service
            )
        else:
            # Without service embeddings, predict all services at once
            # Input: (batch, fusion_dim), Output: (batch, num_services)
            self.ranking_head = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_services)
            )

    def forward(
        self,
        metrics: Optional[torch.Tensor] = None,
        logs: Optional[torch.Tensor] = None,
        traces: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal_weights: Optional[torch.Tensor] = None,
        service_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict root cause service.

        Args:
            metrics: (batch, seq_len, features) metrics data
            logs: (batch, log_dim) logs data
            traces: (node_features, edge_index) traces data
            causal_weights: (batch,) causal scores from PCMCI
            service_mask: (batch, num_services) mask for valid services
            return_attention: Return attention weights

        Returns:
            Dictionary with:
            - 'logits': (batch, num_services) raw scores
            - 'probs': (batch, num_services) probabilities
            - 'ranking': (batch, num_services) service ranking
            - 'attention': Attention weights if requested
        """
        # 1. Multimodal fusion
        fusion_output = self.fusion_model(
            metrics=metrics,
            logs=logs,
            traces=traces,
            causal_weights=causal_weights,
            return_attention=return_attention
        )

        fused_repr = fusion_output['fused']  # (batch, fusion_dim)
        batch_size = fused_repr.size(0)

        # 2. Optionally add service embeddings
        if self.use_service_embedding:
            # Create service indices: (batch, num_services)
            service_indices = torch.arange(
                self.num_services,
                device=fused_repr.device
            ).unsqueeze(0).expand(batch_size, -1)

            # Get service embeddings: (batch, num_services, 64)
            service_embs = self.service_embedding(service_indices)

            # Expand fused representation: (batch, 1, fusion_dim)
            fused_expanded = fused_repr.unsqueeze(1).expand(-1, self.num_services, -1)

            # Concatenate: (batch, num_services, fusion_dim + 64)
            combined = torch.cat([fused_expanded, service_embs], dim=-1)

            # Compute scores for each service: (batch, num_services)
            # Apply ranking head to each service representation
            logits = torch.zeros(
                batch_size,
                self.num_services,
                device=fused_repr.device
            )
            for i in range(self.num_services):
                logits[:, i] = self.ranking_head(combined[:, i, :]).squeeze(-1)
        else:
            # Direct prediction from fused representation
            logits = self.ranking_head(fused_repr)  # (batch, num_services)

        # 3. Apply service mask if provided
        if service_mask is not None:
            # Mask out invalid services with large negative value
            logits = logits.masked_fill(~service_mask.bool(), -1e9)

        # 4. Compute probabilities
        probs = F.softmax(logits, dim=-1)  # (batch, num_services)

        # 5. Compute ranking (argsort in descending order)
        ranking = torch.argsort(logits, dim=-1, descending=True)  # (batch, num_services)

        result = {
            'logits': logits,
            'probs': probs,
            'ranking': ranking,
            'fused_repr': fused_repr,
            'modality_embeddings': fusion_output.get('modality_embeddings', {})
        }

        if return_attention and 'attention' in fusion_output:
            result['attention'] = fusion_output['attention']

        return result

    def predict_top_k(
        self,
        logits: torch.Tensor,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k most likely root cause services.

        Args:
            logits: (batch, num_services) service scores
            k: Number of top predictions

        Returns:
            top_k_services: (batch, k) service indices
            top_k_scores: (batch, k) confidence scores
        """
        probs = F.softmax(logits, dim=-1)
        top_k_scores, top_k_services = torch.topk(probs, k=min(k, logits.size(-1)), dim=-1)
        return top_k_services, top_k_scores

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_services: torch.Tensor,
        loss_type: str = 'cross_entropy'
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            logits: (batch, num_services) predicted scores
            target_services: (batch,) ground truth service indices
            loss_type: 'cross_entropy' or 'ranking_loss'

        Returns:
            loss: Scalar loss value
        """
        if loss_type == 'cross_entropy':
            # Standard cross-entropy loss
            return F.cross_entropy(logits, target_services)

        elif loss_type == 'ranking_loss':
            # Ranking loss: Penalize if ground truth is not ranked high
            batch_size = logits.size(0)
            num_services = logits.size(1)

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Extract ground truth probabilities
            target_probs = probs[torch.arange(batch_size), target_services]

            # Ranking loss: -log(target_prob) + margin penalty
            ranking_loss = -torch.log(target_probs + 1e-9)

            # Add margin penalty: Encourage ground truth to be top-ranked
            # For each sample, penalize if any other service has higher score
            target_logits = logits[torch.arange(batch_size), target_services].unsqueeze(1)
            margins = F.relu(logits - target_logits + 1.0)  # Margin = 1.0
            margins[torch.arange(batch_size), target_services] = 0  # Don't penalize self
            margin_penalty = margins.mean(dim=1)

            total_loss = ranking_loss + 0.1 * margin_penalty
            return total_loss.mean()

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


# Evaluation metrics

def compute_accuracy_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute AC@k: Accuracy at top-k.

    Args:
        predictions: (batch, num_services) ranked service indices
        targets: (batch,) ground truth service indices
        k: Top-k threshold

    Returns:
        AC@k score (0-1)
    """
    batch_size = targets.size(0)
    top_k = predictions[:, :k]  # (batch, k)

    # Check if target is in top-k
    correct = (top_k == targets.unsqueeze(1)).any(dim=1)
    accuracy = correct.float().mean().item()

    return accuracy


def compute_average_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute Avg@k: Position-weighted accuracy.

    Avg@k = (1/batch_size) * sum(1 / rank) for samples where rank <= k

    Args:
        predictions: (batch, num_services) ranked service indices
        targets: (batch,) ground truth service indices
        k: Top-k threshold

    Returns:
        Avg@k score
    """
    batch_size = targets.size(0)
    top_k = predictions[:, :k]  # (batch, k)

    # Find rank of ground truth in top-k (0-indexed)
    ranks = []
    for i in range(batch_size):
        target_service = targets[i].item()
        if target_service in top_k[i]:
            rank = (top_k[i] == target_service).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
            ranks.append(1.0 / rank)
        else:
            ranks.append(0.0)

    avg_at_k = np.mean(ranks)
    return avg_at_k


def compute_mrr(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute MRR: Mean Reciprocal Rank.

    MRR = (1/batch_size) * sum(1 / rank)

    Args:
        predictions: (batch, num_services) ranked service indices
        targets: (batch,) ground truth service indices

    Returns:
        MRR score
    """
    batch_size = targets.size(0)

    reciprocal_ranks = []
    for i in range(batch_size):
        target_service = targets[i].item()
        # Find rank of ground truth (1-indexed)
        rank = (predictions[i] == target_service).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)

    mrr = np.mean(reciprocal_ranks)
    return mrr


def evaluate_rca_model(
    model: RCAModel,
    dataloader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate RCA model on a dataset.

    Args:
        model: RCAModel instance
        dataloader: PyTorch DataLoader with (metrics, logs, traces, target)
        device: Device to run on
        k_values: List of k values for AC@k and Avg@k

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            metrics = batch.get('metrics', None)
            logs = batch.get('logs', None)
            traces = batch.get('traces', None)
            targets = batch['target']  # (batch,) ground truth service indices

            if metrics is not None:
                metrics = metrics.to(device)
            if logs is not None:
                logs = logs.to(device)
            if traces is not None:
                traces = (traces[0].to(device), traces[1].to(device))

            # Forward pass
            output = model(metrics=metrics, logs=logs, traces=traces)
            ranking = output['ranking']  # (batch, num_services)

            all_predictions.append(ranking.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # (total_samples, num_services)
    all_targets = torch.cat(all_targets, dim=0)  # (total_samples,)

    # Compute metrics
    metrics = {}

    # AC@k for each k
    for k in k_values:
        ac_k = compute_accuracy_at_k(all_predictions, all_targets, k=k)
        metrics[f'AC@{k}'] = ac_k

    # Avg@k for each k
    for k in k_values:
        avg_k = compute_average_at_k(all_predictions, all_targets, k=k)
        metrics[f'Avg@{k}'] = avg_k

    # MRR
    mrr = compute_mrr(all_predictions, all_targets)
    metrics['MRR'] = mrr

    return metrics
