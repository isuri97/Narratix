"""
DsQoLA: Domain-specific Query-Only Linear Adapter
===================================================

Architecture:
    Query → Base Embedding Model (frozen) → Linear Adapter (trainable)
            384 → 128 → 384  (bottleneck with residual + GELU + LayerNorm + L2-norm)

Training:
    Hybrid Loss = InfoNCE (in-batch negatives, τ=0.05)
                + 0.3 × Triplet Margin Loss (hard negatives, m=0.5)
    Optimiser:  AdamW, lr=1e-4, linear warmup (100 steps), gradient clipping (1.0)
    Epochs:     10 (tested 10, 30, 40 — no gain beyond 10)

Usage:
    python dsqola.py \
        --train_path  data/filtered_chunks_train.json \
        --val_path    data/filtered_chunks_test.json \
        --neg_path    data/financial_news.txt \
        --output_dir  outputs/
"""

# ===========================================================================
# Imports
# ===========================================================================

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ===========================================================================
# SECTION 1 — ADAPTER ARCHITECTURE
# ===========================================================================

class DsQoLA(nn.Module):
    """
    Domain-specific Query-Only Linear Adapter (DsQoLA).

    A lightweight bottleneck adapter that transforms query embeddings to better
    align with a domain-specific document corpus. Document embeddings remain
    entirely frozen — only query embeddings pass through this adapter at
    retrieval time.

    Architecture:
        f(x) = LayerNorm(W_up · GELU(W_down · x) + x)
        Output is L2-normalised to unit vectors.

    Args:
        input_dim (int):      Embedding dimension of the base model.
                              Default: 384 (all-MiniLM-L6-v2).
        bottleneck_dim (int): Hidden bottleneck dimension. Default: 128.
        dropout (float):      Dropout rate. Default: 0.1.
    """

    def __init__(self, input_dim: int = 384, bottleneck_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Bottleneck layers: 384 → 128 → 384
        self.down_proj  = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj    = nn.Linear(bottleneck_dim, input_dim)

        # Regularisation and normalisation
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: bottleneck transformation with residual connection.

        Args:
            x: Query embeddings (batch_size, input_dim).

        Returns:
            Adapted, L2-normalised query embeddings (batch_size, input_dim).
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)           # Residual connection
        return F.normalize(x, p=2, dim=-1)          # L2 normalisation

    def save(self, path: str) -> None:
        """Save adapter weights and config to disk."""
        torch.save({
            'adapter_state_dict': self.state_dict(),
            'input_dim':          self.input_dim,
            'bottleneck_dim':     self.bottleneck_dim,
        }, path)
        print(f"[Adapter] Saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'DsQoLA':
        """Load a saved adapter from disk."""
        ckpt    = torch.load(path, map_location=device)
        adapter = cls(input_dim=ckpt['input_dim'], bottleneck_dim=ckpt['bottleneck_dim'])
        adapter.load_state_dict(ckpt['adapter_state_dict'])
        adapter.to(device)
        adapter.eval()
        print(f"[Adapter] Loaded from: {path}")
        return adapter


# ===========================================================================
# SECTION 2 — DATA LOADING AND NEGATIVE SAMPLING
# ===========================================================================

def load_jsonl_data(path: str) -> List[Dict]:
    """
    Load query/passage pairs from a JSONL file.

    Expected record format:
        {
          "chunk_index":  0,
          "input_chunk":  "testimony passage ...",
          "questions": {
              "question_1": "Who was deported from ...",
              "question_2": "When did the witness ...",
              ...  (up to question_20)
          }
        }

    Args:
        path: Path to JSONL file.

    Returns:
        List of record dicts.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[Data] Loaded {len(data):,} samples from {path}")
    return data


def create_negative_sampler(corpus_path: str) -> Callable[[], str]:
    """
    Build a random hard-negative sampler from a plain-text corpus.

    The corpus should contain one sentence/paragraph per line.
    For LiLADH, a financial news corpus was used: its semantic and
    stylistic register is sharply distant from Holocaust oral testimony,
    providing strongly contrastive negative signal during training.

    Args:
        corpus_path: Path to plain-text negative corpus.

    Returns:
        Zero-argument callable that returns a random negative string.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Negative corpus is empty: {corpus_path}")

    print(f"[Negatives] Loaded {len(lines):,} samples from {corpus_path}")

    def sampler() -> str:
        return random.choice(lines)

    return sampler


# ===========================================================================
# SECTION 3 — TRIPLET DATASET
# ===========================================================================

class TripletDataset(Dataset):
    """
    PyTorch Dataset for DsQoLA contrastive training.

    Each sample returns a (query_text, positive_embedding) pair.
    At each __getitem__ call, one question is selected at random
    from the 20 synthetic questions per chunk, providing implicit
    data augmentation without additional overhead.

    Positive passage chunks are pre-encoded once using the frozen
    base model — consistent with the query-only adapter design where
    document embeddings are never recomputed.

    Args:
        data:             Loaded JSONL records (from load_jsonl_data).
        base_model:       Frozen sentence transformer.
        negative_sampler: Callable returning a hard negative string.
    """

    def __init__(
        self,
        data: List[Dict],
        base_model: SentenceTransformer,
        negative_sampler: Callable[[], str]
    ):
        self.data             = data
        self.base_model       = base_model
        self.negative_sampler = negative_sampler

        # Pre-encode all positive passage chunks once (base model is frozen)
        print("[Dataset] Pre-encoding positive document chunks...")
        self.positive_embeddings = [
            base_model.encode(item['input_chunk'], convert_to_tensor=True)
            for item in data
        ]
        print(f"[Dataset] Pre-encoded {len(self.positive_embeddings):,} chunks.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        item           = self.data[idx]
        questions_dict = item['questions']
        query_text     = random.choice(list(questions_dict.values()))
        positive_emb   = self.positive_embeddings[idx]
        return query_text, positive_emb


def collate_fn(batch):
    """Collate (query_text, positive_emb) pairs into batched tensors."""
    query_texts   = [item[0] for item in batch]
    positive_embs = torch.stack([item[1] for item in batch])
    return query_texts, positive_embs


# ===========================================================================
# SECTION 4 — LEARNING RATE SCHEDULER
# ===========================================================================

def get_linear_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR:
    """
    Linear warmup followed by linear decay.

    Args:
        optimizer:           AdamW optimiser instance.
        num_warmup_steps:    Steps for linear warmup phase.
        num_training_steps:  Total training steps.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# SECTION 5 — TRAINING
# ===========================================================================

def train_adapter(
    base_model: SentenceTransformer,
    train_data: List[Dict],
    negative_sampler: Callable[[], str],
    num_epochs: int          = 10,
    batch_size: int          = 32,
    learning_rate: float     = 1e-4,
    warmup_steps: int        = 100,
    max_grad_norm: float     = 1.0,
    triplet_margin: float    = 0.5,
    infonce_temperature: float = 0.05,
    triplet_weight: float    = 0.3,
    save_path: Optional[str] = None,
) -> Tuple[DsQoLA, List[float]]:
    """
    Train DsQoLA using hybrid InfoNCE + Triplet Margin Loss.

    The base embedding model is fully frozen throughout training.
    Only the adapter parameters are updated.

    Loss:
        L = L_InfoNCE(τ=0.05) + 0.3 × L_Triplet(m=0.5)

    Args:
        base_model:            Frozen sentence transformer.
        train_data:            Training records from load_jsonl_data().
        negative_sampler:      Hard negative string sampler.
        num_epochs:            Training epochs. Default: 10.
        batch_size:            Batch size. Default: 32.
        learning_rate:         AdamW learning rate. Default: 1e-4.
        warmup_steps:          Linear warmup steps. Default: 100.
        max_grad_norm:         Gradient clipping max norm. Default: 1.0.
        triplet_margin:        Triplet loss margin m. Default: 0.5.
        infonce_temperature:   InfoNCE temperature τ. Default: 0.05.
        triplet_weight:        Weight on Triplet loss term. Default: 0.3.
        save_path:             If set, saves adapter weights here.

    Returns:
        Tuple of (trained DsQoLA adapter, list of per-epoch average losses).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")

    # Freeze base model
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    # Initialise adapter
    input_dim = base_model.get_sentence_embedding_dimension()
    adapter   = DsQoLA(input_dim=input_dim).to(device)
    n_params  = sum(p.numel() for p in adapter.parameters())
    print(f"[Train] Adapter: {input_dim} → 128 → {input_dim} | "
          f"Trainable params: {n_params:,}")

    # Loss functions
    infonce_loss_fn = nn.CrossEntropyLoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=triplet_margin, p=2)

    # Optimiser, dataset, scheduler
    optimizer   = AdamW(adapter.parameters(), lr=learning_rate)
    dataset     = TripletDataset(train_data, base_model, negative_sampler)
    dataloader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    total_steps = len(dataloader) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"[Train] Epochs: {num_epochs} | "
          f"Steps/epoch: {len(dataloader)} | "
          f"Total steps: {total_steps}")
    print("-" * 60)

    adapter.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for query_texts, positive_embs in dataloader:
            positive_embs = positive_embs.to(device)

            # Encode queries through frozen base model
            query_embs = base_model.encode(
                query_texts, convert_to_tensor=True, device=device
            ).detach()

            # Sample and encode hard negatives
            negative_texts = [negative_sampler() for _ in range(len(query_texts))]
            negative_embs  = base_model.encode(
                negative_texts, convert_to_tensor=True, device=device
            ).detach()

            # Adapt query embeddings (only the adapter is differentiable)
            adapted_query_embs = adapter(query_embs)

            # Normalise all embeddings
            adapted_query_embs = F.normalize(adapted_query_embs, p=2, dim=1)
            positive_embs_norm = F.normalize(positive_embs,       p=2, dim=1)
            negative_embs_norm = F.normalize(negative_embs,       p=2, dim=1)

            # InfoNCE loss — in-batch negatives
            cos_sim = torch.mm(
                adapted_query_embs, positive_embs_norm.t()
            ) / infonce_temperature
            labels       = torch.arange(cos_sim.size(0)).to(device)
            loss_infonce = infonce_loss_fn(cos_sim, labels)

            # Triplet Margin loss — hard negatives from external corpus
            loss_triplet = triplet_loss_fn(
                adapted_query_embs,
                positive_embs_norm,
                negative_embs_norm
            )

            # Hybrid combined loss
            loss = loss_infonce + triplet_weight * loss_triplet

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(adapter.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1:>2}/{num_epochs}  |  Avg Loss: {avg_loss:.4f}")

    print("-" * 60)
    print(f"[Train] Complete. "
          f"Loss: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f} "
          f"({((epoch_losses[-1] - epoch_losses[0]) / epoch_losses[0]) * 100:.1f}%)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        adapter.save(save_path)

    return adapter, epoch_losses


# ===========================================================================
# SECTION 6 — EVALUATION METRICS
# ===========================================================================

def hit_rate(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    """Hit@k: 1.0 if ground_truth appears in top-k results, else 0.0."""
    return 1.0 if ground_truth in retrieved_docs[:k] else 0.0


def reciprocal_rank(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    """Reciprocal rank of the first relevant document within top-k."""
    try:
        rank = retrieved_docs.index(ground_truth) + 1
        return 1.0 / rank if rank <= k else 0.0
    except ValueError:
        return 0.0


def encode_query(
    query: str,
    base_model: SentenceTransformer,
    adapter: Optional[DsQoLA] = None
) -> np.ndarray:
    """
    Encode a query string, optionally passing through the adapter.

    Args:
        query:      Query string.
        base_model: Frozen base sentence transformer.
        adapter:    Trained DsQoLA adapter, or None for baseline.

    Returns:
        1D numpy embedding vector.
    """
    emb = base_model.encode(query, convert_to_tensor=True)
    if adapter is not None:
        device = next(adapter.parameters()).device
        emb    = emb.to(device)
        with torch.no_grad():
            emb = adapter(emb)
    return emb.cpu().detach().numpy()


def retrieve_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_texts: List[str],
    k: int = 10
) -> List[str]:
    """
    Retrieve top-k passages by cosine similarity.

    Embeddings are assumed to be L2-normalised, so cosine similarity
    reduces to dot product.
    """
    scores       = corpus_embeddings @ query_embedding
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [corpus_texts[i] for i in top_k_indices]


def evaluate_adapter(
    validation_data: List[Dict],
    base_model: SentenceTransformer,
    adapter: Optional[DsQoLA],
    corpus_embeddings: np.ndarray,
    corpus_texts: List[str],
    k_list: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance over the full validation set.

    For each query/passage pair, all 20 synthetic questions are encoded
    and their embeddings averaged to form a single robust query vector.

    Args:
        validation_data:    Validation records.
        base_model:         Frozen base sentence transformer.
        adapter:            DsQoLA adapter or None for baseline.
        corpus_embeddings:  Pre-computed document embeddings (n_docs, dim).
        corpus_texts:       Corresponding passage strings.
        k_list:             List of k values for Hit@k. Default: [1, 5, 10].

    Returns:
        Dict with keys Hit@1, Hit@5, Hit@10, MRR.
    """
    max_k      = max(k_list)
    hit_scores = {k: [] for k in k_list}
    rr_scores  = []
    label      = "Adapter" if adapter is not None else "Base (no adapter)"
    print(f"\n[Eval] {label} | {len(validation_data):,} samples")

    for item in validation_data:
        ground_truth  = item.get('input_chunk', '')
        questions     = item.get('questions', {})
        if not ground_truth or not questions:
            continue

        question_list = list(questions.values()) if isinstance(questions, dict) else questions

        # Encode all questions and average for a robust query representation
        q_embs         = np.array([encode_query(q, base_model, adapter)
                                   for q in question_list])
        query_embedding = np.mean(q_embs, axis=0)

        retrieved = retrieve_top_k(query_embedding, corpus_embeddings,
                                   corpus_texts, k=max_k)

        for k in k_list:
            hit_scores[k].append(hit_rate(retrieved, ground_truth, k))
        rr_scores.append(reciprocal_rank(retrieved, ground_truth, max_k))

    results = {f"Hit@{k}": float(np.mean(hit_scores[k])) for k in k_list}
    results["MRR"] = float(np.mean(rr_scores))

    print(f"  {'Metric':<10} {'Score':>8}")
    print("  " + "-" * 20)
    for metric, score in results.items():
        print(f"  {metric:<10} {score:>8.4f}")

    return results


def compare_base_vs_adapter(
    validation_data: List[Dict],
    base_model: SentenceTransformer,
    adapter: DsQoLA,
    corpus_embeddings: np.ndarray,
    corpus_texts: List[str],
    k_list: List[int] = [1, 5, 10]
) -> Dict:
    """
    Run baseline and adapter evaluations and print a comparison table.

    Returns:
        Dict with keys 'base' and 'adapter'.
    """
    base_results    = evaluate_adapter(validation_data, base_model, None,
                                       corpus_embeddings, corpus_texts, k_list)
    adapter_results = evaluate_adapter(validation_data, base_model, adapter,
                                       corpus_embeddings, corpus_texts, k_list)

    print(f"\n  {'Metric':<10} {'Base':>8} {'Adapter':>10} {'Abs':>8} {'Rel%':>8}")
    print("  " + "-" * 50)
    for metric in base_results:
        b   = base_results[metric]
        a   = adapter_results[metric]
        ab  = a - b
        rel = (ab / b * 100) if b > 0 else 0.0
        print(f"  {metric:<10} {b:>8.4f} {a:>10.4f} {ab:>+8.4f} {rel:>+7.1f}%")

    return {'base': base_results, 'adapter': adapter_results}


# ===========================================================================
# SECTION 7 — CORPUS UTILITIES
# ===========================================================================

def prepare_corpus_embeddings(
    corpus_texts: List[str],
    base_model: SentenceTransformer,
    batch_size: int = 64,
    cache_path: Optional[str] = None
) -> np.ndarray:
    """
    Pre-compute and optionally cache document embeddings.

    Document embeddings are computed once and remain unchanged
    throughout adapter training and inference — a core efficiency
    property of the query-only adapter design.

    Args:
        corpus_texts: Passage strings to embed.
        base_model:   Frozen sentence transformer.
        batch_size:   Encoding batch size.
        cache_path:   Optional .npy file path for caching.

    Returns:
        L2-normalised embedding matrix (n_docs, embedding_dim).
    """
    if cache_path and Path(cache_path).exists():
        print(f"[Corpus] Loading cached embeddings from: {cache_path}")
        return np.load(cache_path)

    print(f"[Corpus] Computing embeddings for {len(corpus_texts):,} passages...")
    embeddings = base_model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    if cache_path:
        np.save(cache_path, embeddings)
        print(f"[Corpus] Embeddings cached to: {cache_path}")

    return embeddings


# ===========================================================================
# SECTION 8 — VISUALISATION
# ===========================================================================

def plot_training_loss(
    epoch_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Adapter Training Loss Over Epochs"
) -> None:
    """
    Plot training loss curve over epochs (Figure 2 in paper).

    Args:
        epoch_losses: Per-epoch average loss values.
        save_path:    Optional path to save figure.
        title:        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs  = list(range(1, len(epoch_losses) + 1))

    ax.plot(epochs, epoch_losses, marker='o', color='steelblue',
            linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Average Loss', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Training loss saved to: {save_path}")
    plt.show()


def plot_model_comparison(
    results: List[Dict],
    metrics: List[str] = ['Hit@1', 'Hit@5', 'Hit@10'],
    save_path: Optional[str] = None,
    title: str = "Overall Model Evaluation with Linear Adapter"
) -> None:
    """
    Plot adapter Hit@k scores across all base models (Figure 3 in paper).

    Args:
        results:   List of dicts with keys 'model_name', 'base', 'adapter'.
        metrics:   Ordered metric names for x-axis.
        save_path: Optional path to save figure.
        title:     Plot title.
    """
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, r in enumerate(results):
        short_name     = r['model_name'].replace('sentence-transformers/', '')
        adapter_scores = [r['adapter'].get(m, 0) for m in metrics]
        ax.plot(metrics, adapter_scores,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2, markersize=7, label=short_name)

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(title='Model', fontsize=9, title_fontsize=9,
              loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Model comparison saved to: {save_path}")
    plt.show()


def print_improvement_table(results: List[Dict]) -> None:
    """Print consolidated multi-model comparison table (Table 3 in paper)."""
    metrics = ['Hit@1', 'Hit@5', 'Hit@10', 'MRR']
    print("\n" + "=" * 78)
    print("CONSOLIDATED MULTI-MODEL COMPARISON")
    print("=" * 78)
    print(f"{'Model':<30} {'Metric':<10} {'Base':>8} {'Adapter':>10} "
          f"{'Abs':>8} {'Rel%':>8}")
    print("-" * 78)

    for r in results:
        name = r['model_name'].replace('sentence-transformers/', '')
        for j, metric in enumerate(metrics):
            b    = r['base'].get(metric, 0)
            a    = r['adapter'].get(metric, 0)
            ab   = a - b
            rel  = (ab / b * 100) if b > 0 else 0.0
            col  = name if j == 0 else ''
            print(f"{col:<30} {metric:<10} {b:>8.4f} {a:>10.4f} "
                  f"{ab:>+8.4f} {rel:>+7.1f}%")
        print("-" * 78)


# ===========================================================================
# SECTION 9 — EXPERIMENT RUNNER
# ===========================================================================

# Base models evaluated in the paper (Section 4)
MODEL_CONFIGS = [
    'all-MiniLM-L6-v2',            # Small, fast baseline
    'all-MiniLM-L12-v2',           # Larger MiniLM variant
    'multi-qa-mpnet-base-dot-v1',  # Fine-tuned for QA retrieval (best absolute)
    'stsb-roberta-base',           # Trained on STS data (best relative gain)
]

# Training hyperparameters as reported in the paper
ADAPTER_KWARGS = {
    'num_epochs':          10,    # Tested 10, 30, 40 — no gain beyond 10
    'batch_size':          32,
    'learning_rate':       1e-4,
    'warmup_steps':        100,
    'max_grad_norm':       1.0,
    'triplet_margin':      0.5,
    'infonce_temperature': 0.05,
    'triplet_weight':      0.3,
}


def run_experiment(
    model_name: str,
    train_data: List[Dict],
    validation_data: List[Dict],
    negative_sampler: Callable[[], str],
    corpus_embeddings: np.ndarray,
    corpus_texts: List[str],
    output_dir: str
) -> Dict:
    """
    Run a single base model experiment:
        baseline evaluation → adapter training → adapter evaluation.

    Args:
        model_name:        HuggingFace model identifier.
        train_data:        Training records.
        validation_data:   Validation records.
        negative_sampler:  Hard negative sampler.
        corpus_embeddings: Pre-computed document embeddings.
        corpus_texts:      Corresponding passage strings.
        output_dir:        Directory for adapter weights and plots.

    Returns:
        Dict with keys 'model_name', 'base', 'adapter', 'epoch_losses'.
    """
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {model_name}")
    print(f"{'=' * 60}")

    base_model = SentenceTransformer(model_name)
    print(f"[Model] Embedding dim: {base_model.get_sentence_embedding_dimension()}")

    adapter_path = os.path.join(
        output_dir, f"adapter_{model_name.replace('/', '_')}.pth"
    )
    adapter, epoch_losses = train_adapter(
        base_model=base_model,
        train_data=train_data,
        negative_sampler=negative_sampler,
        save_path=adapter_path,
        **ADAPTER_KWARGS
    )

    # Plot training loss for this model
    plot_training_loss(
        epoch_losses,
        save_path=os.path.join(
            output_dir, f"loss_{model_name.replace('/', '_')}.png"
        ),
        title=f"Training Loss — {model_name}"
    )

    comparison = compare_base_vs_adapter(
        validation_data=validation_data,
        base_model=base_model,
        adapter=adapter,
        corpus_embeddings=corpus_embeddings,
        corpus_texts=corpus_texts,
        k_list=[1, 5, 10]
    )

    return {
        'model_name':   model_name,
        'base':         comparison['base'],
        'adapter':      comparison['adapter'],
        'epoch_losses': epoch_losses,
    }


# ===========================================================================
# SECTION 10 — MAIN ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='DsQoLA adapter experiments — LiLADH CIKM 2026'
    )
    parser.add_argument('--train_path',  type=str, required=True,
                        help='Path to training JSONL file.')
    parser.add_argument('--val_path',    type=str, required=True,
                        help='Path to validation JSONL file.')
    parser.add_argument('--neg_path',    type=str, required=True,
                        help='Path to negative corpus (one sentence per line).')
    parser.add_argument('--output_dir',  type=str, default='outputs/',
                        help='Directory for adapter weights and plots.')
    parser.add_argument('--models',      nargs='+', default=MODEL_CONFIGS,
                        help='Base model names to evaluate.')
    parser.add_argument('--embed_cache', type=str, default=None,
                        help='Optional .npy path to cache corpus embeddings.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    train_data      = load_jsonl_data(args.train_path)
    validation_data = load_jsonl_data(args.val_path)
    neg_sampler     = create_negative_sampler(args.neg_path)
    corpus_texts    = [item['input_chunk']
                       for item in train_data + validation_data]
    print(f"[Corpus] Total passages: {len(corpus_texts):,}")

    all_results = []

    for model_name in args.models:
        # Pre-compute corpus embeddings for this base model
        base_model_for_corpus = SentenceTransformer(model_name)
        cache_path = (
            args.embed_cache.replace(
                '.npy', f'_{model_name.replace("/", "_")}.npy'
            ) if args.embed_cache else None
        )
        corpus_embeddings = prepare_corpus_embeddings(
            corpus_texts=corpus_texts,
            base_model=base_model_for_corpus,
            cache_path=cache_path
        )

        result = run_experiment(
            model_name=model_name,
            train_data=train_data,
            validation_data=validation_data,
            negative_sampler=neg_sampler,
            corpus_embeddings=corpus_embeddings,
            corpus_texts=corpus_texts,
            output_dir=args.output_dir
        )
        all_results.append(result)

    # Final consolidated table and comparison plot
    print_improvement_table(all_results)
    plot_model_comparison(
        results=all_results,
        save_path=os.path.join(args.output_dir, 'model_comparison.png')
    )
    print(f"\n[Done] All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
