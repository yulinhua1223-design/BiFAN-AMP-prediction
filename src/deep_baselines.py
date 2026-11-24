"""Architecture-level deep baselines using frozen ESM-2 embeddings.

Architecture-level re-implementations of published deep AMP-classifier
designs, evaluated under the same local protocol as BiFAN: identical FASTA
splits, identical ESM-2 backbone, validation-selected decision threshold, and
mean +- std over the same seeds. They are not presented as official
reproductions of the original authors' code.

    mlp          mean-pooled ESM-2 embedding -> 2-layer MLP (ESM-2 probe)
    amplify      AMPlify (Li et al., BMC Genomics 2023): parallel 1-D
                 convolutions (kernels 1/3/5/7, 256 filters) + global max
                 pooling + 3 fully connected layers
    bilstm       BiLSTM over per-residue embeddings + mean/max pooling
    transformer  2-layer Transformer encoder + attention pooling
                 (AMP-BERT style, frozen encoder)

Usage:
    python deep_baselines.py --datasets genpept inhouse --seeds 100 101 102
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from fasta_utils import parse_fasta
from metrics_utils import compute_classification_metrics, select_threshold

MODELS = ("mlp", "amplify", "bilstm", "transformer")
KEYS = ("ACC", "MCC", "AUC", "AUPRC", "F1", "Sn", "Sp", "BACC")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def embed_split(path: Path, tokenizer, encoder, device, max_length: int, batch_size: int,
                limit: int = 0):
    """Return per-residue ESM-2 embeddings (special tokens removed) + labels."""
    records = parse_fasta(path)
    if limit:
        records = records[:limit]
    labels = np.array([record.label for record in records], dtype=np.int64)
    embeddings = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        encoded = tokenizer([record.sequence for record in chunk], return_tensors="pt",
                            padding=True, truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            hidden = encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
        hidden = hidden.float().cpu()
        counts = mask.sum(dim=1).tolist()
        for row, count in enumerate(counts):
            count = int(count)
            if count <= 2:
                embeddings.append(hidden[row, :1].to(torch.float16))
            else:
                embeddings.append(hidden[row, 1:count - 1].to(torch.float16))
    return embeddings, labels


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.embeddings[index], int(self.labels[index])


def collate(batch):
    vectors = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([v.shape[0] for v in vectors], dtype=torch.long)
    width = int(lengths.max())
    dim = vectors[0].shape[1]
    padded = torch.zeros(len(vectors), width, dim, dtype=torch.float32)
    mask = torch.zeros(len(vectors), width, dtype=torch.bool)
    for row, vector in enumerate(vectors):
        length = vector.shape[0]
        padded[row, :length] = vector.float()
        mask[row, :length] = True
    return {"embeddings": padded, "mask": mask, "labels": labels, "lengths": lengths}


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        pooled = (batch["embeddings"] * batch["mask"].unsqueeze(-1)).sum(1)
        pooled = pooled / batch["lengths"].unsqueeze(-1)
        return self.net(pooled)


class AMPlifyNet(nn.Module):
    """AMPlify-style head: parallel convolutions with global max pooling."""

    def __init__(self, dim: int, filters: int = 256, kernels=(1, 3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, filters, kernel, padding=kernel // 2) for kernel in kernels])
        self.net = nn.Sequential(
            nn.Linear(filters * len(kernels), 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        x = (batch["embeddings"] * batch["mask"].unsqueeze(-1)).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            feature = torch.relu(conv(x))
            feature = feature.masked_fill(~batch["mask"].unsqueeze(1), -1e9)
            pooled.append(feature.max(dim=2).values)
        return self.net(torch.cat(pooled, dim=1))


class BiLSTMNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.net = nn.Sequential(
            nn.Linear(hidden * 4, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, batch):
        out, _ = self.lstm(batch["embeddings"])
        mask = batch["mask"].unsqueeze(-1)
        mean = (out * mask).sum(1) / batch["lengths"].unsqueeze(-1)
        peak = out.masked_fill(~mask, -1e9).max(dim=1).values
        return self.net(torch.cat([mean, peak], dim=1))


class TransformerNet(nn.Module):
    def __init__(self, dim: int, layers: int = 2, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.query = nn.Linear(dim, 1)
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, batch):
        mask = batch["mask"]
        x = self.encoder(batch["embeddings"], src_key_padding_mask=~mask)
        scores = self.query(x).masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled = (x * torch.softmax(scores, dim=1)).sum(dim=1)
        return self.net(pooled)


def build_model(name: str, dim: int) -> nn.Module:
    if name == "mlp":
        return MLPHead(dim)
    if name == "amplify":
        return AMPlifyNet(dim)
    if name == "bilstm":
        return BiLSTMNet(dim)
    if name == "transformer":
        return TransformerNet(dim)
    raise ValueError(f"unknown model {name!r}")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        logits = model({key: value.to(device) for key, value in batch.items()})
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def train_and_evaluate(name, dim, packs, device, seed, epochs, lr, batch_size, patience,
                       num_workers):
    set_seed(seed)
    loaders = {}
    for split, (embeddings, labels) in packs.items():
        loaders[split] = DataLoader(
            EmbeddingDataset(embeddings, labels), batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=collate, num_workers=num_workers,
            pin_memory=(device.type == "cuda"))

    model = build_model(name, dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_mcc, best_state, counter = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        probabilities, labels = predict(model, loaders["val"], device)
        mcc = compute_classification_metrics(labels, probabilities, threshold=0.5)["MCC"]
        if mcc > best_mcc:
            best_mcc = mcc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probabilities, val_labels = predict(model, loaders["val"], device)
    tau, _ = select_threshold(val_labels, val_probabilities)
    test_probabilities, test_labels = predict(model, loaders["test"], device)
    metrics = compute_classification_metrics(test_labels, test_probabilities, threshold=float(tau))
    return (metrics, float(tau), float(best_mcc), test_probabilities, test_labels,
            val_probabilities, val_labels)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Architecture-level deep baselines using frozen ESM-2 embeddings"
    )
    parser.add_argument("--datasets", nargs="+", default=["genpept"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--esm-path", default="models/esm2_t33_650M_UR50D")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="results/deep")
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap each split")
    parser.add_argument("--dump-npz", action="store_true",
                        help="also store per-sequence val/test probabilities (never overwrites json)")
    parser.add_argument("--npz-dir", default="results/deep_preds")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)
    if args.dump_npz:
        npz_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    encoder = AutoModel.from_pretrained(args.esm_path).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    dim = encoder.config.hidden_size
    print(f"encoder: {args.esm_path} | hidden {dim} | device {device}", flush=True)

    for dataset in args.datasets:
        split_dir = Path(args.data_root) / dataset
        packs = {}
        for split in ("train", "val", "test"):
            started = time.time()
            embeddings, labels = embed_split(split_dir / f"{split}.fasta", tokenizer, encoder,
                                             device, args.max_length, args.embed_batch_size,
                                             limit=args.limit)
            packs[split] = (embeddings, labels)
            print(f"[{dataset}] embedded {split}: n={len(labels)} in {time.time() - started:.1f}s",
                  flush=True)

        for model_name in args.models:
            for seed in args.seeds:
                path = out_dir / f"{dataset}_{model_name}_seed{seed}.json"
                npz_path = npz_dir / f"{dataset}_{model_name}_seed{seed}.npz"
                if path.exists() and not (args.dump_npz and not npz_path.exists()):
                    print(f"  {dataset:<10} {model_name:<12} seed {seed} | [skip] already done",
                          flush=True)
                    continue
                if path.exists() and args.dump_npz and not npz_path.exists():
                    print(f"  {dataset:<10} {model_name:<12} seed {seed} | "
                          f"[npz only] recomputing to store per-sequence predictions", flush=True)
                started = time.time()
                (metrics, tau, best_val, test_probabilities, test_labels,
                 val_probabilities, val_labels) = train_and_evaluate(
                    model_name, dim, packs, device, seed, args.epochs, args.lr,
                    args.batch_size, args.patience, args.num_workers)
                payload = {
                    "dataset": dataset,
                    "representation": f"esm2_frozen_{dim}d",
                    "model_id": model_name,
                    "seed": seed,
                    "tau": tau,
                    "seconds": time.time() - started,
                    "n_train": len(packs["train"][1]),
                    "n_val": len(packs["val"][1]),
                    "n_test": len(packs["test"][1]),
                    "best_val_MCC": best_val,
                    "test_at_tau_star": metrics,
                    "test": metrics,
                }
                if not path.exists():
                    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                if args.dump_npz:
                    np.savez(npz_path, val_probs=val_probabilities, val_labels=val_labels,
                             test_probs=test_probabilities, test_labels=test_labels)
                print(f"  {dataset:<10} {model_name:<12} seed {seed} | "
                      f"MCC {metrics['MCC']:.4f} | AUC {metrics['AUC']:.4f} | "
                      f"ACC {metrics['ACC']:.4f} | {time.time() - started:.0f}s", flush=True)
    print("\ndeep baselines finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
