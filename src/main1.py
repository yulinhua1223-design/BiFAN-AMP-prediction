from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
import os
import random
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from amp_model import AmpPredictor  # noqa: E402
from data_utils import (  # noqa: E402
    DynamicBatchCollator,
    build_dataset_from_fasta,
    prepare_datasets,
)
from fusion_variants import FUSION_NAMES  # noqa: E402
from metrics_utils import compute_classification_metrics, select_threshold  # noqa: E402

REPORT_KEYS = ("ACC", "MCC", "AUC", "AUPRC", "F1", "Sn", "Sp", "BACC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiFAN single-run training and evaluation")
    parser.add_argument("--dataset", default="genpept", help="dataset tag; default split dir is data/<tag>")
    parser.add_argument("--data-dir", default=None, help="override the split directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esm-path", default="models/esm2_t33_650M_UR50D",
                        help="local ESM-2 directory")
    parser.add_argument("--out-dir", default=None, help="default runs/bifan_<dataset>_<variant>_seed<seed>")
    parser.add_argument("--fusion", default="add", choices=list(FUSION_NAMES))
    parser.add_argument("--branch", default="full", choices=["full", "esm_only", "hand_only"],
                        help="which branch feeds the classifier (component ablation)")
    parser.add_argument("--aux-lambda", type=float, default=0.0,
                        help="weight of the auxiliary loss on the handcrafted branch")
    parser.add_argument("--tag", default=None, help="override the variant tag used in the output directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--cksaagp-gap", type=int, default=3)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-checkpoint", action="store_true",
                        help="keep best_model.pth (about 2.6 GB) after evaluation")
    parser.add_argument("--max-train-samples", type=int, default=0, help="debug: truncate the training split")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="debug: truncate val/test splits")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"],
                        help="autocast precision for the forward/backward pass")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="max gradient norm; 0 disables clipping")
    parser.add_argument("--cross-eval-fasta", nargs="*", default=None,
                        help="extra FASTA files scored with the trained model (cross-dataset)")
    parser.add_argument("--save-features", action="store_true",
                        help="dump the fused / branch representations of val+test for figures")
    return parser.parse_args()


def _autocast(dtype, device):
    """Return an autocast context manager (or a no-op context for fp32)."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def truncate(dataset, limit: int):
    if limit and limit > 0 and len(dataset) > limit:
        dataset.records = dataset.records[:limit]
        dataset.handcrafted_features = dataset.handcrafted_features[:limit]
    return dataset


def parameter_breakdown(model: AmpPredictor) -> dict:
    groups = {
        "esm_backbone": model.esm,
        "attention_pooling": model.attention_query,
        "sequence_projection": model.global_projection,
        "handcrafted_encoder": model.handcrafted_encoder,
        "fusion": model.fusion_engine,
        "classifier": model.classifier,
    }
    if model.handcrafted_head is not None:
        groups["handcrafted_auxiliary_head"] = model.handcrafted_head
    return {name: sum(p.numel() for p in block.parameters()) for name, block in groups.items()}


def run_epoch(model, loader, optimizer, criterion, device, is_train=True, threshold=0.5,
              aux_lambda=0.0, autocast_dtype=None, grad_clip=1.0, collect_features=False):
    model.train(is_train)
    total_loss = 0.0
    all_logits, all_labels, all_ids = [], [], []
    all_aux = []
    all_fused, all_global, all_hand = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        h_features = batch["handcrafted_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train), _autocast(autocast_dtype, device):
            result = model(input_ids, attention_mask, h_features)
            logits = result["logits"]
            loss = criterion(logits, labels)
            aux_logits = result.get("aux_logits")
            if is_train and aux_lambda > 0.0 and aux_logits is not None:
                loss = loss + aux_lambda * criterion(aux_logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        total_loss += float(loss.item())
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())
        all_ids.extend(batch["identifiers"])
        if aux_logits is not None:
            all_aux.append(aux_logits.detach().float().cpu())
        if collect_features:
            all_fused.append(result["fused_feature"].detach().float().cpu())
            if result.get("global_feature") is not None:
                all_global.append(result["global_feature"].detach().float().cpu())
            all_hand.append(result["handcrafted_feature"].detach().float().cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(logits_tensor.float(), dim=1)[:, 1].numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    metrics = compute_classification_metrics(labels_np, probabilities, threshold=threshold)
    metrics["loss"] = total_loss / max(len(loader), 1)

    aux_probabilities = None
    if all_aux:
        aux_tensor = torch.cat(all_aux, dim=0)
        aux_probabilities = torch.softmax(aux_tensor.float(), dim=1)[:, 1].numpy()

    features = None
    if collect_features:
        features = {
            "fused": torch.cat(all_fused, dim=0).numpy(),
            "handcrafted": torch.cat(all_hand, dim=0).numpy(),
            "global": torch.cat(all_global, dim=0).numpy() if all_global else None,
        }

    return {
        "metrics": metrics,
        "probs": probabilities,
        "labels": labels_np,
        "ids": all_ids,
        "aux_probs": aux_probabilities,
        "features": features,
    }


def plot_single_roc(labels, probs, output_path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()


def variant_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    if args.branch != "full":
        return args.branch
    tag = args.fusion
    if args.aux_lambda > 0.0:
        tag += f"_aux{args.aux_lambda:g}"
    return tag


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir) if args.data_dir else Path("data") / args.dataset
    tag = variant_tag(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"bifan_{args.dataset}_{tag}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {name: data_dir / f"{name}.fasta" for name in ("train", "val", "test")}
    for name, path in splits.items():
        if not path.exists():
            raise SystemExit(f"missing {name} split: {path}")

    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"device  : {device}")
    print(f"dataset : {args.dataset} ({data_dir})")
    print(f"variant : {tag} | fusion={args.fusion} branch={args.branch} aux_lambda={args.aux_lambda}")
    print(f"seed    : {args.seed}")
    print(f"out dir : {out_dir}")

    autocast_dtype = torch.bfloat16 if (args.precision == "bf16" and device.type == "cuda") else None
    print(f"precision: {args.precision}")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    collator = DynamicBatchCollator(tokenizer=tokenizer, max_length=args.max_length)

    print("loading data ...")
    train_ds, _, test_ds, feature_names, scaler = prepare_datasets(
        train_fasta=splits["train"],
        test_fasta=splits["test"],
        tokenizer=tokenizer,
        valid_size=0.0,
        max_length=args.max_length,
        cksaagp_gap=args.cksaagp_gap,
        seed=args.seed,
    )
    # the validation split must reuse the training scaler, otherwise the
    # selection of the decision threshold would leak evaluation statistics
    valid_ds = build_dataset_from_fasta(
        splits["val"],
        tokenizer=tokenizer,
        scaler=scaler,
        max_length=args.max_length,
        cksaagp_gap=args.cksaagp_gap,
    )
    truncate(train_ds, args.max_train_samples)
    truncate(valid_ds, args.max_eval_samples)
    truncate(test_ds, args.max_eval_samples)

    print(f"splits  : train={len(train_ds)} | val={len(valid_ds)} | test={len(test_ds)} | handcrafted={len(feature_names)}")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    loader_kwargs = {"batch_size": args.batch_size, "collate_fn": collator,
                     "num_workers": args.num_workers,
                     "pin_memory": device.type == "cuda",
                     "persistent_workers": args.num_workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = AmpPredictor(
        args.esm_path,
        len(feature_names),
        fusion=args.fusion,
        branch=args.branch,
        aux_lambda=args.aux_lambda,
    ).to(device)

    breakdown = parameter_breakdown(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params  : total={n_params / 1e6:.2f}M trainable={n_trainable / 1e6:.2f}M")
    for name, count in breakdown.items():
        print(f"          {name:<26} {count / 1e6:.4f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_mcc = -1.0
    best_model_path = out_dir / "best_model.pth"
    counter = 0
    history = []
    started = time.time()

    for epoch in range(args.epochs):
        train_out = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True,
                              aux_lambda=args.aux_lambda, autocast_dtype=autocast_dtype,
                              grad_clip=args.grad_clip)
        valid_out = run_epoch(model, valid_loader, None, criterion, device, is_train=False,
                              autocast_dtype=autocast_dtype)
        train_m, valid_m = train_out["metrics"], valid_out["metrics"]

        history.append({"epoch": epoch + 1, "train": train_m, "val": valid_m})
        print(
            f"epoch {epoch + 1:02d} | loss {train_m['loss']:.4f} | "
            f"train MCC {train_m['MCC']:.4f} | val MCC {valid_m['MCC']:.4f} | val AUC {valid_m['AUC']:.4f}"
        )

        if valid_m["MCC"] > best_val_mcc:
            best_val_mcc = valid_m["MCC"]
            torch.save(model.state_dict(), best_model_path)
            counter = 0
            print(f"  --> new best val MCC {best_val_mcc:.4f} (checkpoint saved)")
        else:
            counter += 1

        if counter >= args.patience:
            print(f"early stopping at epoch {epoch + 1}")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_seconds = time.time() - started
    print(f"training finished in {train_seconds / 60:.1f} min")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    # threshold selection follows the GenPept-Curated-2025 protocol: maximise MCC
    # on the validation split, then freeze tau and apply it to the test split
    valid_out = run_epoch(model, valid_loader, None, criterion, device, is_train=False,
                          autocast_dtype=autocast_dtype)
    tau_star, val_metrics_star = select_threshold(valid_out["labels"], valid_out["probs"])

    if args.save_features:
        valid_features = run_epoch(model, valid_loader, None, criterion, device, is_train=False,
                                   autocast_dtype=autocast_dtype, collect_features=True)
        test_features = run_epoch(model, test_loader, None, criterion, device, is_train=False,
                                  autocast_dtype=autocast_dtype, collect_features=True)
        payload = {
            "val_fused": valid_features["features"]["fused"],
            "val_handcrafted": valid_features["features"]["handcrafted"],
            "val_labels": valid_features["labels"],
            "val_probs": valid_features["probs"],
            "test_fused": test_features["features"]["fused"],
            "test_handcrafted": test_features["features"]["handcrafted"],
            "test_labels": test_features["labels"],
            "test_probs": test_features["probs"],
        }
        if test_features["features"]["global"] is not None:
            payload["val_global"] = valid_features["features"]["global"]
            payload["test_global"] = test_features["features"]["global"]
        np.savez(out_dir / "features.npz", **payload)
        print(f"representations written to {out_dir / 'features.npz'}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_started = time.time()
    test_out = run_epoch(model, test_loader, None, criterion, device, is_train=False,
                         autocast_dtype=autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.time() - inference_started
    test_metrics_star = compute_classification_metrics(test_out["labels"], test_out["probs"], threshold=tau_star)
    test_metrics_at_05 = compute_classification_metrics(test_out["labels"], test_out["probs"], threshold=0.5)

    print("=" * 60)
    print(f"validation-selected threshold tau* = {tau_star:.4f}")
    for label, metrics in (("tau*", test_metrics_star), ("tau=0.5", test_metrics_at_05)):
        print(f"held-out test split ({label})")
        for key in REPORT_KEYS:
            print(f" - {key:<6}: {metrics[key]:.4f}")

    aux_evidence = None
    if valid_out["aux_probs"] is not None:
        aux_tau, aux_metrics = select_threshold(valid_out["labels"], valid_out["aux_probs"])
        aux_test = compute_classification_metrics(test_out["labels"], test_out["aux_probs"], threshold=aux_tau)
        aux_evidence = {"tau": aux_tau, "val": aux_metrics, "test": aux_test}
        print(f"handcrafted-branch auxiliary head on the test set: MCC {aux_test['MCC']:.4f} | AUC {aux_test['AUC']:.4f}")

    cross_eval = {}
    for cross_entry in (args.cross_eval_fasta or []):
        cross_path = Path(cross_entry)
        if not cross_path.exists():
            print(f"[warn] cross-eval file missing: {cross_path}")
            continue
        cross_ds = build_dataset_from_fasta(cross_path, tokenizer=tokenizer, scaler=scaler,
                                            max_length=args.max_length, cksaagp_gap=args.cksaagp_gap)
        cross_loader = DataLoader(cross_ds, shuffle=False, **loader_kwargs)
        cross_out = run_epoch(model, cross_loader, None, criterion, device, is_train=False,
                              autocast_dtype=autocast_dtype)
        cross_at_tau = compute_classification_metrics(cross_out["labels"], cross_out["probs"],
                                                      threshold=tau_star)
        cross_at_half = compute_classification_metrics(cross_out["labels"], cross_out["probs"],
                                                       threshold=0.5)
        cross_eval[cross_path.stem] = {
            "source": str(cross_path),
            "n": len(cross_ds),
            "at_tau_star": cross_at_tau,
            "at_0.5": cross_at_half,
        }
        np.savez(out_dir / f"cross_predictions_{cross_path.stem}.npz",
                 ids=np.array(cross_out["ids"]), probs=cross_out["probs"],
                 labels=cross_out["labels"])
        print(f"cross-eval {cross_path.name}: n={len(cross_ds)} | MCC {cross_at_tau['MCC']:.4f} | "
              f"AUC {cross_at_tau['AUC']:.4f} | Sn {cross_at_tau['Sn']:.4f} | Sp {cross_at_tau['Sp']:.4f}")

    report = {
        "dataset": args.dataset,
        "cross_eval": cross_eval,
        "inference_seconds": inference_seconds,
        "inference_throughput_per_second": len(test_ds) / max(inference_seconds, 1e-9),
        "peak_gpu_gib": (torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == "cuda" else 0.0,
        "variant": tag,
        "fusion": args.fusion,
        "branch": args.branch,
        "aux_lambda": args.aux_lambda,
        "seed": args.seed,
        "tau_star": tau_star,
        "best_val_mcc": best_val_mcc,
        "train_seconds": train_seconds,
        "epochs_run": len(history),
        "n_params": n_params,
        "n_trainable": n_trainable,
        "parameter_breakdown": breakdown,
        "train_size": len(train_ds),
        "val_size": len(valid_ds),
        "test_size": len(test_ds),
        "handcrafted_dim": len(feature_names),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "esm_path": args.esm_path,
        "test_at_tau_star": test_metrics_star,
        "test_at_0.5": test_metrics_at_05,
        "val_at_tau_star": val_metrics_star,
        "handcrafted_branch": aux_evidence,
        "config": vars(args),
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "predictions.npz",
        val_ids=np.array(valid_out["ids"]),
        val_probs=valid_out["probs"],
        val_labels=valid_out["labels"],
        test_ids=np.array(test_out["ids"]),
        test_probs=test_out["probs"],
        test_labels=test_out["labels"],
    )
    plot_single_roc(test_out["labels"], test_out["probs"], out_dir / "test_roc_curve.png")

    # the checkpoint is only needed to reload the best weights above; keeping one
    # per run would cost ~2.6 GB each, which adds up fast across a seed sweep
    if not args.keep_checkpoint:
        best_model_path.unlink(missing_ok=True)
        print("checkpoint removed (pass --keep-checkpoint to keep it)")

    print(f"results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
