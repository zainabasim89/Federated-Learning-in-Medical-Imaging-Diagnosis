import sys
import os
import json
import multiprocessing

sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from dataset import EarDiseaseDataset, TransformSubset, get_transforms, set_seed
from model import get_model
from fl_server import evaluate_on_test_set   # <-- SAME metric code as FL runs

# ═══════════════════════════════════════════════════════════
# HYPERPARAMETERS — imported directly from run_flower.py so
# there is a SINGLE source of truth. This guarantees the
# centralized baseline can never silently drift from the FL
# settings (model, lr, batch size, patience, seed, backbone
# freezing, dataset root all come from the same place).
# ═══════════════════════════════════════════════════════════
from run_flower import (
    ROOT, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR,
    ES_PATIENCE, SEED, FREEZE_BACKBONE,
)

# ── Centralized epoch budget ─────────────────────────────────
# Set directly here — independent of the FL run's NUM_ROUNDS /
# LOCAL_EPOCHS, so you can do a quick sanity-check pass first
# and switch to the full budget for your final paper run
# without touching anything else.
#
# CURRENTLY SET FOR A QUICK 10-EPOCH SANITY CHECK.
# For the final paper run, the standard, defensible convention
# used for FL-vs-centralized comparisons is to match the total
# epoch budget an FL client would see across all rounds:
#
#     CENTRAL_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS
#
# With the current run_flower.py settings (NUM_ROUNDS=3,
# LOCAL_EPOCHS=20) that would be 60. Early stopping (same
# ES_PATIENCE as FL) will cut this short anyway if validation
# loss stops improving, exactly as in fl_client.py, so it's
# safe to set this generously for the final run.
CENTRAL_EPOCHS = 10   # <-- quick sanity-check run; switch to NUM_ROUNDS * LOCAL_EPOCHS for the final paper run


VAL_RATIO = 0.15   # same val_ratio used in dirichlet/iid partitioning

RESULTS = os.path.join(os.path.dirname(__file__), 'results_centralized_4class')
os.makedirs(RESULTS, exist_ok=True)


# =========================================================
# STRATIFIED TRAIN/VAL SPLIT OF THE FULL POOL
# Same stratification strategy as dataset.iid_partition's
# per-client split, just applied once to the whole training
# pool instead of once per client (centralized = one "client"
# holding all the data).
# =========================================================
def stratified_train_val_split(dataset, val_ratio, seed):
    labels = np.array([s[1] for s in dataset.samples])
    indices = np.arange(len(dataset))

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    train_pos, val_pos = next(splitter.split(indices, labels))
    return indices[train_pos].tolist(), indices[val_pos].tolist()


# =========================================================
# DATA / CLASS BREAKDOWN SUMMARY (mirrors save_partition_summary
# in fl_server.py, but for the single centralized pool)
# =========================================================
def save_centralized_summary(train_ds, val_ds, test_ds, full_dataset,
                             results_dir):
    os.makedirs(results_dir, exist_ok=True)
    class_names = full_dataset.class_names()

    def count_breakdown(subset):
        per_class = {c: 0 for c in class_names}
        base_ds = subset.dataset
        for i in subset.indices:
            _, _, cname = base_ds.samples[i]
            per_class[cname] += 1
        return per_class, sum(per_class.values())

    rows = []
    print(f"\n{'='*70}")
    print(f"  CENTRALIZED BASELINE — DATA SUMMARY")
    print(f"{'='*70}")
    for split_name, subset in [("Train", train_ds), ("Val", val_ds),
                               ("Test", test_ds)]:
        per_class, total = count_breakdown(subset)
        counts_str = " ".join(f"{cn[:10]}:{per_class[cn]}"
                              for cn in class_names)
        print(f"  {split_name:<6} {total:>5}  ({counts_str})")
        row = {"Split": split_name, "Total": total}
        row.update(per_class)
        rows.append(row)
    print(f"{'='*70}")

    pd.DataFrame(rows).to_excel(
        os.path.join(results_dir, "centralized_data_summary.xlsx"),
        index=False
    )


# =========================================================
# TRAINING LOOP — identical logic to EarDiseaseFlowerClient.fit
# (same optimizer, same loss, same early-stopping rule), just
# run once on the full pooled training set instead of per-round
# on a client shard.
# =========================================================
def train_centralized(model, train_ds, val_ds, test_ds, device,
                      num_classes, class_names, results_dir):

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    best_val_loss  = float('inf')
    patience_count = 0
    best_weights   = copy.deepcopy(model.state_dict())

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_accuracy": [], "test_f1_macro": [], "test_f1_weighted": [],
        "test_precision_macro": [], "test_recall_macro": [],
        "test_auc_macro_ovr": [], "test_f1_per_class": [],
        "test_confusion_matrix": [],
    }

    csv_path = os.path.join(results_dir, "epoch_metrics.csv")
    pd.DataFrame(columns=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "test_accuracy", "test_f1_macro", "test_f1_weighted",
        "test_precision_macro", "test_recall_macro", "test_auc_macro_ovr",
    ]).to_csv(csv_path, index=False)

    print(f"\n[Centralized] Training for up to {CENTRAL_EPOCHS} epochs "
          f"(ES patience={ES_PATIENCE})")

    model.train()
    for epoch in range(CENTRAL_EPOCHS):
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(loader)

        # ── Validation ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.long().to(device)
                logits = model(images)
                v_loss += loss_fn(logits, labels).item()
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / v_total if v_total > 0 else 0.0
        val_loss = v_loss / max(len(val_loader), 1)
        model.train()

        # ── Test evaluation every epoch (dataset is small, cheap;
        #    gives a convergence curve directly comparable to the
        #    FL round-by-round curves) ──
        tm = evaluate_on_test_set(model, test_ds, device, num_classes,
                                  batch_size=BATCH_SIZE)

        print(f"  Ep {epoch+1}/{CENTRAL_EPOCHS} | "
              f"Loss:{avg_loss:.4f} TrAcc:{train_acc:.1f}% | "
              f"ValLoss:{val_loss:.4f} ValAcc:{val_acc*100:.1f}% | "
              f"TestAcc:{tm['accuracy']*100:.1f}%", end="")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_count = 0
            best_weights = copy.deepcopy(model.state_dict())
            print(" *")
        else:
            patience_count += 1
            print(f" [{patience_count}/{ES_PATIENCE}]")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc / 100.0)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        for key in ["accuracy", "f1_macro", "f1_weighted",
                    "precision_macro", "recall_macro", "auc_macro_ovr"]:
            history[f"test_{key}"].append(tm[key])
        history["test_f1_per_class"].append(tm["f1_per_class"])
        history["test_confusion_matrix"].append(tm["confusion_matrix"])

        row = {
            "epoch": epoch + 1, "train_loss": avg_loss,
            "train_acc": train_acc / 100.0, "val_loss": val_loss,
            "val_acc": val_acc, "test_accuracy": tm["accuracy"],
            "test_f1_macro": tm["f1_macro"],
            "test_f1_weighted": tm["f1_weighted"],
            "test_precision_macro": tm["precision_macro"],
            "test_recall_macro": tm["recall_macro"],
            "test_auc_macro_ovr": tm["auc_macro_ovr"],
        }
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False,
                                   index=False)

        with open(os.path.join(results_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if patience_count >= ES_PATIENCE:
            print(f"  [EarlyStop] epoch {epoch+1}")
            break

    # ── Restore best weights, final report ──
    model.load_state_dict(best_weights)
    final_tm = evaluate_on_test_set(model, test_ds, device, num_classes,
                                    batch_size=BATCH_SIZE)

    torch.save(model.state_dict(),
              os.path.join(results_dir, "final_model.pth"))

    df = pd.read_csv(csv_path)
    df.to_excel(os.path.join(results_dir, "epoch_metrics.xlsx"), index=False)

    _save_convergence_plot(history, results_dir)
    _save_confusion_matrix_plot(final_tm, class_names, results_dir)

    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS — held-out test set  [Centralized]")
    print(f"{'='*55}")
    print(f"  Accuracy          : {final_tm['accuracy']:.4f}")
    print(f"  F1 (macro)        : {final_tm['f1_macro']:.4f}")
    print(f"  F1 (weighted)     : {final_tm['f1_weighted']:.4f}")
    print(f"  Precision (macro) : {final_tm['precision_macro']:.4f}")
    print(f"  Recall (macro)    : {final_tm['recall_macro']:.4f}")
    print(f"  AUC (macro OVR)   : {final_tm['auc_macro_ovr']:.4f}")
    for cname, f1c in zip(class_names, final_tm['f1_per_class']):
        print(f"    F1 [{cname}]: {f1c:.4f}")
    print(f"{'='*55}")
    print(f"  Saved to: {results_dir}/")

    return final_tm


def _save_convergence_plot(history, results_dir):
    epochs = history["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    pairs = [
        ("val_acc", "test_accuracy", "Accuracy"),
        (None, "test_f1_macro", "F1 (macro)"),
        (None, "test_f1_weighted", "F1 (weighted)"),
        (None, "test_precision_macro", "Precision (macro)"),
        (None, "test_recall_macro", "Recall (macro)"),
        (None, "test_auc_macro_ovr", "AUC (macro OVR)"),
    ]
    for i, (vm, tm, title) in enumerate(pairs):
        ax = axes[i]
        if vm and vm in history:
            ax.plot(epochs, history[vm], 'b-o', lw=2, ms=4, label='Val')
        ax.plot(epochs, history[tm], 'r-s', lw=2, ms=4, label='Test')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle(
        "Centralized Baseline — Ear Disease Classification (Chile, 4-class)\n"
        "Same model / preprocessing / hyperparameters as FL runs"
    )
    plt.tight_layout()
    path = os.path.join(results_dir, "convergence_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


def _save_confusion_matrix_plot(tm, class_names, results_dir):
    cm = np.array(tm["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Centralized Baseline (Test Set)')
    plt.tight_layout()
    path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load dataset — IDENTICAL loading path/order as run_flower.py ──
    print("\n[1] Loading training pool and locked test set...")
    train_pool_ds = EarDiseaseDataset(ROOT, split='Training')
    test_pool_ds = EarDiseaseDataset(
        ROOT, split='Testing', class_to_idx=train_pool_ds.class_to_idx
    )
    num_classes = train_pool_ds.num_classes
    class_names = train_pool_ds.class_names()

    print("\n" + "="*55)
    print("  Centralized Baseline — Ear Disease Multiclass Classification")
    print("="*55)
    print(f"  Classes        : {num_classes} -> {class_names}")
    print(f"  Epoch budget   : {CENTRAL_EPOCHS} "
          f"(independent of FL settings: NUM_ROUNDS[{NUM_ROUNDS}] x "
          f"LOCAL_EPOCHS[{LOCAL_EPOCHS}] = "
          f"{NUM_ROUNDS * LOCAL_EPOCHS} would be the full-budget value)")
    print(f"  Batch size     : {BATCH_SIZE} | LR: {LR}")
    print(f"  ES patience    : {ES_PATIENCE}")
    print(f"  Freeze backbone: {FREEZE_BACKBONE}")
    print(f"  Seed           : {SEED}")
    print("="*55)

    config = {
        "algorithm": "Centralized",
        "task": "Multiclass (4-class)",
        "num_classes": num_classes,
        "class_names": class_names,
        "central_epochs_budget": CENTRAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "es_patience": ES_PATIENCE,
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "freeze_backbone": FREEZE_BACKBONE,
        "model": "MobileNetV2",
        "dataset": "Ear Disease - Chile (Viscaino et al. 2020)",
    }
    with open(os.path.join(RESULTS, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Config] Saved to {RESULTS}/config.json")

    # ── 2. Same transforms as FL pipeline ──
    aug_tf = get_transforms(train=True)
    clean_tf = get_transforms(train=False)

    test_ds = TransformSubset(
        test_pool_ds, list(range(len(test_pool_ds))), transform=clean_tf
    )

    # ── 3. Stratified train/val split of the WHOLE pool (no client
    #    partitioning — this is the point of the centralized baseline) ──
    print("\n[2] Splitting pooled training set (stratified, "
          f"val_ratio={VAL_RATIO})...")
    train_idx, val_idx = stratified_train_val_split(
        train_pool_ds, VAL_RATIO, SEED
    )
    train_ds = TransformSubset(train_pool_ds, train_idx, aug_tf)
    val_ds = TransformSubset(train_pool_ds, val_idx, clean_tf)
    print(f"  Train: {len(train_ds)} (augmented) | "
          f"Val: {len(val_ds)} (clean)")

    save_centralized_summary(train_ds, val_ds, test_ds, train_pool_ds,
                             RESULTS)

    # ── 4. Same model / freeze_backbone setting as FL runs ──
    model = get_model(num_classes=num_classes,
                      freeze_backbone=FREEZE_BACKBONE).to(device)

    # ── 5. Train ──
    train_centralized(model, train_ds, val_ds, test_ds, device,
                      num_classes, class_names, RESULTS)

    print(f"\n[Done] Results in: {RESULTS}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
