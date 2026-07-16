import sys
import os
sys.path.append(os.path.dirname(__file__))

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score,
)

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitIns,
)
from collections import OrderedDict, defaultdict

from model import get_model


# =========================================================
# PARTITION SUMMARY SAVER
#
# Per-diagnostic-class breakdown (Normal / Earwax plug /
# Myringosclerosis / Chronic otitis media counts) for the
# full training pool, each client's train/val split, and the
# held-out test set. Useful directly as the "heterogeneity
# table" in your paper.
# =========================================================
def save_partition_summary(client_train_sets, client_val_sets,
                           test_dataset, full_dataset,
                           results_dir, alpha, partition_mode="dirichlet"):

    os.makedirs(results_dir, exist_ok=True)

    class_names = full_dataset.class_names()   # ordered by index
    num_classes = len(class_names)

    def count_breakdown(subset):
        """subset is a TransformSubset wrapping EarDiseaseDataset."""
        per_class = {c: 0 for c in class_names}
        base_ds = subset.dataset
        indices = list(subset.indices)

        for i in indices:
            _, _, cname = base_ds.samples[i]
            per_class[cname] = per_class.get(cname, 0) + 1

        total = sum(per_class.values())
        return {"per_class": per_class, "total": total}

    def count_breakdown_full(dataset):
        per_class = {c: 0 for c in class_names}
        for _, _, cname in dataset.samples:
            per_class[cname] = per_class.get(cname, 0) + 1
        return {"per_class": per_class, "total": sum(per_class.values())}

    # ── Sheet 1: Full Dataset (training pool) Overview ────────
    ov = count_breakdown_full(full_dataset)
    total = ov["total"]
    overview_rows = []
    for c in class_names:
        cnt = ov["per_class"][c]
        overview_rows.append({
            "Class"     : c,
            "Count"     : cnt,
            "Percentage": f"{cnt/max(total,1)*100:.1f}%",
        })
    overview_rows.append({"Class": "TOTAL", "Count": total,
                           "Percentage": "100.0%"})

    # ── Sheet 2: Client Partition Detail ──────────────────────
    partition_rows = []
    print(f"\n{'='*70}")
    label = f"Dirichlet (alpha={alpha})" if partition_mode == "dirichlet" \
            else "IID"
    print(f"  PARTITION SUMMARY — {label}")
    print(f"{'='*70}")

    header = f"  {'Client':<10} {'Split':<6} {'Total':>6} " + \
             " ".join(f"{c[:10]:>11}" for c in class_names)
    print(header)
    print(f"  {'-'*len(header)}")

    for i, (tr_ds, vl_ds) in enumerate(
            zip(client_train_sets, client_val_sets)):
        for split_name, split_ds in [("Train", tr_ds),
                                      ("Val",   vl_ds)]:
            c = count_breakdown(split_ds)
            counts_str = " ".join(
                f"{c['per_class'][cn]:>11}" for cn in class_names
            )
            print(f"  Client {i+1:<4} {split_name:<6} "
                  f"{c['total']:>6} {counts_str}")

            row = {
                "Client": f"Client {i+1}",
                "Split" : split_name,
                "Total" : c["total"],
            }
            for cn in class_names:
                row[cn] = c["per_class"][cn]
            partition_rows.append(row)

    # ── Sheet 3: Test Set Detail ───────────────────────────────
    tc = count_breakdown(test_dataset)
    test_rows = []
    for c in class_names:
        cnt = tc["per_class"][c]
        test_rows.append({
            "Class": c, "Count": cnt,
            "Percentage": f"{cnt/max(tc['total'],1)*100:.1f}%"
        })
    test_rows.append({"Class": "TOTAL", "Count": tc["total"],
                       "Percentage": "100.0%"})

    test_breakdown_str = ", ".join(
        f"{cn}:{tc['per_class'][cn]}" for cn in class_names
    )
    print(f"\n  Test Set: {tc['total']} images ({test_breakdown_str})")
    print(f"{'='*70}")

    path = os.path.join(results_dir, "partition_summary.xlsx")
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame(overview_rows).to_excel(
            writer, sheet_name="Dataset Overview", index=False)
        pd.DataFrame(partition_rows).to_excel(
            writer, sheet_name="Client Partition", index=False)
        pd.DataFrame(test_rows).to_excel(
            writer, sheet_name="Test Set", index=False)

    print(f"\n  [Saved] Partition summary -> {path}")


# =========================================================
# EVALUATION — held-out test set  (multiclass)
# =========================================================
def evaluate_on_test_set(model, dataset, device, num_classes,
                         batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    loss_fn    = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.concatenate(all_probs, axis=0) if all_probs else \
                 np.zeros((0, num_classes))

    accuracy = float((all_preds == all_labels).mean())

    f1_macro    = float(f1_score(all_labels, all_preds,
                                  average='macro', zero_division=0))
    f1_weighted = float(f1_score(all_labels, all_preds,
                                  average='weighted', zero_division=0))
    precision_macro = float(precision_score(all_labels, all_preds,
                                             average='macro',
                                             zero_division=0))
    recall_macro = float(recall_score(all_labels, all_preds,
                                       average='macro', zero_division=0))

    # Per-class F1 (useful in the paper's results table)
    f1_per_class = f1_score(all_labels, all_preds,
                            average=None, zero_division=0,
                            labels=list(range(num_classes))).tolist()

    # Macro one-vs-rest AUC — only computable if every class
    # appears in the test set; otherwise reported as 0.0
    try:
        classes_present = np.unique(all_labels)
        if len(classes_present) == num_classes and all_probs.shape[0] > 0:
            auc = float(roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='macro',
                labels=list(range(num_classes))
            ))
        else:
            auc = 0.0
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(num_classes)))

    return {
        "accuracy"        : accuracy,
        "f1_macro"        : f1_macro,
        "f1_weighted"     : f1_weighted,
        "precision_macro" : precision_macro,
        "recall_macro"    : recall_macro,
        "auc_macro_ovr"   : auc,
        "f1_per_class"    : f1_per_class,
        "confusion_matrix": cm.tolist(),
        "loss"            : float(total_loss / max(len(loader), 1)),
    }


# =========================================================
# FEDAVG STRATEGY  (multiclass)
#
# Aggregation rule (McMahan et al., 2017):
#   w_i = n_i / sum(n_j)
# where n_i is the number of training samples on client i.
# No JSD, no distribution awareness — pure sample weighting.
# =========================================================
class FedAvgStrategy(fl.server.strategy.Strategy):

    def __init__(self,
                 num_rounds,
                 num_classes=5,
                 class_names=None,
                 min_clients=4,
                 test_dataset=None,
                 device=None,
                 results_dir="../results_fedavg",
                 strategy_mode="fedavg",
                 mu=0.0):

        super().__init__()

        self.num_rounds   = num_rounds
        self.num_classes  = num_classes
        self.class_names  = class_names or [f"Class{i}"
                                            for i in range(num_classes)]
        self.min_clients  = min_clients
        self.test_dataset = test_dataset
        self.device       = device or torch.device("cpu")
        self.results_dir  = results_dir
        # NOTE: strategy_mode/mu are informational only (used in
        # logging/labels). FedProx's proximal term is applied
        # entirely on the CLIENT side (see fl_client.py); server
        # aggregation (aggregate_fit below) is identical FedAvg
        # sample-count weighting for both fedavg and fedprox.
        self.strategy_mode = strategy_mode
        self.mu             = mu

        os.makedirs(results_dir, exist_ok=True)

        # Clear stale files from any previous interrupted run
        for fname in ["history.json", "round_metrics.csv"]:
            p = os.path.join(results_dir, fname)
            if os.path.exists(p):
                os.remove(p)

        self.global_model = get_model(num_classes=num_classes).to(self.device)

        self.history = {
            "round"               : [],
            "fed_val_accuracy"    : [],
            "fed_val_loss"        : [],
            "test_accuracy"       : [],
            "test_f1_macro"       : [],
            "test_f1_weighted"    : [],
            "test_precision_macro": [],
            "test_recall_macro"   : [],
            "test_auc_macro_ovr"  : [],
            "test_f1_per_class"   : [],
            "test_confusion_matrix": [],
            # Per-client detail (logged for analysis, not used in agg)
            "client_ids"          : [],
            "client_val_accs"     : [],
            "client_weights"      : [],   # sample-count weights
            "client_class_ratios" : [],   # list of lists per round
            "client_n_train"      : [],
            "client_n_val"        : [],
        }

        self.csv_path = os.path.join(results_dir, "round_metrics.csv")
        pd.DataFrame(columns=[
            "round",
            "fed_val_accuracy", "fed_val_loss",
            "test_accuracy",
            "test_f1_macro", "test_f1_weighted",
            "test_precision_macro", "test_recall_macro",
            "test_auc_macro_ovr",
        ]).to_csv(self.csv_path, index=False)

        algo_label = f"FedProx (mu={self.mu})" if self.strategy_mode == "fedprox" \
                     else "FedAvg"
        print(f"\n[{algo_label}] Initialized (multiclass, {num_classes} classes)")
        print(f"  Classes: {self.class_names}")
        print(f"  Rounds : {num_rounds} | Clients: {min_clients}")
        print(f"  Device : {self.device}")
        print(f"  Aggregation: sample-count weighting (n_i / sum n_j) "
              f"[unchanged by mu — mu only affects client-side loss]")

    def initialize_parameters(self, client_manager):
        weights = [v.detach().cpu().numpy()
                   for v in self.global_model.state_dict().values()]
        return ndarrays_to_parameters(weights)

    def configure_fit(self, server_round, parameters,
                      client_manager):
        clients = client_manager.sample(
            num_clients=self.min_clients,
            min_num_clients=self.min_clients,
        )
        return [
            (c, FitIns(parameters, {"round": server_round}))
            for c in clients
        ]

    # =========================================================
    # AGGREGATE FIT — standard FedAvg
    # =========================================================
    def aggregate_fit(self, server_round, results, failures):

        if failures:
            print(f"[WARN] {len(failures)} failures in "
                  f"round {server_round}")
        if not results:
            print("[ERROR] No results received")
            return None, {}

        client_param_list = []
        n_train_samples   = []   # used for FedAvg weighting
        client_val_accs   = []
        client_val_losses = []
        n_val_samples     = []
        client_ids        = []
        client_class_ratios = []

        for fit_res_tuple in results:
            _, fit_res = fit_res_tuple
            client_param_list.append(
                parameters_to_ndarrays(fit_res.parameters)
            )
            n_train_samples.append(fit_res.num_examples)

            acc = fit_res.metrics.get("local_val_acc",  0.0)
            lss = fit_res.metrics.get("local_val_loss", 0.0)
            nv  = fit_res.metrics.get("n_val_samples",  1)
            cid = fit_res.metrics.get("client_id",      0)
            ratios = [
                fit_res.metrics.get(f"class_ratio_{c}", 0.0)
                for c in range(self.num_classes)
            ]

            client_val_accs.append(acc)
            client_val_losses.append(lss)
            n_val_samples.append(nv)
            client_ids.append(str(cid))
            client_class_ratios.append(ratios)

        # ── FedAvg weights: n_i / sum(n_j) ───────────────────
        total_train    = sum(n_train_samples)
        fedavg_weights = np.array(n_train_samples) / total_train

        # ── Weighted aggregation ──────────────────────────────
        aggregated = []
        for layer in range(len(client_param_list[0])):
            aggregated.append(np.sum(
                [fedavg_weights[i] * client_param_list[i][layer]
                 for i in range(len(client_param_list))],
                axis=0
            ))

        state_dict = OrderedDict({
            k: torch.tensor(v)
            for k, v in zip(
                self.global_model.state_dict().keys(), aggregated
            )
        })
        self.global_model.load_state_dict(state_dict)

        # ── Federated validation (sample-weighted proxy) ──────
        total_n      = sum(n_val_samples)
        fed_val_acc  = sum(n_val_samples[i] * client_val_accs[i]
                           for i in range(len(client_ids))
                           ) / max(total_n, 1)
        fed_val_loss = sum(n_val_samples[i] * client_val_losses[i]
                           for i in range(len(client_ids))
                           ) / max(total_n, 1)

        # ── Test set evaluation ───────────────────────────────
        tm = evaluate_on_test_set(
            self.global_model, self.test_dataset, self.device,
            self.num_classes
        )

        # ── Print round summary ───────────────────────────────
        print(f"\n[Round {server_round}] FedAvg Aggregation")
        cls_hdr = " ".join(f"C{c:<5}" for c in range(self.num_classes))
        print(f"  {'Client':<8} {cls_hdr} {'nTrain':>8} "
              f"{'Weight':>8} {'ValAcc':>8} {'nVal':>6}")
        for i, cid in enumerate(client_ids):
            ratios_str = " ".join(
                f"{r*100:5.1f}%" for r in client_class_ratios[i]
            )
            print(f"  C{cid:<7} {ratios_str} "
                  f"{n_train_samples[i]:>8} "
                  f"{fedavg_weights[i]:>8.4f} "
                  f"{client_val_accs[i]*100:>7.1f}% "
                  f"{n_val_samples[i]:>6}")

        print(f"\n  Fed Val Acc      : {fed_val_acc*100:.2f}%")
        print(f"  Test Accuracy    : {tm['accuracy']*100:.2f}%")
        print(f"  Test F1 (macro)  : {tm['f1_macro']:.4f}")
        print(f"  Test F1 (weighted): {tm['f1_weighted']:.4f}")
        print(f"  Precision (macro): {tm['precision_macro']:.4f}")
        print(f"  Recall (macro)   : {tm['recall_macro']:.4f}")
        print(f"  AUC (macro OVR)  : {tm['auc_macro_ovr']:.4f}")
        for cname, f1c in zip(self.class_names, tm['f1_per_class']):
            print(f"    F1 [{cname}]: {f1c:.4f}")

        # ── Save history ──────────────────────────────────────
        self.history["round"].append(server_round)
        self.history["fed_val_accuracy"].append(fed_val_acc)
        self.history["fed_val_loss"].append(fed_val_loss)
        for key in ["accuracy", "f1_macro", "f1_weighted",
                    "precision_macro", "recall_macro", "auc_macro_ovr"]:
            self.history[f"test_{key}"].append(tm[key])
        self.history["test_f1_per_class"].append(tm["f1_per_class"])
        self.history["test_confusion_matrix"].append(tm["confusion_matrix"])
        self.history["client_ids"].append(client_ids)
        self.history["client_val_accs"].append(client_val_accs)
        self.history["client_weights"].append(fedavg_weights.tolist())
        self.history["client_class_ratios"].append(client_class_ratios)
        self.history["client_n_train"].append(n_train_samples)
        self.history["client_n_val"].append(n_val_samples)

        # ── CSV row ───────────────────────────────────────────
        row = {
            "round"                : server_round,
            "fed_val_accuracy"     : fed_val_acc,
            "fed_val_loss"         : fed_val_loss,
            "test_accuracy"        : tm["accuracy"],
            "test_f1_macro"        : tm["f1_macro"],
            "test_f1_weighted"     : tm["f1_weighted"],
            "test_precision_macro" : tm["precision_macro"],
            "test_recall_macro"    : tm["recall_macro"],
            "test_auc_macro_ovr"   : tm["auc_macro_ovr"],
        }
        pd.DataFrame([row]).to_csv(
            self.csv_path, mode="a", header=False, index=False
        )

        # ── Checkpoint ────────────────────────────────────────
        torch.save(
            self.global_model.state_dict(),
            os.path.join(self.results_dir,
                         f"model_round_{server_round}.pth")
        )

        # ── History JSON ──────────────────────────────────────
        with open(os.path.join(self.results_dir, "history.json"),
                  "w") as f:
            json.dump(self.history, f, indent=2)

        if server_round == self.num_rounds:
            self._finalize()

        return ndarrays_to_parameters(aggregated), {}

    # =========================================================
    # FINALIZE
    # =========================================================
    def _finalize(self):
        torch.save(
            self.global_model.state_dict(),
            os.path.join(self.results_dir, "final_model.pth")
        )

        df = pd.read_csv(self.csv_path)
        df.to_excel(
            os.path.join(self.results_dir, "round_metrics.xlsx"),
            index=False
        )

        self._save_convergence_plot()
        self._save_confusion_matrix_plot()
        self._save_client_detail_excel()

        last = self.history
        print(f"\n{'='*55}")
        print(f"  FINAL RESULTS — held-out test set  [FedAvg]")
        print(f"{'='*55}")
        print(f"  Accuracy          : {last['test_accuracy'][-1]:.4f}")
        print(f"  F1 (macro)        : {last['test_f1_macro'][-1]:.4f}")
        print(f"  F1 (weighted)     : {last['test_f1_weighted'][-1]:.4f}")
        print(f"  Precision (macro) : {last['test_precision_macro'][-1]:.4f}")
        print(f"  Recall (macro)    : {last['test_recall_macro'][-1]:.4f}")
        print(f"  AUC (macro OVR)   : {last['test_auc_macro_ovr'][-1]:.4f}")
        for cname, f1c in zip(self.class_names, last['test_f1_per_class'][-1]):
            print(f"    F1 [{cname}]: {f1c:.4f}")
        print(f"{'='*55}")
        print(f"  Saved to: {self.results_dir}/")

    # =========================================================
    # CLIENT DETAIL PER ROUND
    # =========================================================
    def _save_client_detail_excel(self):
        rows = []
        for r_idx, rnd in enumerate(self.history["round"]):
            cids    = self.history["client_ids"][r_idx]
            wts     = self.history["client_weights"][r_idx]
            vaccs   = self.history["client_val_accs"][r_idx]
            ratios  = self.history["client_class_ratios"][r_idx]
            n_train = self.history["client_n_train"][r_idx]
            n_vals  = self.history["client_n_val"][r_idx]

            for i, cid in enumerate(cids):
                row = {
                    "Round"         : rnd,
                    "Client"        : f"C{cid}",
                    "N Train"       : n_train[i],
                    "FedAvg Weight" : round(wts[i], 4),
                    "Local Val Acc" : f"{vaccs[i]*100:.1f}%",
                    "N Val Samples" : n_vals[i],
                }
                for c, cname in enumerate(self.class_names):
                    row[f"{cname} %"] = f"{ratios[i][c]*100:.1f}%"
                rows.append(row)

        path = os.path.join(self.results_dir,
                            "client_detail_per_round.xlsx")
        pd.DataFrame(rows).to_excel(path, index=False)
        print(f"  [Saved] {path}")

    # =========================================================
    # CONVERGENCE PLOT
    # =========================================================
    def _save_convergence_plot(self):
        rounds = self.history["round"]
        if not rounds:
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()

        pairs = [
            ("fed_val_accuracy", "test_accuracy",        "Accuracy"),
            (None,               "test_f1_macro",        "F1 (macro)"),
            (None,               "test_f1_weighted",     "F1 (weighted)"),
            (None,               "test_precision_macro", "Precision (macro)"),
            (None,               "test_recall_macro",    "Recall (macro)"),
            (None,               "test_auc_macro_ovr",   "AUC (macro OVR)"),
        ]

        for i, (vm, tm, title) in enumerate(pairs):
            ax = axes[i]
            if vm and vm in self.history:
                ax.plot(rounds, self.history[vm],
                        'b-o', lw=2, ms=4, label='Fed Val')
            ax.plot(rounds, self.history[tm],
                    'r-s', lw=2, ms=4, label='Test')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('FL Round')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.suptitle(
            "FedAvg Baseline — Ear Disease Classification (Otoscopic, 5-class)\n"
            "Sample-count weighting (n_i / sum n_j) | "
            "Federated privacy-preserving validation"
        )
        plt.tight_layout()
        path = os.path.join(self.results_dir,
                            "convergence_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {path}")

    # =========================================================
    # CONFUSION MATRIX (final round)
    # =========================================================
    def _save_confusion_matrix_plot(self):
        if not self.history["test_confusion_matrix"]:
            return
        cm = np.array(self.history["test_confusion_matrix"][-1])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix — Final Round (Test Set)')
        plt.tight_layout()
        path = os.path.join(self.results_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {path}")

    def configure_evaluate(self, *args, **kwargs):
        return []

    def aggregate_evaluate(self, *args, **kwargs):
        return None, {}

    def evaluate(self, *args, **kwargs):
        return None
