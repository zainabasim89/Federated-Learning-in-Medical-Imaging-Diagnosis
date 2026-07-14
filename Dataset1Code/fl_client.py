import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
import copy

from model import get_model


class EarDiseaseFlowerClient(fl.client.NumPyClient):
    """
    FedAvg / FedProx client for the 4-class ear disease task
    (Normal, Earwax plug, Myringosclerosis, Chronic otitis
    media).

    Set mu=0.0 for plain FedAvg (standard multiclass CE loss
    only). Set mu>0.0 to add the FedProx proximal term:

        L = CE(w) + (mu/2) * ||w - w_global||^2

    which is computed against the global weights received at
    the START of this round (w_t), following Li et al., 2020
    ("Federated Optimization in Heterogeneous Networks").

    Per-class label distribution is reported in metrics so the
    server can log heterogeneity stats for the paper, but it
    plays NO role in aggregation (FedAvg weights by n_samples
    on the server side regardless of mu).
    """

    def __init__(self,
                 client_id,
                 local_dataset,
                 val_dataset,
                 num_classes=4,
                 local_epochs=5,
                 batch_size=16,
                 lr=1e-3,
                 early_stopping_patience=3,
                 mu=0.0):

        self.client_id     = client_id
        self.local_dataset = local_dataset   # TransformSubset (augmented)
        self.val_dataset   = val_dataset     # TransformSubset (clean)
        self.num_classes   = num_classes
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.patience      = early_stopping_patience
        self.mu            = mu   # 0.0 = FedAvg, >0.0 = FedProx

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model     = get_model(num_classes=num_classes).to(self.device)
        self.round_num = 0

        # Compute once — dataset is fixed, indices never change
        self.label_dist = self._get_label_dist()

        dist_str = " | ".join(
            f"C{c}:{self.label_dist[c]:.1%}" for c in range(num_classes)
        )
        algo = f"FedProx (mu={self.mu})" if self.mu > 0 else "FedAvg"
        print(f"\n[Client {client_id}] Init | Device: {self.device} | {algo}")
        print(f"  Train: {len(local_dataset)} | Val: {len(val_dataset)}")
        print(f"  {dist_str}")

    # =========================================================
    # LABEL DISTRIBUTION  (per-class proportions)
    # TransformSubset.dataset is always EarDiseaseDataset
    # (one level, no nesting).
    # =========================================================
    def _get_label_dist(self):
        base    = self.local_dataset.dataset   # EarDiseaseDataset
        indices = self.local_dataset.indices   # flat list of ints
        labels  = [base.samples[i][1] for i in indices]
        total   = max(len(labels), 1)
        counts  = [labels.count(c) for c in range(self.num_classes)]
        return [c / total for c in counts]

    # =========================================================
    # FLOWER INTERFACE
    # =========================================================
    def get_parameters(self, config=None):
        return [v.cpu().numpy()
                for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict({
            k: torch.tensor(v)
            for k, v in zip(keys, parameters)
        })
        self.model.load_state_dict(state_dict, strict=True)

    # =========================================================
    # LOCAL TRAINING  —  CE loss (+ optional FedProx term)
    # =========================================================
    def fit(self, parameters, config):
        self.round_num = config.get("round", self.round_num + 1)
        algo = f"FedProx(mu={self.mu})" if self.mu > 0 else "FedAvg"
        print(f"\n[Client {self.client_id}] Round {self.round_num} | {algo}")

        self.set_parameters(parameters)

        # Snapshot global weights received THIS round — the proximal
        # anchor point w_t for FedProx. Detached, not touched by
        # optimizer.step(), so this stays fixed for the whole round.
        global_params = [p.detach().clone()
                          for p in self.model.parameters()]

        loader = DataLoader(
            self.local_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        loss_fn = nn.CrossEntropyLoss()

        # Early stopping state
        best_val_loss  = float('inf')
        patience_count = 0
        best_weights   = copy.deepcopy(self.model.state_dict())

        self.model.train()

        for epoch in range(self.local_epochs):
            total_loss = 0.0
            correct    = 0
            total      = 0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.long().to(self.device)   # class indices

                optimizer.zero_grad()
                logits = self.model(images)              # (B, num_classes)

                # ── Standard multiclass CE ──
                loss = loss_fn(logits, labels)

                # ── FedProx proximal term (skipped if mu == 0) ──
                if self.mu > 0:
                    prox_term = 0.0
                    for w, w_t in zip(self.model.parameters(),
                                       global_params):
                        prox_term = prox_term + (w - w_t).pow(2).sum()
                    loss = loss + (self.mu / 2.0) * prox_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds   = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

            train_acc = 100 * correct / total if total > 0 else 0.0
            avg_loss  = total_loss / len(loader)

            val_acc, val_loss = self._validate()

            print(f"  Ep {epoch+1}/{self.local_epochs} | "
                  f"Loss:{avg_loss:.4f} "
                  f"TrAcc:{train_acc:.1f}% | "
                  f"ValLoss:{val_loss:.4f} "
                  f"ValAcc:{val_acc*100:.1f}%",
                  end="")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss  = val_loss
                patience_count = 0
                best_weights   = copy.deepcopy(self.model.state_dict())
                print(" *")
            else:
                patience_count += 1
                print(f" [{patience_count}/{self.patience}]")
                if patience_count >= self.patience:
                    print(f"  [EarlyStop] epoch {epoch+1}")
                    break

        # Restore best weights from this round
        self.model.load_state_dict(best_weights)
        final_val_acc, final_val_loss = self._validate()

        print(f"  [Client {self.client_id}] "
              f"Best ValAcc: {final_val_acc*100:.2f}%")

        metrics = {
            "client_id"      : int(self.client_id),
            "local_val_acc"  : float(final_val_acc),
            "local_val_loss" : float(final_val_loss),
            "n_val_samples"  : int(len(self.val_dataset)),
            "round"          : int(self.round_num),
            "mu"             : float(self.mu),
        }
        # Per-class ratios reported individually (class_ratio_0, _1, ...)
        # so they survive Flower's flat metrics dict without nesting.
        for c in range(self.num_classes):
            metrics[f"class_ratio_{c}"] = float(self.label_dist[c])

        return (
            self.get_parameters(),
            len(self.local_dataset),
            metrics
        )

    # =========================================================
    # LOCAL VALIDATION
    # =========================================================
    def _validate(self):
        self.model.eval()
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        loss_fn    = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.long().to(self.device)
                logits = self.model(images)
                total_loss += loss_fn(logits, labels).item()
                preds   = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        self.model.train()
        acc  = correct / total if total > 0 else 0.0
        loss = total_loss / max(len(loader), 1)
        return acc, loss

    # =========================================================
    # FLOWER EVALUATE
    # =========================================================
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, loss = self._validate()
        return (
            float(loss),
            len(self.val_dataset),
            {"accuracy": float(acc)}
        )
