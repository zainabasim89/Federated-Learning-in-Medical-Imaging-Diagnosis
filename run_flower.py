import sys
import os
import time
import json
import multiprocessing

sys.path.append(os.path.dirname(__file__))
multiprocessing.set_start_method('spawn', force=True)

import torch
import flwr as fl
import matplotlib
matplotlib.use('Agg')

from multiprocessing import Barrier

from dataset import (
    EarDiseaseDataset,
    TransformSubset,
    get_transforms,
    stratified_test_split,
    dirichlet_partition,
    iid_partition,
)
from fl_client import EarDiseaseFlowerClient
from fl_server import FedAvgStrategy, save_partition_summary


# ═══════════════════════════════════════════════════════════
# CONFIGURATION  —  the only two switches you need per run
# ═══════════════════════════════════════════════════════════
PARTITION_MODE = "dirichlet"        # "iid" or "dirichlet"
STRATEGY_MODE  = "fedavg"     # "fedavg" or "fedprox"

# FedProx proximal term strength. Ignored when STRATEGY_MODE
# == "fedavg" (mu is forced to 0.0 below regardless of this
# value, so it's safe to leave this set between runs).
# Typical FedProx paper range: 0.001 - 1.0. Start at 0.01.
FEDPROX_MU = 0.01

# ── everything else ──────────────────────────────────────────
NUM_CLIENTS   = 3
NUM_ROUNDS    = 5
ALPHA         = 0.5     # Dirichlet heterogeneity level (used only if
                         # PARTITION_MODE == "dirichlet")
LOCAL_EPOCHS  = 10
BATCH_SIZE    = 16
LR            = 1e-3
ES_PATIENCE   = 3
SERVER_ADDR   = "127.0.0.1:8080"
SEED          = 42      # same seed -> identical splits across runs
FREEZE_BACKBONE = True  # see model.py docstring for rationale

# ── Overall split, across the WHOLE dataset: ──────────────────
#     70% train (across all clients) / 15% val (across all
#     clients) / 15% held-out test.
# TEST_RATIO is taken straight off the full dataset first.
# VAL_RATIO_OF_POOL is then applied *within* the remaining 85%
# pool, so that val ends up at 15% of the WHOLE dataset too:
#     0.15 (pool) * 0.85 (pool fraction) ≈ ... solved below.
TEST_RATIO         = 0.15
VAL_RATIO_OF_POOL  = 0.15 / (1.0 - TEST_RATIO)   # ≈ 0.1765

# ── Resolve effective mu: FedProx only applies when requested ──
if STRATEGY_MODE == "fedprox":
    MU = FEDPROX_MU
elif STRATEGY_MODE == "fedavg":
    MU = 0.0
else:
    raise ValueError(
        f"Unknown STRATEGY_MODE: {STRATEGY_MODE!r}. "
        f"Use 'fedavg' or 'fedprox'."
    )

# ── Path to the Otoscopic dataset folder ──
# This dataset ships FLAT — no train/test split folders. ROOT
# must point DIRECTLY at the folder containing the 5 class
# subfolders:
#   ROOT/<Class1>/
#   ROOT/<Class2>/
#   ROOT/<Class3>/
#   ROOT/<Class4>/
#   ROOT/<Class5>/
# UPDATE THIS to match where you unzipped the Kaggle download.
ROOT = os.path.join(os.path.dirname(__file__), 'Dataset')

# ── Results folder auto-named from BOTH switches, so nothing
#    from a previous combination ever gets overwritten ──────
RESULTS = os.path.join(
    os.path.dirname(__file__),
    f'results_{STRATEGY_MODE}_{PARTITION_MODE}_5class'
)
os.makedirs(RESULTS, exist_ok=True)


def wait_for_server(address, timeout=90):
    import socket
    host, port = address.split(':')
    port = int(port)
    print(f"\n[Sync] Waiting for server at {address}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("[Sync] Server ready")
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server did not start within timeout.")


def client_fn(client_id, train_dataset, val_dataset, num_classes, barrier,
              mu):
    print(f"\n[Client {client_id}] PID={os.getpid()}")
    try:
        barrier.wait(timeout=60)
    except Exception:
        pass

    client = EarDiseaseFlowerClient(
        client_id               = client_id,
        local_dataset            = train_dataset,
        val_dataset              = val_dataset,
        num_classes              = num_classes,
        local_epochs             = LOCAL_EPOCHS,
        batch_size               = BATCH_SIZE,
        lr                       = LR,
        early_stopping_patience  = ES_PATIENCE,
        mu                       = mu,
    )
    fl.client.start_client(
        server_address=SERVER_ADDR,
        client=client.to_client(),
    )
    print(f"[Client {client_id}] Done.")


def main():
    # ── Reproducibility ───────────────────────────────────────
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── 1. Load the full dataset (flat, 5 class folders) ──────
    print("\n[1] Loading full Otoscopic dataset...")
    full_ds = EarDiseaseDataset(ROOT)

    num_classes = full_ds.num_classes
    class_names = full_ds.class_names()

    algo_label = f"FedProx (mu={MU})" if STRATEGY_MODE == "fedprox" else "FedAvg"

    print("\n" + "="*55)
    print(f"  {algo_label} — Otoscopic Ear Disease Classification (5-class)")
    print("="*55)
    print(f"  Classes        : {num_classes} -> {class_names}")
    print(f"  Clients        : {NUM_CLIENTS}")
    print(f"  Rounds         : {NUM_ROUNDS}")
    print(f"  Partition mode : {PARTITION_MODE}"
          + (f" (alpha={ALPHA})" if PARTITION_MODE == "dirichlet" else ""))
    print(f"  Strategy       : {STRATEGY_MODE}"
          + (f" (mu={MU})" if STRATEGY_MODE == "fedprox" else ""))
    print(f"  Freeze backbone: {FREEZE_BACKBONE}")
    print(f"  Max epochs     : {LOCAL_EPOCHS} (ES patience={ES_PATIENCE})")
    print(f"  Weighting      : sample-count (n_i / sum n_j)")
    print(f"  Overall split  : 70% train / 15% val / 15% test")
    print(f"  Seed           : {SEED}")
    print("="*55)

    # ── Save config ───────────────────────────────────────────
    config = {
        "algorithm"      : "FedProx" if STRATEGY_MODE == "fedprox" else "FedAvg",
        "strategy_mode"  : STRATEGY_MODE,
        "mu"             : MU,
        "task"           : "Multiclass (5-class)",
        "num_classes"    : num_classes,
        "class_names"    : class_names,
        "num_clients"    : NUM_CLIENTS,
        "num_rounds"     : NUM_ROUNDS,
        "partition_mode" : PARTITION_MODE,
        "alpha"          : ALPHA if PARTITION_MODE == "dirichlet" else None,
        "local_epochs"   : LOCAL_EPOCHS,
        "batch_size"     : BATCH_SIZE,
        "lr"             : LR,
        "es_patience"    : ES_PATIENCE,
        "seed"           : SEED,
        "test_ratio"          : TEST_RATIO,
        "val_ratio_of_pool"   : VAL_RATIO_OF_POOL,
        "overall_split"       : "70% train / 15% val / 15% test",
        "freeze_backbone": FREEZE_BACKBONE,
        "model"          : "MobileNetV2",
        "dataset"        : "Otoscopic Image Dataset (UCI Machine Learning, Kaggle)",
    }
    with open(os.path.join(RESULTS, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Config] Saved to {RESULTS}/config.json")

    clean_tf = get_transforms(train=False)
    aug_tf   = get_transforms(train=True)

    # ── 2. Carve off the held-out 15% test set ─────────────────
    print(f"\n[2] Splitting off held-out test set ({TEST_RATIO:.0%})...")
    all_idx    = list(range(len(full_ds)))
    all_labels = [s[1] for s in full_ds.samples]

    pool_idx, pool_labels, test_idx = stratified_test_split(
        indices     = all_idx,
        labels      = all_labels,
        test_ratio  = TEST_RATIO,
        seed        = SEED,
    )

    test_ds = TransformSubset(full_ds, test_idx, transform=clean_tf)

    # ── 3. Partition the remaining 85% pool across clients ──────
    print(f"\n[3] Partitioning training pool ({PARTITION_MODE})...")
    if PARTITION_MODE == "dirichlet":
        client_splits = dirichlet_partition(
            indices     = pool_idx,
            labels      = pool_labels,
            num_clients = NUM_CLIENTS,
            alpha       = ALPHA,
            val_ratio   = VAL_RATIO_OF_POOL,
            num_classes = num_classes,
            seed        = SEED,
        )
    elif PARTITION_MODE == "iid":
        client_splits = iid_partition(
            indices     = pool_idx,
            labels      = pool_labels,
            num_clients = NUM_CLIENTS,
            val_ratio   = VAL_RATIO_OF_POOL,
            num_classes = num_classes,
            seed        = SEED,
        )
    else:
        raise ValueError(
            f"Unknown PARTITION_MODE: {PARTITION_MODE!r}. "
            f"Use 'dirichlet' or 'iid'."
        )

    # ── 4. Build TransformSubsets per client ──────────────────
    print("\n[4] Building client datasets...")

    client_train_sets = []
    client_val_sets   = []

    for i, split in enumerate(client_splits):
        tr = TransformSubset(full_ds, split['train'], aug_tf)
        vl = TransformSubset(full_ds, split['val'],   clean_tf)
        client_train_sets.append(tr)
        client_val_sets.append(vl)
        print(f"  Client {i+1}: "
              f"train={len(tr)} (augmented) | "
              f"val={len(vl)} (clean)")

    # ── Sanity check: confirm the overall 70/15/15 split ───────
    n_train_total = sum(len(t) for t in client_train_sets)
    n_val_total   = sum(len(v) for v in client_val_sets)
    n_test_total  = len(test_ds)
    n_all         = n_train_total + n_val_total + n_test_total
    print(f"\n  Overall check -> train={n_train_total} "
          f"({n_train_total/n_all:.1%}) | "
          f"val={n_val_total} ({n_val_total/n_all:.1%}) | "
          f"test={n_test_total} ({n_test_total/n_all:.1%})")

    # ── 5. Save partition summary ─────────────────────────────
    print("\n[5] Saving partition summary...")
    save_partition_summary(
        client_train_sets = client_train_sets,
        client_val_sets   = client_val_sets,
        test_dataset      = test_ds,
        full_dataset      = full_ds,
        results_dir       = RESULTS,
        alpha             = ALPHA,
        partition_mode    = PARTITION_MODE,
    )

    # ── 6. Start server ───────────────────────────────────────
    print(f"\n[6] Starting Flower server ({algo_label})...")
    strategy = FedAvgStrategy(
        num_rounds    = NUM_ROUNDS,
        num_classes   = num_classes,
        class_names   = class_names,
        min_clients   = NUM_CLIENTS,
        test_dataset  = test_ds,
        device        = torch.device('cpu'),
        results_dir   = RESULTS,
        strategy_mode = STRATEGY_MODE,
        mu            = MU,
    )

    server_proc = multiprocessing.Process(
        target=fl.server.start_server,
        kwargs=dict(
            server_address=SERVER_ADDR,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        ),
    )
    server_proc.start()
    wait_for_server(SERVER_ADDR)

    # ── 7. Launch clients ─────────────────────────────────────
    print(f"\n[7] Launching {NUM_CLIENTS} clients...")
    barrier = Barrier(NUM_CLIENTS)
    procs   = []

    for i in range(NUM_CLIENTS):
        p = multiprocessing.Process(
            target=client_fn,
            args=(i + 1,
                  client_train_sets[i],
                  client_val_sets[i],
                  num_classes,
                  barrier,
                  MU),
        )
        p.start()
        procs.append(p)
        print(f"  Client {i+1} started (PID={p.pid})")

    # ── 8. Wait ───────────────────────────────────────────────
    server_proc.join()
    for p in procs:
        p.join()

    print("\n[Done] All processes finished.")
    print(f"[Done] Results in: {RESULTS}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
