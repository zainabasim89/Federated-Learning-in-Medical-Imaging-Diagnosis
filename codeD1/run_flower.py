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
    dirichlet_partition,
    iid_partition,
)
from fl_client import EarDiseaseFlowerClient
from fl_server import FedAvgStrategy, save_partition_summary


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
NUM_CLIENTS   = 3
NUM_ROUNDS    = 3
ALPHA         = 0.5     # Dirichlet heterogeneity level (used only if
                         # PARTITION_MODE == "dirichlet")
LOCAL_EPOCHS  = 20
BATCH_SIZE    = 16
LR            = 1e-3
ES_PATIENCE   = 5
SERVER_ADDR   = "127.0.0.1:8080"
SEED          = 42      # same seed -> identical splits across runs

# "iid" or "dirichlet" — flip this and re-run to get your
# IID-vs-non-IID comparison. Results are saved to separate
# folders automatically so nothing gets overwritten.
PARTITION_MODE = "iid"

# ── Path to the TOP-LEVEL "Dataset" folder you downloaded ──
# Must contain:
#   ROOT/training/<4 class folders>
#   ROOT/testing/<4 class folders>
# UPDATE THIS to match where you unzipped the Chile dataset.
ROOT = os.path.join(os.path.dirname(__file__), 'Dataset')
print(ROOT)
print(os.path.exists(ROOT))
print(os.listdir(ROOT))

RESULTS = os.path.join(
    os.path.dirname(__file__),
    f'results_fedavg_{PARTITION_MODE}_4class'
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


def client_fn(client_id, train_dataset, val_dataset, num_classes, barrier):
    print(f"\n[Client {client_id}] PID={os.getpid()}")
    try:
        barrier.wait(timeout=60)
    except Exception:
        pass

    client = EarDiseaseFlowerClient(
        client_id               = client_id,
        local_dataset           = train_dataset,
        val_dataset             = val_dataset,
        num_classes             = num_classes,
        local_epochs            = LOCAL_EPOCHS,
        batch_size              = BATCH_SIZE,
        lr                      = LR,
        early_stopping_patience = ES_PATIENCE,
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

    # ── 1. Load dataset using the dataset's OWN train/test split ──
    # Build class_to_idx from the TRAINING split first, then reuse
    # it for TESTING so both splits use identical integer labels
    # for the same class names.
    print("\n[1] Loading training pool and locked test set...")
    train_pool_ds = EarDiseaseDataset(ROOT, split='Training')
    test_pool_ds  = EarDiseaseDataset(
        ROOT, split='Testing', class_to_idx=train_pool_ds.class_to_idx
    )

    num_classes = train_pool_ds.num_classes
    class_names = train_pool_ds.class_names()

    print("\n" + "="*55)
    print("  FedAvg — Ear Disease Multiclass Classification (Chile)")
    print("="*55)
    print(f"  Classes        : {num_classes} -> {class_names}")
    print(f"  Clients        : {NUM_CLIENTS}")
    print(f"  Rounds         : {NUM_ROUNDS}")
    print(f"  Partition mode : {PARTITION_MODE}"
          + (f" (alpha={ALPHA})" if PARTITION_MODE == "dirichlet" else ""))
    print(f"  Max epochs     : {LOCAL_EPOCHS} (ES patience={ES_PATIENCE})")
    print(f"  Weighting      : sample-count (n_i / sum n_j)")
    print(f"  Seed           : {SEED}")
    print("="*55)

    # ── Save config ───────────────────────────────────────────
    config = {
        "algorithm"      : "FedAvg",
        "task"           : "Multiclass (4-class)",
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
        "val_ratio"      : 0.15,
        "model"          : "MobileNetV2",
        "dataset"        : "Ear Disease - Chile (Viscaino et al. 2020)",
    }
    with open(os.path.join(RESULTS, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Config] Saved to {RESULTS}/config.json")

    clean_tf = get_transforms(train=False)
    test_ds  = TransformSubset(
        test_pool_ds, list(range(len(test_pool_ds))), transform=clean_tf
    )

    # ── 2. Partition the training pool across clients ─────────
    pool_idx    = list(range(len(train_pool_ds)))
    pool_labels = [s[1] for s in train_pool_ds.samples]

    print(f"\n[2] Partitioning training pool ({PARTITION_MODE})...")
    if PARTITION_MODE == "dirichlet":
        client_splits = dirichlet_partition(
            indices     = pool_idx,
            labels      = pool_labels,
            num_clients = NUM_CLIENTS,
            alpha       = ALPHA,
            val_ratio   = 0.15,
            num_classes = num_classes,
            seed        = SEED,
        )
    elif PARTITION_MODE == "iid":
        client_splits = iid_partition(
            indices     = pool_idx,
            labels      = pool_labels,
            num_clients = NUM_CLIENTS,
            val_ratio   = 0.15,
            num_classes = num_classes,
            seed        = SEED,
        )
    else:
        raise ValueError(
            f"Unknown PARTITION_MODE: {PARTITION_MODE!r}. "
            f"Use 'dirichlet' or 'iid'."
        )

    # ── 3. Build TransformSubsets per client ──────────────────
    print("\n[3] Building client datasets...")
    aug_tf = get_transforms(train=True)

    client_train_sets = []
    client_val_sets   = []

    for i, split in enumerate(client_splits):
        tr = TransformSubset(train_pool_ds, split['train'], aug_tf)
        vl = TransformSubset(train_pool_ds, split['val'],   clean_tf)
        client_train_sets.append(tr)
        client_val_sets.append(vl)
        print(f"  Client {i+1}: "
              f"train={len(tr)} (augmented) | "
              f"val={len(vl)} (clean)")

    # ── 4. Save partition summary ─────────────────────────────
    print("\n[4] Saving partition summary...")
    save_partition_summary(
        client_train_sets = client_train_sets,
        client_val_sets   = client_val_sets,
        test_dataset      = test_ds,
        full_dataset      = train_pool_ds,
        results_dir       = RESULTS,
        alpha             = ALPHA,
        partition_mode    = PARTITION_MODE,
    )

    # ── 5. Start server ───────────────────────────────────────
    print("\n[5] Starting Flower server (FedAvg)...")
    strategy = FedAvgStrategy(
        num_rounds   = NUM_ROUNDS,
        num_classes  = num_classes,
        class_names  = class_names,
        min_clients  = NUM_CLIENTS,
        test_dataset = test_ds,
        device       = torch.device('cpu'),
        results_dir  = RESULTS,
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

    # ── 6. Launch clients ─────────────────────────────────────
    print(f"\n[6] Launching {NUM_CLIENTS} clients...")
    barrier = Barrier(NUM_CLIENTS)
    procs   = []

    for i in range(NUM_CLIENTS):
        p = multiprocessing.Process(
            target=client_fn,
            args=(i + 1,
                  client_train_sets[i],
                  client_val_sets[i],
                  num_classes,
                  barrier),
        )
        p.start()
        procs.append(p)
        print(f"  Client {i+1} started (PID={p.pid})")

    # ── 7. Wait ───────────────────────────────────────────────
    server_proc.join()
    for p in procs:
        p.join()

    print("\n[Done] All processes finished.")
    print(f"[Done] Results in: {RESULTS}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
