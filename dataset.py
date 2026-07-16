#updated dataset file — Otoscopic Image Dataset (5-class, flat layout)
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# TRANSFORM SUBSET
# Base dataset stores only file paths, integer class label,
# and the original diagnostic class name. Transform lives
# here so train/val/test can differ without a second disk
# scan.
#
#   TransformSubset.dataset  ->  EarDiseaseDataset  (always)
# =========================================================
class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset   = dataset
        self.indices   = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label, _ = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================
# EAR DISEASE DATASET  (Otoscopic Image Dataset — UCI Machine
# Learning, Kaggle: ucimachinelearning/otoscopic-image-dataset)
#
# 5-class multiclass task. Class folder names are
# auto-detected from disk, so slight naming differences in
# your download don't require code changes.
#
# IMPORTANT — this dataset ships FLAT, with NO pre-made
# train/test split:
#
#   root_dir/
#     <ClassFolder1>/
#     <ClassFolder2>/
#     <ClassFolder3>/
#     <ClassFolder4>/
#     <ClassFolder5>/
#
# root_dir should point directly at the folder that CONTAINS
# the 5 class subfolders.
#
# Because there's no built-in split here, the held-out test
# set and the client train/val partitions are both carved out
# downstream (see stratified_test_split() below and
# run_flower.py for the correct order):
#   1. Load the whole dataset with this class (100%).
#   2. stratified_test_split() -> 15% test / 85% pool.
#   3. dirichlet_partition() or iid_partition() splits the
#      85% pool across clients, each further split into
#      train/val so that, overall, the dataset is
#      70% train / 15% val / 15% test.
#
# Stores (path, class_idx, original_class_name) per sample.
# =========================================================
class EarDiseaseDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None):
        """
        root_dir: folder that directly contains the class
        subfolders (no 'training'/'testing' split folders for
        this dataset).

        class_to_idx: dict mapping class folder name -> int.
        If None, it is built from this folder's own listing
        (sorted alphabetically for reproducibility).
        """
        self.samples = []   # (path, class_idx, class_name)

        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"[EarDiseaseDataset] Folder not found: {root_dir}\n"
                f"Expected structure:\n"
                f"  {root_dir}/<5 class folders>\n"
                f"Update ROOT in run_flower.py if your folder is named "
                f"differently."
            )

        class_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if not class_folders:
            raise RuntimeError(
                f"[EarDiseaseDataset] No class subfolders found in "
                f"{root_dir}"
            )

        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(class_folders)}
        else:
            self.class_to_idx = dict(class_to_idx)
            missing = [c for c in class_folders
                       if c not in self.class_to_idx]
            if missing:
                print(f"[WARNING] '{root_dir}' contains classes not seen "
                      f"in the supplied class_to_idx mapping: "
                      f"{missing}. These folders will be SKIPPED — "
                      f"check that folder names match exactly "
                      f"(including case/spacing).")

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes  = len(self.class_to_idx)

        for class_name in class_folders:
            if class_name not in self.class_to_idx:
                continue
            class_idx    = self.class_to_idx[class_name]
            class_folder = os.path.join(root_dir, class_name)

            files = sorted([
                f for f in os.listdir(class_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            for fname in files:
                self.samples.append((
                    os.path.join(class_folder, fname),
                    class_idx,
                    class_name
                ))

        self._print_summary("full dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Direct access without transform — use TransformSubset
        # for any split that needs augmentation or normalisation.
        img_path, label, _ = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return image, label

    def class_names(self):
        """Returns class names ordered by their integer index."""
        return [self.idx_to_class[i] for i in range(self.num_classes)]

    def _print_summary(self, label):
        total = len(self.samples)
        by_class = defaultdict(int)
        for _, _, cname in self.samples:
            by_class[cname] += 1

        print(f"\n[Dataset:{label}] Total: {total} | "
              f"Classes: {self.num_classes}")
        for cname in sorted(by_class.keys(),
                             key=lambda c: self.class_to_idx[c]):
            idx = self.class_to_idx[cname]
            print(f"    [{idx}] {cname:<28} "
                  f"({by_class[cname]} images)")


# =========================================================
# TRANSFORMS  (unchanged from the brain tumor pipeline)
# =========================================================
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


# =========================================================
# STRATIFIED TEST SPLIT  (new — needed because this dataset
# has no pre-made train/test folders)
#
# Carves off a stratified `test_ratio` slice of the FULL
# dataset as the locked, held-out test set, seen only at
# server-side evaluation. Everything else ("the pool") goes
# on to dirichlet_partition()/iid_partition() to be divided
# across FL clients.
#
# Returns plain index/label lists so the pool can be fed
# straight into dirichlet_partition/iid_partition, and a
# separate list of test indices to build the test
# TransformSubset from the SAME underlying dataset object.
# =========================================================
def stratified_test_split(indices, labels, test_ratio=0.15, seed=42):
    set_seed(seed)

    indices = np.array(indices)
    labels  = np.array(labels)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    pool_pos, test_pos = next(splitter.split(indices, labels))

    pool_idx    = indices[pool_pos].tolist()
    test_idx    = indices[test_pos].tolist()
    pool_labels = labels[pool_pos].tolist()

    num_classes = int(labels.max()) + 1
    test_labels = labels[test_pos]
    counts_str = " ".join(
        f"C{c}:{int((test_labels == c).sum())}" for c in range(num_classes)
    )
    print(f"\n[Split] Stratified test split | test_ratio={test_ratio}")
    print(f"  Pool (train+val) : {len(pool_idx)}")
    print(f"  Test (held-out)  : {len(test_idx)} ({counts_str})")

    return pool_idx, pool_labels, test_idx


# =========================================================
# DIRICHLET PARTITION  (non-IID) — generalized to N classes
#
# Lower alpha = more heterogeneous (e.g. alpha=0.1 is severe
# skew, alpha=0.5 is moderate).
#
# Accepts plain index arrays + a matching labels array
# (integer class indices, any number of classes).
# Returns a list of dicts with 'train' and 'val' index lists
# (indices refer back into the ORIGINAL dataset, so they can
# be fed straight into TransformSubset).
#
# val_ratio here is relative to the POOL passed in (i.e. after
# the test set has already been carved out by
# stratified_test_split). To land on overall proportions of
# 70% train / 15% val / 15% test across the WHOLE dataset,
# pass val_ratio = 0.15 / (1 - test_ratio) — see run_flower.py.
# =========================================================
def dirichlet_partition(indices, labels,
                        num_clients=4,
                        alpha=0.5,
                        val_ratio=0.15,
                        num_classes=None,
                        seed=42):
    set_seed(seed)

    indices = np.array(indices)
    labels  = np.array(labels)

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    class_indices = defaultdict(list)
    for pos, lbl in enumerate(labels):
        class_indices[int(lbl)].append(pos)   # position inside `indices`

    client_positions = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        pos_c = np.array(class_indices.get(c, []))
        if len(pos_c) == 0:
            continue
        np.random.shuffle(pos_c)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        sizes = (proportions * len(pos_c)).astype(int)

        while sizes.sum() < len(pos_c):
            sizes[np.argmax(proportions)] += 1
        while sizes.sum() > len(pos_c):
            sizes[np.argmax(sizes)] -= 1

        split_points = np.concatenate(([0], np.cumsum(sizes)))
        for i in range(num_clients):
            client_positions[i].extend(
                pos_c[split_points[i]:split_points[i+1]].tolist()
            )

    client_splits = []
    print(f"\n[Partition] Dirichlet (non-IID) alpha={alpha} | "
          f"{num_clients} clients | {num_classes} classes")

    for i, positions in enumerate(client_positions):
        positions = np.array(positions)
        np.random.shuffle(positions)

        pos_labels = labels[positions]

        train_parts, val_parts = [], []
        for c in range(num_classes):
            c_pos = positions[pos_labels == c]
            np.random.shuffle(c_pos)
            if len(c_pos) == 0:
                continue
            n_val = max(1, int(len(c_pos) * val_ratio))
            val_parts.append(c_pos[:n_val])
            train_parts.append(c_pos[n_val:])

        train_pos = (np.concatenate(train_parts)
                     if train_parts else np.array([], dtype=int))
        val_pos   = (np.concatenate(val_parts)
                     if val_parts else np.array([], dtype=int))
        np.random.shuffle(train_pos)
        np.random.shuffle(val_pos)

        train_idx = indices[train_pos].tolist()
        val_idx   = indices[val_pos].tolist()

        tr_labels = labels[train_pos]
        counts_str = " ".join(
            f"C{c}:{(tr_labels == c).sum()}" for c in range(num_classes)
        )
        print(f"  Client {i+1}: train={len(train_idx)} ({counts_str}) "
              f"| val={len(val_idx)}")

        client_splits.append({
            'train': train_idx,
            'val'  : val_idx,
        })

    return client_splits


# =========================================================
# IID PARTITION
#
# Stratified IID partition.
#
# 1. Split every class equally among all clients.
# 2. Inside every client perform a stratified
#    train/validation split using StratifiedShuffleSplit.
#
# Same val_ratio convention as dirichlet_partition() above:
# it's relative to the POOL (post test-split), not the whole
# dataset.
# =========================================================
def iid_partition(indices,
                  labels,
                  num_clients=4,
                  val_ratio=0.15,
                  num_classes=None,
                  seed=42):

    set_seed(seed)

    indices = np.array(indices)
    labels = np.array(labels)

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    # -----------------------------------------
    # Dataset index -> label mapping
    # -----------------------------------------
    index_to_label = {
        idx: lbl for idx, lbl in zip(indices, labels)
    }

    # -----------------------------------------
    # Equal class split across clients
    # -----------------------------------------
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        class_indices = indices[labels == c]

        np.random.shuffle(class_indices)

        splits = np.array_split(class_indices, num_clients)

        for client in range(num_clients):
            client_indices[client].extend(
                splits[client].tolist()
            )

    print(f"\n[Partition] Stratified IID | {num_clients} clients")

    client_splits = []

    # -----------------------------------------
    # Stratified train/validation split
    # -----------------------------------------
    for client_id in range(num_clients):

        client_idx = np.array(client_indices[client_id])

        np.random.shuffle(client_idx)

        client_labels = np.array([
            index_to_label[idx]
            for idx in client_idx
        ])

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed
        )

        train_pos, val_pos = next(
            splitter.split(client_idx, client_labels)
        )

        train_idx = client_idx[train_pos].tolist()
        val_idx = client_idx[val_pos].tolist()

        train_labels = np.array([
            index_to_label[idx]
            for idx in train_idx
        ])

        counts = " ".join(
            f"C{c}:{np.sum(train_labels==c)}"
            for c in range(num_classes)
        )

        print(
            f"Client {client_id+1}: "
            f"train={len(train_idx)} ({counts}) "
            f"| val={len(val_idx)}"
        )

        client_splits.append({
            "train": train_idx,
            "val": val_idx
        })

    return client_splits
