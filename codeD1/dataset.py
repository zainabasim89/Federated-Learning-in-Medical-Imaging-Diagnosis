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
# EAR DISEASE DATASET  (Chile — Viscaino et al., 2020)
# Native 4-class multiclass task:
#   Normal, Earwax plug, Myringosclerosis, Chronic otitis media
# (exact folder names auto-detected from disk, so slight
# naming differences in your download don't require code
# changes).
#
# Expected folder layout (matches the Figshare download):
#
#   root_dir/
#     training/
#       <ClassFolder1>/
#       <ClassFolder2>/
#       <ClassFolder3>/
#       <ClassFolder4>/
#     testing/
#       <same 4 class folders>
#
# root_dir should point at the top-level "Dataset" folder you
# downloaded (the one containing "training" and "testing").
#
# IMPORTANT: pass class_to_idx from the TRAINING instantiation
# into the TESTING instantiation so both splits use identical
# integer label indices for the same class names. See
# run_flower.py for the correct load order.
#
# Stores (path, class_idx, original_class_name) per sample.
# =========================================================
class EarDiseaseDataset(Dataset):
    def __init__(self, root_dir, split='Training', class_to_idx=None):
        """
        split: 'training' or 'testing' — matches the two
        folders provided in the Chile dataset download.

        class_to_idx: dict mapping class folder name -> int.
        If None, it is built from this split's own folder
        listing (sorted alphabetically for reproducibility).
        Always build it from 'training' first, then pass it
        in when loading 'testing' so label indices match.
        """
        self.samples = []   # (path, class_idx, class_name)
        split_dir = os.path.join(root_dir, split)

        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"[EarDiseaseDataset] Split folder not found: {split_dir}\n"
                f"Expected structure:\n"
                f"  {root_dir}/Training/<4 class folders>\n"
                f"  {root_dir}/Testing/<4 class folders>\n"
                f"Update ROOT in run_flower.py if your folder is named "
                f"differently."
            )

        class_folders = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        if not class_folders:
            raise RuntimeError(
                f"[EarDiseaseDataset] No class subfolders found in "
                f"{split_dir}"
            )

        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(class_folders)}
        else:
            self.class_to_idx = dict(class_to_idx)
            missing = [c for c in class_folders
                       if c not in self.class_to_idx]
            if missing:
                print(f"[WARNING] '{split}' contains classes not seen "
                      f"in the training split's class_to_idx mapping: "
                      f"{missing}. These folders will be SKIPPED — "
                      f"check that training/testing folder names match "
                      f"exactly (including case/spacing).")

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes  = len(self.class_to_idx)

        for class_name in class_folders:
            if class_name not in self.class_to_idx:
                continue
            class_idx    = self.class_to_idx[class_name]
            class_folder = os.path.join(split_dir, class_name)

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

        self._print_summary(split)

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

    def _print_summary(self, split):
        total = len(self.samples)
        by_class = defaultdict(int)
        for _, _, cname in self.samples:
            by_class[cname] += 1

        print(f"\n[Dataset:{split}] Total: {total} | "
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
# For your IID-vs-non-IID comparison arm. Implemented as
# Dirichlet with a very large alpha (near-uniform proportions
# across clients), which is a standard equivalent of a random
# stratified split and lets both partition modes share one
# code path.
# =========================================================
# def iid_partition(indices, labels, num_clients=4, val_ratio=0.15,
#                   num_classes=None, seed=42):
#     print(f"\n[Partition] IID | {num_clients} clients")
#     return dirichlet_partition(
#         indices, labels, num_clients=num_clients,
#         alpha=100.0, val_ratio=val_ratio,
#         num_classes=num_classes, seed=seed
#     )

# def iid_partition(indices,
#                   labels,
#                   num_clients=4,
#                   val_ratio=0.10,
#                   num_classes=None,
#                   seed=42):

#     set_seed(seed)

#     indices = np.array(indices)
#     labels = np.array(labels)

#     if num_classes is None:
#         num_classes = int(labels.max()) + 1

#     # Map original dataset index -> label
#     index_to_label = {idx: lbl for idx, lbl in zip(indices, labels)}

#     # ---------------------------------------------------
#     # Step 1: Split each class equally among all clients
#     # ---------------------------------------------------
#     client_indices = [[] for _ in range(num_clients)]

#     for c in range(num_classes):

#         class_idx = indices[labels == c]

#         np.random.shuffle(class_idx)

#         splits = np.array_split(class_idx, num_clients)

#         for client in range(num_clients):
#             client_indices[client].extend(splits[client].tolist())

#     print(f"\n[Partition] Stratified IID | {num_clients} clients")

#     client_splits = []

#     # ---------------------------------------------------
#     # Step 2: Stratified Train/Validation split
#     # ---------------------------------------------------
#     for client in range(num_clients):

#         client_idx = np.array(client_indices[client])

#         np.random.shuffle(client_idx)

#         train_idx = []
#         val_idx = []

#         for c in range(num_classes):

#             cls = [
#                 idx for idx in client_idx
#                 if index_to_label[idx] == c
#             ]

#             np.random.shuffle(cls)

#             n_val = max(1, int(len(cls) * val_ratio))

#             val_idx.extend(cls[:n_val])
#             train_idx.extend(cls[n_val:])

#         np.random.shuffle(train_idx)
#         np.random.shuffle(val_idx)

#         train_labels = np.array(
#             [index_to_label[idx] for idx in train_idx]
#         )

#         counts = " ".join(
#             f"C{c}:{np.sum(train_labels == c)}"
#             for c in range(num_classes)
#         )

#         print(
#             f"Client {client+1}: "
#             f"train={len(train_idx)} ({counts}) "
#             f"| val={len(val_idx)}"
#         )

#         client_splits.append({
#             "train": train_idx,
#             "val": val_idx
#         })

#     return client_splits
def iid_partition(indices,
                  labels,
                  num_clients=4,
                  val_ratio=0.15,
                  num_classes=None,
                  seed=42):

    """
    Stratified IID partition.

    1. Split every class equally among all clients.
    2. Inside every client perform a stratified
       train/validation split using StratifiedShuffleSplit.

    This is the standard IID protocol used in many FL papers.
    """

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