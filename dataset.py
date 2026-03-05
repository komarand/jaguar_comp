import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from models import models

class JaguarDataset(Dataset):
    """
    PyTorch Dataset for generating pairs of images from the Jaguar Re-ID dataset.
    This generates positive pairs (same identity) and negative pairs (different identity)
    for validation and calibration training.
    """
    def __init__(self, df, img_dir, num_pairs=1000, pos_ratio=0.5):
        self.df = df
        self.img_dir = img_dir
        self.num_pairs = num_pairs
        self.pos_ratio = pos_ratio
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        """Generates a list of tuples: (img1_path, img2_path, label, query_identity)"""
        pairs = []
        num_pos = int(self.num_pairs * self.pos_ratio)
        num_neg = self.num_pairs - num_pos

        # Group by identity
        id_to_imgs = self.df.groupby('identity')['image'].apply(list).to_dict()
        identities = list(id_to_imgs.keys())

        # Filter identities with at least 2 images for positive pairs
        multi_img_ids = [i for i in identities if len(id_to_imgs[i]) > 1]

        # Generate Positive Pairs
        for _ in range(num_pos):
            if not multi_img_ids:
                break
            # Randomly select an identity
            ind = random.choice(multi_img_ids)
            # Randomly select two distinct images
            img1, img2 = random.sample(id_to_imgs[ind], 2)
            path1 = os.path.join(self.img_dir, img1)
            path2 = os.path.join(self.img_dir, img2)
            pairs.append((path1, path2, 1, ind))

        # Generate Negative Pairs
        for _ in range(num_neg):
            # Select two distinct identities
            ind1, ind2 = random.sample(identities, 2)
            img1 = random.choice(id_to_imgs[ind1])
            img2 = random.choice(id_to_imgs[ind2])
            path1 = os.path.join(self.img_dir, img1)
            path2 = os.path.join(self.img_dir, img2)
            # Use ind1 as the 'query' identity for evaluation purposes
            pairs.append((path1, path2, 0, ind1))

        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label, identity = self.pairs[idx]
        return path1, path2, label, identity

def get_dataloader(csv_path, img_dir, num_pairs=1000, batch_size=1, num_workers=0):
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Returning empty DataLoader.")
        return None

    df = pd.read_csv(csv_path)
    # Assume CSV has columns 'image' and 'identity'
    dataset = JaguarDataset(df, img_dir, num_pairs=num_pairs)
    # Batch size 1 because variable shapes or local matching might be tricky to batch easily
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
