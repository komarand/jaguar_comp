import torch
import os
import pandas as pd
from engine import extract_global_feature, get_local_matches

class WildFusionInference:
    def __init__(self, calibrator, novelty_threshold=0.5, top_k=5, global_thresh=0.85):
        self.calibrator = calibrator
        self.novelty_threshold = novelty_threshold
        self.top_k = top_k
        self.global_thresh = global_thresh
        self.database = {} # id -> list of image paths
        self.db_features = [] # List of tuples: (id, feature_tensor, img_path)
        self.feature_cache = {} # Cache to avoid redundant ViT passes

    def _get_or_compute_global_feature(self, img_path):
        if img_path in self.feature_cache:
            return self.feature_cache[img_path]
        feat = extract_global_feature(img_path)
        self.feature_cache[img_path] = feat
        return feat

    def add_known_individual(self, individual_id, img_path):
        if not os.path.exists(img_path):
            return

        if individual_id not in self.database:
            self.database[individual_id] = []
        self.database[individual_id].append(img_path)

        # Precompute global feature and cache it
        feat = self._get_or_compute_global_feature(img_path)
        self.db_features.append((individual_id, feat, img_path))

    def build_database(self, csv_path, img_dir):
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping DB build.")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            ind_id = row['ground_truth']
            self.add_known_individual(ind_id, img_path)
        print(f"Database built with {len(self.db_features)} images.")

    def compute_similarity(self, img_path1, img_path2):
        """
        Computes pairwise similarity for Kaggle test format using embedding cache and two-stage verification.
        Returns the calibrated probability P(match).
        """
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            return 0.0

        feat1 = self._get_or_compute_global_feature(img_path1)
        feat2 = self._get_or_compute_global_feature(img_path2)

        cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
        global_dist = 1.0 - cosine_sim

        # Two-stage Local Verification: Only run LightGlue if global distance is promising
        if global_dist < self.global_thresh:
            l_score = get_local_matches(img_path1, img_path2)
            p_match = self.calibrator.predict_proba(global_dist, l_score)
        else:
            p_match = self.calibrator.predict_proba(global_dist, 0)

        return p_match

def generate_submission(system, test_csv, test_dir, output_csv="submission.csv"):
    if not os.path.exists(test_csv):
        print(f"Warning: {test_csv} not found. Skipping submission generation.")
        return

    df_test = pd.read_csv(test_csv)
    predictions = []

    print(f"Generating submission for {len(df_test)} pairs...")
    for _, row in df_test.iterrows():
        query_img_path = os.path.join(test_dir, row['query_image'])
        gallery_img_path = os.path.join(test_dir, row['gallery_image'])

        sim = system.compute_similarity(query_img_path, gallery_img_path)
        predictions.append({'row_id': row['row_id'], 'similarity': sim})

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
