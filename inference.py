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
            img_path = os.path.join(img_dir, row['image'])
            ind_id = row['identity']
            self.add_known_individual(ind_id, img_path)
        print(f"Database built with {len(self.db_features)} images.")

    def identify(self, query_img_path):
        """
        Global-First filter:
        1. Compare query against all global features in DB.
        2. Retrieve Top-K candidates based on global distance.
        3. Two-stage verification: Run local matching ONLY on Top-K candidates
           AND ONLY if global distance is below threshold (e.g., 0.85).
        4. Apply calibrated fusion and novelty threshold.
        """
        if not os.path.exists(query_img_path):
            return "new_individual"

        if not self.db_features:
            return "new_individual"

        # Extract global feature once for the query
        query_feat = self._get_or_compute_global_feature(query_img_path)

        # 1. Global Search
        global_scores = []
        for db_id, db_feat, db_img_path in self.db_features:
            cosine_sim = torch.nn.functional.cosine_similarity(query_feat, db_feat).item()
            global_dist = 1.0 - cosine_sim
            global_scores.append((global_dist, db_id, db_img_path))

        # 2. Retrieve Top-K candidates
        global_scores.sort(key=lambda x: x[0]) # Ascending distance (lower is better)
        top_k_candidates = global_scores[:self.top_k]

        # 3. Two-stage Local Verification on Top-K
        best_match_id = None
        best_p_match = 0.0

        for g_dist, db_id, db_img_path in top_k_candidates:
            # ONLY compute local score if g_dist < global_thresh
            if g_dist < self.global_thresh:
                l_score = get_local_matches(query_img_path, db_img_path)

                # 4. Calibrated Fusion
                p_match = self.calibrator.predict_proba(g_dist, l_score)

                if p_match > best_p_match:
                    best_p_match = p_match
                    best_match_id = db_id
            else:
                # If g_dist is too high, we skip LightGlue and assume probability is very low
                # Or we can predict with 0 local matches
                p_match = self.calibrator.predict_proba(g_dist, 0)
                if p_match > best_p_match:
                    best_p_match = p_match
                    best_match_id = db_id

        # Novelty Filtering
        if best_p_match >= self.novelty_threshold:
            return best_match_id
        else:
            return "new_individual"

def generate_submission(system, test_csv, test_dir, output_csv="submission.csv"):
    if not os.path.exists(test_csv):
        print(f"Warning: {test_csv} not found. Skipping submission generation.")
        return

    df_test = pd.read_csv(test_csv)
    predictions = []

    print(f"Generating submission for {len(df_test)} queries...")
    for _, row in df_test.iterrows():
        img_path = os.path.join(test_dir, row['image'])
        pred_id = system.identify(img_path)
        predictions.append({'image': row['image'], 'identity': pred_id})

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
