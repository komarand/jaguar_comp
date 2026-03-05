import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

class ScoreCalibrator:
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')

    def _combine_scores(self, global_dist, local_score):
        # We want higher score for better matches.
        # Lower global_dist is better, higher local_score is better.
        similarity = 1.0 - global_dist
        return max(0, similarity) * np.log1p(local_score)

    def fit(self, global_dists, local_scores, labels):
        """
        global_dists: list or array of global distances
        local_scores: list or array of local match counts
        labels: 1 for match, 0 for non-match
        """
        combined = [self._combine_scores(g, l) for g, l in zip(global_dists, local_scores)]
        self.iso_reg.fit(combined, labels)
        print("Calibration complete.")

    def predict_proba(self, global_dist, local_score):
        combined = self._combine_scores(global_dist, local_score)
        return self.iso_reg.predict([combined])[0]

    def save(self, filepath):
        joblib.dump(self.iso_reg, filepath)
        print(f"Calibrator saved to {filepath}")

    def load(self, filepath):
        self.iso_reg = joblib.load(filepath)
        print(f"Calibrator loaded from {filepath}")
