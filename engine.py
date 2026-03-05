import torch
from models import models
from metrics import compute_identity_balanced_map
from calibration import ScoreCalibrator
from lightglue.utils import load_image, rbd
from PIL import Image
import os

def extract_global_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = models.global_transforms(img).unsqueeze(0).to(models.device)
    with torch.no_grad():
        # Depending on torch.compile, we use autocast
        with torch.autocast(device_type=models.device.type, dtype=torch.float16):
            feat = models.global_model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat

def get_local_matches(img_path1, img_path2):
    """
    Extracts local features and matches them using LightGlue.
    Returns the number of inlier matches.
    """
    img1 = load_image(img_path1).to(models.device)
    img2 = load_image(img_path2).to(models.device)

    with torch.no_grad():
        with torch.autocast(device_type=models.device.type, dtype=torch.float16):
            # Extract features with ALIKED
            feats1 = models.local_extractor.extract(img1)
            feats2 = models.local_extractor.extract(img2)

            # Match features with LightGlue
            matches01 = models.local_matcher({"image0": feats1, "image1": feats2})

    matches01_unbatched = rbd(matches01)
    matches = matches01_unbatched['matches']
    local_score = matches.shape[0] if matches is not None else 0

    return local_score

def match_pair(img_path1, img_path2):
    """
    Returns both global distance and local match count for two images.
    Global Distance: 1 - cosine_similarity
    Local Score: Number of inlier matches
    """
    feat1 = extract_global_feature(img_path1)
    feat2 = extract_global_feature(img_path2)

    cosine_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
    global_dist = 1.0 - cosine_sim

    local_score = get_local_matches(img_path1, img_path2)
    return global_dist, local_score

def train_calibration(train_loader, save_path="calibrator.pkl"):
    print("Training Calibration (Isotonic Regression)...")
    global_dists = []
    local_scores = []
    labels = []
    identities = []

    for i, (path1, path2, label, identity) in enumerate(train_loader):
        # DataLoaders return tuples of paths and labels, handle batch dim
        p1 = path1[0] if isinstance(path1, (list, tuple)) else path1
        p2 = path2[0] if isinstance(path2, (list, tuple)) else path2
        l = label[0].item() if isinstance(label, torch.Tensor) else label
        query_identity = identity[0] if isinstance(identity, (list, tuple)) else identity

        # Check if the path exists before attempting to open it
        if not os.path.exists(p1) or not os.path.exists(p2):
             continue

        g_dist, l_score = match_pair(p1, p2)
        global_dists.append(g_dist)
        local_scores.append(l_score)
        labels.append(l)
        identities.append(query_identity)

    calibrator = ScoreCalibrator()
    if global_dists and local_scores and labels:
        calibrator.fit(global_dists, local_scores, labels)
        calibrator.save(save_path)
    else:
        print("Warning: No valid data found to train calibration.")

    return calibrator, global_dists, local_scores, labels, identities

def evaluate_pipeline(calibrator, global_dists, local_scores, labels, identities):
    print("Evaluating Pipeline...")
    predictions = []
    for g, l in zip(global_dists, local_scores):
        p = calibrator.predict_proba(g, l)
        predictions.append(p)

    map_score = compute_identity_balanced_map(labels, predictions, identities)
    print(f"Identity-balanced mAP: {map_score:.4f}")
    return map_score
