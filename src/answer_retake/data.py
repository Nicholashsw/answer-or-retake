"""Loading the saved VLM features and score files."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "ourval", "test")
LP_COLS = ["lp_first", "lp_mean", "lp_min", "lp_sum", "n_answer_tokens"]
FEATURE_DIRS = ("features", "team1_features_20261019")


def data_dir(arg=None):
    path = Path(arg or os.environ.get("AR_DATA", "data"))
    if not any((path / d).exists() for d in FEATURE_DIRS):
        raise FileNotFoundError(f"no feature folder in {path.resolve()} (see data/README.md)")
    return path


def feature_dir(root):
    return next(Path(root) / d for d in FEATURE_DIRS if (Path(root) / d).exists())


def features(root, split):
    return pd.read_parquet(feature_dir(root) / f"features_{split}.parquet")


def embedding(root, split, kind, layer="mid"):
    suffix = "_mid" if layer == "mid" else ""
    return np.load(feature_dir(root) / f"emb_{split}_{kind}{suffix}.npy").astype(np.float32)


def baseline3(root, images):
    b3 = pd.read_parquet(feature_dir(root) / "baseline3_test.parquet").set_index("image")
    return b3.loc[list(images)].reset_index()


def vqa_eval_module(root):
    """VQA answer scoring (vqa_eval.py), loaded from the data folder."""
    for path in (Path(root) / "vqa_eval.py", Path(root) / "team1_demo_extractor" / "vqa_eval.py"):
        if path.exists():
            spec = importlib.util.spec_from_file_location("vqa_eval", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("vqa_eval.py not found in the data folder")
