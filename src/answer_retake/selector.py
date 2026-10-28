"""Selector: a small MLP on frozen Qwen2-VL features that predicts if the VLM's answer is right."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .data import LP_COLS, embedding

INPUTS = {
    "full": (["image", "question", "answer"], True),
    "photo_only": (["image"], False),
    "photo_question": (["image", "question"], False),
    "logprob_only": ([], True),
}


def build_x(root, split, frame, kinds, use_lp, layer="mid"):
    parts = [embedding(root, split, k, layer) for k in kinds]
    if use_lp:
        parts.append(frame[LP_COLS].fillna(0).to_numpy(np.float32))
    return np.hstack(parts)


def fit_predict(x_train, y_train, x_eval, hidden=256, seeds=(0, 1, 2, 3, 4), alpha=1e-2):
    """Mean predicted probability over seeds. Plain logistic regression when there are only a few inputs."""
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xe = {name: scaler.transform(x) for name, x in x_eval.items()}
    if x_train.shape[1] <= 16:
        model = LogisticRegression(max_iter=2000).fit(xt, y_train)
        return {name: model.predict_proba(x)[:, 1] for name, x in xe.items()}
    out = {name: np.zeros(len(x)) for name, x in xe.items()}
    for seed in seeds:
        model = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, early_stopping=True,
                              validation_fraction=0.1, max_iter=200, random_state=seed)
        model.fit(xt, y_train)
        for name, x in xe.items():
            out[name] += model.predict_proba(x)[:, 1] / len(seeds)
    return out
