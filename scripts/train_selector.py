"""Train the selector and baselines from the saved VLM features.

Train on train, pick every setting on ourval, touch test once.
Writes results/run2_scores_{ourval,test}.csv, results/run2_table.csv and results/model_selection.csv.

    python scripts/train_selector.py --data data
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from answer_retake import data as D  # noqa: E402
from answer_retake.metrics import aurc, coverage_risk, threshold_at  # noqa: E402
from answer_retake.selector import INPUTS, build_x, fit_predict  # noqa: E402

TARGETS = [1, 5, 10, 20]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="1 seed, small grid (for a fast check)")
    args = ap.parse_args()
    root, out = D.data_dir(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    frames = {s: D.features(root, s) for s in D.SPLITS}
    y_train = frames["train"].correct.values
    c_val, c_test = frames["ourval"].correct.values, frames["test"].correct.values
    scores = {"ourval": {}, "test": {}}

    # B2: VLM's own confidence. Pick the log-prob summary with the lowest ourval AURC.
    lp_options = ["lp_first", "lp_mean", "lp_min", "lp_sum"]
    lp_aurc = {k: aurc(frames["ourval"][k], c_val) for k in lp_options}
    best_lp = min(lp_aurc, key=lp_aurc.get)
    log(f"B2: ourval AURC by summary {dict((k, round(v, 3)) for k, v in lp_aurc.items())} -> {best_lp}")
    for s in scores:
        scores[s]["B2"] = frames[s][best_lp].values

    # Selector settings: layer and hidden size, picked on ourval AURC (full inputs, 1 seed).
    grid = [("mid", 256)] if args.quick else [(layer, h) for layer in ("mid", "last") for h in (64, 256)]
    rows = []
    kinds, use_lp = INPUTS["full"]
    for layer, hidden in grid:
        x = {s: build_x(root, s, frames[s], kinds, use_lp, layer) for s in D.SPLITS}
        p = fit_predict(x["train"], y_train, {"ourval": x["ourval"]}, hidden=hidden, seeds=(0,))
        rows.append({"layer": layer, "hidden": hidden, "ourval_AURC": aurc(p["ourval"], c_val)})
        log(f"selection: layer {layer}, hidden {hidden}: ourval AURC {rows[-1]['ourval_AURC']:.4f}")
    sel = pd.DataFrame(rows)
    sel.to_csv(out / "model_selection.csv", index=False)
    layer, hidden = sel.sort_values("ourval_AURC").iloc[0][["layer", "hidden"]]
    hidden = int(hidden)
    log(f"chosen: layer {layer}, hidden {hidden}")

    # Final models: full with 5 seeds, ablations with 3.
    seeds = {"full": (0,) if args.quick else (0, 1, 2, 3, 4)}
    for name, (kinds, use_lp) in INPUTS.items():
        x = {s: build_x(root, s, frames[s], kinds, use_lp, layer) for s in D.SPLITS}
        p = fit_predict(x["train"], y_train, {"ourval": x["ourval"], "test": x["test"]}, hidden=hidden,
                        seeds=seeds.get(name, (0,) if args.quick else (0, 1, 2)))
        for s in scores:
            scores[s][name] = p[s]
        log(f"{name}: ourval AURC {aurc(p['ourval'], c_val):.4f}")

    for s, frame in (("ourval", frames["ourval"]), ("test", frames["test"])):
        df = pd.DataFrame({"image": frame.image, "correct": frame.correct, **{f"score_{k}": v for k, v in scores[s].items()}})
        df.to_csv(out / f"run2_scores_{s}.csv", index=False)

    # Table on test: thresholds from ourval only.
    table = []
    b1 = {"method": "B1 always answer", "AURC": 1 - c_test.mean()}
    for t in TARGETS:
        b1[f"cov@{t}%"], b1[f"risk@{t}%"] = 1.0, 1 - c_test.mean()
    table.append(b1)
    for name in ["B2", "photo_only", "photo_question", "logprob_only", "full"]:
        row = {"method": name, "AURC": aurc(scores["test"][name], c_test)}
        for t in TARGETS:
            thr = threshold_at(scores["ourval"][name], c_val, t / 100)
            row[f"cov@{t}%"], row[f"risk@{t}%"] = coverage_risk(scores["test"][name] >= thr, c_test)
        table.append(row)
    b3 = D.baseline3(root, frames["test"].image)
    quiet = b3.b3_answer.str.strip().str.lower().eq("unanswerable").values
    cov, risk = coverage_risk(~quiet, b3.b3_correct.values)
    table.append({"method": "B3 prompt 'unanswerable'", "AURC": np.nan, "cov@20%": cov, "risk@20%": risk})
    flaw_file = root / "flaw_probs_test.csv"
    if flaw_file.exists():
        fl = pd.read_csv(flaw_file)
        p_bad = fl[[c for c in fl.columns if c.startswith("p_")]].max(axis=1).values
        table.append({"method": "B4 flaw head only (AURC only)", "AURC": aurc(-p_bad, c_test)})
    tab = pd.DataFrame(table)
    tab.to_csv(out / "run2_table.csv", index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
