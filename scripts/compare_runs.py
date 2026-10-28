"""Compare the main run's selector scores with a second training run on the same test questions.

Writes results/compare_runs.md and results/figures/compare_runs.png.

    python scripts/compare_runs.py --data data
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from answer_retake import data as D  # noqa: E402
from answer_retake.metrics import aurc, coverage_risk, paired_bootstrap_aurc, rc_curve, threshold_at  # noqa: E402

METHODS = ["B2", "photo_only", "photo_question", "logprob_only", "full"]
NAMES = {"B2": "B2 token probability", "photo_only": "Photo only", "photo_question": "Photo + question",
         "logprob_only": "Log-probs only", "full": "Full selector"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    root, out = D.data_dir(args.data), Path(args.out)
    main_t, main_v = pd.read_csv(root / "test_scores.csv"), pd.read_csv(root / "ourval_scores.csv")
    run2_t, run2_v = pd.read_csv(out / "run2_scores_test.csv"), pd.read_csv(out / "run2_scores_ourval.csv")
    assert (main_t.image == run2_t.image).all() and (main_v.image == run2_v.image).all()
    c, cv = main_t.correct.values, main_v.correct.values

    rows = []
    for m in METHODS:
        a, b = main_t[f"score_{m}"].values, run2_t[f"score_{m}"].values
        row = {"method": NAMES[m], "AURC main run": aurc(a, c), "AURC second run": aurc(b, c),
               "score rank corr.": spearmanr(a, b).statistic}
        for t in (10, 20):
            ans_a = main_t[f"answer_{m}@{t}%"].astype(bool).values
            ans_b = b >= threshold_at(run2_v[f"score_{m}"], cv, t / 100)
            ca, ra = coverage_risk(ans_a, c)
            cb, rb = coverage_risk(ans_b, c)
            row[f"cov/risk @{t}% main"] = f"{ca:.3f} / {ra:.3f}"
            row[f"cov/risk @{t}% second"] = f"{cb:.3f} / {rb:.3f}"
            if t == 20:
                row["same decision @20%"] = (ans_a == ans_b).mean()
        rows.append(row)
    tab = pd.DataFrame(rows)

    d_run2 = paired_bootstrap_aurc(run2_t.score_full, run2_t.score_B2, c)
    d_main = paired_bootstrap_aurc(main_t.score_full, main_t.score_B2, c)
    d_gap = paired_bootstrap_aurc(run2_t.score_full, main_t.score_full, c)
    order_main = list(tab.sort_values("AURC main run").method)
    order_run2 = list(tab.sort_values("AURC second run").method)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, col in [("full", "#2a78d6"), ("B2", "#eb6834")]:
        for df, ls, who in [(main_t, "-", "main run"), (run2_t, "--", "second run")]:
            cov, risk = rc_curve(df[f"score_{m}"], c)
            ax.plot(cov, risk, color=col, ls=ls, lw=1.8, label=f"{NAMES[m]}, {who}")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk on test")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.65)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Main run vs second training run (test)", loc="left", fontweight="bold")
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "compare_runs.png", dpi=200, bbox_inches="tight")

    with open(out / "compare_runs.md", "w") as f:
        f.write("# Main run vs second training run\n\n")
        f.write(tab.round(3).to_markdown(index=False) + "\n\n")
        f.write(f"- Full selector minus B2, AURC: main run {d_main[0]:+.3f} ({d_main[1]:+.3f} to {d_main[2]:+.3f}), "
                f"second run {d_run2[0]:+.3f} ({d_run2[1]:+.3f} to {d_run2[2]:+.3f}). 95% paired bootstrap.\n")
        f.write(f"- Second run minus main run (full selector), AURC: {d_gap[0]:+.3f} ({d_gap[1]:+.3f} to {d_gap[2]:+.3f}).\n")
        f.write(f"- Ranking of methods by AURC, main run: {' < '.join(order_main)}\n")
        f.write(f"- Ranking of methods by AURC, second run: {' < '.join(order_run2)}\n")
    print(tab.round(3).to_string(index=False))
    print(open(out / "compare_runs.md").read().split("\n\n", 2)[-1])


if __name__ == "__main__":
    main()
