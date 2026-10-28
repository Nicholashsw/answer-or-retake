"""Check the pipeline end to end: splits, labels, score files, thresholds and the proposal numbers.

Reads only. Writes results/checks_report.md, results/checks.csv and results/checks_numbers.csv.

    python scripts/check_pipeline.py --data data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from answer_retake import data as D  # noqa: E402
from answer_retake.metrics import aurc, coverage_risk, threshold_at  # noqa: E402

TARGETS = [1, 5, 10, 20]
METHODS = ["B2", "photo_only", "photo_question", "logprob_only", "full", "full_prompt_last"]

# Numbers as printed in proposal v6 (Table 2 and Section 7)
PROPOSAL = {
    ("B1", "risk"): 0.613, ("B2", "AURC"): 0.373, ("B2", "cov@10%"): 0.067, ("B2", "risk@10%"): 0.134,
    ("B2", "cov@20%"): 0.261, ("B2", "risk@20%"): 0.266, ("photo_only", "AURC"): 0.445,
    ("photo_only", "cov@10%"): 0.007, ("photo_only", "risk@10%"): 0.156, ("photo_question", "AURC"): 0.372,
    ("photo_question", "cov@10%"): 0.078, ("photo_question", "risk@10%"): 0.146, ("full", "AURC"): 0.317,
    ("full", "cov@1%"): 0.029, ("full", "risk@1%"): 0.024, ("full", "cov@10%"): 0.215, ("full", "risk@10%"): 0.119,
    ("full", "cov@20%"): 0.390, ("full", "risk@20%"): 0.237, ("B3", "coverage"): 0.423, ("B3", "risk"): 0.385,
    ("flaw", "macro F1"): 0.57, ("flaw", "blur F1"): 0.76, ("flaw", "obstruction F1"): 0.39,
    ("flaw", "blurry photos"): 1379, ("flaw", "2+ flaws share"): 0.34, ("medicine", "B1 risk"): 0.747,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    root, out = D.data_dir(args.data), Path(args.out)
    out.mkdir(exist_ok=True)

    tr, va, te = (D.features(root, s) for s in D.SPLITS)
    ts = pd.read_csv(root / "test_scores.csv")
    vs = pd.read_csv(root / "ourval_scores.csv")
    fl = pd.read_csv(root / "flaw_probs_test.csv")
    vqa = D.vqa_eval_module(root)
    splits_dir = D.feature_dir(root) / "splits"

    checks = []

    def check(group, name, passed, detail=""):
        checks.append({"group": group, "check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    # 1. splits
    a, b, c_ = set(tr.image), set(va.image), set(te.image)
    check("splits", "no photo in two splits", not (a & b or a & c_ or b & c_))
    check("splits", "train and ourval come from VizWiz train, test is VizWiz val",
          tr.image.str.startswith("VizWiz_train").all() and va.image.str.startswith("VizWiz_train").all()
          and te.image.str.startswith("VizWiz_val").all())
    check("splits", "sizes 17,445 / 3,078 / 4,319", (len(tr), len(va), len(te)) == (17445, 3078, 4319))
    rows_match = all(list(d.image) == (splits_dir / f"{s}.txt").read_text().split()
                     for s, d in zip(D.SPLITS, (tr, va, te)))
    check("splits", "parquet rows follow splits/*.txt", rows_match)

    # 2. labels
    for name, d in zip(D.SPLITS, (tr, va, te)):
        acc = np.array([vqa.vqa_accuracy(p, list(ans)) for p, ans in zip(d.vlm_answer_raw, d.answers)])
        check("labels", f"{name}: VQA accuracy recomputes from the 10 answers", np.allclose(acc, d.vqa_acc))
        check("labels", f"{name}: correct = (VQA accuracy >= 0.5)", ((d.vqa_acc >= 0.5).astype(int) == d.correct).all())
    check("labels", "99 medicine questions", int(te.is_medicine.sum()) == 99)

    # 3. score files line up with the feature files
    check("score files", "test_scores.csv rows and labels = features_test",
          (ts.image.values == te.image.values).all() and (ts.correct.values == te.correct.values).all()
          and (ts.is_medicine.values == te.is_medicine.values).all())
    check("score files", "ourval_scores.csv rows and labels = features_ourval",
          (vs.image.values == va.image.values).all() and (vs.correct.values == va.correct.values).all())
    check("score files", "flaw_probs_test.csv rows = features_test", (fl.image.values == te.image.values).all())
    check("score files", "B2 score = sum of answer log-probs", np.allclose(ts.score_B2, te.lp_sum))

    # 4. thresholds come from ourval only
    mism = 0
    for m in METHODS:
        for t in TARGETS:
            thr = threshold_at(vs[f"score_{m}"], vs.correct, t / 100)
            mism += int(((ts[f"score_{m}"].values >= thr).astype(int) != ts[f"answer_{m}@{t}%"].values).sum())
    check("thresholds", "ourval thresholds reproduce every saved answer/abstain decision", mism == 0, f"{mism} rows differ")

    # 5. recompute the proposal numbers
    c = te.correct.values
    got = {("B1", "risk"): 1 - c.mean(), ("medicine", "B1 risk"): 1 - c[te.is_medicine.values].mean()}
    for m in METHODS:
        got[(m, "AURC")] = aurc(ts[f"score_{m}"], c)
        for t in TARGETS:
            cov, risk = coverage_risk(ts[f"answer_{m}@{t}%"], c)
            got[(m, f"cov@{t}%")], got[(m, f"risk@{t}%")] = cov, risk
    b3 = D.baseline3(root, te.image)
    quiet = b3.b3_answer.str.strip().str.lower().eq("unanswerable").values
    got[("B3", "coverage")], got[("B3", "risk")] = 1 - quiet.mean(), (1 - b3.b3_correct.values[~quiet]).mean()

    flaws = ["BLR", "BRT", "DRK", "FRM", "OBS", "ROT"]
    human = {k: (te[f"flaw_{k}"].astype(float) >= 3).astype(int).values for k in flaws}  # majority of 5 annotators
    f1 = {k: f1_score(human[k], (fl[f"p_{k}"] >= 0.34).astype(int), zero_division=0) for k in flaws}
    labels = np.stack([human[k] for k in flaws], axis=1)
    got[("flaw", "macro F1")] = np.mean(list(f1.values()))
    got[("flaw", "blur F1")], got[("flaw", "obstruction F1")] = f1["BLR"], f1["OBS"]
    got[("flaw", "blurry photos")] = int(human["BLR"].sum())
    got[("flaw", "2+ flaws share")] = (labels.sum(1)[labels.any(1)] >= 2).mean()

    rows = []
    for key, want in PROPOSAL.items():
        have = got[key]
        tol = 0.5 if isinstance(want, int) else 0.0051
        rows.append({"what": f"{key[0]} {key[1]}", "proposal": want, "recomputed": round(float(have), 4),
                     "match": abs(have - want) <= tol})
    numbers = pd.DataFrame(rows)
    check("proposal numbers", f"{numbers.match.sum()} of {len(numbers)} proposal numbers recomputed", numbers.match.all())

    # 6. was the flaw cut-off tuned on test? (cannot prove; show how flat F1 is around it)
    cuts = np.round(np.arange(0.20, 0.51, 0.02), 2)
    macro = [np.mean([f1_score(human[k], (fl[f"p_{k}"] >= t).astype(int), zero_division=0) for k in flaws]) for t in cuts]
    best = cuts[int(np.argmax(macro))]

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "checks.csv", index=False)
    numbers.to_csv(out / "checks_numbers.csv", index=False)
    with open(out / "checks_report.md", "w") as f:
        f.write("# Pipeline checks\n\n")
        f.write(f"{(checks_df.result == 'PASS').sum()} of {len(checks_df)} checks pass.\n\n")
        f.write(checks_df.to_markdown(index=False) + "\n\n")
        f.write("## Proposal numbers recomputed from the files\n\n" + numbers.to_markdown(index=False) + "\n\n")
        f.write("## Flaw head cut-off\n\n")
        f.write(f"Macro F1 on test at the chosen 0.34 cut-off: {macro[list(cuts).index(0.34)]:.3f}. "
                f"Best cut-off on test would be {best:.2f} (macro F1 {max(macro):.3f}); "
                f"range over 0.20-0.50: {min(macro):.3f} to {max(macro):.3f}.\n"
                "The cut-off is meant to be picked on QualityIssues val minus the test photos; "
                "F1 is flat around it, so the exact choice barely matters.\n")
    print(checks_df.to_string(index=False))
    print(f"\n{numbers.match.sum()} of {len(numbers)} proposal numbers recomputed")
    print(f"flaw macro F1 at 0.34: {macro[list(cuts).index(0.34)]:.3f}, best on test {max(macro):.3f} at {best:.2f}")


if __name__ == "__main__":
    main()
