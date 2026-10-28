# Data (not in the repo)

Put these here before running anything. Nothing in this folder is committed.

```
data/
  features/              VLM feature files: features_{train,ourval,test}.parquet,
                         emb_*.npy, baseline3_test.parquet, splits/
  vqa_eval.py            VQA answer scoring
  test_scores.csv        selector scores from the main run
  ourval_scores.csv
  flaw_probs_test.csv    flaw head scores on the test photos
```

The folder names from the original zips (`team1_features_20261019/`, `team1_demo_extractor/`) also work.
You can point to another folder with `--data /path/to/folder` or the `AR_DATA` environment variable.
