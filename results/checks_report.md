# Pipeline checks

17 of 17 checks pass.

| group            | check                                                           | result   | detail        |
|:-----------------|:----------------------------------------------------------------|:---------|:--------------|
| splits           | no photo in two splits                                          | PASS     |               |
| splits           | train and ourval come from VizWiz train, test is VizWiz val     | PASS     |               |
| splits           | sizes 17,445 / 3,078 / 4,319                                    | PASS     |               |
| splits           | parquet rows follow splits/*.txt                                | PASS     |               |
| labels           | train: VQA accuracy recomputes from the 10 answers              | PASS     |               |
| labels           | train: correct = (VQA accuracy >= 0.5)                          | PASS     |               |
| labels           | ourval: VQA accuracy recomputes from the 10 answers             | PASS     |               |
| labels           | ourval: correct = (VQA accuracy >= 0.5)                         | PASS     |               |
| labels           | test: VQA accuracy recomputes from the 10 answers               | PASS     |               |
| labels           | test: correct = (VQA accuracy >= 0.5)                           | PASS     |               |
| labels           | 99 medicine questions                                           | PASS     |               |
| score files      | test_scores.csv rows and labels = features_test                 | PASS     |               |
| score files      | ourval_scores.csv rows and labels = features_ourval             | PASS     |               |
| score files      | flaw_probs_test.csv rows = features_test                        | PASS     |               |
| score files      | B2 score = sum of answer log-probs                              | PASS     |               |
| thresholds       | ourval thresholds reproduce every saved answer/abstain decision | PASS     | 0 rows differ |
| proposal numbers | 27 of 27 proposal numbers recomputed                            | PASS     |               |

## Proposal numbers recomputed from the files

| what                    |   proposal |   recomputed | match   |
|:------------------------|-----------:|-------------:|:--------|
| B1 risk                 |      0.613 |       0.6129 | True    |
| B2 AURC                 |      0.373 |       0.3734 | True    |
| B2 cov@10%              |      0.067 |       0.0671 | True    |
| B2 risk@10%             |      0.134 |       0.1345 | True    |
| B2 cov@20%              |      0.261 |       0.2614 | True    |
| B2 risk@20%             |      0.266 |       0.2657 | True    |
| photo_only AURC         |      0.445 |       0.4448 | True    |
| photo_only cov@10%      |      0.007 |       0.0074 | True    |
| photo_only risk@10%     |      0.156 |       0.1562 | True    |
| photo_question AURC     |      0.372 |       0.3722 | True    |
| photo_question cov@10%  |      0.078 |       0.0778 | True    |
| photo_question risk@10% |      0.146 |       0.1458 | True    |
| full AURC               |      0.317 |       0.3171 | True    |
| full cov@1%             |      0.029 |       0.0292 | True    |
| full risk@1%            |      0.024 |       0.0238 | True    |
| full cov@10%            |      0.215 |       0.2146 | True    |
| full risk@10%           |      0.119 |       0.1187 | True    |
| full cov@20%            |      0.39  |       0.3897 | True    |
| full risk@20%           |      0.237 |       0.2371 | True    |
| B3 coverage             |      0.423 |       0.423  | True    |
| B3 risk                 |      0.385 |       0.3848 | True    |
| flaw macro F1           |      0.57  |       0.5651 | True    |
| flaw blur F1            |      0.76  |       0.7598 | True    |
| flaw obstruction F1     |      0.39  |       0.3883 | True    |
| flaw blurry photos      |   1379     |    1379      | True    |
| flaw 2+ flaws share     |      0.34  |       0.3423 | True    |
| medicine B1 risk        |      0.747 |       0.7475 | True    |

## Flaw head cut-off

Macro F1 on test at the chosen 0.34 cut-off: 0.565. Best cut-off on test would be 0.36 (macro F1 0.570); range over 0.20-0.50: 0.505 to 0.570.
The cut-off is meant to be picked on QualityIssues val minus the test photos; F1 is flat around it, so the exact choice barely matters.
