# Main run vs second training run

| method               |   AURC main run |   AURC second run |   score rank corr. | cov/risk @10% main   | cov/risk @10% second   | cov/risk @20% main   | cov/risk @20% second   |   same decision @20% |
|:---------------------|----------------:|------------------:|-------------------:|:---------------------|:-----------------------|:---------------------|:-----------------------|---------------------:|
| B2 token probability |           0.373 |             0.373 |              1     | 0.067 / 0.134        | 0.067 / 0.134          | 0.261 / 0.266        | 0.261 / 0.266          |                1     |
| Photo only           |           0.445 |             0.451 |              0.847 | 0.007 / 0.156        | 0.005 / 0.190          | 0.057 / 0.256        | 0.023 / 0.257          |                0.959 |
| Photo + question     |           0.372 |             0.383 |              0.836 | 0.078 / 0.146        | 0.059 / 0.110          | 0.225 / 0.225        | 0.200 / 0.233          |                0.872 |
| Log-probs only       |           0.366 |             0.373 |              0.976 | 0.110 / 0.166        | 0.060 / 0.121          | 0.278 / 0.254        | 0.258 / 0.256          |                0.948 |
| Full selector        |           0.317 |             0.327 |              0.967 | 0.215 / 0.119        | 0.205 / 0.125          | 0.390 / 0.237        | 0.405 / 0.256          |                0.929 |

- Full selector minus B2, AURC: main run -0.056 (-0.065 to -0.047), second run -0.046 (-0.056 to -0.037). 95% paired bootstrap.
- Second run minus main run (full selector), AURC: +0.010 (+0.006 to +0.013).
- Ranking of methods by AURC, main run: Full selector < Log-probs only < Photo + question < B2 token probability < Photo only
- Ranking of methods by AURC, second run: Full selector < Log-probs only < B2 token probability < Photo + question < Photo only
