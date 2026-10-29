# Gold standard evaluation

171 of 400 lines done.

## letters: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 171 | 93.4 % / 0.26 | 90.2 % / 0.57 | 94.3 % / 0.32 | 93.4 % / 0.20 | 95.2 % / 0.34 |
| stratum agreed | 92 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| stratum disputed | 79 | 63.3 % / 1.39 | 45.6 % / 3.11 | 68.4 % / 1.74 | 63.3 % / 1.09 | 73.4 % / 1.84 |
| line type cons_argument | 13 | 96.7 % / 0.15 | 83.6 % / 0.46 | 86.8 % / 0.46 | 86.8 % / 0.39 | 90.1 % / 0.31 |
| line type cons_body | 156 | 93.0 % / 0.26 | 90.6 % / 0.58 | 95.0 % / 0.30 | 93.8 % / 0.19 | 95.5 % / 0.34 |
| line type header | 2 | 100.0 % / 0.00 | 100.0 % / 0.00 | 79.6 % / 2.81 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| volume v1 | 41 | 91.2 % / 0.31 | 92.1 % / 0.31 | 92.1 % / 0.40 | 95.1 % / 0.15 | 94.1 % / 0.25 |
| volume v2 | 34 | 96.4 % / 0.27 | 85.6 % / 0.82 | 97.6 % / 0.05 | 91.6 % / 0.22 | 96.4 % / 0.15 |
| volume v3 | 30 | 92.9 % / 0.26 | 94.1 % / 0.76 | 92.9 % / 0.66 | 96.5 % / 0.17 | 94.1 % / 0.66 |
| volume v4 | 36 | 90.3 % / 0.32 | 87.9 % / 0.69 | 92.7 % / 0.25 | 89.1 % / 0.30 | 95.1 % / 0.47 |
| volume v5 | 30 | 96.5 % / 0.10 | 90.6 % / 0.35 | 96.5 % / 0.20 | 94.1 % / 0.18 | 96.5 % / 0.18 |
| flag segmentation | 7 | 100.0 % / 0.00 | 0.0 % / 7.87 | 28.6 % / 6.74 | 28.6 % / 1.69 | 28.6 % / 7.30 |

## print: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 171 | 83.8 % / 0.39 | 88.6 % / 0.60 | 92.7 % / 0.33 | 91.3 % / 0.26 | 92.5 % / 0.38 |
| stratum agreed | 92 | 90.2 % / 0.18 | 98.9 % / 0.02 | 98.9 % / 0.02 | 98.9 % / 0.02 | 98.9 % / 0.02 |
| stratum disputed | 79 | 54.4 % / 1.32 | 41.8 % / 3.17 | 64.6 % / 1.68 | 57.0 % / 1.32 | 63.3 % / 1.97 |

## Error classes (word differences against gold, unweighted counts)

| class | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| other word | 21 | 27 | 24 | 20 | 19 |
| margin number or † | 2 | 13 | 5 | 13 | 6 |
| abbreviation not expanded or cut word | 11 | 5 | 1 | 2 | 2 |
| i/l/1 | 1 | 3 | 2 | 2 | 2 |
| line end cut | 3 | 3 | 0 | 0 | 2 |
| abbreviation mark left | 7 | 0 | 0 | 0 | 0 |
| s/f | 4 | 0 | 0 | 0 | 0 |

## Lines no reading got right (letters)

- g104 `nullus. extra de vsur. li. 6. si vero sunt occulti, ita qua vix de vsu`  
  G `nullus. extra de vsur. li. 6. si vero sunt occulti, ita quod vix de vsu`  
  P `nullus. extra de vsur. li. 6. si vero sunt occulti, ita quam vix de vsu`  
  K `nullus. extra de vsur. li. 6. si vero sunt occulti, ita quae vix de vsu`
