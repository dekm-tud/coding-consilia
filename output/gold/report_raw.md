# Gold standard evaluation

300 of 300 lines done.

## letters: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 300 | 91.7 % / 0.39 | 91.5 % / 0.41 | 94.7 % / 0.29 | 95.7 % / 0.13 | 97.5 % / 0.10 |
| stratum agreed | 150 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| stratum disputed | 150 | 57.3 % / 1.89 | 56.0 % / 2.01 | 72.7 % / 1.44 | 78.0 % / 0.63 | 87.3 % / 0.47 |
| line type cons_argument | 15 | 100.0 % / 0.00 | 97.3 % / 0.06 | 91.9 % / 0.37 | 97.3 % / 0.06 | 100.0 % / 0.00 |
| line type cons_body | 278 | 91.0 % / 0.41 | 91.2 % / 0.42 | 94.7 % / 0.29 | 95.5 % / 0.13 | 97.3 % / 0.10 |
| line type cons_title | 7 | 100.0 % / 0.00 | 91.2 % / 0.85 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| volume v1 | 69 | 91.1 % / 0.40 | 94.4 % / 0.22 | 93.3 % / 0.46 | 98.9 % / 0.05 | 97.8 % / 0.09 |
| volume v2 | 59 | 87.4 % / 0.46 | 92.7 % / 0.26 | 95.3 % / 0.18 | 95.3 % / 0.17 | 96.7 % / 0.13 |
| volume v3 | 59 | 91.7 % / 0.50 | 89.8 % / 0.44 | 94.3 % / 0.21 | 95.5 % / 0.13 | 96.2 % / 0.17 |
| volume v4 | 55 | 94.8 % / 0.19 | 85.8 % / 0.40 | 93.3 % / 0.50 | 91.8 % / 0.19 | 97.8 % / 0.05 |
| volume v5 | 58 | 94.1 % / 0.36 | 93.4 % / 0.76 | 97.4 % / 0.11 | 96.1 % / 0.13 | 99.3 % / 0.03 |
| flag auto | 151 | 98.3 % / 0.16 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |

## print: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 300 | 89.7 % / 0.35 | 89.1 % / 0.41 | 93.1 % / 0.30 | 93.1 % / 0.16 | 95.3 % / 0.12 |
| stratum agreed | 150 | 98.7 % / 0.02 | 98.0 % / 0.03 | 98.7 % / 0.02 | 98.0 % / 0.03 | 98.0 % / 0.03 |
| stratum disputed | 150 | 52.7 % / 1.62 | 52.0 % / 1.84 | 70.0 % / 1.39 | 72.7 % / 0.63 | 84.0 % / 0.46 |

## Margin numbers (separate from the text above)

| | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| gold lines with a margin number: 28, right | 26 | 9 | 27 | 12 | 13 |
| number read where gold has none | 0 | 0 | 0 | 0 | 0 |

## Error classes (word differences against gold, unweighted counts)

| class | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| other word | 43 | 57 | 45 | 26 | 16 |
| abbreviation not expanded or cut word | 15 | 9 | 5 | 8 | 2 |
| s/f | 13 | 4 | 1 | 5 | 4 |
| line end cut | 5 | 2 | 4 | 2 | 2 |
| abbreviation mark left | 7 | 0 | 0 | 1 | 1 |
| i/l/1 | 2 | 1 | 2 | 0 | 2 |

## Lines no reading got right (letters)

- g177 `xime quia dicunt vsq; ad riuum senij longaeuis, et antiquissimis`  
  G `xime quia dicunt vsq; ad riuum senij lógaeuis, et antiquissimis`  
  P `xime quia dicunt usque ad riuum fenii longaeuis, et antiquissimis`  
  K `xime quia dicunt vsque ad rium fenii longeuis, et antiquissimi`
- g278 `illa aut. res quae videtur posse praeindicare ecclesiis de consti. c. ecclesia S.`  
  G `illa aut. res quae videtur posse praeiudicare ecclesiis de consti. c. ecclesia S.`  
  P `illa aut. res quae videtur posse praeiudicare ecclesiis de consti. c. ecclesia S.`  
  K `lili au. re qua vu posse praeudicare eccretiis de costi. c eclesia §.`
