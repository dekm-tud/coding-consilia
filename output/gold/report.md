# Gold standard evaluation

300 of 300 lines done.

## letters: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 300 | 92.1 % / 0.37 | 91.5 % / 0.40 | 94.7 % / 0.29 | 95.7 % / 0.12 | 97.5 % / 0.10 |
| stratum agreed | 150 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| stratum disputed | 150 | 59.3 % / 1.81 | 56.0 % / 1.98 | 72.7 % / 1.41 | 78.0 % / 0.60 | 87.3 % / 0.47 |
| line type cons_argument | 15 | 100.0 % / 0.00 | 97.3 % / 0.06 | 91.9 % / 0.37 | 97.3 % / 0.06 | 100.0 % / 0.00 |
| line type cons_body | 278 | 91.4 % / 0.39 | 91.2 % / 0.42 | 94.7 % / 0.29 | 95.5 % / 0.13 | 97.3 % / 0.10 |
| line type cons_title | 7 | 100.0 % / 0.00 | 91.2 % / 0.85 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| volume v1 | 69 | 91.1 % / 0.40 | 94.4 % / 0.22 | 93.3 % / 0.46 | 98.9 % / 0.05 | 97.8 % / 0.09 |
| volume v2 | 59 | 88.0 % / 0.43 | 92.7 % / 0.24 | 95.3 % / 0.15 | 95.3 % / 0.14 | 96.7 % / 0.13 |
| volume v3 | 59 | 92.3 % / 0.47 | 89.8 % / 0.44 | 94.3 % / 0.21 | 95.5 % / 0.13 | 96.2 % / 0.17 |
| volume v4 | 55 | 94.8 % / 0.19 | 85.8 % / 0.40 | 93.3 % / 0.50 | 91.8 % / 0.19 | 97.8 % / 0.05 |
| volume v5 | 58 | 94.8 % / 0.33 | 93.4 % / 0.76 | 97.4 % / 0.11 | 96.1 % / 0.13 | 99.3 % / 0.03 |
| flag auto | 151 | 98.3 % / 0.16 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |

## print: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 300 | 90.9 % / 0.32 | 89.1 % / 0.40 | 93.4 % / 0.29 | 93.3 % / 0.15 | 95.5 % / 0.12 |
| stratum agreed | 150 | 98.7 % / 0.02 | 98.0 % / 0.03 | 98.7 % / 0.02 | 98.0 % / 0.03 | 98.0 % / 0.03 |
| stratum disputed | 150 | 58.7 % / 1.46 | 52.0 % / 1.83 | 71.3 % / 1.33 | 74.0 % / 0.59 | 85.3 % / 0.43 |

## Margin numbers (separate from the text above)

| | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| gold lines with a margin number: 28, right | 26 | 9 | 27 | 12 | 13 |
| number read where gold has none | 0 | 0 | 0 | 0 | 0 |

## Error classes (word differences against gold, unweighted counts)

| class | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| other word | 36 | 57 | 45 | 24 | 14 |
| abbreviation not expanded or cut word | 13 | 8 | 4 | 7 | 2 |
| s/f | 13 | 4 | 1 | 5 | 4 |
| line end cut | 4 | 2 | 4 | 2 | 2 |
| abbreviation mark left | 9 | 0 | 0 | 1 | 1 |
| i/l/1 | 2 | 1 | 2 | 0 | 2 |

## Lines no reading got right (letters)

- g177 `xime quia dicunt vsque ad riuum senij longaeuis, et antiquissimis`  
  G `xime quia dicunt vsque ad riuum senij logaeuis, et antiquissimis`  
  P `xime quia dicunt usque ad riuum fenii longaeuis, et antiquissimis`  
  K `xime quia dicunt vsque ad rium fenii longeuis, et antiquissimi`
- g278 `illa aut. res quae videtur posse praeindicare ecclesiis de consti. c. ecclesia S.`  
  G `illa aut. res quae videtur posse praeiudicare ecclesiis de consti. c. ecclesia S.`  
  P `illa aut. res quae videtur posse praeiudicare ecclesiis de consti. c. ecclesia S.`  
  K `lili au. re qua vu posse praeudicare eccretiis de costi. c eclesia §.`
