# Gold standard evaluation

15 of 300 lines done.

## letters: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 15 | 86.9 % / 0.44 | 80.4 % / 0.66 | 93.5 % / 0.15 | 90.2 % / 0.22 | 90.2 % / 0.29 |
| stratum agreed | 5 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| stratum disputed | 10 | 60.0 % / 1.18 | 40.0 % / 1.78 | 80.0 % / 0.39 | 70.0 % / 0.59 | 70.0 % / 0.79 |
| line type cons_argument | 2 | 100.0 % / 0.00 | 0.0 % / 1.89 | 100.0 % / 0.00 | 50.0 % / 0.94 | 100.0 % / 0.00 |
| line type cons_body | 12 | 83.7 % / 0.50 | 83.7 % / 0.58 | 91.8 % / 0.17 | 91.8 % / 0.17 | 87.7 % / 0.33 |
| line type cons_title | 1 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| volume v1 | 4 | 90.2 % / 0.39 | 90.2 % / 0.19 | 90.2 % / 0.19 | 90.2 % / 0.19 | 90.2 % / 0.19 |
| volume v2 | 4 | 86.0 % / 0.29 | 71.9 % / 0.88 | 100.0 % / 0.00 | 100.0 % / 0.00 | 86.0 % / 0.59 |
| volume v3 | 1 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| volume v4 | 3 | 66.7 % / 1.33 | 66.7 % / 2.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| volume v5 | 3 | 83.7 % / 0.33 | 67.3 % / 0.66 | 83.7 % / 0.33 | 67.3 % / 0.66 | 83.7 % / 0.33 |
| flag segmentation | 2 | 100.0 % / 0.00 | 50.0 % / 2.11 | 100.0 % / 0.00 | 100.0 % / 0.00 | 50.0 % / 2.11 |

## print: line accuracy / CER

| subset | n | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|---|
| corpus (weighted) | 15 | 83.7 % / 0.41 | 77.1 % / 0.58 | 86.9 % / 0.47 | 86.9 % / 0.23 | 86.9 % / 0.29 |
| stratum agreed | 5 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 | 100.0 % / 0.00 |
| stratum disputed | 10 | 50.0 % / 1.11 | 30.0 % / 1.58 | 60.0 % / 1.27 | 60.0 % / 0.63 | 60.0 % / 0.79 |

## Margin numbers (separate from the text above)

| | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| gold lines with a margin number: 4, right | 4 | 1 | 3 | 3 | 3 |
| number read where gold has none | 0 | 0 | 0 | 0 | 0 |

## Error classes (word differences against gold, unweighted counts)

| class | gemini | pylaia | kraken | online | majority |
|---|---|---|---|---|---|
| other word | 6 | 5 | 1 | 3 | 3 |
| abbreviation not expanded or cut word | 0 | 2 | 2 | 1 | 1 |
| abbreviation mark left | 2 | 0 | 0 | 0 | 0 |

## Lines no reading got right (letters)

- g015 `tamen modo allegati non potest, quia videtur tacite renun`  
  G `tamen modo allegari non potest, quia videtur tacitè renun¬`  
  P `tamen modo allegati non potest, quia videtut tacite renun¬`  
  K `tamen modo allegati non potest, quia videtur tacites renun¬`
