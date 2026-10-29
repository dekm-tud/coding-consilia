# Layer B evaluation — split dev, 99 gold units

| run | units | relevance P/R/F1 | pair P/R/F1 (counted roles) | role acc | evidence valid | per code F1 (support) |
|---|---|---|---|---|---|---|
| body_gemini-3-8-flash_p1 | 92 | 1.00/1.00/1.00 | 0.85/1.00/0.92 | 0.74 | 1.00 | illegitimate_birth 0.94 (8), pregnancy_unborn 1.00 (6), adultery 1.00 (5), legitimation 0.91 (5), marital_violence 1.00 (2), concubinage_fornication 0.80 (2), paternity_filiation 1.00 (2), marital_cohabitation 1.00 (1), birth 0.67 (1), consummation 1.00 (1), stuprum 1.00 (1), conception 0.50 (1), fertility_sterility 1.00 (1), chastity_conditions 1.00 (1), prostitution 1.00 (1), nursing 1.00 (1) |

Codebook v1.0; gold annotated by a model (Claude Opus 5.5) on the editor's behalf. Counted roles: analogy, argument, case_fact, topic.
