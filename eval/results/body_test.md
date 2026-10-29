# Layer B evaluation — split test, 54 gold units

| run | units | relevance P/R/F1 | pair P/R/F1 (counted roles) | role acc | evidence valid | per code F1 (support) |
|---|---|---|---|---|---|---|
| body_gemini-3-8-flash_p1 | 54 | 1.00/0.94/0.97 | 0.73/0.92/0.81 | 0.77 | 1.00 | illegitimate_birth 0.83 (5), legitimation 1.00 (3), pregnancy_unborn 1.00 (2), incest 0.67 (2), birth 1.00 (2), concubinage_fornication 0.80 (2), adultery 0.67 (1), marital_cohabitation 1.00 (1), puberty_body 0.00 (1), paternity_filiation 0.67 (1), fertility_sterility 1.00 (1), chastity_conditions 0.67 (1), conception 1.00 (1), abduction 1.00 (1) |

Codebook v1.0; gold annotated by a model (Claude Opus 5.5) on the editor's behalf. Counted roles: analogy, argument, case_fact, topic.
