# SpeechEvalX evaluation

All rates are proportions; paired loss_pp is in percentage points.

Instrumentation failures are excluded from scored denominators and reported as coverage.

| Agent | Input | Scored / eligible | JCA | Tool | Arg-EM | No-call | Corpus WER |
|---|---|---:|---:|---:|---:|---:|---:|
| replay | asr:whisper | 4/4 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1579 |
| replay | entity | 3/3 | 0.0000 | 1.0000 | 0.0000 | N/A | 0.2000 |
| replay | function_word | 3/3 | 1.0000 | 1.0000 | 1.0000 | N/A | 0.2000 |
| replay | number | 3/3 | 0.0000 | 1.0000 | 0.0000 | N/A | 0.2000 |
| replay | reference | 4/4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Matched roles: replay

Common scored sources: 3

| Role | N | H | R | Loss (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| entity | 3 | 3 | 0 | 100.00 | [100.00, 100.00] |
| function_word | 3 | 0 | 0 | 0.00 | [0.00, 0.00] |
| number | 3 | 3 | 0 | 100.00 | [100.00, 100.00] |

| Harmed-pair category | Count | Share |
|---|---:|---:|
| argument_value | 6 | 1.0000 |

| Role contrast | Sources | Exact p | Holm p |
|---|---:|---:|---:|
| number_vs_entity | 3 | 1 | 1 |
| number_vs_function_word | 3 | 0.25 | 0.75 |
| entity_vs_function_word | 3 | 0.25 | 0.75 |

## Observed ASR: replay / asr:whisper

Distinct harmed pairs: 4. Role counts below overlap.

| Role | Hypotheses with role error | Action failures | Harmed |
|---|---:|---:|---:|
| number | 3 | 3 | 3 |
