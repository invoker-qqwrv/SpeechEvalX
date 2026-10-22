# SpeechEvalX

**Task-role diagnostics for tool-using voice agents: when does an ASR error actually matter?**

SpeechEvalX is a paired diagnostic protocol, with a reference implementation, for measuring how ASR errors propagate through a cascaded voice agent:

```
speech  →  ASR  →  LLM  →  structured tool call
```

Word error rate counts wrong words, but it does not tell you whether the *action* changed. In "Set the temperature to 22", turning *the* into *a* is harmless, while turning *22* into *32* is not. Both are a single substituted word. SpeechEvalX compares such edits under matched lexical severity and asks one question: **does the task role of the edited span (number, negation, entity, …) explain action failure beyond what lexical, semantic, and binary "critical span" descriptions already capture?**

> **Paper:** *When Does an ASR Error Matter? Task-Role Diagnostics for Tool-Using Voice Agents*
> Xiaolong Wang, Shijie Su, Naqi Xiong. Leibniz University Hannover.
>
> **Status: v0.1.0 (Phase 0 MVP).** The paired execution, matched-corruption, scoring, tracing and paired-statistics core is implemented and runs without a GPU. Some parts of the paper's analysis are not in this repository yet; see [Paper ↔ code status](#paper--code-status) for the exact list.

---

## Key findings

| | Result |
|---|---|
| **ASR front ends hurt actions** | Joint call accuracy (JCA) drops from **82.3 %** with reference transcripts to **74.9 / 73.2 / 71.9 %** with Whisper large-v3 / SenseVoice-Small / Paraformer-large (corpus WER 6.3 / 7.2 / 8.1 %). |
| **Equal edit count ≠ equal risk** | With source, edit count and edit operation held fixed, **negation (25.7 pp)** and **number (21.9 pp)** edits cost the most JCA. Function-word and other non-critical edits cost **2.9 pp** and **1.4 pp**. |
| **Task-bearing vs. control roles** | Mean paired JCA loss is **17.3 pp** across the five task-bearing roles and **2.1 pp** across the two control roles. The contrasts remain significant after Holm correction (p < 0.01). |
| **Role identity adds predictive information** | Source-grouped ROC-AUC for predicting action failure rises from **0.73** (binary critical-span flag) to **0.78** with detailed role labels (Δ = 0.05, 95 % CI [0.02, 0.08], p = 0.003). AP rises from 0.62 to 0.68. |
| **Pattern transfers across agents** | Applied without refitting to Llama-3.1-8B-Instruct, the detailed roles reach 0.74 vs. 0.69 (direction consistent, but no confidence interval was computed). |
| **Matching matters** | Dropping within-source severity matching inflates the mean task-bearing loss from 17.3 to 20.6 pp. |
| **Where harm shows up** | Among 216 harmed controlled pairs, 47.7 % are argument-value errors, while 30.1 % change the call/no-call decision, the tool, or output validity. |

---

## The protocol

For each source utterance `i` there is a verified reference transcript `r_i`, a reference action `z_i` (one tool call, or an explicit *no-call*), and a hypothesis `h_i` (a real ASR output, or a controlled single edit of `r_i`). The same frozen agent is run on both inputs, and **both outputs are scored against the same reference action `z_i`**, not against the agent's output on the clean transcript, because that output can itself be wrong.

```
                         ┌─ reference transcript r_i ──► Agent ──► ẑ(r_i) ─┐
  source utterance i ────┤                                                  ├─► compare with the SAME z_i
                         └─ one controlled edit  h_i ──► Agent ──► ẑ(h_i) ─┘
                               (same source, edit count, edit operation)
```

Each pair lands in one of four cells:

| Reference transcript | Corrupted transcript | Outcome |
|:-:|:-:|---|
| ✔ | ✔ | preserved success |
| ✔ | ✘ | **harmed (H)** |
| ✘ | ✔ | **rescued (R)** |
| ✘ | ✘ | persistent failure (not counted as new harm) |

The paired JCA loss for role `c` over eligible set `I_c` is

```
L_c = (1 / |I_c|) · Σ_i [ y(r_i) − y(h_i^(c)) ] = (H_c − R_c) / |I_c|
```

so a positive value means a net loss in action accuracy caused by the edit.

### Task-role taxonomy

| Role | Type | Inclusion rule (short) |
|---|---|---|
| `number` | task-bearing | Quantity, time, date, ordinal, or identifier aligned to an action value |
| `entity` | task-bearing | Named person, place, organization, product, or account aligned to an action value |
| `negation` | task-bearing | Overt cue whose scope changes whether or how the action is executed |
| `intent_cue` | task-bearing | Predicate or phrase that selects the call or the no-call intent |
| `other_slot` | task-bearing | Schema-grounded argument value not covered by the roles above |
| `function_word` | control | Closed-class token outside every task-bearing span |
| `other_noncritical` | control | Remaining material outside every task-bearing span |

"Non-critical" only means that no annotated schema role is present. It does not guarantee that an edit is harmless. Two annotators labeled 120 sources (684 candidate edit spans): exact span-boundary F1 = 0.93, role Cohen's κ = 0.89, eligibility agreement = 95.0 %.

### Metrics

- **JCA** = (correct calls + correct no-calls) / all eligible examples. It measures structured-action accuracy, not end-to-end task completion.
- **Tool accuracy** = `T / N_C`, the share of call-reference examples with a correct tool name.
- **Arg-EM** = `C / T`, argument exact match conditional on a correct tool. Hence call-only accuracy = `(T/N_C)·(C/T) = C/N_C`.
- **No-call accuracy** = `C0 / N0`.
- Malformed outputs count as failures. Corpus WER is total word edits divided by total reference words (not the mean of per-utterance WERs).

---

## Experimental setup (paper)

| | |
|---|---|
| **Data** | 600 candidates sampled with seed 2027 from the English single-tool subset of [VoiceAgentBench](https://arxiv.org/abs/2510.07978) and the English single-turn subset of BFCL Audio. 6 failed eligibility checks, leaving **594 sources** (297 + 297; 540 call and 54 no-call references). |
| **Intervention set** | 210 sources that admit one unambiguous single-edit variant for *every* compared role. The other 384 sources are used only in the benchmark-audio ASR analysis. One variant per source and role is chosen by an outcome-independent frozen rule. |
| **ASR front ends** | Whisper large-v3, SenseVoice-Small, Paraformer-large |
| **Agents** | Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (sensitivity) |
| **Inference** | bfloat16, temperature 0, top-p 1, ≤ 256 new tokens, seed 2027. Prompt, schema, parser, canonicalizer and scoring are fixed across paired conditions. |
| **Failure-prediction features** | Cumulative sets B0 (lexical severity, edit operation, length, benchmark) → B1 (+ BERT cosine) → B2 (+ Semantic-WER) → B3 (+ binary critical-span flag) → B4 (+ detailed task roles). Regularized logistic regression, five source-grouped folds. |

---

## Results

### ASR hypotheses on benchmark audio (594 sources, %)

| Input | WER | Tool | Arg-EM | No-call | **JCA** |
|---|--:|--:|--:|--:|--:|
| Reference transcript | – | 91.3 | 89.5 | 88.9 | **82.3** |
| Whisper large-v3 | 6.3 | 86.9 | 85.1 | 85.2 | **74.9** |
| SenseVoice-Small | 7.2 | 86.1 | 83.9 | 83.3 | **73.2** |
| Paraformer-large | 8.1 | 85.2 | 83.3 | 81.5 | **71.9** |

Correct actions out of 594: 489 (reference), 445, 435 and 427.

### Matched interventions: paired JCA loss by role (210 sources, percentage points)

| Role | H | R | **L_c** (Qwen2.5-7B) | 95 % CI | L_c (Llama-3.1-8B) |
|---|--:|--:|--:|---|--:|
| Negation | 58 | 4 | **25.7** | [20.0, 31.9] | 23.3 |
| Number | 51 | 5 | **21.9** | [16.7, 27.6] | 19.0 |
| Entity | 40 | 4 | **17.1** | [12.4, 22.4] | 15.7 |
| Intent cue | 30 | 4 | **12.4** | [8.1, 17.1] | 10.5 |
| Other slot value | 24 | 4 | **9.5** | [5.7, 13.8] | 8.6 |
| *Function word (control)* | 8 | 2 | *2.9* | [0.5, 5.7] | 3.3 |
| *Other non-critical (control)* | 5 | 2 | *1.4* | [−0.5, 3.8] | 1.0 |

Intervals are source-bootstrap 95 % CIs. The role ordering is the same for both agents. Restricting to deterministic substitutions gives 16.8 pp (task-bearing) vs. 2.0 pp (control).

### Does knowing the role help predict failure?

Endpoint: action failure after corruption, among cases that are correct with the reference transcript. Out-of-fold, source-grouped five-fold CV.

| Model | Added information | ROC-AUC | AP |
|---|---|--:|--:|
| B0 | Lexical controls | 0.62 | 0.48 |
| B1 | + BERT embedding cosine | 0.68 | 0.55 |
| B2 | + Semantic-WER | 0.70 | 0.58 |
| B3 | + binary critical-span flag | 0.73 | 0.62 |
| **B4** | **+ detailed task roles** | **0.78** | **0.68** |
| **B4 − B3** | | **+0.05** (95 % CI [0.02, 0.08], p = 0.003) | +0.06 (point estimate) |

Robustness of B4: 0.74 vs. 0.69 for B3 on Llama-3.1-8B (no refit, no CI); leave-domain-out ROC-AUC 0.72 (train VoiceAgentBench → test BFCL Audio) and 0.70 (reverse).

### Why harmed pairs fail (216 harmed controlled pairs)

| Category (precedence order) | Count | Share |
|---|--:|--:|
| Malformed output | 6 | 2.8 % |
| Call/no-call decision | 17 | 7.9 % |
| Wrong tool | 42 | 19.4 % |
| Argument structure | 48 | 22.2 % |
| Argument value | 103 | 47.7 % |

Precedence only resolves overlaps. It is not a timeline of the model's internal reasoning.

### Observed ASR errors (descriptive only)

Pooled over the three ASR conditions, with overlapping role labels. Among hypotheses that contain at least one error of the given role, the share followed by an action failure:

| Role | Failed / with error | Rate |
|---|--:|--:|
| Negation | 20 / 42 | 47.6 % |
| Number | 35 / 104 | 33.7 % |
| Entity | 37 / 128 | 28.9 % |
| Intent cue | 23 / 87 | 26.4 % |
| Other slot value | 29 / 159 | 18.2 % |
| Function word | 19 / 201 | 9.5 % |
| Other non-critical | 17 / 236 | 7.2 % |

The 180 numerators are role-label occurrences, not necessarily 180 distinct failed hypotheses. These are observational associations with correlated multi-edit errors, not isolated single-role effects. The B4 − B3 benefit has **not** been tested on observed ASR errors.

---

## Repository layout

The repository root is itself the Python package (`speechevalx`).

| Path | What it provides |
|---|---|
| `experiment.py` | `run_paired_variant(...)`: runs clean and corrupted transcripts through the same agent, scores both against the same reference action and returns a `PairedTrace`. **This is the intended path for aggregate results.** |
| `pipeline.py` | `run_single_case(...)`: simple audio → ASR → agent → score path producing a `CaseTrace` with WER/CER, tool correctness and argument exact match. Single-call references only (no no-call handling); `task_success` is left `None`. |
| `datasets/` | Benchmark-neutral `SpeechEvalSample` and `ReferenceAction` (`tool_name=None` means *no-call*), plus `VoiceAgentBenchAdapter` and `BFCLAudioAdapter`. |
| `corruption/` | `Corruption` record and `generate_matched_substitutions(...)`, which emits only eligible one-token, exactly-one-match substitutions. |
| `alignment/` | `word_alignment(ref, hyp)`: deterministic Levenshtein backtrace keeping substitution / insertion / deletion operations. |
| `parsing/` | `parse_tool_call(...)`: safe AST-based parser for a single `name(key=literal, ...)` call. It never evaluates code. |
| `scoring/` | `ScoringConfig`, `score_action(...)`, `canonicalize_action(...)` for frozen, configuration-driven action scoring. |
| `evaluation/` | `evaluate_single_tool(...)`, used by `pipeline.py`. |
| `metrics/` | `wer`, `cer`, `normalize_value`. |
| `statistics/` | `cei_by_class`, `cluster_bootstrap_cei`, `epr_matrix`, `mcnemar_exact`, `holm_adjust`. |
| `trace/` | `PairedTrace` (JSONL) and `CaseTrace` (JSON) audit-trail schemas. |
| `models/asr/` | `ASRModel` interface, `WhisperASR` (lazy Hugging Face), `MockASR`. |
| `models/agent/` | `AgentModel` interface, `QwenAgent` (lazy Hugging Face causal LM), `RuleBasedAgent`, `TemplateAgent`, `MockAgent`. |

---

## Installation

The core library uses only the Python standard library (Python 3.10+; smoke-tested on 3.12). `transformers` and `torch` are needed only for the real `WhisperASR` / `QwenAgent` adapters.

The code imports itself as `speechevalx`, and there is no `pyproject.toml` yet, so clone into a folder with exactly that name and run your scripts from its **parent** directory:

```bash
mkdir workspace && cd workspace
git clone https://github.com/invoker-qqwrv/SpeechEvalX.git speechevalx

# optional: only for real ASR / LLM back ends
pip install transformers torch
```

> ⚠️ **Do not run Python with the repository root as the working directory.** The sub-packages `statistics/` and `trace/` would shadow the standard-library modules of the same name. Keep `speechevalx/` one level down, as above.

---

## Quick start (no GPU, no data download)

This reproduces the paper's Figure 1 idea on a synthetic example with the built-in deterministic `TemplateAgent`: one source utterance, three matched single-token substitutions, one reference action.

> `TemplateAgent`, `RuleBasedAgent` and the `Mock*` classes exist for smoke tests only. Do not report numbers from them.

```python
# run from the directory that contains speechevalx/
from pathlib import Path

from speechevalx.corruption import generate_matched_substitutions
from speechevalx.datasets import ReferenceAction, SpeechEvalSample
from speechevalx.experiment import run_paired_variant
from speechevalx.models import TemplateAgent
from speechevalx.statistics import cei_by_class

tools = [{"name": "transfer_money",
          "parameters": {"recipient": "string", "amount": "number"}}]

sample = SpeechEvalSample(
    sample_id="demo-001",
    benchmark="synthetic",
    audio_path=Path("demo.wav"),                      # unused by run_paired_variant
    reference_transcript="Transfer 22 dollars to Alice",
    tool_schema=tools,
    reference_action=ReferenceAction("transfer_money",
                                     {"recipient": "alice", "amount": 22}),
)

# one eligible single-token substitution per role
variants = generate_matched_substitutions(
    sample.sample_id, sample.reference_transcript,
    {"number": ("22", "32"),
     "entity": ("Alice", "Bob"),
     "function_word": ("to", "for")},
)

agent, rows = TemplateAgent(), []
for corruption, corrupted_transcript in variants:
    trace = run_paired_variant(sample, corruption, corrupted_transcript, agent)
    print(trace.error_class, "|", corrupted_transcript, "->",
          trace.paired_outcome, "|", trace.first_failure)
    trace.save_jsonl(Path("out/paired.jsonl"))        # auditable per-pair record
    rows.append(trace.to_dict())

print(cei_by_class(rows))                             # per-role n / harmed / rescued / loss
```

Expected output:

```
number | Transfer 32 dollars to Alice -> harmed | argument_correctness
entity | Transfer 22 dollars to Bob -> harmed | argument_correctness
function_word | Transfer 22 dollars for Alice -> preserved_success | None
[{'error_class': 'entity', 'n': 1, 'harmed': 1, 'rescued': 0, 'cei': 1.0}, {'error_class': 'function_word', 'n': 1, 'harmed': 0, 'rescued': 0, 'cei': 0.0}, {'error_class': 'number', 'n': 1, 'harmed': 1, 'rescued': 0, 'cei': 1.0}]
```

The two task-bearing edits break the action and the function-word edit does not, even though all three are one-word substitutions.

---

## Using real models

```python
from pathlib import Path
from speechevalx.models import WhisperASR, QwenAgent

asr = WhisperASR("openai/whisper-large-v3", device=0)     # default device=-1 (CPU)
hyp = asr.transcribe(Path("sample.wav")).text

agent = QwenAgent("Qwen/Qwen2.5-7B-Instruct")             # greedy decoding, ≤256 new tokens
out = agent.generate(hyp, tools, instruction="")          # -> AgentResult(raw_output=...)
```

To add another ASR front end or agent, subclass `ASRModel.transcribe(audio_path) -> ASRResult` or `AgentModel.generate(transcript, tools, instruction) -> AgentResult`.

**Expected agent output format.** One Python-like call with keyword arguments only, for example `find_restaurants(location='mumbai', number=3)`. The literal `no_call`, `none` or an empty string is read as a no-call decision. Anything else that fails to parse (positional arguments, `**kwargs`, non-literal values, extra text) is treated as a malformed output and scored as a failure. Surrounding `[...]` and Markdown code fences are stripped.

---

## Input data

Adapters read benchmark files that you have already downloaded. They do not fetch data.

```python
from pathlib import Path
from speechevalx.datasets import VoiceAgentBenchAdapter, BFCLAudioAdapter

for sample in VoiceAgentBenchAdapter(Path("data/voiceagentbench")).iter_samples():
    ...
for sample in BFCLAudioAdapter(Path("data/bfcl_audio")).iter_samples():
    ...
```

Both adapters recursively read `*.json` / `*.jsonl` under the root and accept several field-name variants:

| Adapter | Accepted fields |
|---|---|
| `VoiceAgentBenchAdapter` | `id` / `case_id`; `path` / `audio_path`; `query` / `reference`; `functions` / `available_tools`; `expected_tool_call` / `expected_call`; optional `instruction` |
| `BFCLAudioAdapter` | `id` / `case_id` / `sample_id`; `transcript` / `query` / `instruction`; `audio_path` / `path`; `tools` / `functions` / `tool_schema`; `reference_action` / `expected_tool_call` / `expected_call` (`null` or `"no_call"` → no-call). Unrecognized fields are kept in `sample.metadata`. |

Note that `VoiceAgentBenchAdapter` requires a non-empty expected call, so no-call references are supported through `BFCLAudioAdapter` or by constructing `ReferenceAction(None)` yourself.

### Scoring rules

`score_action(prediction, expected, ScoringConfig(...))` is configuration-driven:

- Strings are stripped and case-folded unless the field is listed in `case_sensitive_fields`; `string_aliases` maps normalized strings to canonical ones (use lower-case keys).
- Numeric strings are normalized, so `"5"` and `5` compare equal.
- Arguments are compared on the *required* keys: `required_arguments[tool]` if given, otherwise every key in the reference. Additional predicted keys are not penalized.
- Output: `valid_call`, `no_call`, `tool_correct`, `argument_exact_match` (requires a correct tool), `jca`.

---

## Outputs: the paired trace

`run_paired_variant(...)` returns a `PairedTrace`. Append it to a JSONL file with `trace.save_jsonl(path)`. Each record links the source and corruption IDs, both transcripts, the reference action, raw and parsed agent outputs for the clean and corrupted paths, the word-level alignment, the edited-span record, tool/argument/JCA flags for both paths, `paired_outcome` (`preserved_success` / `harmed` / `rescued` / `persistent_failure`), `first_failure` (`tool_selection` / `argument_correctness` / `terminal_jca`), `first_changed_failure` (set only for harmed pairs), configuration IDs, the random seed and corruption provenance.

`pipeline.run_single_case(...)` writes a simpler per-case `CaseTrace` JSON file with the ASR hypothesis, WER/CER, parsed prediction and ground truth.

---

## Paired statistics

All helpers take a list of dicts, i.e. `PairedTrace.to_dict()` rows.

| Function | Purpose |
|---|---|
| `cei_by_class(rows)` | Per `error_class`: `n`, `harmed`, `rescued`, and `cei = mean(clean_jca − corrupted_jca) = (H − R)/n`. This is the paper's **L_c** (multiply by 100 for percentage points). |
| `cluster_bootstrap_cei(rows, iterations=1000, seed=7)` | Percentile 95 % CI per class, resampling whole sources (`source_sample_id`). |
| `epr_matrix(rows)` | Share of corrupted variants with a failed action, overall and per class. It includes persistent failures, so it is **not** a harm estimate. |
| `mcnemar_exact(rows)` | Exact two-sided McNemar test on the discordant `clean_jca` / `corrupted_jca` pairs. |
| `holm_adjust({name: p})` | Holm step-down correction for a set of prespecified contrasts. |

Practical notes:

- Select **one variant per source and role** before calling `cei_by_class` (the paper uses an outcome-independent frozen rule). The helper does not enforce this.
- For a role-vs-role contrast, build one row per source with role A's corrupted outcome in `clean_jca` and role B's in `corrupted_jca`, then pass them to `mcnemar_exact` and correct with `holm_adjust`.

---

## Paper ↔ code status

What is in this repository today versus what the paper reports:

| Paper component | Status |
|---|---|
| Paired clean/corrupted execution against one reference action | ✅ `experiment.run_paired_variant` |
| Matched single-token substitutions (same source, count, operation) | ✅ `corruption.generate_matched_substitutions` |
| Safe tool-call parsing, no-call handling, malformed = failure | ✅ `parsing`, `experiment._prediction` |
| Frozen, configurable action scoring (tool / args / JCA) | ✅ `scoring` |
| Word alignment with S/I/D provenance | ✅ `alignment` |
| Auditable paired traces | ✅ `trace.PairedTrace` |
| Paired loss `L_c`, source-cluster bootstrap, exact McNemar, Holm | ✅ `statistics` |
| VoiceAgentBench and BFCL Audio adapters | ✅ `datasets` (local files; the seed-2027 sampling and eligibility filtering scripts are not included) |
| Whisper large-v3 front end | ✅ `models.asr.WhisperASR` |
| SenseVoice-Small and Paraformer-large front ends | ⏳ not yet implemented |
| Qwen2.5-7B-Instruct agent | 🟡 `QwenAgent` (greedy, ≤256 tokens). It does not set bfloat16 or a seed, and the paper's frozen prompt/config is not included. |
| Llama-3.1-8B-Instruct sensitivity agent | ⏳ no dedicated adapter yet |
| Role annotation tooling and annotated span files | ⏳ roles enter only through `Corruption.error_class` |
| Final-output error categories with the paper's precedence (Table 7) | 🟡 only a coarse `first_failure` (tool / argument / terminal) is recorded |
| Corpus WER with NFKC + lower-case + punctuation removal | 🟡 `metrics.wer` is per-utterance, lower-cased and whitespace-tokenized |
| Aggregate JCA / Tool / Arg-EM / No-call tables | ⏳ per-case flags are stored; no aggregation script yet |
| B0–B4 failure-prediction ablation (BERT cosine, Semantic-WER, grouped CV) | ⏳ not yet included |
| Second-model transfer and leave-domain-out evaluation | ⏳ not yet included |
| Repeated-inference audit (50 sources × 3 runs) | ⏳ not yet included |

✅ implemented · 🟡 partial or simplified · ⏳ not in the repository yet

---

## Limitations

- Matched edits control lexical severity, **not** semantic distance or acoustic likelihood.
- Observed ASR hypotheses contain correlated multi-edit errors. Single-edit results should not be generalized to them without separate validation.
- Evidence comes from English data, synthetic speech, current tool schemas, and only 54 no-call cases. Broader validation needs more languages, APIs and agent families.
- The analysis uses reference transcripts and reference actions, so it is an **offline** diagnostic for ASR / model selection, error analysis and regression-test prioritization, not a deployment-time error detector.
- Multi-call cases are out of scope for the primary analysis.

---

## Citation

If you use SpeechEvalX, please cite the paper (update with venue / arXiv details when available):

```bibtex
@misc{wang2026speechevalx,
  title  = {When Does an ASR Error Matter? Task-Role Diagnostics for Tool-Using Voice Agents},
  author = {Wang, Xiaolong and Su, Shijie and Xiong, Naqi},
  year   = {2026},
  note   = {Leibniz University Hannover. Code: https://github.com/invoker-qqwrv/SpeechEvalX}
}
```

## Contact

Corresponding author: Xiaolong Wang (wang_xiaolong1@outlook.com)
