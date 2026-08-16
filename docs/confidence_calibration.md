# Confidence Calibration

## Definition and system position

Confidence gating decides whether retrieved evidence is strong enough to send to the LLM:

```text
query -> retrieval -> rerank -> confidence gate -> generation or refusal
```

It reduces unsupported generation risk, but an over-conservative gate creates false refusals and makes answerable requests appear unsupported.

## Inputs, processing, and output

The current implementation uses the Hybrid retrieval score belonging to the candidate placed first after CrossEncoder reranking:

```text
retrieval_score = 0.7 * embedding_score + 0.3 * normalized_bm25_score
confidence_score = reranked_top1.retrieval_score
passed_gate = confidence_score >= threshold
```

CrossEncoder `rerank_score` is not compared with the threshold. CrossEncoder still affects the decision indirectly because it decides which candidate becomes Top1.

The production threshold remains `0.6`.

## Four different states

These terms must not be treated as synonyms:

- **Grounded**: the evaluation dataset says the corpus contains an answer.
- **Accepted**: the selected score passed the confidence gate.
- **Evidence-complete**: all required evidence reached the actual LLM context.
- **Answer-correct**: the generated answer satisfied the evaluation expectation.

A grounded query can be refused. An accepted grounded query can still lack complete context. Complete context can still lead to a generation failure.

## Evaluation design

The evaluation reused the current Query Rewrite, embedding, Hybrid retrieval, CrossEncoder, and Top10 candidate behavior. Each query was retrieved once, then its score was replayed against thresholds `0.40` through `0.70`.

Cases were separated into:

- 54 grounded cases;
- 4 document-unanswerable cases evaluated by the confidence gate;
- 6 sensitive or out-of-scope cases rejected by pre-retrieval guardrails.

The six guardrail cases do not have confidence scores and must not be mixed into the score distribution.

## Results

| Threshold | Grounded accepted | False refusals | False accepts |
|---:|---:|---:|---:|
| 0.40 | 27 | 27 | 0 |
| 0.45 | 19 | 35 | 0 |
| 0.50 | 9 | 45 | 0 |
| 0.55 | 3 | 51 | 0 |
| 0.60 | 1 | 53 | 0 |
| 0.65 | 1 | 53 | 0 |
| 0.70 | 0 | 54 | 0 |

At `0.6`, 46 refused grounded cases already had complete expected evidence in Top5 context. This is strong evidence that the current threshold is over-conservative on the development benchmark.

Grounded scores ranged from `-0.0180` to `0.6617`; document-unanswerable scores ranged from `0.0783` to `0.3222`. The overlap means no scalar threshold can perfectly separate the two labels on current scores.

## Interpretation and trade-offs

`0.40-0.45` is a useful future diagnostic range, not a production-optimal setting. All four current unanswerable cases were refused at `0.40`, but four negatives across seven PDFs cannot represent production traffic.

Lowering a threshold generally improves answerable-query recall, but also gives irrelevant high-scoring evidence more opportunities to reach generation. A ranking score measures relative candidate quality for one query; it is not automatically a calibrated probability that the question is answerable.

Production should remain at `0.6` until a larger confidence-specific negative set and end-to-end safety checks exist.

## Recommended evolution

1. Expand hard unanswerable and partially answerable cases without changing the threshold.
2. Diagnose false refusals and false accepts by retrieval, context, confidence, and generation stage.
3. Re-evaluate the `0.40-0.45` diagnostic region with the expanded negative set.
4. If score overlap remains large, evaluate simple additional signals such as Top1/Top2 margin and evidence coverage.
5. Only after offline and end-to-end validation consider a production change with monitoring and rollback.

Do not begin with a learned confidence model or LLM judge. The current benchmark must first establish whether simple, explainable signals solve the measured problem.

## Common failure modes

- Threshold too high: evidence-backed false refusal.
- Threshold too low: irrelevant evidence is accepted and may cause hallucination.
- Correct source but wrong chunk: acceptance without complete evidence.
- CrossEncoder regression: a weaker Hybrid-scored candidate becomes Top1 and changes the gate input.
- Multi-evidence limitation: Top1 score passes but one required evidence component is absent from context.
- Misleading evaluation: treating every accepted grounded label as a correct final answer.

## Important code and artifacts

- `RAGService.ask()`: selects reranked Top1 retrieval score and applies the gate.
- `is_confident()`: implements `score >= threshold`.
- `eval/run_corpus_v2_confidence_calibration.py`: evaluation-only threshold replay.
- `eval/corpus_v2_confidence_calibration_latest.json`: per-case decisions and aggregate results.

## Transferable skills

- **MUST MASTER**: confusion-matrix terms, gate position, score provenance, stage-based failure classification.
- **PROJECT-LEVEL FAMILIARITY**: score-distribution overlap, evidence-aware acceptance, threshold experiments.
- **AWARENESS ONLY**: learned calibration, LLM-as-judge confidence, self-consistency, statistical production calibration.

The same approach transfers to agent tool execution gates and multimodal document pipelines: distinguish whether input evidence exists, whether the system accepted it, and whether the downstream result was correct.

## Interview explanation

**中文：** 我对 RAG confidence gate 做了离线阈值校准。系统并不直接使用 CrossEncoder 分数，而是使用重排后 Top1 候选的 Hybrid retrieval score。实验发现生产阈值 `0.6` 在当前开发集上产生大量 evidence-backed false refusals，同时 grounded 与 unanswerable 分数存在重叠。因此我没有直接降低生产阈值，而是建议先扩充困难负例，再判断是否需要加入简单的多特征置信信号。

**English:** I built an evaluation-only calibration sweep for the RAG confidence gate. The gate uses the Hybrid retrieval score of the candidate placed first by the CrossEncoder, rather than the CrossEncoder score itself. The current `0.6` threshold caused many evidence-backed false refusals, while grounded and unanswerable score distributions overlapped. I therefore kept production unchanged and recommended expanding hard negative coverage before testing a lower threshold or additional confidence signals.

## Truthful resume wording

Designed an evidence-aware confidence calibration benchmark for a multi-document RAG system, separating retrieval, context, guardrail, and confidence failures; identified severe false-refusal behavior and score-distribution overlap without changing production thresholds.
