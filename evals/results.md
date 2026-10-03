# Evaluation results (milestone 2.5)

Date of the runs: 2026-10-04. Data: the 4 in-scope 10-Ks (AAPL FY2024 and FY2025, NVDA FY2025 and FY2026), sections Risk Factors and MD&A. Question set: `evals/questions.json`, 40 questions (28 single-company, 5 cross-company, 7 unanswerable, 6 of them on-topic). Raw per-question results: `evals/runs/<run name>.json`.

## How to read this

- A retrieved chunk is **relevant** when its ticker and section match an expected entry and its text contains one of that entry's phrases (case-insensitive, whitespace collapsed). Phrases are used instead of chunk ids because ids change on every re-embed.
- **hit@k**: share of the 33 answerable questions with a relevant chunk in the top k. **MRR@5**: mean of 1/rank of the first relevant chunk (0 if none in the top 5). One question is worth 0.030 of a hit rate.
- Retrieval metrics are computed on the top 5 chunks that `retrieve` returns, BEFORE the relevance threshold. The threshold only feeds the rejection metrics.
- **unanswerable rejected**: share of the 7 unanswerable questions whose best chunk is below `RELEVANCE_THRESHOLD` (0.30) OR (only with `--answers`) whose answer abstains. Retrieval-only runs can only count the threshold, so their value is not comparable with the `--answers` runs. **answerable wrongly rejected**: share of the 33 answerable questions below the threshold.
- **faithfulness**: supported claims / claims, averaged over the answerable answers that did not abstain. **citation coverage**: claims with at least one valid citation (a number from 1 to the number of excerpts) / claims, averaged the same way. **abstention**: share of the 7 unanswerable answers that abstained (the fixed "I don't know" text counts).
- The answers and the judge are both `gpt-5.4-mini`. The judge rule counts a claim without a citation as unsupported, so a missing citation lowers both faithfulness and coverage.

## Comparison

| run | chunk size | rerank | hit@1 | hit@3 | hit@5 | MRR@5 | unanswerable rejected | answerable wrongly rejected | faithfulness | citation coverage | abstention |
|-----|-----------|--------|-------|-------|-------|-------|-----------------------|-----------------------------|--------------|-------------------|------------|
| baseline_1500 | 1500 | off | 0.515 | 0.758 | 0.848 | 0.642 | 0.714 | 0.000 | 0.981 | 0.989 | 0.714 |
| rerank_fast_1500 | 1500 | rerank-v4.0-fast | 0.636 | 0.939 | 1.000 | 0.788 | 0.857 | 0.000 | 0.978 | 0.994 | 0.857 |
| rerank_pro_1500 | 1500 | rerank-v4.0-pro | 0.697 | 0.909 | 1.000 | 0.816 | 0.143 | 0.000 | - | - | - |
| baseline_800 | 800 | off | 0.424 | 0.606 | 0.667 | 0.507 | 0.143 | 0.000 | - | - | - |
| rerank_fast_800 | 800 | rerank-v4.0-fast | 0.606 | 0.788 | 0.879 | 0.713 | 0.143 | 0.000 | - | - | - |

Settings common to all runs: top 5 chunks returned, 20 candidates per search before fusion, `RERANK_CANDIDATES_K` 20 for the rerank runs, `RELEVANCE_THRESHOLD` 0.30, `RRF_K` 60, embeddings `text-embedding-3-small`, chunk overlap 200 for 1500 and 100 for 800 (read back from the stored chunks: the longest chunk was 1,499 and 797 characters, overlap up to 197 and 99). `rerank_pro_1500`, `baseline_800` and `rerank_fast_800` are retrieval only (no `--answers`), so they have no answer columns.

Rank of the first relevant chunk per answerable question (1 to 5, "miss" = not in the top 5):

| question | baseline_1500 | rerank_fast_1500 | rerank_pro_1500 | baseline_800 | rerank_fast_800 |
|----------|---------------|------------------|-----------------|--------------|-----------------|
| a01 | 3 | 1 | 1 | miss | 1 |
| a02 | 1 | 1 | 1 | 1 | 1 |
| a03 | 1 | 2 | 1 | 3 | 2 |
| a04 | 1 | 1 | 1 | 1 | 1 |
| a05 | 2 | 1 | 1 | 1 | 1 |
| a06 | 1 | 1 | 1 | 1 | 1 |
| a07 | 4 | 2 | 2 | 1 | 2 |
| a08 | 4 | 2 | 2 | miss | 2 |
| a09 | miss | 4 | 4 | miss | 4 |
| a10 | 1 | 2 | 4 | 1 | 1 |
| a11 | 1 | 1 | 1 | 1 | 1 |
| a12 | 1 | 2 | 1 | 1 | 1 |
| a13 | 1 | 1 | 1 | 1 | 1 |
| a14 | 1 | 1 | 1 | 1 | 1 |
| n01 | 1 | 1 | 1 | miss | 1 |
| n02 | 3 | 1 | 1 | miss | 1 |
| n03 | 2 | 1 | 1 | 1 | 1 |
| n04 | 3 | 1 | 2 | 3 | 2 |
| n05 | 1 | 1 | 1 | 3 | 1 |
| n06 | 2 | 1 | 1 | 5 | 1 |
| n07 | miss | 2 | 2 | 2 | 1 |
| n08 | 2 | 1 | 1 | miss | miss |
| n09 | 1 | 1 | 1 | 1 | 1 |
| n10 | 1 | 1 | 1 | 1 | 1 |
| n11 | 1 | 3 | 3 | miss | 5 |
| n12 | 1 | 1 | 1 | miss | miss |
| n13 | 5 | 1 | 1 | miss | 2 |
| n14 | miss | 1 | 1 | 1 | 1 |
| c01 | 1 | 2 | 1 | 2 | 3 |
| c02 | 1 | 1 | 1 | 3 | 1 |
| c03 | miss | 3 | 3 | miss | miss |
| c04 | miss | 4 | 2 | miss | miss |
| c05 | 2 | 3 | 4 | 5 | 4 |

## Findings

**Reranking helps a lot, on retrieval.** At chunk size 1500, `rerank-v4.0-fast` raised hit@1 from 0.515 to 0.636 (17 to 21 of 33 questions), hit@3 from 0.758 to 0.939 and hit@5 from 0.848 to 1.000; MRR@5 went from 0.642 to 0.788. The 5 questions the baseline missed completely (a09, n07, n14, c03, c04) were all found in the top 4. Reranking made the top result better on 9 questions (a01, a05, n02, n03, n04, n06, n08, n13, n14) and worse on 5 (a03, a10, a12, n11, c01, each moved to rank 2 or 3).

**Fast or pro: no real difference.** `rerank-v4.0-pro` had hit@1 0.697 (23 of 33) against 0.636 (21) for fast, hit@3 0.909 (30) against 0.939 (31), hit@5 1.000 for both and MRR@5 0.816 against 0.788. The two disagree on 7 questions in both directions, and no metric differs by more than 2 questions, which on 33 questions is noise. Pro is the larger model, fast is described by Cohere as its lighter version, so `rerank-v4.0-fast` stays the default.

**Chunk size 1500 beats 800 clearly.** Without rerank, 800 characters dropped hit@5 from 0.848 to 0.667 (6 questions) and MRR@5 from 0.642 to 0.507; with `rerank-v4.0-fast` it dropped hit@5 from 1.000 to 0.879 (4 questions) and MRR@5 from 0.788 to 0.713. Both comparisons point the same way, and the difference (4 to 6 questions in hit@5, and 0.15 in hit@3 in both comparisons) is larger than the noise between the two rerank models. A likely reason, not tested: at 800 characters a risk factor's heading and its sentences are split into different chunks, so fewer chunks carry both the topic words and the answer. The phrases were checked against both layouts before any paid call, so the drop is not an artefact of the labels. Cost of 1500 versus 800: half as many chunks (440 against 881) and fewer, longer excerpts in the prompt.

**Answer quality did not change measurably.** Faithfulness was 0.981 without rerank and 0.978 with it, citation coverage 0.989 and 0.994: both are above 0.97, and a difference of 0.003 is a fraction of one claim. On this set the model answers faithfully from the excerpts it gets; the gain from reranking is that the right excerpts are present and first. The unsupported claims that the judge found were mostly the model's own hedging sentences ("The excerpts do not mention free or open AI models specifically", "This suggests demand effects are uncertain") and one answer to a hard question (a09).

**The threshold alone does not reject on-topic unanswerable questions.** In every run only the off-topic question (u07, banana bread, similarity 0.095) was below 0.30: 1 of 7 (0.143) in the retrieval-only runs. The six on-topic unanswerable questions had a best similarity of 0.33 to 0.73 across the five runs, in the range of real questions (the lowest best similarity of an answerable question was 0.44 to 0.51 depending on the run). The language model did the rest: it abstained on 4 of the 6 without rerank and on 5 of the 6 with `rerank-v4.0-fast` (abstention 0.714 and 0.857 including the banana-bread question). The difference of one question is the judge's `abstained` flag on u04, whose two answers look alike (both state one supported claim), so it is not a real change.

| id | unanswerable question | best similarity (baseline) | best similarity (rerank fast) | abstained (baseline) | abstained (rerank fast) |
|----|-----------------------|----------------------------|-------------------------------|----------------------|-------------------------|
| u01 | What was the total pay of NVIDIA's chief executive last year, including stock awards? | 0.548 | 0.541 | yes | yes |
| u02 | What will Apple's stock price be at the end of next year? | 0.574 | 0.567 | NO | NO |
| u03 | How much revenue did NVIDIA report for fiscal year 2022? | 0.714 | 0.660 | yes | yes |
| u04 | How many iPhone units did Apple sell in fiscal 2025? | 0.635 | 0.635 | NO | yes |
| u05 | What is the price and driving range of Apple's electric car? | 0.412 | 0.329 | yes | yes |
| u06 | What is the battery life of NVIDIA's smartphone? | 0.412 | 0.382 | yes | yes |
| u07 | What is a good recipe for banana bread? | 0.095 | 0.095 | yes | yes |

An observation, not a change: the two lowest-similarity on-topic questions (u05 and u06, best similarity 0.33 to 0.41 in all five runs) lie below the lowest answerable question (0.44), so a threshold of about 0.42 would have rejected them in every run without wrongly rejecting an answerable question. The margin is thin (0.02) and it concerns two questions, so the threshold stays at 0.30 until a larger set can support a change.

u02 (Apple's stock price next year) was answered in both runs from risk-factor text about stock-price volatility, with citations: not a refusal, and not investment advice either, but not "I don't know". The rerank score is not a substitute for the similarity threshold: the banana-bread chunks still got rerank scores of 0.35 and 0.34, a scale that changes with the model, which is why the threshold stays on `vector_similarity`.

## Reranking on three questions

Before is the top chunk of `baseline_1500`, after is the top chunk of `rerank_fast_1500` (first 150 characters of each).

**n14**: "How might free, open AI models change demand for NVIDIA's products?" — first relevant chunk at rank miss without rerank, rank 1 with `rerank-v4.0-fast`.

- Before (top result, no rerank): NVDA FY2025 risk_factors, chunk 528, similarity 0.526, rerank score -, relevant: no
  > •changes in business and economic conditions; •sudden or sustained government lockdowns or public health issues; •rapidly changing technology or custo
- After (top result, rerank fast): NVDA FY2026 mdna, chunk 412, similarity 0.503, rerank score 0.806, relevant: yes
  > In February 2026, the USG granted a license that would allow us to ship small amounts of H200 products to specific China-based customers. To date, we 

**n08**: "How does NVIDIA depend on TSMC and other chip manufacturers?" — first relevant chunk at rank 2 without rerank, rank 1 with `rerank-v4.0-fast`.

- Before (top result, no rerank): NVDA FY2025 risk_factors, chunk 588, similarity 0.477, rerank score -, relevant: no
  > 26 * * * Table of Contents Over the past three years, we have been subject to a series of shifting and expanding export control restrictions, impactin
- After (top result, rerank fast): NVDA FY2026 risk_factors, chunk 323, similarity 0.443, rerank score 0.759, relevant: yes
  > Dependency on third-party suppliers and their technology to manufacture, assemble, test, or package our products reduces our control over product quan

**n13**: "What risks does NVIDIA see in buying or investing in other companies?" — first relevant chunk at rank 5 without rerank, rank 1 with `rerank-v4.0-fast`.

- Before (top result, no rerank): NVDA FY2026 mdna, chunk 409, similarity 0.667, rerank score -, relevant: no
  > The following discussion and analysis of our financial condition and results of operations should be read in conjunction with "Item 1A. Risk Factors,"
- After (top result, rerank fast): NVDA FY2025 risk_factors, chunk 567, similarity 0.526, rerank score 0.834, relevant: yes
  > We face additional risks related to acquisitions and strategic investments, including the diversion of capital and other resources, including manageme

n13 shows why the similarity score cannot replace a reranker: the baseline's top chunk had a similarity of 0.667, higher than the relevant chunk's 0.526 that reranking put first, and it is the opening paragraph of the MD&A, which only mentions "Item 1A. Risk Factors".

## Chosen defaults

- `CHUNK_SIZE` 1500, `CHUNK_OVERLAP` 200 (unchanged). The 800/100 experiment was re-embedded back to 1500/200; the stored chunks are the same 440 as before (AAPL FY2024 76, FY2025 83; NVDA FY2025 142, FY2026 139) with new ids.
- `RERANK_MODEL` `rerank-v4.0-fast`, `RERANK_ENABLED` true, `RERANK_CANDIDATES_K` 20 (unchanged). The candidate count and `top_k` were not tuned in this milestone.
- `RELEVANCE_THRESHOLD` 0.30 (unchanged, see the threshold finding).

## Cost and calls

| run | embed_query | rerank | answer | judge |
|-----|-------------|--------|--------|-------|
| baseline_1500 | 40 | 0 | 39 | 39 |
| rerank_fast_1500 | 40 | 40 | 39 | 39 |
| rerank_pro_1500 | 40 | 40 | 0 | 0 |
| baseline_800 | 40 | 0 | 0 | 0 |
| rerank_fast_800 | 40 | 40 | 0 | 0 |
| **total** | 200 | 120 | 78 | 78 |

Besides the runs: two full re-embeds (881 and 440 chunks, about 100,000 tokens each), 2 real Cohere calls in the inspection and 9 more in manual checks (the retrieval samples and one chat question). That is 131 Cohere calls in total against the trial key's limit of 1,000 a month (and 10 a minute, which is why the rerank runs wait 6.5 seconds between questions). The Cohere SDK retries 429 and 5xx answers itself, so the real number of HTTP calls can be slightly higher. `baseline_1500` took 2 minutes 19 seconds; a rerank run takes at least 4.3 minutes because of the pacing. OpenAI cost was not read from a billing page; from the token volume (about 78 answer calls and 78 judge calls of roughly 2,000 to 3,000 input tokens, plus 200 query embeddings and 2 re-embeds) it is well under one dollar.

## Known limitations of this evaluation

- **Small set.** 33 answerable questions: one question is 0.030 of a hit rate, so differences of one or two questions (fast against pro, the faithfulness numbers) are noise. Only the reranking effect and the chunk-size effect are larger than that.
- **The same model answers and judges.** `gpt-5.4-mini` judges its own answers, which favours faithfulness. The judge also showed inconsistencies: it listed "the excerpts do not mention ..." sentences as claims, and it marked answers as `abstained` while also listing claims (u04 and u06 in the rerank run). Abstention and faithfulness are therefore soft numbers.
- **Phrase labels under-count.** A relevant chunk that words the idea differently is not counted, so the hit rates are lower bounds. The labels were written by reading the same filings the system uses, and each answerable question has 1 to 16 matching chunks. Two labels are broad (c03 `licensing requirements`, n09 `AI Diffusion`), and for text that repeats in two fiscal years either chunk counts.
- **Retrieval metrics ignore the threshold.** They are computed on the top 5 before the 0.30 filter; no answerable question was below the threshold, so this changed nothing here.
- **Rejection numbers are not comparable across run types.** Retrieval-only runs can only apply the threshold; `--answers` runs add the model's abstention.
- **One embedding model, one prompt, one run per configuration.** Answers vary between runs (no temperature setting is possible for this model), and no run was repeated.
- **Unanswerable questions were checked against the text I read**, not exhaustively; u02 and u04 touch topics the sections mention (stock-price risk, iPhone net sales) without giving the asked-for number.
