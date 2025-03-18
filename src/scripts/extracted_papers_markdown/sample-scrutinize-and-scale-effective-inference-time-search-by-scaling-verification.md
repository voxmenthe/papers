# Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification

*Source: [arXiv:2502.01839](https://arxiv.org/abs/2502.01839)*

*[Submitted on Mon, 3 Feb 2025 21:31:07 UTC]*

# Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification 

Eric Zhao ${ }^{1,2}$ Pranjal Awasthi ${ }^{1}$ Sreenivas Gollapudi ${ }^{1}$<br>${ }^{1}$ Google Research, ${ }^{2}$ UC Berkeley


#### Abstract

Sampling-based search, a simple paradigm for utilizing test-time compute, involves generating multiple candidate responses and selecting the best one-typically by having models self-verify each response for correctness. In this paper, we study the scaling trends governing sampling-based search. Among our findings is that simply scaling up a minimalist implementation of sampling-based search, using only random sampling and direct self-verification, provides a practical inference method that, for example, elevates the reasoning capabilities of Gemini v1.5 Pro above that of o1-Preview on popular benchmarks. We partially attribute the scalability of sampling-based search to a phenomenon of implicit scaling, where sampling a larger pool of responses in turn improves self-verification accuracy. We further identify two useful principles for improving self-verification capabilities with test-time compute: (1) comparing across responses provides helpful signals about the locations of errors and hallucinations, and (2) different model output styles are useful for different contexts - chains of thought are useful for reasoning but harder to verify. We also find that, though accurate verification can be elicited, frontier models demonstrate remarkably weak out-of-box verification capabilities and introduce a benchmark to measure progress on these deficiencies.


## 1 Introduction

Recent advances in language models highlight the importance of test-time compute scaling wherein one uses more compute during inference to enhance reasoning capabilities [OpenAI, 2024, Team, 2025, Agarwal et al., 2024, Wei et al., 2022, Yao et al., 2023, Akyürek et al., 2024]. There are many methods for increasing test-time compute usage, including implicitly encouraging longer responses via reinforcement learning [OpenAI, 2024, Team, 2025] or explicitly via prompting [Wei et al., 2022, Yao et al., 2023]. However, sampling-based search-an instance of the generate-and-test approach where a model generates many responses in parallel, e.g. via random sampling or delegation, and selects what the model guesses to be the best one-remains one of the most natural and fundamental paradigms. In addition to being complementary with other test-time compute scaling strategies, it also has the unique advantage of being embarrassingly parallel and allowing for arbitrarily scaling: simply sample more responses [Cobbe et al., 2021, Wang et al., 2023]. As a result, sampling-based search plays an increasingly crucial role as language

[^0]
[^0]:    Corresponding author: eric.zh@berkeley.edu.
    Code and benchmark: github.com/google-research/google-research/sampling_based_search.

models are set loose on frontier mathematical and scientific problems where inference compute budgets reach thousands of dollars or more per problem.

Though recent works demonstrate the benefits of sampling-based search [Cobbe et al., 2021, Wang et al., 2023, Xue et al., 2023], many questions remain as to what scaling trends govern this fundamental test-time compute scaling strategy. To develop this understanding, we study a minimalist-yet remarkably effective-instantiation of sampling-based search that uses a language model [Gemini Team, 2024] to both generate a set of candidate responses via random sampling and select the best one by attempting to verify each response with natural language. Specifically, we consider the case where models must self-verify their responses to select the best answer, and do not make the strong assumption that one can access groundtruth answers or symbolic systems that exactly verify correctness. In this setup, we address the question: what test-time scaling trends emerge as we scale both the number of sampled responses and verification capabilities? In particular, what are the limits of scaling this simple sampling-based search paradigm and how much does one need to continuously scale verification capability as one scales up search?

Our findings. We first identify scaling trends demonstrating that reasoning performance continues to improve with sampling-based search even as test-time compute is scaled well beyond the point where the performance of self-consistency [Wang et al., 2023] saturates. At sufficient scale, even our minimalist implementation provides a significant leap in reasoning accuracy, lifting Gemini v1.5 Pro performance beyond o1-Preview, and Gemini v1.5 Flash beyond Gemini v1.5 Pro, on reasoning benchmarks such as LiveBench [White et al., 2024] and the AIME [MAA, 2024], exhibiting sustained power-law scaling on the latter. This not only highlights the importance of sampling-based search for scaling capability, but also suggests the utility of sampling-based search as a simple baseline on which to compare other test-time compute scaling strategies and measure genuine improvements in models' search capabilities.

We then attribute much of the strong scaling trends of sampling-based search to an implicit scaling phenomenon. Contrary to the intuition that sampling more responses should impose a greater burden on the verifier and reduce verification accuracy, we observe that scaling sampling indirectly enhances verification accuracy. At a high-level, this is because well-written responses are easier to verify than poorly written responses, and scaling sampling widens the pool of well-written candidates.

We further identify two effective strategies for scaling verification capabilities using test-time compute: (1) directly comparing candidate responses and (2) task-specific rewriting of candidate responses. The former mitigates a core weakness of language models, which struggle to identify mistakes and hallucinations unless given their locations [Tyen et al., 2024], by leveraging the fact that differences between candidate responses provide a strong signal for where errors might be located. The latter leverages our observation of output style suitability where chain-of-thought output formats are beneficial when generating responses but harder to verify than more formal, mathematically conventional writing styles. Surprisingly, while effective verification can be easily elicited from frontier models by communicating these strategies, we observe that frontier models have remarkably poor out-of-box verification capabilities and introduce a new benchmark to quantify these deficits.

Preview and outline. Table 1 summarizes our first finding: that, with effective self-verification, simply scaling sampling-based search is sufficient to approach state-of-art performance on reasoning and math benchmarks (AIME 2024 [MAA, 2024], LiveBench Math, LiveBench Reasoning [White et al., 2024], and the Berkeley MATH dataset [Hendrycks et al., 2021]). It depicts the accuracy of the Gemini v1.5 Pro model

[^0]
[^0]:    ${ }^{1}$ The o1-preview-2024-09-12 numbers in Table 1 use publicly reported figures, with MATH and AIME figures sourced from the OpenAI blog post [OpenAI, 2024], and LiveBench figures sourced from the LiveBench leaderboard (livebench.ai). We found the performance of o1-Preview as accessed through the OpenAI API to slightly differ with publicly reported figures, e.g. scoring $26 \%$ not $44 \%$ on AIME, and scoring $77 \%$ not $67 \%$ on LiveBench Reasoning.

| Method | AIME | MATH | LiveBench Math | LiveBench Reasoning |
| :-- | :--: | :--: | :--: | :--: |
| Pass@1 | $1 / 15$ | $426 / 500$ | $104 / 200$ | $63 / 140$ |
| Consistency@200 | $4 / 15$ | $460 / 500$ | $118 / 200$ | $75 / 140$ |
| Consistency@1,000 | $3 / 15$ | $460 / 500$ | $120 / 200$ | $73 / 140$ |
| Verification@200 | $\mathbf{8} / 15$ | $\mathbf{4 6 7} / 500$ | $\mathbf{1 3 5} / 200$ | $\mathbf{9 7} / 140$ |
| o1-Preview@1 | $7 / 15$ | $428 / 500$ | $131 / 200$ | $95 / 140$ |

Table 1: Accuracy rates of the Gemini v1.5 Pro model using sampling-based search (Verification@200) on reasoning benchmarks, compared to other inference methods and o1-Preview performance. Verification@200 consistently improves on Consistency@200 and surpasses o1-Preview. ${ }^{1}$ Each score reflects a single run, due to the high expense of search at this scale (see Section 5).
[Gemini Team, 2024] when only one solution ${ }^{2}$ is attempted per question (Pass@1), when 200 solutions are attempted and the most common final answer is selected (Consistency@200, Wang et al. [2023]), and under sampling-based search, when 200 solutions are attempted and scored for correctness with the highest scorer selected (Verification@200, Algorithm 1). With sampling-based search (Verification@200), Gemini v1.5 surpasses the performance of o1-Preview, a model explicitly trained on reasoning problems to leverage significant test-time compute and perform internal search.

The rest of this paper is devoted to studying the three key factors behind the numbers in Table 1. Section 2.1 analyzes the remarkable scalability of sampling-based search, as one varies both the compute spent on search and verification; Section 2.2 analyzes the phenomenon of implicit scaling and its role in driving this scalability; and Section 3 discusses important principles for scaling self-verification capability, which may be of independent interest. We also highlight deficits in the verification capabilities of frontier models with a new benchmark in Section 6. Technical details and detailed discussion of related work are found in Sections 5 and 7 respectively.

```
Algorithm 1 Sampling-Based Search (Verification@ \(k_{\text {inf }}\)
Require: Prompt \(Q\), language model LM, scaling parameters \(k_{\text {inf }}, k_{\text {verif }}, k_{\text {tie }}\).
    1: Populate \(\mathcal{S}\) with \(k_{\text {inf }}\) samples from LM("Answer \(Q\) "). \(\triangleright\) Stage 1: Generate Responses
    for each candidate response \(s_{i} \in \mathcal{S}\) do \(\triangleright\) Stage 2: Verify Responses
        Populate \(\mathcal{V}_{i}\) with \(k_{\text {verif }}\) samples from LM("Return 1[response \(s_{i}\) to \(Q\) is correct]").
    4: Gather the highest-scored response \(\mathcal{S}_{\text {Best }}=\left\{s_{i} \mid i \in\left[k_{\text {inf }}\right], \operatorname{Avg}\left(\mathcal{V}_{i}\right) \geq \max _{j \in\left[k_{\text {inf }}\right]} \operatorname{Avg}\left(\mathcal{V}_{j}\right)-0.05\right\}\).
    if \(\left|\mathcal{S}_{\text {Best }}\right|=1\) then
        Return response \(s_{i^{*}}\) where \(i^{*}=\max _{j \in\left[k_{\text {inf }}\right]} \operatorname{Avg}\left(\mathcal{V}_{j}\right)\).
    else
        for each pair of candidate responses \(\left(s_{i}, s_{j}\right) \in\left(\frac{\mathcal{S}_{\text {Best }}}{2}\right)\) do \(\triangleright\) Tie-Break: Compare Responses
            Populate \(\mathcal{C}_{i, j}\) with \(k_{\text {tie }}\) samples from LM("Which of responses \(\left\{s_{i}, s_{j}\right\}\) to \(Q\) is correct?").
        Return response \(s_{i^{*}}\) where \(i^{*}\) is the winner of the most matchups \(\left\{\mathcal{C}_{i, j} \mid s_{i}, s_{j} \in \mathcal{S}_{\text {Best }}\right\}\).
```


# 2 Scaling Trends of Sampling-Based Search 

This section examines how reasoning capability scales with two fundamental test-time compute axes:

[^0]
[^0]:    ${ }^{2}$ As we focus on answering reasoning problems, we use "model responses" and "model solutions" interchangeably.

![img-0.jpeg](img-0.jpeg)

Figure 2.1: Heatmap of Gemini v1.5 Pro accuracy rates using sampling-based search (without tiebreaking) as the number of responses generated (x-axis) and verification attempts (y-axis) increase. Warmer colors indicate higher accuracy (cubic scale). The largest gains occur when scaling both search and verification, with the strongest trend on AIME.

- Search refers to the compute used to discover candidate solutions. In this section, our knob for scaling search is the number of responses sampled for each reasoning problem ( $k_{\text {inf }}$ in Algorithm 1).
- Verification refers to the compute used to scrutinize candidate solutions. Our knob for scaling verification is the number of verification scores we compute and average over per solution ( $k_{\text {verif }}$ ).

For computational reasons, this section uses a streamlined form of Algorithm 1 that omits tie-breaking. This, for example, results in significant underestimates of Verification@k on MATH (see Table 3). All figures are averaged over 20 random seeds, where each run subsamples solutions and verification scores from a primary run that sampled 200 solutions per question and 50 verification scores per solution.

# 2.1 Scaling Trends 

Figure 2.1 provides a heatmap of Verification@k on each benchmark in Table 1 as we scale search and verification. In addition to clear burn-in costs along both axes of scale, we can observe that the largest performance gains are realized when search and verification are both scaled. These trends also indicate

![img-1.jpeg](img-1.jpeg)

Figure 2.2: Plot of Gemini v1.5 Pro accuracy rates using sampling-based search (without tie-breaking and with $k_{\text {verif }}=50$ ) on ambiguous questions only as the number of responses generated increases. A question is ambiguous when the model generates at least one candidate response with a correct final answer. Accuracy on ambiguous questions increases with search.
that the performances of sampling-based search, as reported in Table 1, have not yet been scaled to saturation on these benchmarks. This scaling trend is strongest on the AIME benchmark, where performance is bottlenecked by $k$ (search); we attribute this bottleneck to the difficulty of the AIME questions resulting in correct solutions only appearing with very low probability (see Table 2).

# 2.2 Implicit Scaling 

Scaling sampling-based search along the search axis by sampling more solutions, i.e. increasing $k$, should have two effects on performance that partially cancel out: (1) the verifier must discriminate between more solutions, increasing the likelihood of error and (2) the generator is more likely to produce at least one solution that reaches a correct final answer, i.e. Pass@k increases.

To isolate the first effect, we study the model's Verification@k accuracy on "ambiguous" questions: questions where at least one of the model's $k$ candidate solutions reaches the correct final answer (note that Pass@k equals the number of ambiguous questions). Figure 2.2 and Figure 2.3 do exactly this, plotting Verification@k accuracy measured only on ambiguous questions from each benchmark. To reduce noise in these figures, we deterministically omit benchmark questions that Consistency@200 answers correctly or where, with high probability, 50 random responses result in either all correct or all incorrect final answers.

After controlling for the growth of Pass@k, we should expect a trend of decreasing accuracy if we increase $k$ but keep the number of verification attempts constant. However, Figure 2.2 shows the reverse trend: accuracy increases with $k$. This demonstrates an implicit scaling of verification accuracy, where increasing the number of generated responses increases not only the chance that at least one response is correct (Pass@k) but also the chance that at least one of the correct responses is of higher quality. Here, quality can be understood as the rigour or flawlessness of a response; a lower quality solution may be generally correct but fail to justify a non-trivial step or err in a non-critical step of its reasoning.

Implicit scaling suggests that verification should become more accurate, and sampling-based search should become more effective, with the use of more capable base models that produce more sound reasoning and compelling proofs of correctness. Because the number of ambiguous questions strictly increases with more candidate solutions, the implicit scaling effect also explains the overall accuracy scaling gains in Figure 2.1: larger $k$ increases both the number of ambiguous questions (Pass@k) and accuracy on the set of ambiguous questions.

![img-2.jpeg](img-2.jpeg)

Figure 2.3: Heatmap of Gemini v1.5 Pro accuracy rates using sampling-based search (without tiebreaking) on ambiguous questions only as the number of responses generated (x-axis) and verification attempts (y-axis) increase. Warmer colors indicate higher accuracy (linear scale). A question is ambiguous when the model generates at least one candidate response with a correct final answer. Accuracy on ambiguous questions increases with search (x-axis).

# 2.3 The Long Tail of Response Distributions 

![img-3.jpeg](img-3.jpeg)

Figure 2.4: Line graph depicting the accuracy rates of the Gemini v1.5 Pro model using samplingbased search as the number of candidate responses generated is scaled upwards. The number of verification attempts is fixed at 50 for all plots. The depicted accuracies are obtained without tiebreaking and may be lower than reported elsewhere. Verification@k improves with $k$ even when Consistency@k stagnates on AIME and LiveBench Reasoning.

We can directly observe Verification@k scaling beyond the saturation point of Consistency@k in Figure 2.4, where we plot their performance after fixing the number of verification attempts at 50 . On AIME, the most technically challenging benchmark, Verification@k demonstrates power law scaling even as Consistency@k begins to plateau. The rapid saturation of Consistency@k can be attributed to the fact that, while it is effective at small scales in averaging out noisy mistakes, it necessarily plateaus as it converges on the most probable response; for example, Consistency@50 has the same accuracy as Consistency@10,000 on AIME. Consider cheaply sampling a vast set of solutions from a weak but ergodic model: Consistency@k is unlikely to return a correct solution, but an effective verifier should still be expected to detect rare but correct solutions in the long-tail of the response distribution. We find an example of this on the AIME 2024 exam, where the Gemini v1.5 model struggles to identify the correct answer to Problem 11 on Exam II. Table 2 shows the final answers from 200 randomly sampled Gemini v1.5 solutions, of which only one is correct ("601," in green). Consistency returns the incorrect answer of " 1 " (in red), which appears in over half the responses. In contrast, Verification successfully identifies the solution reaching the correct answer from the response distribution's long-tail, assigning a $\leq 36 \%$ score to each solution reaching a final answer of " 1 " but a $98 \%$ score to the single solution reaching " 601 ". Scaling verification capability is key to driving improved search, allowing for discerning between answers that appear correct with $98 \%$ vs. $76 \%$ confidence. The fact that verification can be used to so effectively leverage the long-tail of model response distributions also suggests that Pass@k, not Pass@1, should be the key performance metric for search applications. Existing post-training techniques (e.g., reinforcement learning from human feedback (RLHF) [Ouyang et al., 2022]) which explicitly optimize for Pass@1 may potentially be doing so at the expense of Pass@k and inhibiting search capability.

## 3 Effective Self-Verification in Natural Language

In the process of scaling sampling-based search, we identified two general principles for eliciting more accurate language model self-verification, that may be of independent interest.

1. Compare responses to localize errors. Disagreements between candidate solutions strongly signal the potential locations of their errors. This can be leveraged to combat the fact that language models

| Problem 11, AIME 2024 | Verification Score | Final Answer | \# Solutions |
| :--: | :--: | :--: | :--: |
|  | 0.95 | 101 | (Wrong) |
| Find the number of triples of nonnegative integers $(a, b, c)$ | 0.76 | 6 | (Wrong) |
| satisfying $a+b+c=300$ and $a^{2} b+a^{2} c+b^{2} a+b^{2} c+c^{2} a+c^{2} b=$ $6,000,000$. | 0.52 | 0 | (Wrong) |
|  | 0.40 | 7 | (Wrong) |
|  | 0.38 | 4 | (Wrong) |
|  | 0.36 | 1 | (Wrong) |
|  | 0.22 | 10 | (Wrong) |
|  | 0.20 | 3 | (Wrong) |
|  | 0.18 | 301 | (Wrong) |
|  | 0.16 | 45451 | (Wrong) |
|  | 0.14 | 101 | (Wrong) |
|  | 0.06 | 2 | (Wrong) |
|  | 0.04 | 45151 | (Wrong) |
|  | 0.04 | 303 | (Wrong) |
|  | 0.00 | 100 | (Wrong) |

Table 2: The final answers identified by the Gemini v1.5 Pro model to Problem 11 on AIME 2024, sorted by verification score and annotated with their multiplicity in 200 solution generations. The correct final answer (green) is only found by 1 generated response whereas Consistency@200 selects an incorrect final answer (red) that is found by 124 generated responses.
have low recall (i.e., often overlook) when asked to identify mistakes and hallucinations [Tyen et al., 2024, Kamoi et al., 2024a], as models are able to identify errors when provided their locations [Tyen et al., 2024]. Specifically, we can improve the self-verification of a candidate response by providing the verifier with other responses to compare the candidate against - an instance of implicit scaling.
2. Rewrite responses for output style suitability. The optimal output style of a language model should depend on the task. Writing in a linear chain of thought - which includes detailing reasoning before committing to a claim-is effective when generating responses (search) [Wei et al., 2022]. However, responses are easier to verify when written rigorously, hierarchically, and modularly. This can be leveraged by having verifiers first rewrite candidate responses in, e.g., an expanded mathematically conventional theorem-lemma-proof format rather than directly evaluating chains-of-thought.

These principles also provide levers for scaling self-verification capability with test-time compute, including by (1) sampling and providing verifiers with more responses to compare between and (2) rewriting responses with increasing rigour and structure.

# 3.1 Sampling-Based Search Implementation 

We now detail our minimalist implementation of sampling-based search (summarized in Algorithm 1) that uses only parallelizable blackbox queries to a language model. It generates candidate responses by randomly sampling from models and select responses by asking models to self-verify; prompts are identical across all benchmarks and provided in the source code.

Step 1: Generate Candidate Responses. A language model generates $k_{\text {inf }}$ candidate responses (candidate solutions) in parallel to each question, using temperature $\sigma_{\text {inf }}$.

Step 2: Verify Candidate Responses. A language model generates $k_{\text {verif }}$ binary "verification scores" for each candidate in parallel, indicating whether its final answer is correct. Each scoring attempt is a single conversation thread that rewrites the response as a theorem, supporting lemmas, and proofs (examples in Appendix E) and systematically scans for errors. The highest scoring response is selected.

Tie-Break: Compare Candidate Responses. When the three highest scoring candidates score within $5 \%$ of one another and disagree on the final answer, a language model directly compares the responses in pairwise matchups. Each matchup is a single conversation thread that identifies where responses diverge and, at each such point, determines which side is correct. Each matchup is repeated $k_{\text {tie }}=100$ times. The response with the most wins in the round-robin tournament is selected.

# 3.2 Ablation Studies 

We can individually ablate the practices of comparing and rewriting candidate responses to confirm their role in eliciting greater verification capability.

Ablating comparisons. The step of asking models to directly compare candidate solutions with similar verification scores significantly increases sampling-based search performance. This is demonstrated in Table 3, where we depict the accuracy rates from Table 1 alongside the accuracy rates after ablating the tie-breaking step. These comparisons have the greatest impact when models struggle from low recall and excessively assign high verification scores. On the MATH benchmark, which sees the greatest lift from comparisons, the average verification score of the top 3 candidate responses is nearly $90 \%$. Recall that, as a result, the figures reported in Section 2 that omit tiebreaking significantly underestimate sampling-based search performances (Verification@k).

| Dataset | Cons@200 | Verification@200 |  |
| :-- | :--: | :--: | :--: |
|  |  | Without | With Tie-Break |
| MATH | $460 / 500$ | $457 / 500$ | $467 / 500$ |
| LiveBench Math | $118 / 200$ | $125 / 200$ | $135 / 200$ |
| LiveBench Reasoning | $75 / 140$ | $94 / 140$ | $97 / 140$ |
| AIME | $4 / 14$ | $7 / 14$ | $8 / 14$ |

![img-4.jpeg](img-4.jpeg)

Table 3: Accuracy rates of Gemini v1.5 Pro using sampling-based search, with and without tiebreaking. Tie-breaking provides most of Verification@200's gains on Consistency@200 (Cons@200) on MATH and LiveBench Math, and smaller gains on AIME and LiveBench Reasoning.

Ablating rewritings. We explored a limited number of prompts for self-verification, including prompts which omit instructing the model to rewrite responses. We did not perform further prompt optimization and expect refinements would boost accuracy. Table 4 shows each prompt's probability of mislabeling correct solutions (false positive) and incorrect solutions (false negative), with the former generally having a more severe impact on downstream performance. We evaluated these prompts on 1,080 candidate responses to 54 level-5 questions from the MATH training split, and 120 candidate responses to 6 questions from AIME 2023. A response is marked as incorrect if, of 20 verification attempts, the number finding an error in the solution exceeds the equal error rate threshold.

Main refers to manually written prompts used in our experiments. Shortened refers to a shorter variant of "Main" that omits, e.g., instructions to avoid truncation. Without Rewrite refers to a variant of "Main" that omits instructing the verifier to first rewrite responses. Split-Context refers to a variant of "Main" that creates separate conversation threads to individually verify pieces of the response.

The gap between the performance of "Main" and "Without Rewrite" demonstrates that ablating the rewriting of solutions negatively impacts verification performance. Similarly, the gap with "Split-Context" demonstrates that splitting the verification process into separate conversation threads sharply decreases performance due to low precision, which we attribute to miscalibration.

| Prompt Style | MATH |  | AIME |  |
| :-- | :--: | :--: | :--: | :--: |
|  | FPR | FNR | FPR | FNR |
| Main | $14 \%$ | $17 \%$ | $7 \%$ | $7 \%$ |
| Shortened | $17 \%$ | $17 \%$ | $7 \%$ | $7 \%$ |
| Without Rewrite | $16 \%$ | $18 \%$ | $11 \%$ | $12 \%$ |
| Split-Context | $19 \%$ | $23 \%$ | $11 \%$ | $14 \%$ |

![img-5.jpeg](img-5.jpeg)

Table 4: Verification scoring accuracy rates of the Gemini v1.5 Pro model for various prompts. False positive rate (FPR) refers to how often a correct response is labeled as incorrect; false negative rate (FNR) refers to how often an incorrect response is labeled as correct.

# 4 Additional Experiments 

### 4.1 Smaller Models

We also observe sampling-based search to be a powerful tool for enhancing smaller, lower-cost models. Here, we apply sampling-based search to Gemini v1.5 Flash model, which has a nearly 20x lower inference cost than Gemini v1.5 Pro. Table 5 lists the performance of using the Flash model to evaluate candidate responses generated by the Pro model (Pro+Flash), and the performance of using the Flash model end-to-end for sampling-based search (Flash). Sampling-based search still provides a significant improvement in performance for both Flash and Pro+Flash. Moreover, Verification@200 still provides significant improvements over Consistency@200, albeit lesser in magnitude than for end-to-end use of Gemini Pro. In addition, Flash Verification@200 using Gemini Flash is competitive with Pro Consistency@200, while Pro+Flash Verification@200 exceeds Pro Consistency@200. We highlight that Pro+Flash Verification@200 has roughly the compute cost of Consistency@500-as our sampling-based search implementation is minimally optimized for efficiency, we expect costs to further decrease.

### 4.2 Performance by Subtask

The LiveBench benchmarks each consist of multiple subtasks. In Table 6, we break down the numbers reported in Table 1 for each of these subtasks. We also provide in Table 6 the Pass@200 scores of the Gemini Pro model, which measure the probability that of 200 attempted responses to a question at least one is correct. Pass@200 upper bounds what one can hope to achieve through Verification or Consistency. Verification provides the greatest gains on AIME 2024, Web-of-Lies, Competition, and Zebra Puzzle. In contrast, Verification does not improve on Consistency on the Olympiad task of the LiveBench Math

| Model | Method | AIME | MATH | LiveBench Math | LiveBench Reasoning |
| :--: | :--: | :--: | :--: | :--: | :--: |
| Pro v1.5 | Pass@1 | $1 / 15$ | $426 / 500$ | $104 / 200$ | $63 / 140$ |
|  | Consistency@200 | $4 / 15$ | $460 / 500$ | $118 / 200$ | $75 / 140$ |
|  | Verification@200 | $8 / 15$ | $467 / 500$ | $135 / 200$ | $97 / 140$ |
| Flash v1.5 | Pass@1 | $2 / 15$ | $407 / 500$ | $96 / 200$ | $65 / 140$ |
|  | Consistency@200 | $3 / 15$ | $440 / 500$ | $92 / 200$ | $84 / 140$ |
|  | Verification@200 | $5 / 15$ | $445 / 500$ | $104 / 200$ | $84 / 140$ |
| Pro+Flash v1.5 | Verification@200 | $7 / 15$ | $456 / 500$ | $119 / 200$ | $84 / 140$ |

![img-6.jpeg](img-6.jpeg)

Table 5: Accuracy rates with sampling-based search using either the Gemini v1.5 Pro model to both generate and verify responses (Pro), Gemini v1.5 Flash to both generate and verify responses (Flash), or Gemini v1.5 Pro model to generate responses and v1.5 Flash to verify responses (Pro+Flash). Verification@200 exceeds Consistency@200 for all model choices, while Pro+Flash Verification@200 matches or exceeds Pro Consistency@200.

| Dataset | Cons@200 | Verif@200 | Improvement (\%) |  | Pass@200 | \# Questions |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  |  |  | (Abs) | (Rel) |  |  |
| Berkeley MATH | $92.0 \%$ | $93.4 \%$ | $2 \% \uparrow$ | $20.0 \%$ | $99.0 \%$ | 500 |
| AIME 2024 | $26.7 \%$ | $53.3 \%$ | $100 \% \uparrow$ | $57.1 \%$ | $73.3 \%$ | 15 |
| Web-of-Lies-v2* | $75.5 \%$ | $91.8 \%$ | $22 \% \uparrow$ | $66.5 \%$ | $100.0 \%$ | 49 |
| Spatial* | $33.3 \%$ | $46.7 \%$ | $40 \% \uparrow$ | $21.5 \%$ | $95.6 \%$ | 45 |
| Zebra Puzzle* | $50.0 \%$ | $67.4 \%$ | $35 \% \uparrow$ | $36.4 \%$ | $97.8 \%$ | 46 |
| Competition ${ }^{\dagger}$ | $66.2 \%$ | $83.1 \%$ | $26 \% \uparrow$ | $63.1 \%$ | $93.0 \%$ | 71 |
| AMPS Hard ${ }^{\dagger}$ | $70.6 \%$ | $77.7 \%$ | $10 \% \uparrow$ | $33.5 \%$ | $91.8 \%$ | 85 |
| Olympiad ${ }^{\dagger}$ | $25.0 \%$ | $22.7 \%$ | $9 \% \downarrow$ | $-11.2 \%$ | $45.5 \%$ | 44 |

Table 6: The Pass@200, Consistency@200 (Cons@200), and Verification@200 (Verif@200) accuracy rates of the Gemini v1.5 Pro model using sampling-based search. LiveBench Math ${ }^{\dagger}$ and LiveBench Reasoning ${ }^{*}$ numbers are divided per task. Absolute \% Increase (Abs) is the percentage improvement of Verification@200 over Consistency@200. Relative \% Increase (Rel) is (Verification@200 Consistency@200) / (Pass@200 - Consistency@200).

benchmark. We attribute this to the unique question design of LiveBench Olympiad task questions, which is incompatible with our implementation of Verification (see Appendix B.2).

# 5 Technical Details 

All experiments are run on Google Cloud with Gemini v1.5-Pro-002 and Gemini v1.5-Flash-002 models dated to September 2024. Unless otherwise specified, the default parameters for our implementation of sampling-based search (Section 3) are $k_{\text {inf }}=200, \sigma_{\text {inf }}=1.5, k_{\text {verif }}=50, \sigma_{\text {verif }}=1$, and a maximum of 8,192 output tokens per query. For all benchmarks, the scoring of candidate responses is performed using a language model rather than literal string comparison; details are in Appendix A.2.

Preliminary scoring. When generating $k_{\text {verif }}=50$ verification scores per candidate solution is too expensive, we first generate $k_{\text {verif }}=10$ preliminary verification scores and discard candidate solutions with an average score below 0.2 . If a final answer is represented by more than 15 candidate responses, only the top 15 -as measured by average preliminary score, tie-breaking randomly-are kept. This results in a smaller pool of candidate solutions for which we compute all $k_{\text {verif }}=50$ verification scores. Preliminary scoring is used on all datasets except AIME, which consists of 15 questions.

Compute. On AIME, the verification process involves 32,000 characters (roughly 13,000 tokens) of model output. Extrapolating from these figures, running the full sampling-based search pipeline on a question for $k_{\text {inf }}=200$ and $k_{\text {verif }}=50$ requires $200 \cdot 50 \cdot 13,000 \approx 130 \mathrm{M}$ output tokens. At around $\$ 5 / 1 \mathrm{M}$ output tokens (public pricing of Gemini v1.5 Pro), this evaluates to approximately $\$ 650$ in cost. Preliminary scoring reduces usage of output tokens by roughly $70 \%$, resulting in a per-question cost of $\$ 200$. The use of Gemini Flash for verification further decreases cost to $\$ 12$ per question.

Datasets. Our MATH benchmark consists of 500 questions from the PRM800K [Lightman et al., 2024] test split of Berkeley MATH [Hendrycks et al., 2021]. Our LiveBench Math benchmark consists of 200 randomly subsampled questions from the 368 available as of October 21st 2024, including AMC12 2023, AIME 2024, SMC 2023, USAMO 2023, IMO 2023, and synthetic math questions [White et al., 2024]. Our LiveBench Reasoning benchmark consists of 140 questions from the 150 available as of October 21st 2024, including Zebra puzzles, Web-Of-Lies, and Spatial reasoning questions [White et al., 2024]. Our AIME benchmark consists of the 15 questions in Exam II of AIME 2024 [MAA, 2024].

## 6 A Verification Benchmark

Frontier language models demonstrate a remarkable mismatch between their problem-solving capabilities and poor out-of-box verification capabilities. These limitations have largely been attributed to the inability of current language models to self-diagnose hallucinations or enforce rigour [Zhang et al., 2023, Orgad et al., 2024, Snyder et al., 2024, Kamoi et al., 2024a, Tyen et al., 2024, Huang et al., 2024]. However, our findings that models can be directed to accurately perform verifications at scale suggest that these out-of-box limitations can be addressed with standard methods like instruction tuning. We compiled a set of challenging reasoning problems and candidate solutions to provide a benchmark for these deficits.

Each entry in this benchmark consists of a question, a correct candidate response, and an incorrect candidate response, and is manually curated from the residuals of our sampling-based search experiments (Section 3). An example entry from this benchmark can be found below (see Appendix D for more).

# Question from LiveBench Reasoning (Web-of-Lies Puzzle) 

In this question, assume each person either always tells the truth or always lies. The person at the campground thinks their friend is lying. Mateo is at the aquarium. The person at the restaurant says the person at the hotel lies. Farid is at the movie theater. The person at the movie theater says the person at the campground lies. Ryan is at the shopping mall. The person at the cafe says the person at the campground lies. The person at the observatory says the person at the museum lies. The person at the museum says the person at the restaurant tells the truth. The person at the campground says the person at the airport tells the truth. The person at the campground tells the truth. Luna is at the restaurant. The person at the movie theater saw a firetruck. Isabella is at the observatory. The person at the amusement park says the person at the museum tells the truth. The person at the restaurant says the person at the shopping mall tells the truth. Liam is at the beach. Ayaan is at the airport. The person at the restaurant says the person at the beach tells the truth. Maya is at the amusement park. The person at the restaurant says the person at the aquarium lies. The person at the restaurant says the person at the movie theater lies. The person at the library says the person at the campground tells the truth. Elowen is at the hotel. Priya is at the library. The person at the gym says the person at the restaurant tells the truth. Kehinde is at the cafe. Anika is at the museum. Quan is at the campground. Soren is at the gym. Does the person at the museum tell the truth? Does the person at the restaurant tell the truth? Does the person at the movie theater tell the truth? Think step by step, and then put your answer in ${ }^{* *}$ bold ${ }^{* *}$ as a list of three words, yes or no (for example, ${ }^{* *}$ yes, no, yes ${ }^{* *}$ ). If you don't know, guess.

## Correct response

Here's how we can break down this logic puzzle:

1. **Quan is at the campground and tells the truth.** This is a given fact.
2. **The person at the restaurant (Luna) says the person at the hotel (Elowen) lies.** We don't know yet if Luna is lying or telling the truth.
3. **Since Luna tells the truth about Farid, Luna *always* tells the truth.** This unlocks several other pieces of information.
4. **The person at the cafe (Kehinde) says the person at the campground lies.** Since Quan tells the truth, Kehinde is lying. Therefore: * **Anika (museum):** Tells the truth. * **Luna (restaurant):** Tells the truth. * **Farid (movie theater):** Lies. **yes, yes, no**

## Incorrect Response Due to Hallucination

Here's how we can solve this logic puzzle:

1. **Quan is at the campground and tells the truth.** This is given.
2. **The person at the airport (Ayaan) tells the truth.** Quan (campground) says the person at the airport tells the truth, and Quan tells the truth.
3. **Luna (restaurant) lies.** The person at the restaurant says the person at the hotel (Elowen) lies. Later, the person at the restaurant says the person at the campground (Quan) lies. Since Quan tells the truth, Luna (restaurant) must be lying.
.. Therefore: * **Anika (museum): lies** * **Luna (restaurant): lies*** **Farid (movie theater): lies**
**Answer: no, no, no**
Figure 6.1: Example of an entry in our verification benchmark. The question is sourced from the LiveBench Reasoning benchmark, and the two responses are generated by Gemini v1.5 Pro. The green response has the correct final answer; that red response has the wrong final answer due to hallucinating a non-existent clause.

On each entry, our benchmark studies verification accuracy on two tasks:

1. Scoring task. When given only the question and one of the responses, is the model able to discern the correctness of the response?
2. Comparison task. When provided the whole tuple with the correctness labels of the responses masked and a guarantee that at least one response is correct, is the model able to discern which response is correct and which is incorrect?

The scoring task is also evaluated over a separate set of (question, response) pairs where the response reaches the correct final answer by coincidence but contains fatal errors and should be labeled by a reasonable verifier as being incorrect; an example can be found in Appendix D. In the scoring task, models are provided only with the task description; in the comparison task, models are provided only with the task description and a suggestion to identify disagreements between responses in its reasoning.

Table 7 lists the baseline performances of current commercial model offerings on this benchmark. Gemini v1.5 Pro is omitted from the benchmark as the entries in the benchmark are curated from the residuals of Gemini v1.5 Pro. The prompts used in Table 7 are provided in Appendix A.4.

As we previously observed, and has been noted in prior works [Tyen et al., 2024, Kamoi et al., 2024a], verification errors are typically due to low recall. Even the easier comparison task, models perform only marginally better-and often worse-than random chance. In many cases, Consistency@5 performs worse than one-shot inference because Consistency simply averages out noise from an output distribution, meaning that a model biased towards producing an incorrect answer will do so with higher probability under Consistency. Addressing these deficits in verification capabilities-which we see as low-hanging fruit for post-training-would enable not only better sampling-based search, but also other downstream applications of verification including reinforcement learning [e.g. OpenAI, 2024, Team, 2025], data flywheeling [e.g., Welleck et al., 2022], and end-user experience (see Section 7 for further discussion).

# 7 Related Work 

Test-time compute. Many of the recent advances in language model reasoning capabilities can be traced to increasing use of test-time compute. Inference strategies like chain-of-thought reasoning [Wei et al., 2022], tree-of-thoughts [Yao et al., 2023] and self-critique [Valmeekam et al., 2023] result in improved reasoning performance at the cost of forming longer responses. Reinforcement learning has emerged as a particularly successful strategy for effectively leveraging more test-time compute, wherein models learn from exploration to form lengthy chain-of-thought outputs that incorporate backtracking and search, despite not being explicitly taught to do so [OpenAI, 2024, Team, 2025]. Inference-time model adaptation, whether through many-shot learning [Agarwal et al., 2024, Anil et al., 2024] or finetuning [Akyürek et al., 2024], provides another avenue when training data is available. We study sampling-based search: obtain a set of candidate responses from a model and apply an aggregation method to select a response, such as self-consistency/plurality voting [Wang et al., 2023] or selecting a response with a reward/verifier model [Cobbe et al., 2021]. These various methods for scaling test-time compute are complementary; for example, sampling-based search can also be used on models trained to produce longer outputs. We note that it is possible for models trained to produce long chains of thought to perform something resembling samplingbased search internally, in which case we still expect our observed scaling trends to hold. However, we also expect explicit sampling-based search will remain indispensable, due to its greater parallelism and robustness than internally implemented search.

Scaling sampling-based search. The paradigm of sampling-based search provides three main knobs for scaling: generation, sampling, and selection. While the cost of generating each individual response

| Model | Metric | Scoring Accuracy |  |  | Comparison Accuracy |
| :--: | :--: | :--: | :--: | :--: | :--: |
|  |  | Correct | Wrong | Flawed |  |
| GPT-4o | Pass@1 | $76.5 \%$ | $31.0 \%$ | $22.2 \%$ | $43.2 \%$ |
|  | Consistency@5 | $77.4 \%$ | $30.0 \%$ | $11.1 \%$ | $35.4 \%$ |
| Claude 3.5 Sonnet | Pass@1 | $89.6 \%$ | $22.5 \%$ | $33.3 \%$ | $56.1 \%$ |
|  | Consistency@5 | $90.3 \%$ | $17.5 \%$ | $33.3 \%$ | $61.2 \%$ |
| o1-preview | Pass@1 | $100 \%$ | $68.8 \%$ | $80.0 \%$ | $84.5 \%$ |
|  | Consistency@5 | $100 \%$ | $79.4 \%$ | $88.8 \%$ | $92 \%$ |
| Gemini 2.0 Flash | Pass@1 | $73.5 \%$ | $44.5 \%$ | $60 \%$ | $58 \%$ |
|  | Consistency@5 | $77.4 \%$ | $42.5 \%$ | $66.6 \%$ | $58.7 \%$ |
| Gemini 2.0 Thinking Flash | Pass@1 | $75.4 \%$ | $56.5 \%$ | $53.3 \%$ | $80 \%$ |
|  | Consistency@5 | $77.4 \%$ | $55 \%$ | $55.5 \%$ | $89.1 \%$ |
| Random guessing |  | $80 \%$ | $20 \%$ | $20 \%$ | $50 \%$ |

Table 7: Accuracy rates of commercial language models on our verification benchmark. For the task of response scoring (Scoring Accuracy), accuracy rates are broken down for entries that require identifying a correct response as being correct (Correct), entries that require identifying a wrong response as being wrong (Wrong), and entries that require identifying a wrong response that coincidentally reaches the correct answer as being wrong (Flawed). GPT-4o and Claude 3.5 Sonnet only perform marginally better than random guessing across all tasks. o1-Preview performs better, but still fails to identify $20-30 \%$ of wrong responses.

can be scaled with previously mentioned interventions, such as chain-of-thought [e.g. Wei et al., 2022], reinforcement learning [e.g. OpenAI, 2024], or inference-time adaptation [e.g. Anil et al., 2024], the cost of sampling a set of responses can be scaled by increasing the number of responses generated [Wang et al., 2023, Snell et al., 2024]. We use random sampling to generate each set of candidate responses, which means the latter corresponds to simply taking more random draws. However, this sampling can also be implemented in an agentic fashion, with a central model delegating the generation of responses so as to perform search more systematically. The process of selecting a response can be scaled by using more expensive rules: self-consistency provides a simple plurality voting rule at the lowest-cost end of the spectrum [Wang et al., 2023], while language model self-verification [e.g. Xue et al., 2023, see below] and learned verification/reward models [e.g. Cobbe et al., 2021, see below] provide a range of selection strategies that vary in cost and capability. For more fine-grained control over the scaling of self-verification in our experiments, we apply plurality voting [Wang et al., 2023] to self-verification and vary our number of verification attempts per response.

Verification of language model outputs. A large body of recent work has studied the self-verification capabilities of large language models [e.g., Cobbe et al., 2021, Kadavath et al., 2022, Saunders et al., 2022, Kim et al., 2023, Xie et al., 2023, Weng et al., 2023, Zhang et al., 2023, Xue et al., 2023, Li et al., 2023, Liu et al., 2024, Chow et al., 2024, Jiang et al., 2024, Dhuliawala et al., 2024, Snyder et al., 2024, Wu et al., 2024, Huang et al., 2024, Kamoi et al., 2024a,b, Orgad et al., 2024, Wen et al., 2024, Tyen et al., 2024, Chen et al., 2024, Kumar et al., 2024, Qu et al., 2024, Zhang et al., 2024, Ko et al., 2025, Havrilla et al., 2024]. While some works-including ours-simply ask models to perform verification and parse the response, others have proposed custom methods of performing self-verification, including: recreating the problem from the response [Xue et al., 2023, Wu et al., 2024], masking and re-filling parts of the response [Weng et al., 2023, Jiang et al., 2024], creating a rubric [Dhuliawala et al., 2024], or asking models to choose from options [Xie et al., 2023, Chen et al., 2024]. Our work does not focus on optimizing for selfverification or advocate for any particular strategy. However, in the course of performing our scaling study, we did identify several previously unstudied principles of self-verification that only arise at sufficiently large scale and may be of independent interest, including implicit scaling, output style suitability, and the importance of directly comparing responses. Other related bodies of work study the learning of verifiers, often on top of a pretrained large language model [e.g. Cobbe et al., 2021, Saunders et al., 2022, Li et al., 2023, Havrilla et al., 2024, Kumar et al., 2024, Qu et al., 2024, Chow et al., 2024, Zhang et al., 2024], and the use of external tools for verification [e.g. Min et al., 2023, Gou et al., 2024, Gao et al., 2024, Kim et al., 2023]. We did not train customized verification models or permit verifier use of external tools in the listed experiments, as we found blackbox model access to be sufficient for effective verification at scale. The limitations of model self-verification capabilities are also well-studied [Kamoi et al., 2024a, Tyen et al., 2024, Huang et al., 2024], and can be remedied with external information [Huang et al., 2024] or hints for localizing errors [Tyen et al., 2024]. Models especially struggle with self-diagnosing hallucinations [Zhang et al., 2023, Orgad et al., 2024, Snyder et al., 2024], despite awareness of their own limitations [Kadavath et al., 2022], and are often incentivized to obfuscate errors [Wen et al., 2024].

Applications of verification. In addition to being used to select from candidate responses [Cobbe et al., 2021, Li et al., 2023, Weng et al., 2023, Jiang et al., 2024, Chen et al., 2024, Xie et al., 2023], verifiers can be used to guide iterative improvements to a model's output by providing feedback to the generating model [Kim et al., 2023, Xue et al., 2023, Valmeekam et al., 2023, Wu et al., 2024, Huang et al., 2024, Dhuliawala et al., 2024, Stechly et al., 2024a,b, Qu et al., 2024, Havrilla et al., 2024, Ko et al., 2025]. Another important application of verification is in enhancing model capabilities. For example, verification results for model outputs can be fed back into models as feedback via in-context

reinforcement learning [Shinn et al., 2023], reinforcement learning [Uesato et al., 2022, Peng et al., 2023, Madaan et al., 2023, Kumar et al., 2024, Chow et al., 2024], or finetuning [Welleck et al., 2022, Paul et al., 2024, An et al., 2024, Singh et al., 2024], in an approach known as data flywheeling. Verification has also been explored as a means of encouraging models to produce better written responses [Anil et al., 2021, Kirchner et al., 2024]. From a product perspective, verification capabilities are also important to the workflow of end users [Collins et al.].

# 8 Conclusion 

This paper studied the scaling trends governing sampling-based search, finding that (1) it scales remarkably well even with simple implementations, (2) implicit scaling plays a big role in this scalability, and (3) self-verification capability can be scaled with test-time compute using two key principles: comparisons localize errors, and responses should be rewritten for output style suitability. To this end, we scaled a minimalist, embarrassingly parallel implementation of sampling-based search that, with sufficient test-time compute, is sufficient to attain state-of-art performance on a range of reasoning benchmarks.

Our results underscore the importance of the sampling-based search paradigm. Given that it complements other test-time compute scaling strategies, is parallelizable and allows for arbitrarily scaling, and admits simple implementations that are demonstrably effective, we expect sampling-based search to play a crucial role as language models are tasked with solving increasingly complex problems with increasingly large compute budgets. We also see the performance of sampling-based search as providing both a strong baseline scaling trend that any non-trivial inference strategy should exceed, and a meaningful measure of a model's search capability when Pass@k is uninformative (e.g. on multiple choice exams). We anticipate model self-verification capabilities to rapidly improve in the short term, as models learn to leverage the principles of implicit scaling and output style suitability, and drive improved scaling rates for samplingbased search. Finally, our results also highlight the importance of being able to effectively sample massive and diverse sets of solutions for search. This calls for more systematic inference alternatives to random sampling, such as agentic approaches that delegate search, and inference-aware optimization methods that maximize, e.g., Pass@k performance rather than Pass@1.

# References 

OpenAI. Introducing OpenAI o1-preview. https://openai.com/index/ introducing-openai-o1-preview/, 2024.

DeepSeek-AI Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025 .

Rishabh Agarwal, Avi Singh, Lei M. Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, John D. Co-Reyes, Eric Chu, Feryal Behbahani, Aleksandra Faust, and Hugo Larochelle. Many-shot in-context learning, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Ekin Akyürek, Mehul Damani, Linlu Qiu, Han Guo, Yoon Kim, and Jacob Andreas. The surprising effectiveness of test-time training for abstract reasoning, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, November 2021.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

Tianci Xue, Ziqi Wang, Zhenhailong Wang, Chi Han, Pengfei Yu, and Heng Ji. RCOT: Detecting and Rectifying Factual Inconsistency in Reasoning by Reversing Chain-of-Thought, October 2023.

Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024 .

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. Livebench: A challenging, contamination-free llm benchmark, 2024.

Mathematical Association of America MAA. AIME 2024 Problem Set, 2024.
Gladys Tyen, Hassan Mansoor, Victor Cârbune, Peter Chen, and Tony Mak. LLMs cannot find reasoning errors, but can correct them given the error location, June 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

Ryo Kamoi, Sarkar Snigdha Sarathi Das, Renze Lou, Jihyun Janice Ahn, Yilun Zhao, Xiaoxin Lu, Nan Zhang, Yusen Zhang, Ranran Haoran Zhang, Sujeeth Reddy Vummanthala, Salika Dave, Shaobo Qin, Arman Cohan, Wenpeng Yin, and Rui Zhang. Evaluating LLMs at Detecting Errors in LLM Responses, July 2024a.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A. Smith. How Language Model Hallucinations Can Snowball, May 2023.

Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov. LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations, October 2024.

Ben Snyder, Marius Moisescu, and Muhammad Bilal Zafar. On Early Detection of Hallucinations in Factual Question Answering. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 2721-2732, Barcelona Spain, August 2024. ACM. ISBN 9798400704901.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. Generating Sequences by Learning to Self-Correct, October 2022.

Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. Can large language models really improve by self-critiquing their own plans?, 2023.

Cem Anil, Esin Durmus, Nina Rimsky, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Meg Tong, Jesse Mu, Daniel J Ford, Francesco Mosconi, Rajashree Agrawal, Rylan Schaeffer, Naomi Bashkansky, Samuel Svenningsen, Mike Lambert, Ansh Radhakrishnan, Carson Denison, Evan J Hubinger, Yuntao Bai, Trenton Bricken, Timothy Maxwell, Nicholas Schiefer, James Sully, Alex Tamkin, Tamera Lanham, Karina Nguyen, Tomasz Korbak, Jared Kaplan, Deep Ganguli, Samuel R. Bowman, Ethan Perez, Roger Baker Grosse, and David Duvenaud. Many-shot jailbreaking, 2024.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language Models (Mostly) Know What They Know, November 2022.

William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing models for assisting human evaluators, June 2022.

Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Qizhe Xie. Self-evaluation guided beam search for reasoning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 2550-2575. Association for Computational Linguistics, 2023.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making Large Language Models Better Reasoners with Step-Aware Verifier, May 2023.

Dancheng Liu, Amir Nassereldine, Ziming Yang, Chenhui Xu, Yuting Hu, Jiajie Li, Utkarsh Kumar, Changjae Lee, Ruiyang Qin, Yiyu Shi, and Jinjun Xiong. Large Language Models have Intrinsic SelfCorrection Ability, December 2024.

Yinlam Chow, Guy Tennenholtz, Izzeddin Gur, Vincent Zhuang, Bo Dai, Sridhar Thiagarajan, Craig Boutilier, Rishabh Agarwal, Aviral Kumar, and Aleksandra Faust. Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models, December 2024.

Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T. Kwok. Forwardbackward reasoning in large language models for mathematical verification. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 6647-6661. Association for Computational Linguistics, 2024.

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. Chain-of-verification reduces hallucination in large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, $A C L$

---
*[BibTeX citation](sample-scrutinize-and-scale-effective-inference-time-search-by-scaling-verification.bib)*
