• Others: DeepSeek-R1 also excels in a wide range of tasks, including creative writing,
general question answering, editing, summarization, and more. It achieves an impressive
length-controlled win-rate of 87.6% on AlpacaEval 2.0 and a win-rate of 92.3% on Are-
naHard, showcasing its strong ability to intelligently handle non-exam-oriented queries.
Additionally, DeepSeek-R1 demonstrates outstanding performance on tasks requiring
long-context understanding, substantially outperforming DeepSeek-V3 on long-context
benchmarks.

2. Approach

2.1. Overview

Previous work has heavily relied on large amounts of supervised data to enhance model
performance. In this study, we demonstrate that reasoning capabilities can be significantly
improved through large-scale reinforcement learning (RL), even without using supervised
fine-tuning (SFT) as a cold start. Furthermore, performance can be further enhanced with
the inclusion of a small amount of cold-start data. In the following sections, we present: (1)
DeepSeek-R1-Zero, which applies RL directly to the base model without any SFT data, and
(2) DeepSeek-R1, which applies RL starting from a checkpoint fine-tuned with thousands of
long Chain-of-Thought (CoT) examples. 3) Distill the reasoning capability from DeepSeek-R1 to
small dense models.

2.2. DeepSeek-R1-Zero: Reinforcement Learning on the Base Model

Reinforcement learning has demonstrated significant effectiveness in reasoning tasks, as ev-
idenced by our previous works (Shao et al., 2024; Wang et al., 2023). However, these works
heavily depended on supervised data, which are time-intensive to gather. In this section, we
explore the potential of LLMs to develop reasoning capabilities without any supervised data,
focusing on their self-evolution through a pure reinforcement learning process. We start with a
brief overview of our reinforcement learning algorithm, followed by the presentation of some
exciting results, and hope this provides the community with valuable insights.

2.2.1. Reinforcement Learning Algorithm

Group Relative Policy Optimization In order to save the training costs of RL, we adopt Group
Relative Policy Optimization (GRPO) (Shao et al., 2024), which foregoes the critic model that is
typically the same size as the policy model, and estimates the baseline from group scores instead.
Specifically, for each question $q$, GRPO samples a group of outputs ${o_1, o_2,…, o_G}$ from the old
policy $\pi_{\theta_{old}}$ and then optimizes the policy model $\pi_{\theta}$ by maximizing the following objective:

$I_{GRPO}(\theta) = E[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]$

$$
\begin{equation}
\frac{1}{G} \sum_{i=1}^{G} (min (\frac{\pi_{\theta} (o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, clip(\frac{\pi_{\theta} (o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1 - \epsilon, 1+ \epsilon) A_i) - \beta D_{KL}(\pi_{\theta}||\pi_{ref}))
\end{equation}
$$

$$
\begin{equation}
D_{KL} (\pi_{\theta}||\pi_{ref}) = -\frac{\pi_{ref}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} log \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} - 1,
\end{equation}
$$
where $\epsilon$ and $\beta$ are hyper-parameters, and $A_i$ is the advantage, computed using a group of
rewards ${r_1,r_2, . . ., r_G }$ corresponding to the outputs within each group:

$$
\begin{equation}
A_i = \frac{r_i – mean(\{r_1, r_2,……,r_G\})}{std(\{r_1,r_2,···,r_G\})}
\end{equation}
$$

A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: prompt. Assistant:

Table 1 | Template for DeepSeek-R1-Zero. prompt will be replaced with the specific reasoning
question during training.

2.2.2. Reward Modeling

The reward is the source of the training signal, which decides the optimization direction of RL.
To train DeepSeek-R1-Zero, we adopt a rule-based reward system that mainly consists of two
types of rewards:
• Accuracy rewards: The accuracy reward model evaluates whether the response is correct.
For example, in the case of math problems with deterministic results, the model is required
to provide the final answer in a specified format (e.g., within a box), enabling reliable
rule-based verification of correctness. Similarly, for LeetCode problems, a compiler can be
used to generate feedback based on predefined test cases.
• Format rewards: In addition to the accuracy reward model, we employ a format reward
model that enforces the model to put its thinking process between ‘<think>' and ‘</think>'
tags.

We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero,
because we find that the neural reward model may suffer from reward hacking in the large-scale
reinforcement learning process, and retraining the reward model needs additional training
resources and it complicates the whole training pipeline.

2.2.3. Training Template

To train DeepSeek-R1-Zero, we begin by designing a straightforward template that guides
the base model to adhere to our specified instructions. As depicted in Table 1, this template
requires DeepSeek-R1-Zero to first produce a reasoning process, followed by the final answer.
We intentionally limit our constraints to this structural format, avoiding any content-specific
biases—such as mandating reflective reasoning or promoting particular problem-solving strate-
gies to ensure that we can accurately observe the model's natural progression during the
reinforcement learning (RL) process.

2.2.4. Performance, Self-evolution Process and Aha Moment of DeepSeek-R1-Zero

Performance of DeepSeek-R1-Zero Figure 2 depicts the performance trajectory of DeepSeek-
R1-Zero on the AIME 2024 benchmark throughout the reinforcement learning (RL) training
process. As illustrated, DeepSeek-R1-Zero demonstrates a steady and consistent enhancement
in performance as the RL training advances. Notably, the average pass@1 score on AIME 2024
shows a significant increase, jumping from an initial 15.6% to an impressive 71.0%, reaching
performance levels comparable to OpenAI-01-0912. This significant improvement highlights the
efficacy of our RL algorithm in optimizing the model's performance over time.

Table 2 provides a comparative analysis between DeepSeek-R1-Zero and OpenAI's 01-0912
models across a variety of reasoning-related benchmarks. The findings reveal that RL empowers