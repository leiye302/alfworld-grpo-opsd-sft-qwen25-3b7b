# Frozen method and comparison contract

Backbone: Qwen2.5-3B-Instruct or Qwen2.5-7B-Instruct, full-parameter training. Each backbone has its own freshly initialized baseline and method run; neither continues a previous1.5B checkpoint.

The shared foundation is the selected SDAR implementation of **GRPO+OPSD**, not the earlier signed-gap modification. Preserve native rewards, invalid-action penalties, grouping/advantages, old-policy probabilities, Reference KL, entropy regularization, mini/microbatch subdivision, Adam and learning-rate scheduling. Teacher scoring finishes before the native Actor updates and is detached.

$$L_k=L_{\mathrm{native\ GRPO+OPSD}}+\mu_k L_{\mathrm{SFT}}.$$

The original SDAR OPSD utility uses

$$\delta_j=\log p_T(y_j)-\operatorname{stopgrad}(\log p_\theta(y_j)),\quad g_j=\operatorname{stopgrad}(\sigma(\beta\delta_j)),$$
$$L_{\mathrm{OPSD}}=\operatorname{agg}\left[g_j(\operatorname{stopgrad}(\log p_T(y_j))-\log p_\theta(y_j))\right].$$

Here `gate_beta=0`, hence the gate is0.5; its coefficient remains0.01 in both groups. This is the code reproduction being compared, not a claim that this sampled-token objective implements the full original OPSD distributional KL. Do not replace it with a signed log-probability-gap advantage or backpropagate into Teacher.

The method adds independent offline expert CE. Each active outer training iteration samples **16 complete trajectories globally with replacement** from the frozen409-episode mixed library:353 screened legacy expert trajectories with72B-written reasoning, plus56 real closed-loop72B trajectories including search/recovery. Their actual supervised decisions are flattened into per-decision inputs with the native history_length2 prompt. This does not mean concatenating the whole episode into an unisolated causal sequence. Context-only Student-prefix or rejected Teacher steps retain the real state chain but do not receive CE.

$$L_{\mathrm{SFT}}=-\frac{\sum_{i\in\mathcal E_k}\sum_j m_{i,j}\log\pi_\theta(y^E_{i,j}\mid h_i^E,y^E_{i,<j})}{\sum_{i\in\mathcal E_k}\sum_j m_{i,j}}.$$

Mask1 covers each approved assistant response: reasoning, action, protocol text and the first native EOS. Mask0 covers prompt/template-supplied tokens, environment observations, padding and excluded decisions. Teacher-only privileged information is not added to Student inputs. Legacy reasoning was written with the expert action supplied to the annotation model; it is not an action-blind reasoning benchmark and this provenance is preserved.

With one-based outer iteration $k$,

$$\mu_k=0.05\left[1+\cos\left(\pi\frac{k-1}{49}\right)\right]\quad(1\le k<50),\qquad\mu_k=0\quad(k\ge50).$$

Thus iteration1 uses0.1 and iteration50 already uses0. Zero-coefficient SFT skips loading/sampling/forward/backward. The OPSD coefficient stays0.01 throughout; the earlier complementary OPSD-growth experiment is not this method.

SFT is added only before the **last existing native mini-batch optimizer step** of each outer iteration, and never creates an additional optimizer step. Rewards, GRPO groups, Student rollout and the original losses remain intact. Eight-rank SFT uses $W/T$ times each rank's supervised token CE sum before FSDP averages ranks, preserving global token-mean normalization. Dummy zero-contribution work gives all ranks matching collective counts. SFT sampling and computation preserve native Student random streams.

Hardware adaptation is shared: one node, eight80GB A100s, inferenceTP2, global train tasks16×group8, PPOmini256, micro32perGPU. The only additional SFT-specific hardware edit permits eight ranks for the complete-trajectory branch; the already dynamic $W/T$ scaling is unchanged. FP32 master parameters/Adam and BF16 compute follow the frozen native backend. Shared response-head memory and SQLite I/O adapters remain enabled in both groups.

Each run ends at150 outer iterations, evaluates before training and every10 iterations, and saves complete resumable FSDP state every25 iterations. The same128-task `valid_seen` manifest fixes task IDs, environment seeds/initial states, per-task-per-turn sampling seeds, temperature0.4 and max50environment steps. Report the prescribed150-step endpoint and the complete curve, not only the peak. One seed and128tasks do not establish robustness or equivalence to an external paper's different evaluation split.
