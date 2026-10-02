# GRPO + OPSD with decaying expert supervision

The default experiment trains Qwen2.5-3B-Instruct from its official base weights for 150 outer iterations on one node with eight A100 40GB GPUs. The root [README](../README.md) is the installation and launch guide. A fresh clone does not need the author's directories, servers, credentials or historical experiment files.

## Training objective

The foundation is the selected SDAR implementation of GRPO + OPSD, upstream commit `d511f043afe9199b8576fc99ea52e12d84841f32`. Preserve its rewards, invalid-action penalties, advantages, old-policy log probabilities, Reference KL, entropy term, mini/microbatch subdivision, optimizer steps and learning-rate schedule. Teacher scores are fixed before Actor updates and detached.

$$L_k=L_{\mathrm{native\ GRPO+OPSD}}+\mu_k L_{\mathrm{SFT}}.$$

The selected native sampled-token OPSD utility is

$$\delta_j=\log p_T(y_j)-\operatorname{stopgrad}(\log p_\theta(y_j)),\qquad g_j=\operatorname{stopgrad}(\sigma(\beta\delta_j)),$$

$$L_{\mathrm{OPSD}}=\operatorname{agg}\!\left[g_j\left(\operatorname{stopgrad}(\log p_T(y_j))-\log p_\theta(y_j)\right)\right].$$

Here `gate_beta=0`, so the gate is 0.5, and its loss coefficient stays 0.01. This identifies the selected code reproduction; it does not claim that this sampled-token objective is the full distributional KL of the original OPSD paper. Do not substitute the earlier signed-gap variant or differentiate through Teacher.

## Expert SFT

Every active outer iteration samples **16 complete expert trajectories globally with replacement**. The supplied library has 409 episodes: 353 screened expert-action episodes with 72B-written reasoning, and 56 real closed-loop 72B episodes including search and recovery. Flatten approved decisions into separate examples with their own native history-length-2 prompt. Do not concatenate different examples into one ordinary causal sequence.

$$L_{\mathrm{SFT}}=-\frac{\sum_{i\in\mathcal E_k}\sum_j m_{i,j}\log\pi_\theta(y^E_{i,j}\mid h_i^E,y^E_{i,<j})}{\sum_{i\in\mathcal E_k}\sum_j m_{i,j}}.$$

Mask 1 covers the approved assistant response: reasoning, action, protocol text and first native EOS. Mask 0 covers prompt/template-supplied content, environment observations, padding and excluded decisions. Student takeover prefixes and rejected Teacher steps remain in the real history, but receive no CE. The normal Student input has no hindsight or other Teacher-only information.

Legacy reasoning was generated with the expert action supplied to the annotation model. It is not an action-blind reasoning benchmark. The real closed-loop portion preserves the Teacher's actual responses and simulator transitions. Data source and supervision rules are described in [data/README.md](../data/README.md).

With one-based **outer** iteration $k$,

$$\mu_k=0.05\left[1+\cos\left(\pi\frac{k-1}{49}\right)\right]\quad(1\le k<50),\qquad\mu_k=0\quad(k\ge50).$$

Iteration 1 uses 0.1; iteration 50 already uses zero. Once zero, skip SFT loading, sampling and forward/backward. OPSD remains at 0.01 throughout.

SFT contributes immediately before the **last existing native mini-batch optimizer step** of the outer iteration. It adds no optimizer step and does not alter Student rollouts or GRPO grouping. If $T$ is the global supervised-token count and $W$ the number of ranks, each rank backpropagates $W/T$ times its local CE sum before distributed averaging. Dummy zero-contribution work ensures matching collective counts. Separate SFT sampling/computation preserves native Student random streams.

The episode index uses portable IDs and selected original step indices. It has no file-checksum gate. Explicit per-round random seeds preserve the supplied experiment's default draw sequence even when file locations or JSON layout change.

## Hardware, validation and saving

The eight-card profile keeps tasks 16 × rollouts 8, PPO mini-batch 256, per-GPU logical micro-batch 32, PPO epochs 1, training temperature 1, prompt/response budgets 2048/512, maximum environment steps 50, history length 2 and inference TP 2. FP32 persistent parameters/Adam and BF16 compute follow the supplied backend.

The 40GB adapter moves saved tensors to CPU earlier at 16,384 packed tokens, 512 MiB head storage, or less than 8 GiB free device memory. It preserves computation shapes, logical microbatches, loss reduction and optimizer placement. The mapped-host allocator reserves 48 GiB per rank, 384 GiB total; require at least 450 GiB actually available host RAM. CPU tests do not constitute an eight-A100 GPU training benchmark.

Evaluate initially and every 10 outer iterations on the same 128 `valid_seen` tasks. Task IDs, environment seeds, initial observations and per-task/per-turn generation seeds are fixed; temperature is 0.4 and the environment budget is 50 steps. Game paths resolve beneath the user's own asset directory. These are not `valid_unseen` paper test scores.

Save complete native training state at 50, 100 and 150. After a newer save returns, confirm all eight rank state archives, dataloader state, tokenizer and the latest-step marker before retiring the previous checkpoint. A failed save retains the previous complete checkpoint and restores its marker. Native pre-write rotation is disabled. Final completion retains step 150, all metrics, evaluations and SFT sampling records.

Report the prescribed endpoint and complete validation curve. A single seed with 128 tasks does not establish robustness or equivalence to an external paper's evaluation.
