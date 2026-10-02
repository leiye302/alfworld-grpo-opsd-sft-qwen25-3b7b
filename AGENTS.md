# Project instructions

Read README.md, docs/METHOD.md and data/README.md. This is a standalone public project. Work on the training host explicitly selected by the current user, using their SSH hostname/alias, port and username. Existing local SSH keys do not select a destination. Never infer a source machine from provenance or old documents. No author-server access, GitHub authentication, external API key or manual hash calculation is required.

The default task is ONE fresh Qwen2.5-3B-Instruct method experiment to 150 outer iterations on one node with 8×A10040GB. Use scripts/launch.sh <work-dir>; it prepares only the required 3B model and public Release, then starts checks/training in tmux. Do not start 7B or baseline unless the current user explicitly asks. Download/check only the selected backbone. A created tmux session is not proof that training passed.

Preserve the original selected SDAR GRPO+OPSD, Reference KL, rewards, masks, global batch, rollout count, mini/micro subdivision, optimizer placement and fixed validation protocol. The method adds 16 global complete expert-trajectory draws per active outer round, token-mean full reasoning/action CE, initial coefficient 0.1 with cosine decay to zero at iteration 50. OPSD stays 0.01. No extra optimizer step. Do not replace OPSD with signed-gap reweighting or the complementary coefficient experiment.

Data paths are portable and relative. Do not require old dataset/manifest hashes to load or train. Retain structural checks of identities, histories, actions, token masks, supervision flags and loss denominators. Explicit sampling-stream seeds preserve the existing default draw sequence independently of dataset formatting. Optional download integrity checks are not a training prerequisite.

Use only allocated GPUs; inspect both compute and graphics processes and never stop someone else's job. Keep environments, downloaded models, outputs, caches and temporary files inside the chosen project/work directory. Keep private keys, credentials, model weights and checkpoints out of Git. Consolidate logs instead of making per-token files.

Save complete native checkpoints at 50/100/150. Retire the previous checkpoint only after the new eight-rank save finishes and its structural checks pass. Do not enable pre-write rotation. Resume only after the old process exits; absence of a complete checkpoint must raise rather than silently initialize from the base model. Retain metrics, fixed validation and sampling records.

The 40GB profile changes saved-tensor storage placement and GPU capacity checks only. Keep logical micro32, mini256, TP2, vLLM fraction0.6, FP32 persistent state and the 384GiB mapped-host reservation/450GiB available-RAM requirement. Report actual test coverage; CPU tests are not an eight-A100 FSDP training acceptance.
