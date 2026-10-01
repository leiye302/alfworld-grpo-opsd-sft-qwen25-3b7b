# Handoff contract

Read README.md and docs/METHOD.md before working. The authorized task is the four fresh Qwen2.5-3B/7B experiments, original SDAR GRPO+OPSD baseline and expert full-trajectory SFT, each to150 outer iterations, one node8×A10080GB.

Use scripts/bootstrap.sh and scripts/handoff.py prepare/check/run-all. Only obtain the locked Release asset and the two locked Qwen model repositories. Do not download unrelated source-server files, an Analyzer, a72B model, or old checkpoints. Do not request or publish source-server credentials. Private GitHub access must use the recipient's own authorized account.

Do not modify the objective, coefficient schedule, optimizer placement, masks, global mini-batch, rollout count, fixed validation manifest, normalization, rewards or evaluation protocol. Preserve the original SDAR OPSD code; the signed-gap alteration and complementary OPSD-growth experiment are out of scope. The method alone adds16 global trajectory draws per active round, token-mean reasoning/action CE, initial0.1cosine to0 at outer iteration50.

Verify allocation before using all eight GPUs; never stop someone else's processes. Keep weights/checkpoints/logs/tokens out of Git. Keep outputs inside the chosen work directory, preserve complete resumable checkpoints, consolidate observations into per-rank/round logs, and report actual verification rather than claiming GPU tests from CPU evidence. Use --resume only after the old process exits.
