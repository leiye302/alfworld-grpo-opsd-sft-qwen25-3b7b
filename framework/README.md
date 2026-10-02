# Vendored training framework

This directory contains the selected [ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR) training framework at upstream commit `d511f043afe9199b8576fc99ea52e12d84841f32`, with the expert-SFT integration and runtime adaptations described in this project's [method document](../docs/METHOD.md).

**Use the root [README](../README.md) and `scripts/launch.sh` to prepare and launch this project.** The upstream example launchers retained inside the framework are reference source; they are not this project's default experiment or installation entry point.

Upstream work: *Self-Distilled Agentic Reinforcement Learning*, [paper](https://arxiv.org/abs/2605.15155). The framework also includes components from [verl](https://github.com/volcengine/verl), [ALFWorld](https://github.com/alfworld/alfworld) and other projects. Preserve their notices and licenses. The vendored framework's Apache 2.0 license remains in [LICENSE](LICENSE).
