# ALFWorld：GRPO + OPSD + Expert SFT

在 Qwen2.5-3B-Instruct 上，将专家完整 reasoning/action 的监督学习加入 SDAR 原生 GRPO＋OPSD。默认使用**单机 8×A100 40GB**，从基模训练 **150 轮**。

这是独立项目：代码、专家数据和固定验证任务随仓库提供，环境资产由公开 Release 下载，基模从官方模型仓库下载。**无需访问作者的服务器、获得 GitHub 授权、调用外置 API，或手动计算任何哈希。** 数据不绑定作者的目录，也不要求文件匹配旧实验的 SHA。开跑检查关注格式、历史/动作对应、训练配置和机器是否能够运行。

## 快速开始

先用**你自己指定的 SSH 地址或别名**登录训练服务器。“已有 SSH 密钥”是认证方式，不是选择目标主机的依据。本仓库不提供远端登录目标；缺少地址时先向服务器所有者确认。

机器需要 Linux x86_64、Python 3.12、`git`、`tmux`、`g++`、支持 CUDA 12.8 的 NVIDIA 驱动，以及分配给本任务的八张 A100 40GB。当前存储适配器要求至少 **450 GiB 实际可用主机内存**，建议安装 1 TiB；数据盘预留 **200～250 GB**。不要停止其他用户的进程来取得资源。

下面的 `/data/alfworld` 是示例。换成自己的数据盘路径，路径尽量短，避免 Ray socket 路径过长。在已经登录的训练服务器执行：

```bash
mkdir -p /data/alfworld
git clone https://github.com/leiye302/alfworld-grpo-opsd-sft-qwen25-3b7b.git /data/alfworld/repo
cd /data/alfworld/repo
bash scripts/launch.sh /data/alfworld
```

该入口在 `tmux` 中依次安装独立环境、下载环境资产和 **3B 基模**、完成运行检查并启动 **3B SFT 方法组**。默认不会下载或训练 7B，也不会启动 baseline。安装和下载阶段也可断开 SSH；训练服务器本身仍需保持运行。

```bash
tmux attach -t alfworld-3b-sft
tail -n 40 /data/alfworld/3b_sft_queue.log
tail -n 40 /data/alfworld/runs/3b_sft/logs/driver.log
```

**tmux 创建成功只代表进程已启动，不代表检查通过或第一轮训练成功。** 日志中应先出现检查结果，再出现真实训练指标。如果 Python 3.12 的命令不是 `python3.12`，在启动前设置 `PYTHON312=/实际路径/python`。

## 方法和训练设置

$$L_k=L_{\mathrm{GRPO+OPSD}}+\mu_k L_{\mathrm{SFT}}.$$

保留选定 SDAR 原生 GRPO＋OPSD、Reference KL、奖励处理、动作有效性惩罚、mini-batch 划分和 optimizer 更新位置。OPSD 系数固定为 **0.01**。SFT 每个活跃外层轮次全局随机抽 **16 条完整专家轨迹**，对被选中监督的 reasoning、action、协议标签和原生 EOS 按有效输出 token 取全局平均；历史、环境反馈和 padding 不计入 CE。

$$\mu_k=\frac{0.1}{2}\left[1+\cos\left(\pi\frac{k-1}{49}\right)\right]\quad(1\le k<50),\qquad\mu_k=0\quad(k\ge50).$$

第 1 轮系数 0.1，第 50 轮归零。SFT 加在本轮**最后一次已有 mini-batch 更新**前，不增加 optimizer step。归零后不再采样或计算 SFT。完整定义和实现边界见 [docs/METHOD.md](docs/METHOD.md)。

| 设置 | 默认值 |
|---|---|
| 基模 | Qwen2.5-3B-Instruct |
| 外层训练轮数 | 150 |
| 每轮任务数 × 每组 rollout | 16 × 8 |
| PPO mini-batch / 每 GPU 逻辑 micro-batch | 256 / 32 |
| PPO epochs | 1 |
| 训练采样温度 | 1.0 |
| 最大 prompt / response token | 2048 / 512 |
| 最大环境步数 / 历史长度 | 50 / 2 |
| 推理 TP / vLLM 显存比例 | 2 / 0.6 |
| 验证 | 训练前及每 10 轮，固定 128 个 valid_seen 任务 |
| 保存 | 第 50、100、150 轮 |

八卡 40GB 适配保留逻辑 micro-batch、计算形状和归一化，只调整反向保存张量的 CPU 存储时机及显存容量检查。FP32 持久参数/Adam、BF16 计算沿用训练后端。当前映射主机内存适配器每 rank 预留 48 GiB，八 rank 共 384 GiB；不能只根据显卡容量判断主机内存是否足够。维护者的 CPU 检查不等于目标八卡的完整训练实测。

## 数据与下载

- `data/expert/episodes.jsonl`：409 条专家轨迹；353 条筛选后的专家动作轨迹及补写 reasoning，56 条真实闭环 Teacher 轨迹，含搜索和恢复。被排除步骤与 Student 接管前缀保留为历史，但不监督。
- `data/expert/manifest.json`：轨迹 ID、选择的步骤和固定随机流元数据，不要求校验旧文件哈希。修改不相关的路径或 JSON 排版不会改变默认 SFT 抽样流。
- `data/fixed_validation/manifest.json`：固定任务、相对游戏路径、环境种子和初始观察；运行时映射到自己的工作目录。验证温度 0.4，开启采样，逐任务/逐决策使用固定生成种子。
- Release `assets-v1/alfworld-runtime-assets-v1.tar.gz`：约 464 MB 压缩、3 GB 展开，含 ALFWorld SQLite 游戏资产、训练/验证 parquet 和依赖模块。没有模型权重、Analyzer 或训练检查点。
- 官方 `Qwen/Qwen2.5-3B-Instruct`：下载固定 revision，默认不下载无关权重。

ALFWorld 游戏通过一个只读 SQLite 文件提供，避免展开大量小文件。用户不需要创建任何作者的旧目录。详细字段、监督边界与数据来源见 [data/README.md](data/README.md)。基模不上传 GitHub，使用时遵守其官方许可证。

## 分步执行与恢复

需要逐步操作时：

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python scripts/handoff.py prepare --work /data/alfworld
python scripts/handoff.py check --work /data/alfworld --gpu
python -u scripts/handoff.py run --work /data/alfworld
```

最后一条适合在已有 tmux 会话中执行。`run` 和 `run-all` 默认都只运行 `3b_sft`；`plan` 显示选定的单组生效配置。只在明确要做其他实验时显式指定 `--size 7b` 或 `--method baseline`。准备和检查也按所选 backbone 执行。

检查自动核对数据格式、样本/历史/动作对应、真实 tokenizer、训练预算和 GPU/内存。不要求旧数据 SHA。`prepare --verify-downloads` 可选进行自动下载完整性校验，普通启动不需要使用它。

确认旧进程已退出后，在同一个工作目录续跑：

```bash
bash scripts/launch.sh /data/alfworld --resume
```

恢复完整 Actor、Adam、学习率、dataloader/RNG 和外层轮次，继续到总计 150。**没有完整检查点时 `--resume` 会报错，不会悄悄从基模重跑。** 已存在实验的新跑入口也不会覆盖它。

每次新 checkpoint 完整保存后检查八 rank 的训练状态与最新轮数标记；100 轮保存成功后删除 50，150 轮保存成功后删除 100。保存失败保留上一份完整状态。最终只保留 150 轮完整可续训状态，验证与训练记录始终保留。

## 结果与运行检查

运行目录为 `runs/3b_sft`：

- `reports/launch_args.json`、`reports/identity.json`：生效参数、代码版本、基模版本和数据集名称。
- `logs/metrics.jsonl`：reward、成功率、entropy、length、梯度、KL、OPSD/SFT loss 和系数等原生指标。
- `logs/actor_audit_rank*.jsonl`：各 rank 更新次数、参数/Adam 精度、Teacher detach 和 GPU 峰值。
- `fixed_validation/results`：每次固定验证的分项结果与压缩真实轨迹。
- `checkpoints`：最新完整训练状态、SFT 抽样记录和保存轮数标记。

第 1 轮应跨过至少两个 mini-batch，检查更新次数、实际参数变化、权重同步和显存；报错时保留日志，不擅自改变 batch、精度、loss 或验证设置。不要用训练 reward 代替验证成功率，也不要把 valid_seen 写成 valid_unseen 论文测试成绩。

```bash
python scripts/collect_results.py --work /data/alfworld
```

结果只汇总选定的 3B 组，输出 CSV、合并指标和验证曲线到 `results/3b_sft`。

当前 3B 完整 checkpoint 估计约 38 GB；保存新的一份时暂时同时保留旧的一份，峰值约 76 GB，再计约 6 GB 基模、环境、缓存与日志。至少预留 200 GB，不按最终单份容量准备磁盘。没有目标 A100 实测速率前，不给出确定完工时间；用前 3～5 轮的实际耗时估算，前后 SFT 阶段分别统计。

## 开发与来源

CPU 检查入口：`python scripts/check_public_project.py`、`python scripts/check_checkpoint_retention.py`、`python scripts/check_a10040_profile.py`。真实 tokenizer/环境检查由 `handoff.py check` 执行；八卡 GPU 检查由 `check --gpu` 执行。它们检查实现是否能够正确运行，不要求用户手动做哈希。

训练框架基于 [ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)，选定上游 commit `d511f043afe9199b8576fc99ea52e12d84841f32`；环境基于 [ALFWorld](https://github.com/alfworld/alfworld)。保留上游源码许可证和各依赖许可证，新增部分沿用仓库 LICENSE。模型许可按对应官方 model card。维护检查结果见 [evidence/open_source_review.json](evidence/open_source_review.json)。
