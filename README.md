# ALFWorld：Qwen2.5-3B /7B，GRPO＋OPSD＋专家轨迹SFT

这是可交给另一位研究者或AI agent的训练交接仓库。本次任务是**单机8×A10080GB，只跑3B和7B两个SFT方法组，各自从对应基模训到150轮**。仓库保留baseline源码和配置供核对配方；本次交接不启动baseline实验。

| 组别 | 初始化 | 训练目标 | 结束轮数 |
|---|---|---|---:|
| `3b_sft` | Qwen2.5-3B-Instruct | 选定SDAR源码的原版GRPO＋OPSD＋专家完整reasoning/action SFT |150|
| `7b_sft` | Qwen2.5-7B-Instruct | 同一方法 |150|

**OPSD固定系数0.01；SFT初始系数0.1，按外层轮数余弦衰减，第50轮归零；活跃轮每轮全局抽16条专家轨迹，监督有效输出token的全局token-mean。** 两组均保留Reference KL、奖励处理、动作有效性惩罚和原生mini-batch更新。不是先前修改过的带符号OPSD，也不是OPSD增长/SFT下降的互补日程。方法公式和实现边界见[docs/METHOD.md](docs/METHOD.md)。

固定验证使用已有的**128个valid_seen任务**，包括初始状态、环境种子和每个任务每个决策的生成种子。训练前验证一次，随后每10轮验证；每50轮保存完整可续训检查点，150轮结束。**第100轮保存并验收后删50轮，第150轮保存并验收后删100轮，最终每组只留150轮。** 两个backbone各自从对应基模开始，不共享训练后的权重。

## 给接手AI的执行指令

任务已经授权为上述两个SFT方法组训练。先读本README和METHOD，准备环境/下载资产，完成检查，然后启动两组队列。**仅运行`3b_sft`和`7b_sft`，不启动baseline。** 不要修改方法、全局batch、采样/验证协议、损失归一化或训练预算；不要使用之前1.5B实验的checkpoint初始化3B/7B。不要把OPSD换成另一个版本。只下载本仓库Release的指定资产和下文两个锁定基模，不下载来源服务器的其它内容，不需要公司服务器登录权限或任何外置模型API密钥。

两组默认**依次**使用全部8张卡：`3b_sft → 7b_sft`。接手者自己的GitHub账号需先被仓库所有者邀请为协作者并接受邀请。本仓库继续保持私有；网页访问、代码拉取和Release下载都需要该账号授权。训练服务器先用`gh auth login`登录自己的账号，再执行下文命令。也可安全设置自己的`GH_TOKEN`；不要把token写入代码、文档、命令行日志或Git提交。

## 1. 机器准备

- Linux x86_64、Python3.12，建议Ubuntu22.04；具备`git`、`g++`、Python venv、正常NVIDIA驱动和CUDA12.8运行能力。
- 单机**8×A10080GB**均已分配给本实验；发现其它用户的计算进程时先等待，禁止停掉别人任务。
- 建议1TiB主机内存，开训检查要求至少450GiB空闲内存。冻结的内存适配器每rank预留48GiB固定页主机内存，八rank合计384GiB。不要擅自删掉适配器或改状态精度以绕过检查。
- 按当前保留规则，两组最终完整检查点合计约130GB；新旧检查点交替时约220GB峰值，另需约21GB基模、环境、日志与缓存空间。建议**至少350GB空闲磁盘，预留400～500GB更宽裕**，并持续监测空间。只能在新检查点完整保存后删除旧的，不能只按最终容量准备磁盘。容量估算见第8节。
- 使用较短的工作目录，例如`/data/alfwork`，避免Ray本地socket路径超长。所有缓存/日志/临时文件和结果都放在指定工作目录。

## 2. 拉仓库、安装冻结环境

已有GitHub CLI可直接：

```bash
gh auth login
gh repo clone leiye302/alfworld-grpo-opsd-sft-qwen25-3b7b
cd alfworld-grpo-opsd-sft-qwen25-3b7b
bash scripts/bootstrap.sh
source .venv/bin/activate
```

也可用配置好认证的`git clone https://github.com/leiye302/alfworld-grpo-opsd-sft-qwen25-3b7b.git`。Python3.12可执行文件不是`python3.12`时，用`PYTHON312=/实际路径/python bash scripts/bootstrap.sh`。

bootstrap只建立本仓库独立环境，使用已通过Linux/Python3.12依赖解析的版本约束。运行时还会优先使用Release中的冻结模块快照，关键版本包括PyTorch2.8.0+cu128、vLLM0.11.0、Transformers4.57.3、Ray2.50.0、ALFWorld0.4.2、TextWorld1.6.2、FlashAttention2.7.4.post1。**不要执行上游的`pip install -r framework/requirements.txt`，不要升级整套依赖**，它不是这次实验的已选环境。

## 3. 下载且只下载所需资产

```bash
python scripts/handoff.py prepare --work /data/alfwork
```

本命令只取得：

1. 本仓库Release `assets-v1`内指定的`alfworld-runtime-assets-v1.tar.gz`，约464MB压缩/约3GB展开：ALFWorld SQLite环境文件、原训练/验证parquet、冻结依赖模块。没有基模、Analyzer权重、旧实验检查点。
2. 官方[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。
3. 官方[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)，revision `a09a35458c702b33eeacc393d103063234e8bc28`。

Git本体包含冻结训练代码、409条专家轨迹及认证manifest、固定验证manifest、配置锁和启动工具。**不需要下载72BTeacher或Analyzer，也不需要调用外置API**；专家库已离线构建。基模权重仅存在接手者自己的工作盘，不上传本仓库。

下载会检查Release整包和内部成员SHA、模型配置/tokenizer SHA。网络失败重新执行同一命令，不要改锁文件“通过”检查。ALFWorld使用一个SQLite文件避免展开海量小文件；manifest里的旧绝对路径是虚拟逻辑路径，由适配器映射到当前工作盘，**不要手动替换manifest路径/哈希，也不用创建公司原目录**。

## 4. 开跑前检查

```bash
python scripts/handoff.py check --work /data/alfwork --gpu
```

必须检查并留下`/data/alfwork/preflight/acceptance.json`：

- 两个真实tokenizer全量编码专家库；目标token、EOS、长度、mask与原响应相符，prompt/环境反馈/padding不参与CE；超长不静默截断。
- 409条轨迹身份/历史/质量掩码完整；16条是**全局**抽样数量。八rank输入补齐不串样本，归一化分母是全局监督token数量。第50轮SFT为0。
- 固定128任务的game字节和重新reset出的初始状态逐条符合SHA；训练parquet保持16行，150epoch对应150外层轮次。
- 3B/7B配置可由实际Hydra训练入口解析，八卡、mini256、micro32、group8、maxenv50等生效。
- 目标机器A100的BF16 FlashAttention前向/反向可用，显卡和主机内存满足条件。

本仓库发布时的实际检查和限制记录在[evidence/handoff_acceptance.json](evidence/handoff_acceptance.json)。维护者的CPU数据/配置检查**不等于对方8×A100上已经完成3B/7B训练**；目标机器GPU检查通过后才开训。

## 5. 启动两个SFT方法组，各150轮

```bash
mkdir -p /data/alfwork
nohup python -u scripts/handoff.py run-all --work /data/alfwork > /data/alfwork/queue.log 2>&1 < /dev/null &
```

`run-all`和`plan`的默认组别均为`3b_sft → 7b_sft`，不会启动baseline。每组150轮，检查点50轮一次，固定验证10轮一次。检查点保留为50→100→150：新检查点验收成功才清理上一份；未通过时保留旧检查点并报错。主机不关机、不停止进程即可独立运行；合上接手者的电脑不影响远端训练。

已拉过旧版仓库的接手者，需要先`git pull --ff-only`再启动。正在运行的训练进程不会自动采用新的保存规则；不要运行中替换源码，需要停止后按完整检查点续跑才生效。`configs/{3b,7b}_{baseline,sft}.json`是Linux示例路径，实际生效路径以`handoff.py plan --work ...`输出为准。

只启动一组：

```bash
python -u scripts/handoff.py run --work /data/alfwork --size 3b --method sft
python -u scripts/handoff.py run --work /data/alfwork --size 7b --method sft
```

上述两条是单组选择示例，**不要同时启动，也不要在队列运行时重复启动**。需要查看完整生效参数时执行`python scripts/handoff.py plan --work /data/alfwork`，输出只包含这两个SFT组。

## 6. 中断后续跑

确认原队列已经退出后：

```bash
nohup python -u scripts/handoff.py run-all --work /data/alfwork --resume > /data/alfwork/queue_resume.log 2>&1 < /dev/null &
```

自动跳过完整结束的组，其余使用原生完整FSDP checkpoint恢复Actor、Adam、学习率、dataloader/RNG和外层计数。从最近保存的完整轮次继续到总计150，不另加150轮。没有完整检查点时原生逻辑会重新从基模开始；不要把中途半写目录当可恢复状态。新跑默认拒绝覆盖已有实验目录。

报错时保留日志、完整checkpoint和identity；不要擅自改变batch、梯度累积、KL、精度或SFT库。显存不足若确需micro/offload硬件适配，必须单独记录；后续若做baseline对照，同一backbone必须使用相同硬件适配，本次交接仍只运行两个SFT组。

## 7. 记录和交回

每组目录`/data/alfwork/runs/{3b_sft,7b_sft}`保存：

- `reports/launch_args.json`、`identity.json`：生效配置、Git commit、模型revision、专家库和固定val哈希。
- `logs/metrics.jsonl`：reward及其标准差、轨迹成功率、entropy、response length、梯度范数、KL、OPSD和SFT loss/coefficient等训练器原生指标。
- `logs/actor_audit_rank*.jsonl`：各rank实际mini/microbatch、每轮optimizer.step次数、FP32参数/Adam、早期实际参数更新与Teacher detach证据。
- `fixed_validation/results/iteration_*.json`及`.jsonl.gz`：每次固定128任务的成功数、分项指标和真实轨迹。
- `checkpoints/global_step_{50,100,150}`：每50轮保存的原生完整FSDP分片/优化器/调度器/随机状态；新的一份验收后删除上一份，结束时只有`global_step_150`。不能只交一份合并HF权重作为续训备份。
- `logs/checkpoint_retention.jsonl`：各次完整保存验收、旧检查点清理及失败恢复标记的合并记录。
- `checkpoints/expert_action_sft_sampling`：SFT抽样身份和监督分母证据，后50轮不会继续生成SFT批次。

第一轮跨至少两次mini-batch更新后检查FP32小更新、optimizer.step次数及权重同步；不能只看初始化时的probability ratio=1。每次保存后检查八rank的模型/Adam/随机状态、数据状态、tokenizer和最新轮数标记。原生保存返回且状态归档结构通过后，只删除本组被替代的50／100轮目录；保存失败不删旧检查点，并恢复上一份有效检查点的续跑标记。这个结构检查不替代加载模型的内容验收。不创建海量逐token文件，原始结果尽量按轮压缩/合并。

两组结束后：

```bash
python scripts/collect_results.py --work /data/alfwork
```

输出合并的验证CSV、两个SFT组训练指标JSONL和3B/7B的验证曲线。交回完整0～150固定val曲线及150轮终点，附配置与运行日志。不要用训练reward代替验证成功率，也不要把valid_seen称为valid_unseen论文测试成绩。该单种子实验不能单独证明鲁棒性。

## 8. 空间和时间估算

下表按两组顺序运行、共享基模下载、每组只保留最新完整checkpoint计算；GB为十进制单位。

| 内容 | 估计空间 |
|---|---:|
| 两个BF16基模 | 约21GB |
| 3B每份完整FSDP训练状态 | 约38GB |
| 7B每份完整FSDP训练状态 | 约91GB |
| 两组最终checkpoint合计 | 约130GB |
| 顺序训练时checkpoint最高同时占用 | 约220GB |
| 环境、下载缓存、运行缓存与两组日志 | 预留约50～100GB |
| 建议开跑前空闲空间 | 最低350GB，推荐400～500GB |

峰值出现在3B已结束、7B保存新checkpoint而旧checkpoint尚未删除时：

$$S_{\mathrm{checkpoint,peak}}\approx S_{3B}+2S_{7B}\approx38+2\times91=220\ \mathrm{GB}.$$

这是FP32模型参数和Adam状态的完整续训档容量估计；具体还受FSDP分片、共享embedding序列化及少量元数据影响。额外导出合并权重、保留旧日志副本或失败的半写checkpoint会增加占用，应单独计入；容量估计不授权删除其它资料。

**本仓库尚未在目标8×A10080GB上实测完整3B/7B训练耗时。此前另用四卡4090日志外推得到的绝对小时/天数已撤回，不能当作八卡A100实测速度或可靠完工时间。**

本次仅把原四组任务缩减为两个SFT组，硬件和各组训练设置没有改变。沿用同一估算口径，训练总时间应约为原四组计划的一半：

$$T_{\mathrm{two\ runs}}\approx\tfrac12 T_{\mathrm{four\ runs}}.$$

这个比例假设每个backbone的baseline与方法组耗时接近，实际受SFT开销及轨迹长度变化影响；两种基模的下载和环境准备仍需完成。接手AI应在每个backbone前3～5轮后，按`logs/metrics.jsonl`中的`timing_s/step`、验证和保存耗时重新估算剩余时间；SFT第50轮归零，前后阶段分开统计。在拿到目标硬件实测前，不承诺具体完工天数或据此设置固定自动关机。

## 来源

底座：[ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)，选定上游commit `d511f043afe9199b8576fc99ea52e12d84841f32`。当前冻结源码/硬件补丁见`evidence/source_snapshot.json`、`hardware_patch.json`；I/O配额保护范围的路径适配另见`runtime_portability_patch.json`。该适配只允许释放当前实验自己预留的空文件，不删除数据/检查点，不改变训练计算。保留上游许可证与各组件许可证。专家库和新增方法用于本次授权的研究交接。Qwen模型遵守对应官方model card/license。
