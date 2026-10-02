# ALFWorld：Qwen2.5-3B，GRPO＋OPSD＋专家轨迹SFT

> **先确认训练服务器（2026-10-02交接澄清）**：本次只在接手者指定的单机8×A10040GB服务器上工作。SSH目标必须来自接手者明确给出的主机地址/SSH别名、端口和用户；“本地已有密钥”只说明认证方式，不能确定目标机器。若这些目标信息缺失，先询问接手者，不能任选本地SSH配置中的机器登录。
>
> 本仓库没有提供来源服务器的登录指令，也不需要连接来源服务器。数据清单、冻结配置和适配器中的历史绝对路径只用于数据身份/虚拟文件索引，**不构成SSH目标、下载地址或在该机器上执行操作的授权**。不要根据它们查找或登录公司的机器；准备、检查和训练都在接手者指定的八卡机器完成，资产通过下文GitHub Release和官方模型仓库下载。

这是可交给另一位研究者或AI agent的训练交接仓库。**当前只授权单机8×A10040GB跑一组`3b_sft`，从Qwen2.5-3B-Instruct基模开始训练到150轮。不跑7B，不跑baseline。** 这条指令替代此前的两组安排。本次将显存适配更新为40GB：保留原训练配置、资产、验证集和方法，只调整显存检查与反向中间张量的存储策略。

| 组别 | 初始化 | 训练目标 | 结束轮数 |
|---|---|---|---:|
| `3b_sft` | Qwen2.5-3B-Instruct | 选定SDAR源码的原版GRPO＋OPSD＋专家完整reasoning/action SFT |150|

**OPSD固定系数0.01；SFT初始系数0.1，按外层轮数余弦衰减，第50轮归零；活跃轮每轮全局抽16条专家轨迹，监督有效输出token的全局token-mean。** 保留Reference KL、奖励处理、动作有效性惩罚和原生mini-batch更新。不是先前修改过的带符号OPSD，也不是OPSD增长/SFT下降的互补日程。方法公式和实现边界见[docs/METHOD.md](docs/METHOD.md)。

固定验证使用已有的**128个valid_seen任务**，包括初始状态、环境种子和每个任务每个决策的生成种子。训练前验证一次，随后每10轮验证；每50轮保存完整可续训检查点，150轮结束。**第100轮保存并验收后删50轮，第150轮保存并验收后删100轮，最终只留150轮。** 从锁定的3B基模开始，不使用旧1.5B实验权重。

## 40GB适配（2026-10-01更新）

目标硬件为**单机8×A10040GB**。已拉过旧仓库的接手者先执行`git pull --ff-only`，再重新执行`check --gpu`，然后使用原单组启动命令。

- 保留TP2、PPOmini256、每GPU逻辑micro32、评分micro32及全部原生损失/更新逻辑。原生损失按各micro-batch分别取token平均，直接缩小micro可能改变token权重，因此本次没有这样改。
- 保留vLLM显存预算0.6，在40GB卡上约为24GB的生成引擎预算。冻结后端的`FSDPVLLMShardingManager`在生成阶段结束调用`sleep(level=1)`，释放生成权重与KV缓存后训练，不新增采样顺序或改变温度。不要为凑满显存盲目提高预算。
- 保留原输出头/映射主机内存适配器；40GB模式在累计输入≥16,384token、输出头张量≥512MiB，或当前空闲GPU显存低于8GiB时，更早把反向所需的保存张量转存CPU。原计算形状、全词表归一化、loss分母及FP32参数/Adam不变。大块临时梯度仍优先使用GPU，容量不足才使用原主机存储路径。
- GPU检查接受每卡至少35GiB的A100，支持40GB型号；记录真实显卡容量。主机内存要求仍为至少450GiB空闲，显存40GB不代表可以减少原固定页主机内存预算。
- `reports/identity.json`记录硬件适配，`actor_audit_rank*.jsonl`记录每rank的GPU分配/预留峰值。先检查第1轮真实显存、更新次数及日志，再根据实测报告速度。不要用占满40GB代替有效算力利用率。

本次维护者完成的是CPU配置/存储分支检查，**尚未在对方8×A10040GB上实测完整训练或显存峰值**。接手服务器必须通过GPU检查；若实际训练OOM，保留证据并报告，不要擅自改batch、损失或参数精度。可复查：`python scripts/check_a10040_profile.py`。

## 给接手AI的执行指令

先读本README、AGENTS.md和METHOD，准备环境/下载资产，完成检查，然后只启动`3b_sft`。**必须使用下文的`run --size 3b --method sft`单组命令；不要使用`run-all`，其未修改的默认队列仍会接着训练7B。** 不要修改方法、全局batch、采样/验证协议、损失归一化或训练预算；不要使用之前1.5B实验的checkpoint初始化3B。不要把OPSD换成另一个版本。只下载本仓库Release的指定资产和下文锁定基模，不下载来源服务器的其它内容，不需要公司服务器登录权限或任何外置模型API密钥。

只有`3b_sft`使用全部8张卡。7B及baseline代码/配置继续保留，但不属于本次训练任务。**本仓库和指定Release已公开，接手者不需要GitHub账号、协作者邀请、GitHub CLI、登录或token。** 直接用HTTPS拉取代码，下载工具匿名获取Release。训练服务器的SSH登录使用接手者指定目标和已有密钥，和GitHub访问是两回事；不要从旧配置或历史数据路径推断登录目标。

## 1. 机器准备

- Linux x86_64、Python3.12，建议Ubuntu22.04；具备`git`、`tmux`、`g++`、Python venv、正常NVIDIA驱动和CUDA12.8运行能力。
- 单机**8×A10040GB**均已分配给本实验；发现其它用户的计算进程时先等待，禁止停掉别人任务。
- 建议1TiB主机内存，开训检查要求至少450GiB空闲内存。冻结的内存适配器每rank预留48GiB固定页主机内存，八rank合计384GiB。不要擅自删掉适配器或改状态精度以绕过检查。
- 按当前单组保留规则，3B最终完整检查点约38GB；新旧检查点交替时约76GB峰值，另计基模、环境、日志与缓存。建议**至少200GB空闲磁盘，250GB以上更宽裕**，并持续监测空间。只能在新检查点完整保存后删除旧的，不能只按最终容量准备磁盘。容量估算见第8节。
- 先确认真实数据盘挂载点，在其中使用较短的工作目录，例如`/data/alfwork`，避免Ray本地socket路径超长。以下`/data/alfwork`是示例；若数据盘不挂载在`/data`，统一替换为实际数据盘路径。代码、环境、模型、缓存、日志、临时文件和结果都放在该项目目录内。

## 2. 拉仓库、安装冻结环境

在训练服务器上执行，不需要GitHub认证：

```bash
mkdir -p /data/alfwork
git clone https://github.com/leiye302/alfworld-grpo-opsd-sft-qwen25-3b7b.git /data/alfwork/repo
cd /data/alfwork/repo
bash scripts/bootstrap.sh
source .venv/bin/activate
```

Python3.12可执行文件不是`python3.12`时，用`PYTHON312=/实际路径/python bash scripts/bootstrap.sh`。拉取或下载返回404时先检查URL和网络，不要索要仓库所有者的密码或token。

bootstrap只建立本仓库独立环境，使用已通过Linux/Python3.12依赖解析的版本约束。运行时还会优先使用Release中的冻结模块快照，关键版本包括PyTorch2.8.0+cu128、vLLM0.11.0、Transformers4.57.3、Ray2.50.0、ALFWorld0.4.2、TextWorld1.6.2、FlashAttention2.7.4.post1。**不要执行上游的`pip install -r framework/requirements.txt`，不要升级整套依赖**，它不是这次实验的已选环境。

## 3. 下载且只下载所需资产

```bash
python scripts/handoff.py prepare --work /data/alfwork
```

本命令只取得：

**准备工具和preflight代码保持原样，因此仍会下载并检查两个锁定基模。7B是现有检查流程的依赖，不会被本次单组训练命令启动。不要跳过检查，也不要把准备阶段下载7B误当成7B开训授权。**

1. 本仓库Release `assets-v1`内指定的`alfworld-runtime-assets-v1.tar.gz`，约464MB压缩/约3GB展开：ALFWorld SQLite环境文件、原训练/验证parquet、冻结依赖模块。没有基模、Analyzer权重、旧实验检查点。
2. 官方[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`。
3. 官方[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)，revision `a09a35458c702b33eeacc393d103063234e8bc28`，仅供现有准备/检查流程使用，不训练。

Git本体包含冻结训练代码、409条专家轨迹及认证manifest、固定验证manifest、配置锁和启动工具。**不需要下载72BTeacher或Analyzer，也不需要调用外置API**；专家库已离线构建。基模权重仅存在接手者自己的工作盘，不上传本仓库。

下载会检查Release整包和内部成员SHA、模型配置/tokenizer SHA。网络失败重新执行同一命令，不要改锁文件“通过”检查。ALFWorld使用一个SQLite文件避免展开海量小文件；manifest里的旧绝对路径是虚拟逻辑路径，由适配器映射到当前工作盘，**不要手动替换manifest路径/哈希，也不用创建来源机器的原目录或登录该机器**。专家数据及manifest的SHA还参与固定抽样身份，修改路径元数据也会改变文件SHA；它们不能当成尚未下载的数据再去来源机器找。

## 4. 开跑前检查

```bash
python scripts/handoff.py check --work /data/alfwork --gpu
```

必须检查并留下`/data/alfwork/preflight/acceptance.json`：

- 两个真实tokenizer全量编码专家库；目标token、EOS、长度、mask与原响应相符，prompt/环境反馈/padding不参与CE；超长不静默截断。
- 409条轨迹身份/历史/质量掩码完整；16条是**全局**抽样数量。八rank输入补齐不串样本，归一化分母是全局监督token数量。第50轮SFT为0。
- 固定128任务的game字节和重新reset出的初始状态逐条符合SHA；训练parquet保持16行，150epoch对应150外层轮次。
- 现有3B/7B配置检查仍可由实际Hydra训练入口解析，八卡、mini256、micro32、group8、maxenv50等生效；检查配置不会启动7B/baseline训练。
- 目标机器A100的BF16 FlashAttention前向/反向可用，显卡和主机内存满足条件。

本仓库发布时的实际检查和限制记录在[evidence/handoff_acceptance.json](evidence/handoff_acceptance.json)。维护者的CPU数据/配置检查**不等于对方8×A100上已经完成3B/7B训练**；目标机器GPU检查通过后才开训。

2026-10-01公开后已在禁用GitHub凭据的独立目录完成新克隆，并用实际下载工具匿名取得完整464MB Release、校验SHA-256一致，见[evidence/public_access_acceptance.json](evidence/public_access_acceptance.json)。这验证了无需GitHub授权的代码和资产入口，不代替目标机器的环境安装、两个基模下载或GPU检查。

## 5. 只启动3B的我们方法，150轮

```bash
tmux new-session -d -s alfworld-3b-sft -c /data/alfwork/repo \
  '/data/alfwork/repo/.venv/bin/python -u scripts/handoff.py run --work /data/alfwork --size 3b --method sft > /data/alfwork/3b_sft_queue.log 2>&1'
```

上述`run`单组命令只训练`3b_sft`，不会接着训练7B或baseline。150轮预算、检查点每50轮一次、固定验证每10轮一次保持原样。检查点保留为50→100→150：新检查点验收成功才清理上一份；未通过时保留旧检查点并报错。主机不关机、不停止进程即可独立运行；合上接手者的电脑不影响远端训练。用`tmux ls`检查会话、`tail -n 40 /data/alfwork/3b_sft_queue.log`查看启动日志；实际训练日志为`/data/alfwork/runs/3b_sft/logs/driver.log`，不能只看到tmux创建成功就报告训练已经正常。

已拉过旧版仓库的接手者，需要先`git pull --ff-only`再启动。更新文档不会自动取消已有旧队列的7B任务；如果旧队列已经启动，需要先检查其实际命令和进程，不能仅凭README更新宣称旧队列已改变。40GB适配需更新本仓库代码；已有训练进程不会因git pull而自动切换。`configs/{3b,7b}_{baseline,sft}.json`继续保留；本次3B实际生效参数查看`runs/3b_sft/reports/launch_args.json`。

不使用tmux时的同一单组入口（不要和已有训练重复启动）：

```bash
python -u scripts/handoff.py run --work /data/alfwork --size 3b --method sft
```

`run-all`和`plan`的原代码未改，仍采用历史两组范围；`plan`的输出不是本次任务清单，**不要运行`run-all`或为了完成它的队列而启动7B**。

## 6. 中断后续跑

确认原训练进程已经退出后，仅续跑同一3B组：

```bash
tmux new-session -d -s alfworld-3b-sft-resume -c /data/alfwork/repo \
  '/data/alfwork/repo/.venv/bin/python -u scripts/handoff.py run --work /data/alfwork --size 3b --method sft --resume > /data/alfwork/3b_sft_resume.log 2>&1'
```

已完整结束的3B组自动跳过，否则使用原生完整FSDP checkpoint恢复Actor、Adam、学习率、dataloader/RNG和外层计数。从最近保存的完整轮次继续到总计150，不另加150轮。没有完整检查点时原生逻辑会重新从基模开始；不要把中途半写目录当可恢复状态。新跑默认拒绝覆盖已有实验目录。不要使用`run-all --resume`，以免启动不在本次授权范围内的7B。

报错时保留日志、完整checkpoint和identity；不要擅自改变batch、梯度累积、KL、精度或SFT库。显存不足若确需micro/offload硬件适配，必须单独记录；后续若做baseline对照，同一backbone必须使用相同硬件适配，本次交接仍只运行`3b_sft`。

## 7. 记录和交回

本次目录`/data/alfwork/runs/3b_sft`保存：

- `reports/launch_args.json`、`reports/identity.json`：生效配置、Git commit、模型revision、专家库和固定val哈希。
- `logs/metrics.jsonl`：reward及其标准差、轨迹成功率、entropy、response length、梯度范数、KL、OPSD和SFT loss/coefficient等训练器原生指标。
- `logs/actor_audit_rank*.jsonl`：各rank实际mini/microbatch、每轮optimizer.step次数、FP32参数/Adam、早期实际参数更新与Teacher detach证据。
- `fixed_validation/results/iteration_*.json`及`.jsonl.gz`：每次固定128任务的成功数、分项指标和真实轨迹。
- `checkpoints/global_step_{50,100,150}`：每50轮保存的原生完整FSDP分片/优化器/调度器/随机状态；新的一份验收后删除上一份，结束时只有`global_step_150`。不能只交一份合并HF权重作为续训备份。
- `logs/checkpoint_retention.jsonl`：各次完整保存验收、旧检查点清理及失败恢复标记的合并记录。
- `checkpoints/expert_action_sft_sampling`：SFT抽样身份和监督分母证据，后50轮不会继续生成SFT批次。

第一轮跨至少两次mini-batch更新后检查FP32小更新、optimizer.step次数及权重同步；不能只看初始化时的probability ratio=1。每次保存后检查八rank的模型/Adam/随机状态、数据状态、tokenizer和最新轮数标记。原生保存返回且状态归档结构通过后，只删除本组被替代的50／100轮目录；保存失败不删旧检查点，并恢复上一份有效检查点的续跑标记。这个结构检查不替代加载模型的内容验收。不创建海量逐token文件，原始结果尽量按轮压缩/合并。

3B组结束后：

```bash
python scripts/collect_results.py --work /data/alfwork
```

该汇总工具未改代码，仍保留3B/7B的旧布局。在新的独立工作目录中只跑本次3B时，CSV和指标仅含3B，图中的7B子图为空；这不代表漏跑任务，**不要为了填图去训练7B**。如工作目录已有旧7B结果，应区分它们与本次结果。交回本次3B的完整0～150固定val曲线及150轮终点，附配置与运行日志。不要用训练reward代替验证成功率，也不要把valid_seen称为valid_unseen论文测试成绩。该单种子实验不能单独证明鲁棒性。

## 8. 空间和时间估算

下表只计本次3B组训练，但按未修改的准备工具保留两个基模下载；每次只保留最新完整checkpoint，GB为十进制单位。

| 内容 | 估计空间 |
|---|---:|
| 现有准备/检查工具的两个BF16基模 | 约21GB |
| 3B每份完整FSDP训练状态 | 约38GB |
| 3B最终checkpoint | 约38GB |
| 3B保存新checkpoint时的峰值同时占用 | 约76GB |
| 环境、下载缓存、运行缓存与日志 | 预留约50～100GB |
| 建议开跑前空闲空间 | 至少200GB，250GB以上更宽裕 |

checkpoint峰值出现在3B保存新的一份而旧的一份尚未删除时：

$$S_{\mathrm{checkpoint,peak}}\approx2S_{3B}\approx2\times38=76\ \mathrm{GB}.$$

这是FP32模型参数和Adam状态的完整续训档容量估计；具体还受FSDP分片、共享embedding序列化及少量元数据影响。额外导出合并权重、保留旧日志副本或失败的半写checkpoint会增加占用，应单独计入；容量估计不授权删除其它资料。

**本仓库尚未在目标8×A10040GB上实测完整3B训练耗时。此前另用四卡4090日志外推得到的绝对小时/天数已撤回，不能当作八卡A100实测速度或可靠完工时间。**

本次只估计一组3B的150轮，不计7B或baseline训练。前3～5轮实测后，接手AI应按`logs/metrics.jsonl`中的`timing_s/step`、验证和保存耗时重新估计余下时间；SFT第50轮归零，前后阶段分别统计：

$$T_{\mathrm{remaining}}\approx\sum_{k=k_{\mathrm{now}}+1}^{150}t_k+T_{\mathrm{validation}}+T_{\mathrm{save}}.$$

准备时间另计，原准备工具仍会下载两个基模。没有目标硬件实测前，不承诺具体完工天数，也不要据此设置固定自动关机。

## 来源

底座：[ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)，选定上游commit `d511f043afe9199b8576fc99ea52e12d84841f32`。当前冻结源码/硬件补丁见`evidence/source_snapshot.json`、`hardware_patch.json`；I/O配额保护范围的路径适配另见`runtime_portability_patch.json`。该适配只允许释放当前实验自己预留的空文件，不删除数据/检查点，不改变训练计算。保留上游许可证与各组件许可证。专家库和新增方法用于本次授权的研究交接。Qwen模型遵守对应官方model card/license。
