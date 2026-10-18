> **提交位置说明：**代码、README、分析报告、loss 曲线、训练与评测日志等，必须提交到**面试方提供并分配给该候选人的独立 GitHub 仓库**。训练好的 LoRA SFT 和第二阶段优化模型，必须上传到**候选人自己的 Hugging Face 账号下创建的模型仓库**（`https://huggingface.co/<你的用户名>/<模型仓库名>`），设置为 **Public**，并将两个模型链接及 revision 写入面试方提供的 GitHub 仓库 README 后提交给面试方。

# 模型训练研究员笔试提交指南

以 [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 完成自建数据、LoRA/QLoRA SFT、SFT 后第二阶段优化，并在 **SWE-bench Pro v2-hard、HLE、ASI-Bench、Terminal-Bench** 上完成三组模型对比。

> **训练好的模型必须上传 Hugging Face，并设置为 Public（公开）。** 至少提供 LoRA SFT 与第二阶段优化两个 checkpoint，供面试方及公众下载、验证和展示。GitHub 提交代码、报告、曲线和日志；模型权重存放在 Hugging Face。

## 仓库使用方式

本仓库为公开题目模板仓库：https://github.com/apexin-ai/qwen36-35b-takehome 。面试方从模板为每位候选人创建独立私有仓库，并单独邀请对应 GitHub 账号。候选人只在分配给自己的仓库提交，不向模板仓库提交答案，也不使用共享候选人分支。模板设置不会自动创建候选人仓库或发送邀请。

在个人候选人仓库的 main 分支提交，或按面试通知提交 Pull Request。最终提供仓库或 PR 链接、最终 commit SHA、两个 Hugging Face 模型链接及 revision，并发送至面试通知指定的邮箱。私有代码仓库与公开模型仓库是两项独立要求。

## 推荐目录结构

以下为提交结构示例，不代表模板已包含这些实现文件。文件名可调整，请在 README 标明实际路径。

```text
./
├── README.md                 # 实验说明、复现命令、结果及公开模型链接
├── report.pdf                # 一页以内分析报告
├── requirements.lock         # 或等价依赖锁定文件
├── .gitignore                # 排除密钥、缓存、权重；保留要求的日志
├── configs/
│   ├── sft.yaml              # SFT 完整配置
│   ├── stage2.yaml           # 第二阶段训练配置
│   └── eval/                 # 四项 benchmark 的版本、任务清单和固定配置
├── data/
│   ├── README.md             # 来源、许可、规模、划分、清洗和去重
│   └── manifests/            # 数据版本、清单及校验值
├── src/                      # 数据处理、训练和评测实现
├── scripts/
│   ├── prepare_data.sh
│   ├── train_sft.sh
│   ├── train_stage2.sh
│   ├── evaluate.sh           # 三组模型 × 四项 benchmark
│   ├── summarize.py          # 汇总原始结果、计算置信区间
│   └── plot_curves.py        # 从源数据重建曲线
├── logs/<run_id>/
│   ├── metadata.json         # 版本、硬件、配置、命令、时间和种子
│   ├── train.log             # 训练 run 的完整日志
│   ├── eval.log              # 评测 run 的完整日志
│   ├── metrics.jsonl         # 逐 step 原始指标
│   ├── errors.log            # 错误、重试、超时；无错误则说明
│   └── resources.csv         # 显存、耗时、GPU 小时和费用
├── curves/
│   ├── sft_loss.png          # 训练及独立验证集 loss
│   ├── stage2_metrics.png    # 第二阶段 loss 及方法相关指标
│   └── source/              # CSV/JSON 等曲线源数据
├── results/
│   ├── base/<benchmark>/<run_id>/
│   ├── sft/<benchmark>/<run_id>/
│   ├── stage2/<benchmark>/<run_id>/
│   └── summary.csv          # 三组模型 × 四项 benchmark 汇总
├── analysis/
│   ├── failures.md          # 至少 20 个失败案例及归因
│   └── uncertainty.json     # 置信区间和重复运行波动
└── models/
    └── README.md            # 两个公开 HF 链接、revision、加载命令和许可证
```

## 实验与复现说明

README 须包含研究问题、实验矩阵、环境安装、数据构建、SFT、第二阶段优化、模型加载、四项评测、结果汇总和绘图的完整可执行命令。注明执行目录、输入输出路径、资源需求和预计耗时。

记录基座 revision、tokenizer、chat template、代码 commit、依赖版本、数据版本、LoRA 目标模块和专家层处理、rank、量化、学习率、batch size、训练步数、上下文长度、随机种子及 checkpoint 选择方式。训练集与独立验证集分离；benchmark 答案、参考补丁、隐藏测试和测试反馈不得用于训练或挑选 checkpoint。

Base 指本次微调前的原始发布权重。SFT 本身属于 post-training；本题第二阶段优化指在 SFT 权重上继续训练，可采用 DPO/IPO 或执行反馈筛选后继续 SFT 等方法，不能仅用推理时拒绝采样替代。允许自行租用算力，必须提交完整实验，不接受缩小模型或 dry run 替代。

## Loss 曲线与日志

- 分别提交 SFT 和第二阶段训练的训练 loss、独立验证集 loss 曲线；第二阶段同时报告适用于所选方法的指标，如 DPO chosen/rejected reward、reward margin 或偏好准确率，并解释含义。不同目标函数的 loss 不应直接比较数值高低。
- 曲线标注 step 或处理 token 数、纵轴含义、run_id 和关键 checkpoint。保留原始数据；如使用平滑，披露窗口并保留未平滑数据。提交 PNG/PDF 图、CSV/JSON 源数据和绘图脚本，不能只交截图或外部仪表盘链接。
- 日志覆盖完整训练与评测过程，包括错误、重试、超时和恢复。每个 run_id 应对应代码版本、配置、数据版本、模型 checkpoint 和结果文件。
- 提交训练/评测耗时、峰值显存、GPU 型号与数量、GPU 小时、费用及计算方式。日志可压缩；在 README 提供位置、格式和读取方法。超出 GitHub 文件限制时，在个人仓库 Release 附压缩日志，并提交索引、校验值和访问说明。

## 四项 benchmark 与评测协议

固定采用 **SWE-bench Pro v2-hard、HLE、ASI-Bench、Terminal-Bench**。训练前锁定并保存官方数据源 URL、版本/commit、split、完整任务清单、评测器、主指标、框架、容器、prompt、工具、上下文、推理预算和采样设置。实验期间不随网页更新切换版本，三组模型使用相同口径。

| Benchmark | 必须说明与提交 |
| --- | --- |
| SWE-bench Pro v2-hard | 指定版本完整任务集、hard 划分依据、任务 ID、官方 resolved rate、Agent 框架及 commit、工具权限、步数/token/超时预算、逐任务补丁与判定结果（许可允许范围内） |
| HLE | 固定版本及 split 的准确率；明确是否包含图像题、是否使用工具、答案抽取规则及判分器版本；不得混报 text-only 与完整多模态结果 |
| ASI-Bench | 训练前确认并记录官方来源、版本和完整任务集；依官方协议报告主指标及分项结果，固定运行模式、种子、工具、预算和聚合方式，保留原始输出与评分凭据 |
| Terminal-Bench | 固定数据集版本（沿用题目中的 2.0）、任务及镜像、Harbor 评测框架与 Terminus-2 Agent 版本；提交五次运行的任务成功率、均值与标准差及相同五组种子 |

Terminal-Bench 沿用题目配置：3 小时超时、32 CPU、48 GB RAM、temperature=1.0、top_p=0.95、top_k=20、max_tokens=80K、256K context。三组模型保持一致。每项应记录 thinking 模式、preserve_thinking、量化和采样设置。

**如指定版本或数据无法获取，须在开始实验前联系面试方确认，不得自行替换或将子集结果标为完整结果。** 未披露的官方配置明确标为本实验设置；如与模型卡设置不同，应说明差异，不声称完全复现官方成绩。官方分数不能替代本次实测 Base。

每组结果应包含逐题/逐任务输出、判定、运行状态、错误原因、耗时及 token 用量。记录全部任务，环境失败、超时和模型失败分别统计，不得删除失败任务后提高分数。

## 结果对比

请将下表填写为真实结果。主指标统一为百分制；如原始指标不是百分比，应同时报告原值和训练前固定的归一化方法。另行报告任务数、运行失败率、成本、配对 bootstrap 置信区间及重复运行波动。

| Benchmark | Base | LoRA SFT | 第二阶段优化 | 最终相对 Base 变化（百分点） |
| --- | --- | --- | --- | --- |
| SWE-bench Pro v2-hard | 待填写 | 待填写 | 待填写 | 待填写 |
| HLE | 待填写 | 待填写 | 待填写 | 待填写 |
| ASI-Bench | 待填写 | 待填写 | 待填写 | 待填写 |
| Terminal-Bench | 待填写 | 待填写 | 待填写 | 待填写 |
| 四项等权宏平均 | 待填写 | 待填写 | 待填写 | 待填写 |

宏平均提升目标为 3 个百分点，5 个百分点作为优秀参考；这是挑战目标，并非已验证的可达门槛或官方标准。未达到仍须如实报告和分析，不得改变口径制造提升。至少分析 20 个失败案例，说明单项退化、局限及至少三个下一步实验。

## Hugging Face 模型公开上传（必须）

1. 将训练好的 **LoRA SFT 与第二阶段优化两个 checkpoint** 上传到 Hugging Face，模型仓库必须设置为 **Public**。可使用两个仓库，或同一仓库中明确区分的 checkpoint/revision。
2. 可提交完整 adapter 权重及配置，或合并权重；原始 Base 引用官方模型即可。模型卡写明基座及 revision、训练方法、数据来源与许可、评测结果、限制、适用许可证和加载方法。
3. 在本 README 与 models/README.md 写明两个 HF URL、checkpoint revision、基座 revision、tokenizer/chat template、量化信息、对应 run_id 和完整加载命令。
4. 提交前验证两个 checkpoint 无需额外授权即可公开下载，并可加载复现。面试方可公开展示模型链接与实验结果；公开材料须具备相应分发权限。
5. GitHub 不存放模型权重。最终提交时将两个 HF 模型链接及 revision 一并发送给面试方。

## 一页分析报告

report.pdf 不超过一页，结构固定为 **摘要（150–200 个中文字符）—过程—结果**。结果包含三组模型在四项 benchmark 的分数、宏平均提升、运行失败率和主要成本。详细日志、曲线、置信区间和失败分析放在仓库其他目录，不计入一页限制。

## 提交检查清单

- [ ] 提交到分配的独立私有仓库，附最终 commit SHA 或 PR 链接
- [ ] README 含可执行复现命令、版本、路径、实验矩阵和已知限制
- [ ] 数据构造代码、来源许可、清单、去重与泄漏检查可核验
- [ ] SFT 和第二阶段训练代码、配置与完整日志齐全
- [ ] loss/方法指标曲线、原始数据和绘图脚本齐全
- [ ] Base/SFT/第二阶段优化的四项完整评测、逐任务结果及失败记录齐全
- [ ] 汇总脚本、置信区间、至少 20 个失败案例及资源成本记录齐全
- [ ] 两个训练 checkpoint 已上传 Hugging Face，设置 Public，URL、revision 和加载命令可用
- [ ] report.pdf 不超过一页，摘要、过程、结果完整
- [ ] 提交前移除密钥、token、个人数据及无权分发内容

## 面试方使用模板

通过 Use this template 创建候选人专属仓库，例如 qwen36-35b-takehome-candidate-001，选择 Private，逐仓库邀请对应候选人并给予 Write 权限。检查组织默认权限和团队授权，确保其他候选人不能访问该仓库。已从模板创建的仓库不会自动同步后续模板更新。
