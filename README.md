# One-Year PD Model Development and Validation Toolkit

可安装、可配置、可测试的一年期违约概率（PD）建模与验证项目。使用**独立借款人的合成数据**，演示信用风险模型开发、统计判断、模型验证和工程交付。本项目不声称可用于银行生产环境或满足监管要求。

## 快速开始

需要 Python 3.11 或以上版本。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pd-toolkit.exe run --config configs/demo.yaml --output output/demo
.\.venv\Scripts\pd-toolkit.exe verify output/demo
```

macOS / Linux：

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pd-toolkit run --config configs/demo.yaml --output output/demo
pd-toolkit verify output/demo
```

需要较快体验时使用 `configs/quick.yaml`。输出目录必须为空；使用新的目录名保存每次实验的证据。

完整情景实验：

```sh
pd-toolkit suite --config configs/demo.yaml --output output/scenarios
python scripts/check_demo.py output/scenarios
```

切换模型或校准方法：复制配置后修改 `models.names`，可选 `logistic`、`gradient_boosting`；修改 `models.calibration`，可选 `sigmoid`、`isotonic`、`none`。默认值会补全未填写字段；拼错配置项会报错。生成完整配置：`pd-toolkit init-config my-config.yaml`。

## 已实现的流程

合成数据 → 数据完整性检查 → 训练与调参 → 独立概率校准 → 跨期及分群验证 → 模型比较 → PDF 报告与证据包。

- 基准：逻辑回归，训练期年度系数稳定性与变量方向检查。
- 挑战者：直方图梯度提升，固定随机种子，关闭隐式随机早停。
- 时间隔离：2013–2017 训练、2019 调参、2021 校准、2023–2024 最终测试。间隔年份用于保证十二个月结果先成熟。
- 预处理：训练集拟合中位数、缺失标志、标准化和类别编码；不读取未来数据。
- 验证：AUC、Gini、Brier、log loss、校准斜率/截距、校准曲线、观测违约率 Wilson 区间、bootstrap 区间、配对模型比较、年度/客群评估、PSI。
- 概念审查：合成机制对照、系数稳定性、训练集局部敏感度、独立 oracle 基准。
- 复现：随机种子、配置、依赖版本、Git 状态、源码/数据/产物 SHA-256、模型参数及序列化模型。

## 情景与故障

| 情景 | 设计 | 预期结果 |
|---|---|---|
| `baseline` | 低违约率、关联变量、非线性、缺失值 | 建立正常基准；仍可能发现统计限制 |
| `downturn` | 测试期失业上升、GDP 下降、负债及使用率上升 | 压力验证，可能需要限制使用或重校准 |
| `composition_shift` | 测试期高风险客群占比上升 | PSI 和分群评估识别组合变化 |
| `mechanism_shift` | 测试期违约机制改变 | 识别显著 PD 低估 |
| `calibration_error` | 测试预测 logit 人为下移 1.5 | 识别校准错误；排序大致保持 |
| `omitted_variable` | 建模时排除历史逾期 | 披露错误设定并评估影响 |
| `leakage` | 引入违约后的催收变量 | 训练前阻断并留存错误证据 |
| `immature` | 结果尚未成熟 | 训练前阻断 |
| `duplicate` | 重复借款人 | 训练前阻断 |
| `unavailable_feature` | 收入在观察日以后才可用 | 训练前阻断 |

压力变化只作用于最终测试期；同一随机种子下，各情景的开发数据保持一致。`suite` 将识别预设完整性故障视为成功；普通 `run` 对这些故障返回退出码 2。

## 输出与审阅

每个成功实验生成 `report.pdf` 和 `evidence.zip`，并保留便于检查的文件：

| 文件 | 内容 |
|---|---|
| `predictions.csv` | 最终测试的原始/校准预测，模型、年份、客群及实际违约 |
| `metrics.csv` / `validation.json` | 总体、年度与客群指标、区间、限制和结论 |
| `findings.csv` | 严重程度、受影响人群、问题及建议行动 |
| `calibration_bins.csv` / `figures/` | 校准分箱、置信区间和图形 |
| `tuning.json` / `comparisons.json` | 独立调参结果、配对比较 |
| `coefficients.csv` / `sensitivities.csv` | 系数稳定性与局部变量敏感度 |
| `observations.csv` | 可观察变量、时间戳与实际结果 |
| `oracle.csv` | 单独保存的真实合成 PD，模型不可读取 |
| `feature_availability.csv` / `integrity.json` | 变量可用时间与完整性检查 |
| `models/` | 已训练的模型及校准器；只加载自己信任的 joblib 文件 |
| `config.yaml` / `metadata.json` | 配置、环境、版本、Git、参数和哈希 |
| `manifest.json` | 逐文件 SHA-256 校验；不包含自身或 ZIP 的哈希 |

失败情景保存 `integrity_failure.json`，不生成暗示验证通过的报告。

## 测试与打包

```sh
ruff check .
ruff format --check .
pytest --cov=pdtoolkit --cov-report=term-missing
python -m build
```

测试覆盖公式已知值、无违约样本、非法概率、结果成熟时点、预处理隔离、时间边界、预设缺陷、最终标签变化不影响训练、报告生成、模型保存、哈希篡改检测和重复运行的一致性。

严格复现环境使用 `requirements-lock.txt` 后再安装项目：

```sh
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
pd-toolkit run --config configs/demo.yaml --output output/repeat
python scripts/reproduce.py output/demo output/repeat
```

同源代码、配置、依赖和平台下，预测/指标绝对容差为 `1e-10`。运行时间、时间戳与 PDF 元数据不要求逐字节一致。模型拟合及关键预测使用单线程数值运算。

## GitHub 交付

`.github/workflows/ci.yml` 在 Python 3.11/3.12 下执行格式检查、测试和打包；随后在固定 Python 3.12 环境执行完整演示，检查预设失败是否被发现，并保存 PDF/证据包 30 天。工作流仅有读取仓库权限。

本地测试成功不等于 GitHub Actions 线上执行成功。连接 GitHub 仓库、推送代码并检查 Actions 成功后，才能将该项交付认定为完成。仓库创建与 PR 流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 项目结构

```text
src/pdtoolkit/   config, data, checks, preprocessing, models, calibration,
                metrics, validation, reporting, pipeline, cli
configs/        完整演示与快速配置
tests/          数据、指标、工作流与缺陷测试
scripts/        演示验收与复现比较
docs/           方法论、架构和项目完成记录
.github/        Actions 与 PR 模板
output/         本地生成报告与证据（不提交）
```

详细方法见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)，工程设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。初始范围仅涵盖一年期 PD；LGD、EAD/CCF 与 IFRS 9 留作未来扩展。
