# 独立材料研发引擎开发计划

## 1. 定位与目标

独立引擎面向材料研发中的数据预处理、机器学习建模、性能预测和配方优化，先以离线 Python 包形式开发，再通过薄适配层接入 Demo3 或其他完整交付系统。

详细设计：

- 数据预处理与建模门禁见 [data-preprocessing-design.md](./data-preprocessing-design.md)；
- 机器学习建模与预测见 [modeling-design.md](./modeling-design.md)；
- 配方优化与 BO 见 [optimization-design.md](./optimization-design.md)。

引擎边界：

- 不依赖 Demo3 的 FastAPI、Agent、SSE 或 React；
- 不复制 MatDesign 代码、目录、接口和命名；
- 不把业务示例名写死为实现常量；
- 配方优化细节以 [optimization-design.md](./optimization-design.md) 为唯一详细设计，本计划只管理开发顺序、交付物和集成方式；
- 业务 MySQL、权限、知识问答、前端展示由宿主系统负责，引擎只消费标准化输入并输出可追溯 Artifact。

最终能力：

```text
数据接入
→ 数据质量检查
→ Modeling Gate
→ Dataset Builder / Version
→ 策略分层 ML 建模
→ 指标评价与适用域
→ 性能预测
→ 配方 / 工艺优化
→ BO 实验推荐
→ Artifact 持久化
```

## 2. 总体架构

```text
engine/
  contracts/       输入输出契约与配置对象
  ingestion/       CSV / Parquet / DataFrame 接入
  governance/      数据探查、清洗、特征工程、数据门禁
  dataset/         Dataset Builder、版本、血缘
  modeling/        策略分层训练与模型持久化
  evaluation/      指标、残差、交叉验证、适用域
  prediction/      模型推理与输入校验
  optimization/    配方优化与 BO，详细设计见 optimization-design.md
  registry/        Dataset / Model / Result Artifact 管理
  cli/             离线命令行入口
  adapters/        Demo3 与外部系统适配层
  tests/           单元、集成、回归和验收测试
```

依赖方向：

```text
contracts
  ← ingestion
  ← governance
  ← dataset
  ← modeling
  ← evaluation
  ← prediction
  ← optimization
  ← registry
  ← cli
  ← adapters
```

禁止反向依赖：

```text
engine 不得依赖 Demo3 server / agent / client
adapters 可以依赖 engine，Demo3 只通过 adapters 调用 engine
```

## 3. 开发原则

1. **契约先行**
   先定义稳定输入输出对象，再实现算法，避免函数参数随业务系统漂移。

2. **Artifact 化**
   Dataset、模型、评价结果、预测结果、优化结果均保存为可追溯文件，并记录配置、版本、血缘和哈希。

3. **策略分层**
   算法选择由数据规模、目标数量、变量类型和约束类型自动决定，不依赖人工猜测。

4. **测试先行**
   每个阶段先建立确定性测试夹具，再实现功能；所有随机算法固定 seed。

5. **通用性**
   目标名、变量名、单位、阈值、约束均来自请求和元数据，示例只出现在文档与测试夹具中。

6. **可移植性**
   引擎包可整体复制到其他仓库；适配层单独维护，不污染核心算法。

7. **真实执行**
   指标、预测、优化候选必须来自真实计算，禁止 LLM 生成或补全。

## 4. 核心 Artifact 契约

### 4.1 DatasetArtifact

```text
DatasetArtifact
  artifact_id
  version
  parent_dataset_id
  source_uri
  data_hash
  feature_schema
  target_schema
  cleaning_rules
  lineage
  created_at
  file_path
```

要求：

- 原始数据只读；
- 清洗后数据生成新版本；
- 不覆盖父版本；
- `data_hash` 由数据内容和 Schema 共同计算；
- 清洗规则必须可序列化和可重放。

### 4.2 DataQualityReport

```text
DataQualityReport
  dataset_artifact_id
  sample_count
  feature_count
  target_count
  sample_feature_ratio
  missing_report
  duplicate_report
  outlier_report
  sample_closure_report
  test_consistency_report
  leakage_report
  warnings
```

报告必须定位到字段和样本，不允许只给布尔值。

### 4.3 ModelingGateResult

```text
ModelingGateResult
  decision
    PASS
    CONDITIONAL_PASS
    FAIL
  reasons
  blocking_items
  warning_items
  recommended_strategy
  evaluated_at
```

规则：

- `FAIL`：不得训练正式模型；
- `CONDITIONAL_PASS`：可训练实验模型，但不得注册为正式可用模型；
- `PASS`：可进入正式建模和注册流程；
- 每个决策必须输出原因和证据。

### 4.4 ModelArtifact

```text
ModelArtifact
  model_id
  version
  dataset_artifact_id
  target_name
  model_type
  hyperparameters
  feature_schema
  preprocessing_schema
  metrics
  applicability_domain
  status
  file_path
```

状态：

```text
EXPERIMENTAL
→ CANDIDATE
→ VALIDATED
→ APPROVED
→ ACTIVE
→ DEPRECATED
```

### 4.5 PredictionResult

```text
PredictionResult
  model_id
  model_version
  dataset_version
  input_values
  predicted_value
  prediction_uncertainty
  applicability_domain
  warnings
```

### 4.6 OptimizationResult

`OptimizationResult` 的详细字段、策略选择、约束和排序规则见 [optimization-design.md](./optimization-design.md)。本计划只约束其开发顺序和验收标准。

## 5. 分阶段开发计划

### 阶段 0：基线与骨架

目标：建立可测试的独立包骨架。

任务：

```text
1. 创建 engine 包和子模块；
2. 建立 contracts 基础对象；
3. 建立 CLI 骨架；
4. 建立 tests 目录；
5. 增加包导出；
6. 建立统一异常体系。
```

交付：

```text
engine/
engine/cli.py
engine/exceptions.py
engine/tests/
```

完成标准：

```text
python -m engine.cli --help
pytest engine/tests
```

### 阶段 1：数据接入契约

目标：统一接收 CSV、Parquet 和 DataFrame。

核心接口：

```python
ingest(source: DataSource, config: IngestionConfig) -> IngestionResult
```

功能：

- 识别文件格式；
- 校验文件存在性和后缀；
- 读取为 DataFrame；
- 统一列名清理；
- 保留原始数据快照；
- 输出字段类型、行数、列数和读取错误。

不做：

- 业务权限判断；
- 目标语义判断；
  这些由宿主系统或 governance 配置提供。

完成标准：

```text
CSV / Parquet / DataFrame 三类输入均可生成 IngestionResult；
坏文件、空文件、缺列均返回结构化异常。
```

### 阶段 2：数据探查与质量检查

详细规则见 [data-preprocessing-design.md](./data-preprocessing-design.md)。

用户不提供预处理配置时，系统使用 `default_safe_v1`；用户提供配置时，用户覆盖可覆盖项，安全规则仍然强制生效。目标字段优先读取数据集旁路 metadata，标识字段可按唯一性和类型推断。

目标：在建模前识别数据是否可信。

检查项：

```text
Sample Count
Feature Count
Missing
Duplicate
Outlier
Constant Feature
Sample Closure
Target Availability
Test Consistency
Leakage
Sample Feature Ratio
```

实现模块：

```text
engine/governance/
  profiling.py
  missing.py
  duplicates.py
  outliers.py
  closure.py
  consistency.py
  leakage.py
  gate.py
```

策略：

```text
1. 每个检查独立输出子报告；
2. gate 汇总所有子报告；
3. blocking 问题输出 FAIL；
4. 可修复或风险性问题输出 warning；
5. 所有阈值来自配置，不写死。
```

完成标准：

```text
构造缺列、重复、异常、泄漏、样本不足、口径不一致夹具；
每种情况均得到预期 gate 决策和原因。
```

### 阶段 3：清洗与 Dataset Builder

目标：生成可复现、可追溯的数据集版本。

能力：

```text
缺失值处理
重复样本处理
异常样本标记或剔除
类型转换
单位映射
目标列标准化
特征列标准化
派生特征
数据集拆分元数据
```

核心接口：

```python
build_dataset(
    ingestion_result: IngestionResult,
    quality_report: DataQualityReport,
    config: DatasetBuildConfig,
) -> DatasetArtifact
```

清洗规则必须记录：

```text
field
rule_name
params
input_count
output_count
reason
```

完成标准：

```text
同一输入 + 同一配置 + 同一代码版本生成相同 data_hash；
父版本和新版本血缘完整；
清洗规则可重放。
```

### 阶段 4：策略分层 ML 建模

详细规则见 [modeling-design.md](./modeling-design.md)。

目标：根据数据规模自动选择候选算法和交叉验证策略。

输入：

```text
DatasetArtifact
TrainingConfig
ModelingGateResult
```

#### 数据分层规则

```text
n = 有效样本数
p = 特征数
ratio = n / p
```

| 层级 | 条件 | CV 策略 | 候选算法 |
|---|---|---|---|
| Tier 1 | `ratio < 1` | LOOCV | Linear Regression、Ridge、Lasso、ElasticNet、BayesianRidge、PLS |
| Tier 2 | `1 <= ratio < 3` | 5-Fold CV | Tier 1 + Gaussian Process、Random Forest、XGBoost |
| Tier 3 | `ratio >= 3` | 5-Fold CV | Tier 2 + LightGBM、Gradient Boosting、SVR |

#### 超参搜索规则

```text
p < 20:
    optuna_trials = 30
20 <= p <= 50:
    optuna_trials = 50
p > 50:
    optuna_trials = 50
```

搜索对象：

```text
Random Forest
XGBoost
LightGBM
```

固定规则：

- 线性和正则模型使用内置 CV 或解析解；
- GP、SVR 使用固定基础参数和配置覆盖；
- Optuna 目标为 CV RMSE；
- 固定 random seed；
- 每个目标独立训练；
- 每个候选模型输出完整指标。

#### 指标

```text
R²
MAE
RMSE
CV Mean
CV Std
```

所有指标在原始目标空间计算；若目标使用变换，预测后必须逆变换再评估。

#### 模型选择

```text
1. 只在通过 Modeling Gate 的数据上训练；
2. 按 CV RMSE 排名；
3. R²、MAE、RMSE、CV Std 同时输出；
4. 小样本时同时考虑模型复杂度惩罚；
5. 并列时优先可解释模型；
6. 用户显式指定算法时记录 override 原因。
```

完成标准：

```text
三个 Tier 的合成数据均可自动选择预期候选集；
每个目标生成 ModelArtifact；
同一数据和 seed 可复现指标。
```

### 阶段 5：模型评价与适用域

目标：保证预测和优化候选可信。

模块：

```text
engine/evaluation/
  metrics.py
  residuals.py
  calibration.py
  applicability_domain.py
```

评价输出：

```text
train_metrics
test_metrics
cv_metrics
residual_plot_data
feature_importance
model_comparison
```

适用域：

```text
IN_DOMAIN
EDGE
OUT_OF_DOMAIN
```

实现策略：

```text
1. 使用模型训练时的标准化器；
2. 计算候选到训练集的 kNN 距离；
3. k = min(5, n_train - 1)；
4. 用训练距离分布确定边界；
5. 未见类别或类别组合直接判为 OUT_OF_DOMAIN。
```

完成标准：

```text
训练分布内、边界、外推候选分别得到三种 AD 状态；
OUT_OF_DOMAIN 不进入高可信推荐。
```

### 阶段 6：预测 API

目标：为宿主系统提供稳定推理接口。

核心接口：

```python
predict(model_artifact: ModelArtifact, inputs: DataFrame) -> list[PredictionResult]
```

校验：

```text
必填特征
数值范围
类别白名单
单位一致性
缺失值
固定值补齐
```

完成标准：

```text
合法输入输出预测、AD 和模型版本；
缺字段、未知类别、越界输入返回结构化错误。
```

同时暴露 UI 无关的 `VisualizationDataset`。正式引擎只输出图表/表格的
`columns + records + chart_type + x_field + y_fields` 数据契约，不渲染图片、
不生成 ECharts option、不依赖前端库。训练产物必须保留 reserved test 的
`y_true / y_pred / residual` 逐样本记录，供前端绘制预测-实测和残差诊断图。

### 阶段 7：配方优化引擎

详细设计、策略选择、变量类型、约束、Pareto、DE 和 BO 规则统一见：

[optimization-design.md](./optimization-design.md)

本阶段开发顺序：

```text
1. OptimizationRequest / Result 契约；
2. SearchSpace；
3. 硬约束修复与过滤；
4. Candidate Rank；
5. DE；
6. Active Set DE；
7. DE-RAG；
8. NSGA-II；
9. Mixed NSGA-II；
10. BO；
11. Pareto / 多样性 / 信任等级；
12. 优化回归测试。
```

验收映射：

```text
多目标阈值推荐
总量和上限约束
Pareto 正确性
外推候选过滤
固定历史样本推荐下一批实验
```

### 阶段 8：Artifact Registry

目标：管理数据和模型资产，不依赖外部平台。

本地结构：

```text
engine/artifacts/
  datasets/
    dataset_id/version/
  models/
    model_id/version/
  model-registry.json
  predictions/
  optimizations/
  reports/
```

能力：

```text
save
load
list
get_lineage
deprecate
```

完成标准：

```text
Artifact 不可变；
同 ID 新版本不覆盖旧版本；
血缘可从优化结果追到模型和数据集。
```

真实数据与 Artifact 复用链：

```text
1. 数据治理测试以 Demo3 已治理数据为基准输入：
   data/PC_ABS_intersection.parquet
   data/ABS_intersection.parquet

2. TestDatasetFactory 只在内存副本上生成针对性微扰数据集；

3. 微扰数据集写入：
   engine/artifacts/datasets/

4. 建模测试以治理好的基准 DatasetArtifact 为输入；

5. 模型写入：
   engine/artifacts/models/

6. 模型注册信息写入：
   engine/artifacts/models/model-registry.json

7. 配方优化测试只读取已注册 ModelArtifact；

8. 优化结果写入：
   engine/artifacts/optimizations/
```

约束：

```text
原始 data/ 目录只读；
模型版本不删除、不覆盖；
优化测试不得重复执行数据治理和模型训练；
缺少可用模型时返回 MODEL_ARTIFACT_NOT_FOUND。
```

### 阶段 9：离线 CLI

目标：不接入 UI 也能完整测试。

命令：

```powershell
python -m engine.cli ingest
python -m engine.cli test-data
python -m engine.cli quality
python -m engine.cli build-dataset
python -m engine.cli train
python -m engine.cli evaluate
python -m engine.cli predict
python -m engine.cli chart-data
python -m engine.cli optimize
python -m engine.cli recommend-experiments
python -m engine.cli lineage
```

每个命令支持：

```text
--config
--input
--output
--seed
--quiet
--model-registry
```

完成标准：

```text
从 CSV 到模型、预测、优化的全链路可通过 CLI 完成；
所有输出为结构化 JSON 或 Artifact 引用。
```

`chart-data` 仅从当前任务报告生成可视化数据契约。PNG 渲染器属于
`engine/tests/render_chart_data.py` 的开发验收工具，不进入正式运行依赖。

### 阶段 10：Demo3 适配层

目标：最小改动接入 Demo3。

适配器：

```text
engine/adapters/demo3/
  data_governance_adapter.py
  training_adapter.py
  prediction_adapter.py
  optimization_adapter.py
```

接入策略：

```text
1. 保留 Demo3 现有函数签名；
2. 内部委托 engine；
3. 将 engine Artifact 转成 Demo3 当前返回结构；
4. 新旧实现并行对比；
5. 通过回归后切换默认实现；
6. 保留 legacy fallback 配置。
```

完成标准：

```text
Demo3 Agent / SSE / 前端不改；
输入输出兼容；
现有回归通过；
可配置回退旧实现。
```

### 阶段 11：外部系统适配

目标：迁移到另一套完整交付系统。

适配内容：

```text
数据源 Adapter
权限上下文 Adapter
模型存储 Adapter
任务状态 Adapter
结果展示 Adapter
```

原则：

```text
外部系统负责权限、会话、任务调度和展示；
engine 只要求标准化 DataFrame、Schema 和 Artifact URI。
```

完成标准：

```text
复制 engine/ 后无需复制 Demo3 业务代码即可运行核心能力。
```

## 6. 测试体系

### 6.1 测试目录

```text
engine/tests/
  contracts/
  governance/
  dataset/
  modeling/
  evaluation/
  prediction/
  optimization/
  registry/
  cli/
  integration/
```

### 6.2 测试类型

| 类型 | 目标 |
|---|---|
| Unit | 单个规则、算法和契约 |
| Integration | 数据 → Dataset → 模型 → 预测 / 优化 |
| Regression | 固定 seed 和输入，校验输出稳定性 |
| Acceptance | 对应手册验收场景 |
| Adapter | Demo3 新旧实现兼容 |

### 6.3 夹具

```text
small_high_dim.csv
medium_medium_dim.csv
large_low_dim.csv
missing.csv
duplicate.csv
outlier.csv
leakage.csv
mixed_variables.csv
multi_target.csv
```

夹具使用通用字段名，如 `feature_001`、`target_001`，不使用业务固定名。

除静态夹具外，集成测试必须包含真实数据复用链：

```text
治理好的 Parquet
→ TestDatasetFactory 微扰数据集
→ DatasetArtifact
→ ModelArtifact
→ model-registry.json
→ OptimizationResult
```

TestDatasetFactory 使用固定 seed，可生成：

```text
normal_jitter
missing
duplicate
outlier
target_conflict
leakage
missing_field
```

### 6.4 必测场景

```text
数据门禁 FAIL / CONDITIONAL_PASS / PASS
Dataset 版本与血缘
三个建模 Tier
CV 指标完整性
AD 三分类
预测输入校验
硬约束修复
候选排序
多目标 Pareto
BO 推荐实验
CLI 全链路
Demo3 adapter 兼容
```

## 7. 里程碑

| 里程碑 | 内容 | 退出标准 |
|---|---|---|
| M0 | 骨架与契约 | CLI 可运行，空测试通过 |
| M1 | 数据治理 | 三种 gate 决策可复现 |
| M2 | Dataset Builder | 版本、血缘、哈希稳定 |
| M3 | ML Engine | 三个 Tier 自动训练并输出完整指标 |
| M4 | AD / Prediction | 预测含版本与 AD |
| M5 | Optimization | 手册优化验收通过 |
| M6 | Registry / CLI | 全链路离线运行 |
| M7 | Demo3 Adapter | 新旧实现兼容并通过回归 |
| M8 | 可迁移包 | 复制 engine 即可在宿主系统集成 |

## 8. 规范性检查

开发过程中持续检查：

```text
1. 核心模块不得依赖 UI / Agent / FastAPI；
2. 示例业务名不得进入实现；
3. 随机过程必须显式 seed；
4. 每个 Artifact 有版本和血缘；
5. 每个策略选择必须输出 reason；
6. 指标和候选必须来自真实计算；
7. 硬约束违反者不得进入最终推荐；
8. OUT_OF_DOMAIN 不进入高可信推荐；
9. 优化细节不得在多个文档中重复维护；
10. 每阶段必须先测试后合入。
```

## 9. 风险与应对

| 风险 | 应对 |
|---|---|
| 数据质量差导致模型不可信 | Modeling Gate 先行，FAIL 阻断正式建模 |
| 小样本高维过拟合 | Tier 1 限制模型复杂度并使用 LOOCV |
| 目标量纲差异影响优化 | 使用各目标 RMSE 归一化 |
| 混合变量优化失败 | Mixed NSGA-II 和专门修复算子 |
| 外推候选看似优异 | AD 过滤并降低信任等级 |
| 引擎与 Demo3 耦合 | adapters 独立，核心包不反向依赖 |
| 文档重复漂移 | 优化细节统一引用 optimization-design.md |

## 10. 首个开发切片

首个切片只做以下内容：

```text
1. engine 包骨架；
2. contracts 基础对象；
3. DataQualityReport / ModelingGateResult；
4. missing、duplicate、outlier 三类检查；
5. 对应单元测试；
6. CLI quality 命令。
```

原因：

```text
数据门禁是建模和优化的前置条件；
该切片小、可测、独立，能最快验证包结构和契约方向。
```
