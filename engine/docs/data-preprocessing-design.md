# 数据预处理与建模门禁详细设计

## 1. 目标与边界

本模块负责把原始研发数据转换为可复现、可追溯、可建模的 `DatasetArtifact`，并在训练前判断数据是否满足正式建模条件。

模块边界：

- 输入：CSV、Parquet、DataFrame 及宿主系统提供的字段元数据；
- 输出：`IngestionResult`、`DataQualityReport`、`ModelingGateResult`、`DatasetArtifact`；
- 不负责：业务权限、知识问答、模型训练、性能预测和配方优化；
- 不假设目标名、变量名、材料体系或测试口径，所有业务语义均来自配置和元数据。

核心流程：

```text
数据接入
→ 字段与类型校验
→ 数据探查
→ 缺失 / 重复 / 异常检查
→ 样本闭合检查
→ 测试口径一致性检查
→ 泄漏检查
→ 样本量与维度检查
→ Modeling Gate
→ 清洗与特征工程
→ Dataset 版本化
```

## 2. 输入输出契约

### 2.1 DataSource

```text
DataSource
  kind
    csv
    parquet
    dataframe
  uri
  dataframe
  encoding
  separator
```

### 2.2 FieldMetadata

```text
FieldMetadata
  name
  role
    identifier
    feature
    target
    group
    timestamp
    condition
    ignored
  dtype
  unit
  allowed_values
  lower_bound
  upper_bound
  is_post_experiment
```

字段元数据是数据治理的核心输入。引擎不通过字段名猜测语义。

### 2.3 RelationMetadata

用于 Sample Closure 和数据关联。

```text
RelationMetadata
  entity
    sample
    batch
    product
    formula
    process
    test
    performance
  keys
  parent_key
  child_key
  relation_type
    one_to_one
    one_to_many
    many_to_many
```

### 2.4 IngestionResult

```text
IngestionResult
  source
  dataframe
  row_count
  column_count
  detected_schema
  field_metadata
  relation_metadata
  warnings
  source_hash
```

### 2.5 DataQualityReport

```text
DataQualityReport
  row_count
  column_count
  feature_count
  target_count
  sample_feature_ratio
  missing_report
  duplicate_report
  outlier_report
  constant_report
  closure_report
  consistency_report
  leakage_report
  target_report
  warnings
```

每个子报告必须包含：

```text
status
severity
affected_fields
affected_rows
metric_value
threshold
reason
suggestion
```

### 2.6 ModelingGateResult

```text
ModelingGateResult
  decision
    PASS
    CONDITIONAL_PASS
    FAIL
  reasons
  blocking_items
  warning_items
  recommended_tier
  evaluated_at
```

### 2.7 DatasetArtifact

```text
DatasetArtifact
  artifact_id
  version
  parent_dataset_id
  data_hash
  feature_schema
  target_schema
  cleaning_rules
  lineage
  file_path
  created_at
```

## 3. 配置对象

### 3.1 QualityThresholdConfig

```text
QualityThresholdConfig
  min_total_samples
  min_samples_per_target
  min_sample_feature_ratio
  max_target_missing_ratio
  max_feature_missing_ratio
  max_duplicate_ratio
  max_global_outlier_ratio
  max_single_feature_outlier_ratio
  max_constant_feature_ratio
  min_feature_count
  min_target_count
```

这些默认值属于 `default_safe_v1`；生产环境可通过用户配置覆盖允许覆盖的项。

### 3.3 预置策略与用户覆盖

数据预处理必须支持两种输入方式：

```text
用户只提供数据集：
  系统使用 default_safe_v1 预置策略；
  目标字段优先读取同目录 metadata.json；
  标识字段可按唯一性和类型推断；
  系统生成清洗计划并输出 DatasetArtifact。

用户提供 preprocessing config：
  用户配置覆盖预置策略中允许覆盖的项；
  安全规则不可覆盖；
  ResolvedConfig 记录用户覆盖项和决策原因。
```

配置优先级：

```text
不可覆盖安全规则
> 用户显式配置
> default_safe_v1 预置策略
> 数据画像推导
```

`default_safe_v1` 策略：

```text
完全重复样本：
  删除，保留第一条

目标缺失样本：
  删除该样本，不插补目标

特征缺失率 > 50%：
  删除字段

数值特征缺失率 ≤ 50%：
  中位数插补，并添加 was_missing 指示列

类别特征缺失率 ≤ 50%：
  众数插补，并添加 was_missing 指示列

异常值：
  默认只标记，不删除、不截尾
  用户可显式选择 winsorize

显式泄漏或疑似目标派生字段：
  强制剔除，用户不能关闭

测试口径不一致：
  FAIL，用户不能静默合并

最终 Modeling Gate 为 FAIL：
  不生成正式 DatasetArtifact
```

解析输出：

```text
ResolvedPreprocessingConfig
  ├─ target_fields
  ├─ feature_fields
  ├─ identifier_fields
  ├─ thresholds
  ├─ closure_config
  ├─ leakage_config
  ├─ consistency_specs
  ├─ cleaning_config
  ├─ user_overrides
  ├─ metadata_source
  └─ resolution_reasons
```

### 3.2 CleaningConfig

```text
CleaningConfig
  missing_strategy
    reject
    drop_row
    drop_column
    mean
    median
    mode
    constant
  duplicate_strategy
    reject
    drop_exact
    drop_by_keys
    keep_first
    keep_latest
  outlier_strategy
    reject
    keep
    winsorize
    drop
    mask
  outlier_method
    iqr
    zscore
    target_residual
    isolation_forest
  transform_config
  feature_derivation_rules
```

所有策略必须写入 `DatasetArtifact.cleaning_rules`，保证同一输入可重放。

## 4. 数据接入与字段校验

### 4.1 接入规则

```text
CSV：
  校验文件存在、后缀、编码、分隔符；
  读取后不得立即修改原始列名；
  记录原始行数和列数。

Parquet：
  校验文件存在和可读性；
  保留 dtypes；
  记录 row group 信息可选。

DataFrame：
  浅拷贝后进入引擎；
  不得修改调用方原始对象。
```

### 4.2 字段校验

必查项：

```text
1. 元数据字段是否存在于数据表；
2. 是否存在未声明的非忽略字段；
3. 标识字段是否唯一或符合关系约束；
4. 目标字段是否为数值或可转换数值；
5. 类别字段是否包含未知取值；
6. 时间字段是否可解析；
7. 单位是否与目标配置一致。
```

处理规则：

```text
缺少 identifier / feature / target：
  FAIL

多余字段未声明：
  warning，默认忽略，不进入特征

目标不可转数值：
  FAIL

类别未知取值：
  CONDITIONAL_PASS 或 FAIL，取决于配置
```

## 5. 数据探查

输出每个字段：

```text
dtype
non_null_count
missing_count
missing_ratio
unique_count
min
max
mean
median
standard_deviation
quartiles
value_counts
constant_flag
dtype_mismatch_count
```

目标字段额外输出：

```text
target_name
valid_sample_count
distribution_summary
zero_or_negative_count
extreme_value_count
recommended_transform
```

`recommended_transform` 只建议，不自动执行。是否执行由 `CleaningConfig` 决定。

## 6. 缺失值检查

### 6.1 目标缺失

```text
target_missing_ratio = missing_count / row_count
```

决策：

| 条件 | 决策 |
|---|---|
| 目标整列缺失 | `FAIL` |
| `target_missing_ratio > max_target_missing_ratio` | `FAIL` |
| `valid_samples_per_target < min_samples_per_target` | `FAIL` |
| 目标存在少量缺失 | 标记样本，训练该目标时剔除 |

目标缺失不允许用均值、中位数或模型插补，避免制造虚假标签。

### 6.2 特征缺失

```text
feature_missing_ratio = missing_count / row_count
```

决策：

| 条件 | 默认处理 |
|---|---|
| `missing_ratio = 1.0` | 删除特征 |
| `missing_ratio > max_feature_missing_ratio` | 删除特征或要求人工确认 |
| 标识或关联键缺失 | 该行不可闭合，进入 closure 处理 |
| 数值特征少量缺失 | 按配置插补 |
| 类别特征少量缺失 | 众数、常量类别或保留为缺失类别 |

插补规则必须记录：

```text
field
strategy
parameters
training_only
```

插补器必须随 `DatasetArtifact` 保存，预测阶段复用，不得重新拟合。

## 7. 重复检查

### 7.1 样本级重复

```text
duplicate_ratio = duplicate_row_count / row_count
```

识别级别：

```text
1. 全字段完全重复；
2. 标识键重复；
3. 特征键重复但目标不同；
4. 同一样品同测试口径多条记录。
```

决策：

| 情况 | 默认决策 |
|---|---|
| 全字段重复 | 去重，保留第一条或最新时间 |
| 标识键重复且目标不同 | `FAIL` 或要求口径分组 |
| 特征相同但目标差异超过阈值 | `FAIL` 或进入冲突报告 |
| 重复率超过配置阈值 | `FAIL` |

### 7.2 冲突目标

同一特征和同一测试口径出现多个目标值时：

```text
if range <= tolerance:
    aggregate by configured aggregation
else:
    conflict
```

可选聚合：

```text
mean
median
first
latest
max
min
```

冲突必须输出样本 ID、字段、原始值、差异和冲突原因。

## 8. 异常值检查

### 8.1 IQR

```text
Q1
Q3
IQR = Q3 - Q1
lower = Q1 - k * IQR
upper = Q3 + k * IQR
k 默认 1.5，可配置
```

适用：

```text
连续特征
连续目标
```

### 8.2 Z-Score

```text
z = abs(value - mean) / standard_deviation
```

默认阈值：

```text
z > 3
```

适用：

```text
近似正态分布字段
```

不适用于强偏态分布，除非先做配置的变换。

### 8.3 目标残差异常

在初步模型不可用前，使用稳健基线识别目标异常：

```text
1. 按配置分组；
2. 每组拟合简单模型或稳健统计量；
3. 计算残差；
4. 标记超过阈值的样本。
```

此检查只标记，不自动删除。

### 8.4 异常处理决策

| 条件 | 决策 |
|---|---|
| 全局异常率超过阈值 | `FAIL` |
| 单特征异常率超过阈值 | 删除特征或人工确认 |
| 少量异常 | 标记、保留、截尾或剔除 |
| 目标异常 | 默认只标记，不自动剔除 |

## 9. Sample Closure

目标：确认“原料—样品—配方—工艺—测试—性能”记录能闭合。

### 9.1 检查项

```text
1. 样品主记录存在；
2. 样品至少有一条配方或工艺记录；
3. 配方原料编码能解析；
4. 工艺字段能解析；
5. 测试记录关联到样品；
6. 性能记录关联到测试或样品；
7. 不存在孤儿子记录；
8. 不存在多父关联；
9. 关联键类型一致；
10. 关联键不为空。
```

### 9.2 输出

```text
total_samples
closed_samples
unclosed_samples
closures_ratio
orphan_records
missing_relations
ambiguous_relations
```

决策：

| 条件 | 决策 |
|---|---|
| `closed_samples = 0` | `FAIL` |
| 闭合率低于配置阈值 | `FAIL` |
| 少量样本不闭合 | 剔除并记录 |
| 存在孤儿记录 | warning 或 error，取决于配置 |

## 10. 测试口径一致性

### 10.1 输入元数据

```text
TestConsistencyMetadata
  test_key
  test_name_field
  test_method_field
  unit_field
  condition_fields
  expected_test_names
  expected_units
  expected_methods
  required_target_mapping
```

### 10.2 检查规则

```text
1. 同一目标是否映射到多个测试项；
2. 同一测试项是否存在不同单位；
3. 同一目标是否存在不同方法；
4. 同一方法是否缺少条件；
5. 条件是否超出常规范围；
6. 测试时间是否异常；
7. 样本是否能关联测试条件。
```

决策：

| 情况 | 决策 |
|---|---|
| 同目标单位不一致且不可换算 | `FAIL` |
| 同目标方法不一致 | `FAIL` 或分组建模 |
| 单位可换算 | 统一换算并记录 |
| 缺测试条件 | `CONDITIONAL_PASS` 或 `FAIL` |

不得静默合并不同口径。

## 11. 泄漏检查

### 11.1 泄漏来源

```text
1. target 本身或 target 派生字段；
2. post-experiment 字段；
3. 实验结果字段；
4. 由目标统计得到的分组特征；
5. 未来时间信息；
6. 样本唯一标识；
7. 与目标强相关且业务上预测时不可得的字段；
8. 数据预处理时在全量数据上拟合造成的泄漏。
```

### 11.2 规则

```text
if field.role == target:
    leak

if field.is_post_experiment == true:
    leak

if field.name in configured_leakage_fields:
    leak

if feature is derived from target:
    leak
```

决策：

| 情况 | 处理 |
|---|---|
| 明确泄漏字段 | 剔除并记录 |
| 疑似泄漏 | 输出 warning，要求确认 |
| 标识字段 | 不进入特征 |
| 预处理泄漏 | 禁止，所有拟合必须限制在训练折内 |

## 12. 样本量与维度检查

计算：

```text
n = 有效样本数
p = 特征数
ratio = n / p
```

按每个目标分别计算：

```text
n_target = 该目标非缺失样本数
```

决策：

| 条件 | 决策 |
|---|---|
| `n < min_total_samples` | `FAIL` |
| `n_target < min_samples_per_target` | 该目标 `FAIL` |
| `p < min_feature_count` | `FAIL` |
| 无有效目标 | `FAIL` |
| `ratio` 低于阈值 | `CONDITIONAL_PASS`，进入高维策略 |

`ratio` 同时输出给建模模块作为分层依据。

## 13. Modeling Gate 决策矩阵

| 检查项 | FAIL | CONDITIONAL_PASS | PASS |
|---|---|---|---|
| 样本量 | 低于最小样本 | 接近阈值 | 高于阈值 |
| 目标缺失 | 超阈值或全缺失 | 少量缺失 | 有效样本充足 |
| 特征缺失 | 关键特征缺失 | 非关键特征处理后可用 | 可控 |
| 重复 | 高重复或目标冲突 | 少量可去重 | 无重要冲突 |
| 异常 | 异常率超阈值 | 少量已标记 | 可控 |
| Sample Closure | 闭合率不足 | 少量不闭合已剔除 | 良好 |
| 测试口径 | 不可比口径 | 分组或标记后可用 | 一致 |
| 泄漏 | 明确泄漏未处理 | 疑似项已剔除或确认 | 无泄漏 |
| 样本特征比 | 过低且无可用策略 | 高维策略可用 | 正常 |

决策规则：

```text
if any blocking item:
    FAIL
else if any warning item requiring review:
    CONDITIONAL_PASS
else:
    PASS
```

## 14. 清洗与特征工程

### 14.1 清洗顺序

```text
1. 字段选择；
2. 类型转换；
3. 单位统一；
4. 剔除明确泄漏字段；
5. 处理重复；
6. 处理目标缺失样本；
7. 处理特征缺失；
8. 处理异常值；
9. 生成派生特征；
10. 特征标准化配置；
11. 保存 DatasetArtifact。
```

### 14.2 类型转换

```text
数值字段：
  去除空白和千分位分隔符；
  统一小数点；
  转换失败计数；

类别字段：
  strip；
  统一大小写配置；
  未知类别处理；

时间字段：
  统一 UTC 或业务时区；
  解析失败进入报告。
```

### 14.3 派生特征

规则必须显式声明：

```text
DerivedFeatureRule
  name
  expression
  input_fields
  dtype
  unit
  missing_strategy
```

禁止自动猜测业务公式。

### 14.4 标准化

训练前只保存标准化配置，不在 Dataset Builder 中预先拟合最终模型标准化器，除非配置明确要求。

原因：

```text
1. 交叉验证折内必须重新拟合；
2. 不同模型可能需要不同预处理；
3. 避免全量数据泄漏。
```

## 15. Dataset 版本与血缘

### 15.1 版本规则

```text
同一 parent + 同一 cleaning config + 同一代码版本 + 同一输入 hash
→ 相同 dataset hash
→ 可复用

任一项变化
→ 新版本
```

### 15.2 血缘记录

```text
source_uri
source_hash
parent_dataset_id
quality_report_id
gate_result_id
cleaning_rules
removed_fields
removed_rows
added_features
transform_rules
engine_version
```

### 15.3 基于治理数据的微扰测试数据集

数据预处理能力必须使用治理好的真实数据做集成测试。当前基准输入为：

```text
data/PC_ABS_intersection.parquet
data/ABS_intersection.parquet
```

规则：

```text
1. 原始治理数据只读；
2. 微扰只发生在内存副本；
3. 微扰数据集作为新 DatasetArtifact 版本保存；
4. 输出目录固定为 engine/artifacts/datasets/；
5. 每个版本包含 dataset.parquet、metadata.json、lineage.json；
6. lineage 记录 source_uri、source_hash、perturbation_spec 和 random_seed；
7. 同一 source + perturbation_spec + seed + engine_version 生成相同 data_hash。
```

TestDatasetFactory 输出类型：

| 类型 | 用途 |
|---|---|
| `normal_jitter` | 在训练特征范围内做小幅数值扰动，验证 PASS 路径 |
| `missing` | 注入目标或特征缺失，验证缺失报告和 gate |
| `duplicate` | 复制样本或标识，验证重复和冲突处理 |
| `outlier` | 注入极端特征或目标值，验证异常检查 |
| `target_conflict` | 构造同键不同目标值，验证冲突报告 |
| `leakage` | 添加后实验字段或目标派生字段，验证泄漏拦截 |
| `missing_field` | 删除关键元数据字段，验证 Schema FAIL |

`PerturbationSpec`：

```text
kind
target_fields
affected_ratio
magnitude
random_seed
params
```

微扰幅度默认不超过字段训练分布的可配置比例，且不得修改原始文件。

## 16. 模块设计

```text
engine/
  ingestion/
    reader.py
    schema.py
  governance/
    profiling.py
    missing.py
    duplicates.py
    outliers.py
    closure.py
    consistency.py
    leakage.py
    gate.py
  dataset/
    test_data_factory.py
    cleaner.py
    feature_engine.py
    builder.py
    lineage.py
```

核心 API：

```python
ingest(source: DataSource, config: IngestionConfig) -> IngestionResult

profile_data(result: IngestionResult) -> DataProfile

check_quality(
    result: IngestionResult,
    config: QualityThresholdConfig,
) -> DataQualityReport

apply_modeling_gate(
    report: DataQualityReport,
    config: GateConfig,
) -> ModelingGateResult

build_dataset(
    result: IngestionResult,
    report: DataQualityReport,
    gate: ModelingGateResult,
    config: DatasetBuildConfig,
) -> DatasetArtifact
```

## 17. 测试计划

### 17.1 单元测试

```text
CSV / Parquet / DataFrame 读取
字段缺失
类型转换
目标缺失
特征缺失
完全重复
标识重复
目标冲突
IQR 异常
Z-Score 异常
样本闭合
口径不一致
单位不一致
泄漏字段
样本量不足
gate 三分支决策
cleaning rule 重放
dataset hash 稳定性
```

### 17.2 集成测试

```text
原始 CSV
→ IngestionResult
→ DataQualityReport
→ ModelingGateResult
→ DatasetArtifact
→ reload dataset
→ 校验 hash 和 lineage
```

### 17.3 确定性要求

```text
同输入
+ 同配置
+ 同代码版本
→ 同报告摘要
→ 同 data_hash
→ 同 lineage
```

## 18. 验收标准

```text
1. 每个质量检查均有结构化报告；
2. FAIL 能阻断正式建模；
3. CONDITIONAL_PASS 能阻止模型注册为正式模型；
4. 清洗规则可序列化和重放；
5. Dataset 版本不可变；
6. 血缘可追溯；
7. 无业务字段硬编码；
8. 预处理不会在全量数据上拟合造成泄漏。
```
