# 配方优化功能交叉验证方案

## 1. 文档目的

本文档用于让一个仅了解“配方优化需求”的智能体或开发者交叉验证独立配方优化引擎的功能设计是否完整、算法选择是否合理、输入输出是否可实现、测试是否充分。读者不需要了解 Demo3 Agent、前端、数据治理和模型训练细节。

被验证模块的唯一职责是：

```text
输入目标性能、搜索变量、约束和已注册预测模型
→ 生成并评估候选配方 / 工艺
→ 输出可信、可追溯、满足约束的推荐结果
```

本文档不替代 [optimization-design.md](./optimization-design.md)，而是面向交叉验证抽取和补充关键契约、场景、策略和验收标准。实现时应以两份文档共同约束为准；若出现冲突，以本文档的用户场景、API 边界和 BO 历史数据契约为准。

## 2. 边界与前提

### 2.1 必须依赖

```text
Model Registry
ModelArtifact
模型预测接口
模型评估指标
Applicability Domain 元数据
```

优化阶段必须加载已注册模型，不允许为了优化而重新训练模型，也不允许重新执行数据治理。

### 2.2 不负责

```text
自然语言理解
权限控制
数据库查询
知识检索
图表渲染
模型训练
实验结果回流
```

智能体可以在调用前把自然语言解析成结构化请求，但引擎只信任显式传入的目标、变量和约束。

### 2.3 商业与来源边界

- MatDesign 只作为功能参照，不复制代码、接口、命名和依赖。
- 不引入 Bgolearn、pymoo 或 MatDesign 专有实现。
- 允许使用通用库：`numpy / pandas / scipy / scikit-learn / DEAP`。
- 业务字段名、目标名、单位、阈值和约束值不得写死在实现中。
- 文档中的 `x1 / x2 / target_1 / 100` 等仅是通用示例，不是实现常量。

## 3. 用户场景覆盖

### 3.1 场景分类

| 场景 | 用户意图 | 目标表达 | 默认策略 | 输出要求 |
|---|---|---|---|---|
| 精确目标收敛 | 希望预测性能尽量接近一组目标值 | 多个 `equal` 目标 | 低维可枚举用 Candidate Rank；连续高维用 DE / DE-RAG / Active Set DE | 每个目标输出预测值、误差、可信度和候选配方 |
| 单边阈值推荐 | 性能必须大于或小于阈值 | `greater_or_equal` 或 `less_or_equal` | 低维用 Candidate Rank；高维连续用 NSGA-II；混合变量用 Mixed NSGA-II | 只输出满足硬目标的候选；不足时明确警告 |
| 混合阈值推荐 | 同一请求中有的目标要求更高，有的要求更低 | 多个不同方向的目标 | NSGA-II / Mixed NSGA-II | 每个目标逐项展示满足情况 |
| 区间目标推荐 | 性能希望落在指定区间 | `in_range(lower_value, upper_value)` | Candidate Rank / NSGA-II / Mixed NSGA-II | 区间内候选优先，区间外候选不得标记为满足 |
| 多目标权衡 | 用户要 Pareto 前沿或多个目标同时较优 | `maximize / minimize / threshold` 混合 | NSGA-II / Mixed NSGA-II | 输出 Pareto rank、拥挤距离和目标值 |
| 历史经验约束 | 参考相近历史配方，减少无效搜索 | 目标 + `historical_candidates` | DE-RAG | 输出历史覆盖率、搜索维度和固定变量策略 |
| 下一轮实验推荐 | 已有实验结果，希望推荐最值得做的下一批 | `observed_values` + BO | GP BO | 输出 acquisition、预测改善和多样性信息 |

### 3.2 预测与优化的边界

用户只输入一个配方并要求预测性能时，应调用独立预测 API，不进入优化流程。

用户输入目标性能并要求推荐配方时，进入优化流程。优化过程内部会调用模型预测能力，但预测能力本身不属于优化模块对外职责。

## 4. API 与智能体 Tool 边界

对外 API 按用户任务拆分，内部实现仍归属配方优化模块：

```text
POST /optimization/formula
  推荐满足目标性能和约束的配方 / 工艺候选

POST /optimization/next-experiments
  基于历史实验结果推荐下一批实验
```

智能体 Tool 建议同名暴露：

```text
optimize_formula
recommend_next_experiments
```

### 4.1 智能体识别约束的规则

用户通常不会显式提供 `x1+x2≤100%` 这类配置。允许的处理方式是：

```text
用户自然语言
→ 智能体识别变量语义、单位、总量或互斥关系
→ 转成显式 HardConstraintSpec
→ 调用优化引擎
```

引擎不做隐式猜测：

1. 智能体识别出的约束必须显式传入；
2. 未传入的约束不生效；
3. 若模型元数据声明了约束，适配层可转换成显式请求；
4. 智能体无法确认总量、单位或互斥语义时，应向用户确认；
5. 引擎不得因为字段名相似而自动推断业务约束。

这保证了优化结果可复现、可审计，也避免错误约束 silently 改变搜索空间。

## 5. 请求契约

### 5.1 OptimizationRequest

`POST /optimization/formula` 请求：

```text
OptimizationRequest
  request_id
  objectives
  variables
  hard_constraints
  soft_constraints
  model_registry_path
  model_selection
  historical_candidates
  top_n
  random_seed
  max_evaluations
  time_limit
  execution_options
  strategy_thresholds
  model_quality_gate
  algorithm_override
  output_options
```

默认值：

| 字段 | 默认规则 | 用户配置 |
|---|---|---|
| `variables` | 缺省时由 ModelArtifact feature schema 和 feature bounds 构建 | 可指定类型、角色、边界、类别、固定值 |
| `hard_constraints` | 空列表，表示没有额外硬约束 | 可传入总量、上限、互斥、目标阈值等 |
| `soft_constraints` | 空列表 | 可传入成本、历史距离、工艺稳定性等 |
| `top_n` | 默认 5 | 可配置，必须为正整数 |
| `random_seed` | 固定默认值 | 可配置 |
| `max_evaluations` | 按策略自动计算并可被时间上限截断 | 可配置 |
| `time_limit` | 安全上限 | 可配置 |
| `execution_options` | 超时返回当前最优可行结果，状态标记为 `PARTIAL` | 可配置 |
| `strategy_thresholds` | 使用安全默认值，见第 8.3 节 | 可配置，但必须为正数 |
| `model_quality_gate` | 离线默认 `warn`，生产可配置为 `block` | 请求只能比部署策略更严格，不能放宽 |
| `algorithm_override` | 不覆盖 | 可指定候选策略，必须记录原因 |

用户给出的变量边界只能与模型边界取交集。用户显式扩大边界时必须携带 `allow_exploration=true`；这类候选只能进入探索集合，不得进入高可信推荐。

### 5.2 ObjectiveSpec

```text
ObjectiveSpec
  target_name
  operator
    equal
    greater_or_equal
    less_or_equal
    in_range
    maximize
    minimize
  value
  lower_value
  upper_value
  tolerance
  weight
  requirement
    hard
    preferred
  model_id
  model_version
  unit
```

语义：

1. `equal`：预测值尽量接近目标值；
2. `greater_or_equal`：预测值应不低于阈值；
3. `less_or_equal`：预测值应不高于阈值；
4. `in_range`：预测值应位于区间内；
5. `maximize / minimize`：用于方向优化和 Pareto；
6. `hard`：最终候选必须满足，否则删除；
7. `preferred`：参与排序和优化，但未满足时输出 warning，不直接删除。

同一目标名可以出现在多个 ObjectiveSpec 中，但必须避免冲突。例如同一目标同时要求 `≥a` 和 `<a` 会触发 `OBJECTIVE_CONFLICT`。

### 5.3 VariableSpec

```text
VariableSpec
  name
  type
    continuous
    integer
    categorical
  role
    mixture
    process
    additive
    cost
    other
  lower
  upper
  categories
  unit
  min_effective_value
  fixed_value
  allow_exploration
```

变量校验：

1. `continuous / integer` 必须有有效边界；
2. `integer` 边界必须可转换为整数；
3. `categorical` 必须提供无重复类别白名单；
4. `fixed_value` 会从自由搜索维度移除，但输出时保留；
5. 所有变量必须能映射到模型 feature schema；
6. 未出现在请求中的模型必需特征，按模型默认值或训练中位数补齐，并写入 diagnostics。

### 5.4 HardConstraintSpec

```text
HardConstraintSpec
  name
  kind
    linear_sum
    bound
    mutex
    categorical_incompatibility
    target_threshold
  variables
  coefficients
  constant
  lower
  upper
  target_name
  operator
  tolerance
```

通用示例：

```text
x1+x2≤100
→ linear_sum(
    variables=[x1,x2],
    coefficients=[1,1],
    upper=100
  )

x3≤15
→ bound(variable=x3, upper=15)

x4 与 x5 不能同时使用
→ mutex(variables=[x4,x5])
```

硬约束规则：

1. 变量类硬约束在候选生成或修复阶段处理；
2. 目标类硬约束在模型预测后复核；
3. 硬约束违反者不得进入最终推荐；
4. 每个候选必须输出逐项约束满足情况；
5. 修复失败不是错误数据，必须保留为不可行候选或计入不可行统计。

### 5.5 SoftConstraintSpec

```text
SoftConstraintSpec
  name
  kind
    minimize_expression
    maximize_expression
    history_distance
    process_stability
    custom
  variables
  weight
  ranking_policy
    tie_breaker
    additional_objective
    prefilter
  filter_threshold
  normalization
  params
```

软约束默认只影响排序，不淘汰候选。惩罚分数统一归一化到 `[0,1]`，越大越差。

| 策略 | 语义 | 使用条件 |
|---|---|---|
| `tie_breaker` | 作为排序或 Pareto 同 rank 内的附加排序项 | 默认策略 |
| `additional_objective` | 作为附加最小化目标参与 Pareto 排序 | 用户显式声明 `ranking_policy=additional_objective` |
| `prefilter` | 惩罚分数超过 `filter_threshold` 时删除候选 | 用户显式声明阈值，且只用于非安全性偏好 |

不允许根据权重大小自动切换策略。高权重成本、历史距离等偏好必须由请求显式声明为附加目标或前置过滤。

软约束归一化基准：

```text
1. 用户显式 normalization.lower / upper；
2. 与请求一起确定的搜索空间理论边界或历史距离 q95；
3. 仅当前候选池可用的有限分位数，并输出非确定性归一化警告；
4. 归一化后截断到 [0,1]，并记录截断比例。
```

多软约束聚合：

```text
soft_score = sum(normalized_weight_i * penalty_i) / sum(normalized_weight_i)
weight_i >= 0
```

当两个非零权重软约束对同一变量给出完全相反的方向，或聚合后候选分数极差小于 `soft_conflict_zero_discrimination_tolerance`，必须输出 `SOFT_CONSTRAINT_CONFLICT` warning；引擎不自动选择一方。

## 6. 模型加载契约

### 6.1 ModelSelection

```text
ModelSelection
  strategy
    latest_valid
    explicit_model_id
  target_mappings
```

默认策略为 `latest_valid`：

1. 只选择可用状态模型；
2. 按目标名匹配；
3. 同一数据集血缘下优先最新版本；
4. 用户显式指定 `model_id + model_version` 时优先使用；
5. 指定模型缺少指标或 AD 元数据时拒绝高可信优化。

### 6.2 模型必备元数据

```text
feature_schema
target_schema
feature_bounds
preprocessing_schema
metrics: RMSE / MAE / R2 / CV Mean / CV Std
applicability_domain
dataset_artifact_id
model_id
model_version
```

错误：

| 错误 | 条件 |
|---|---|
| `MODEL_ARTIFACT_NOT_FOUND` | Registry 中没有可用模型 |
| `MODEL_TARGET_NOT_FOUND` | 请求目标没有匹配模型 |
| `MODEL_SCHEMA_MISMATCH` | 变量无法映射到模型特征 |
| `NO_MODEL_METRICS` | 缺少目标归一化所需指标 |
| `NO_APPLICABILITY_DOMAIN` | 缺少适用域元数据 |
| `LOW_MODEL_QUALITY` | 指标低于配置阈值 |

模型质量门禁：

```text
ModelQualityGate
  mode
    warn
    block
  thresholds
    min_cv_r2
    min_test_r2
    max_cv_rmse
    max_rmse_to_target_range_ratio
```

`warn` 模式下，低于阈值可继续输出实验性候选，但必须降级可信度并写入 `LOW_MODEL_QUALITY`。`block` 模式下返回 `MODEL_QUALITY_BLOCKED`，不得输出推荐候选。阈值必须来自部署配置或请求，不在实现中写死。

质量门禁优先级：

```text
部署配置 block
> 部署配置 warn

请求可以在部署 warn 时请求 block；
请求不得把部署 block 放宽为 warn。
```

若引擎无法获得部署级配置，则使用请求配置，但适配层必须记录配置来源。

## 7. 统一执行流水线

### 7.1 配方推荐流程

```text
请求校验
→ 加载并校验 ModelArtifact
→ 构建 SearchSpace
→ 选择策略
→ 候选生成或优化搜索
→ 变量硬约束修复与过滤
→ 模型预测
→ 目标硬约束复核
→ Applicability Domain 判断
→ 目标归一化与软约束评分
→ 单目标排序或 Pareto 排序
→ 多样性选择
→ Top-N 输出
→ 持久化 OptimizationArtifact
```

执行状态：

```text
COMPLETE
PARTIAL
EMPTY
FAILED
```

`time_limit` 触发时，若已有硬约束可行的已评估候选，返回状态 `PARTIAL` 和当前最优可行结果；若尚无可评估候选，返回状态 `FAILED` 和 `OPTIMIZATION_TIMEOUT_NO_CANDIDATES`。中断结果必须记录已完成评估数、当前代数、停止原因，并持久化 Artifact。

该顺序必须保证：

1. 硬约束违反者不进入最终推荐；
2. `OUT_OF_DOMAIN` 不进入高可信推荐；
3. 所有预测值来自真实模型调用；
4. 所有策略选择和超参数可追溯；
5. 固定 seed 下结果可复现。

### 7.2 执行过程输出

执行过程只允许输出结构化阶段事件或阶段图表数据，不允许输出完整文字报告。

允许：

```text
策略已选择
搜索空间已构建
候选生成进度
约束过滤统计
模型评估统计
收敛曲线数据
```

完整报告只在任务完成时输出。

## 8. 策略选择

### 8.1 选择表

| 条件 | 策略 | 依据 |
|---|---|---|
| 请求下一轮实验 | `bo` | 需要同时考虑提升潜力和信息量 |
| 自由维度低且候选总数可枚举 | `candidate_rank` | 直接评估比启发式搜索更透明 |
| 连续变量、目标为精确值 | `de` | DE 适合连续目标匹配 |
| 精确目标且历史配方支撑足够 | `de_rag` | 历史活跃变量可降低搜索维度 |
| 高维小样本、连续精确目标、历史支撑不足 | `active_set_de` | 两阶段搜索降低稀疏配方维度 |
| 连续变量、多目标或混合阈值 / 区间 | `nsga2` | 需要 Pareto 和多目标排序 |
| 存在整数或分类自由变量，且候选规模不可枚举，同时有多目标或硬目标 | `mixed_nsga2` | 需要混合变量算子、分层初始化和修复 |
| 用户显式指定算法 | `algorithm_override` | 允许专家控制，但必须复用同一约束与 AD 流水线 |

精确目标且低维可枚举时，Candidate Rank 优先于 DE。这样可以让简单问题得到确定性完整评估，而不是引入随机进化搜索。

离散组合必须在生成前估算规模：

```text
estimated_discrete_count = product(len(categories_i) or legal_integer_count_i)
```

若总候选数超过 `candidate_rank_max_count`，不得物化完整笛卡尔积，应改用 Mixed NSGA-II，并用分层随机采样构建初始种群。分类变量不使用 PMX；PMX 适用于排列问题，不适合普通多字段类别白名单。

### 8.2 策略选择输出

每次运行必须输出：

```text
selected_strategy
strategy_reason
strategy_inputs
fallback_used
algorithm_parameters
```

自动回退时必须记录：

```text
原策略
回退原因
回退目标策略
```

### 8.3 策略阈值配置

| 配置 | 默认值 | 用途 |
|---|---:|---|
| `candidate_rank_max_count` | 5000 | Candidate Rank 最大可评估候选数 |
| `candidate_rank_max_free_dimensions` | 3 | Candidate Rank 最大自由维度 |
| `candidate_rank_max_points_per_continuous_dimension` | 100 | 单个连续维度最大网格点数 |
| `de_rag_min_history_count` | 5 | DE-RAG 最少历史候选数 |
| `de_rag_min_active_dimension` | 3 | DE-RAG 最少历史活跃维度 |
| `de_rag_min_coverage_ratio` | 0.20 | 历史活跃维度 / 自由维度 |
| `active_set_min_search_dimension` | 20 | Active Set DE 启动维度 |
| `active_set_stage1_top_ratio` | 0.20 | Stage 1 可行候选头部比例 |
| `mixed_nsga2_categorical_pool_multiplier` | 3 | 分类组合过大时的初始池倍数 |
| `soft_conflict_zero_discrimination_tolerance` | 1e-6 | 软约束聚合后视为无区分的阈值 |

所有阈值可通过 `strategy_thresholds` 覆盖，但必须通过正数和上下界校验，并写入 diagnostics。

## 9. 算法实现要求

### 9.1 Candidate Rank

适用：

```text
自由维度低
估算候选数不超过默认 5000
```

生成方式：

```text
continuous: 网格或 LHS
integer: 枚举合法整数
categorical: 类别笛卡尔积
```

连续维度网格点数：

```text
effective_points_per_dimension =
min(
  candidate_rank_max_points_per_continuous_dimension,
  floor(candidate_rank_max_count ** (1 / continuous_dimension_count))
)
```

网格必须包含边界点；若连续维度为 0，则只估算离散组合数。

要求：

1. 先过滤变量硬约束；
2. 再调用模型预测；
3. 再复核目标硬约束和 AD；
4. 输出完整排序，不使用随机进化；
5. 候选数超过上限时自动改用 DE、NSGA-II 或 Mixed NSGA-II；
6. 分类或整数组合必须先估算数量，禁止直接生成超大笛卡尔积。

### 9.2 DE

适用：

```text
连续变量
一个或多个 equal 目标
低维不可枚举或历史支撑不足
```

目标：

```text
weighted_error = sum(weight_i * abs(pred_i - target_i) / scale_i) / sum(weight_i)
scale_i = max(model_rmse_i, epsilon)
```

默认参数：

```text
pop_size = clamp(5 * d_search, 30, 120)
generations = 50
```

用户可覆盖种群、代数、容差和输出数量。所有覆盖必须写入 diagnostics。

### 9.3 DE-RAG

适用：

```text
连续变量
equal 目标
历史候选数量 ≥5
历史活跃变量维度 ≥3
历史活跃维度 / 自由维度 ≥0.20
```

上述三个阈值分别对应 `de_rag_min_history_count`、`de_rag_min_active_dimension`、`de_rag_min_coverage_ratio`，均可配置。

搜索变量：

```text
model_features ∩ historical_active_variable_union
```

非搜索变量补齐优先级：

```text
历史中位数
训练数据中位数
模型默认值
0，仅当变量语义允许
```

支撑不足时回退 DE，并输出 `HISTORY_SUPPORT_WEAK`。

### 9.4 Active Set DE

适用：

```text
连续变量
equal 目标
d_search > 20
sample_feature_ratio < 1
DE-RAG 条件不满足
```

流程：

```text
Stage 1 全维度 DE
→ 选择活跃变量集合
→ Stage 2 活跃集合内 DE
→ 输出两阶段指标和变量裁剪记录
```

活跃阈值优先级：

```text
用户显式配置
VariableSpec.min_effective_value
训练数据非零分布分位数
保守默认值
```

禁止使用固定业务阈值。被剔除变量必须全部写入 diagnostics。

活跃集合选择规则：

```text
1. 取 Stage 1 硬约束可行且目标排序前 top_ratio 的候选；
2. 对每个变量计算 usage_ratio =
   median(top_candidate_value / variable_effective_range)；
3. usage_ratio ≥ normalized_active_threshold 的变量进入活跃集合；
4. 活跃集合为空时保留 Stage 1 目标误差最小的前 10% 变量；
5. 输出每个变量的 usage_ratio、阈值、阈值来源和是否保留。
```

`variable_effective_range = max(upper - lower, epsilon)`；当下界为 0 时可直接使用有效上界。若 `active_threshold` 来源是绝对用量，必须先除以 `variable_effective_range` 转成 `normalized_active_threshold`；若来源本身是比例，则直接使用并记录来源类型。该规则只做变量集裁剪，不改变模型特征输入。

### 9.5 NSGA-II

适用：

```text
连续变量
多目标
混合阈值
区间目标
需要 Pareto 前沿
```

要求：

1. 所有目标转换为最小化方向；
2. 使用非支配排序；
3. 同 rank 内使用拥挤距离；
4. 每代先修复变量硬约束；
5. 模型预测后复核目标硬约束；
6. `OUT_OF_DOMAIN` 不参与高可信支配排序；
7. 输出收敛历史。

默认参数：

```text
population = clamp(5 * d_search, 50, 200)
generations <= 100
patience = 20
relative_tolerance = 1e-3
```

### 9.6 Mixed NSGA-II

适用：

```text
存在自由 integer 或 categorical 变量
且存在多目标、混合阈值、区间目标或硬目标约束
```

算子：

| 变量类型 | 交叉 | 变异 |
|---|---|---|
| continuous | SBX | 多项式变异 |
| integer | 均匀交叉后取整 | 边界内整数均匀变异 |
| categorical | 字段交换 | 类别白名单均匀变异 |

修复顺序：

```text
bound
→ mutex / categorical_incompatibility
→ linear_sum
→ integer rounding
→ linear_sum 复核
```

输出必须保留原始类别值，而不是模型内部编码。

当分类组合数超过 Candidate Rank 上限时：

```text
1. 不生成完整笛卡尔积；
2. 初始种群采用分层随机采样，覆盖每个分类字段的高频类别、边界类别和随机类别；
3. 连续和整数变量用 LHS 或边界均匀采样；
4. 后续仍使用字段交换、类别均匀变异和硬约束修复；
5. 输出分类组合规模和初始采样方法。
```

### 9.7 Bayesian Optimization

`POST /optimization/next-experiments` 专用。

输入：

```text
HistoricalExperiment
  experiment_id
  values
  observed_values
  constraints_report
```

硬性规则：

1. `observed_values` 必须覆盖全部 BO 目标；
2. 缺实测值的历史样本不得参与 GP 训练；
3. 不得用预测值冒充实测值；
4. 历史样本数量不足时不训练 GP；
5. 与历史样本重复的候选不得作为新实验推荐。

GP BO 最小有效样本数：

```text
min_gp_samples = max(10, 2 * d_search + 1)
```

其中 `d_search` 是标准化后仍有自由变化的数值维度；分类维度按独立类别字段计入维度，不按 one-hot 展开重复计数。

冷启动规则：

```text
有效历史实验数 < min_gp_samples
→ selected_strategy = cold_start_design
→ 生成硬约束可行的 LHS / 分层混合采样候选池
→ 用 max-min 多样性选择 top_n
→ 输出 BO_COLD_START_FALLBACK warning
```

冷启动结果不得输出 EI / PI / UCB acquisition 值，不得标记为 GP BO；仍可输出模型预测和 AD 作为参考，但推荐理由必须是多样性和探索覆盖，而不是期望提升。

代理模型：

```text
GaussianProcessRegressor
```

训练数据：

```text
X = 标准化历史实验变量
y = 标准化目标 reward
```

采集函数：

| 条件 | 默认采集函数 | 目的 |
|---|---|---|
| 默认 | EI | 平衡提升与不确定性 |
| 已有可行较优结果且用户要求利用 | PI | 更倾向开发当前较优区域 |
| 无可行结果或历史信息不足 | UCB | 更倾向探索 |

Batch 推荐：

```text
生成候选池
→ GP 预测 mean / std
→ 计算 acquisition
→ 硬约束违反者 acquisition = -inf
→ 选择最大 acquisition 候选
→ 移除邻域内候选
→ 重复直到 top_n
```

默认多样性邻域：

```text
standardized Euclidean distance = 0.1
categorical mismatch penalty = 1
```

BO 输出必须包含每个推荐实验的：

```text
acquisition_name
acquisition_value
predicted_mean
predicted_std
hard_constraint_report
distance_to_nearest_history
exploration_score
```

## 10. 目标归一化与排序

### 10.1 归一化

每个目标使用模型误差尺度归一化：

```text
scale_i = max(model_rmse_i, epsilon)
```

不同量纲目标不得直接相加。若缺少 RMSE，则拒绝高可信优化，除非用户显式指定备用 scale 并接受实验模式。

### 10.2 误差定义

```text
equal:
abs(predicted - target) / scale

greater_or_equal:
max(0, threshold - predicted) / scale

less_or_equal:
max(0, predicted - threshold) / scale

in_range:
max(lower - predicted, 0, predicted - upper) / scale

maximize:
-normalized_predicted

minimize:
normalized_predicted
```

### 10.3 推荐排序

单目标或加权目标：

```text
hard feasible
→ AD 可信
→ objective score 升序
→ soft score 升序
→ diversity
```

多目标：

```text
hard feasible
→ AD 可信
→ pareto rank 升序
→ crowding distance 降序
→ soft score 升序
→ diversity
```

可信候选不足时不得用 `OUT_OF_DOMAIN` 候选补齐高可信列表，只能返回不足数量和警告。

若 Pareto rank 1 数量少于 `top_n`，可以按 rank 升序继续补充硬约束可行且 AD 可信的候选，但必须：

```text
1. 输出 INSUFFICIENT_PARETO_FRONT_CANDIDATES；
2. 标注被补充候选的 pareto_rank；
3. 将其 trust_level 降为 MEDIUM；
4. 不使用 OUT_OF_DOMAIN 候选补齐。
```

若所有已评估候选都违反硬约束：

```text
selected_candidates = []
status = EMPTY
warning = NO_FEASIBLE_CANDIDATES
```

可以在 `diagnostic_candidates` 中返回少量“约束违反量最小”的候选用于诊断，但它们不得进入推荐列表，trust_level 必须为 `REJECTED`。

统一混合变量多样性距离：

```text
numeric_difference_i = abs(a_i - b_i) / max(upper_i - lower_i, epsilon)
categorical_difference_i = 1 if category differs else 0
distance = sqrt(mean(all component differences squared))
```

选择 top_n 时使用 max-min 距离，避免只围绕单个局部最优聚簇。

## 11. 适用域与可信度

### 11.1 AD 判断

使用模型 Artifact 中的 AD 元数据：

```text
IN_DOMAIN
EDGE
OUT_OF_DOMAIN
```

分类或整数变量补充规则：

1. 训练集中未见过的类别为 `OUT_OF_DOMAIN`；
2. 未见过的类别组合为 `OUT_OF_DOMAIN`；
3. 整数变量只判断距离，不做连续取整。

连续数值 AD 必须使用模型训练时保存的规则：

```text
1. 使用与训练一致的标准化器；
2. 计算候选到训练集的第 k 近邻平均距离；
3. k = min(5, n_train - 1)；
4. distance ≤ q75 → IN_DOMAIN；
5. q75 < distance ≤ q95 → EDGE；
6. distance > q95 → OUT_OF_DOMAIN。
```

`k / q75 / q95` 来自 ModelArtifact，不在优化阶段重新拟合或修改。

### 11.2 Trust Level

| 级别 | 条件 | 输出行为 |
|---|---|---|
| `HIGH` | 硬约束满足 + `IN_DOMAIN` + 模型质量达标 | 可进入默认推荐 |
| `MEDIUM` | 硬约束满足 + `EDGE` 或存在非阻断 warning | 可推荐，必须展示风险 |
| `EXPLORATORY` | 硬约束满足 + `OUT_OF_DOMAIN` 或显式探索边界 | 不进入默认推荐，只进入探索集合 |
| `REJECTED` | 硬约束不满足 | 不进入最终推荐 |

## 12. 输出契约

### 12.1 CandidateResult

```text
CandidateResult
  candidate_id
  values
  predicted_values
  prediction_uncertainty
  objective_values
  objective_errors
  hard_constraint_report
  soft_constraint_score
  applicability_domain
  pareto_rank
  crowding_distance
  diversity_score
  trust_level
  model_refs
```

### 12.2 OptimizationResult

```text
OptimizationResult
  request_id
  status
  selected_candidates
  exploratory_candidates
  diagnostic_candidates
  rejected_summary
  diagnostics
  warnings
  artifact_ids
```

`rejected_summary` 只输出计数和原因分布，不需要保留全部被拒绝候选。

`diagnostic_candidates` 只用于解释失败或边界情况，不能被前端当作推荐结果展示。

diagnostics 必须包含：

```text
selected_strategy
strategy_reason
search_dimension
generated_count
repaired_count
hard_feasible_count
model_evaluated_count
in_domain_count
edge_count
out_of_domain_count
selected_count
algorithm_parameters
removed_variables
elapsed_ms
stop_reason
completed_evaluations
strategy_thresholds
model_refs
```

### 12.3 OptimizationArtifact

每次运行持久化：

```text
optimization_result.json
request_snapshot.json
selected_candidates.csv / parquet
diagnostics.json
warnings.json
visualization_datasets.json
```

Artifact 必须记录：

```text
request_id
model_id
model_version
dataset_artifact_id
request_hash
random_seed
created_at
```

## 13. 可视化数据契约

正式引擎只输出结构化数据，不渲染图片、不生成前端图表配置。测试代码可以渲染 PNG 用于人工验收。

必须暴露：

| 数据集 | 形态 | 用途 |
|---|---|---|
| 推荐候选表 | table | 查看变量、预测值、目标误差、可信度 |
| 约束满足表 | table | 逐项检查硬 / 软约束 |
| AD 分布 | table / bar | 检查可信候选数量 |
| Pareto 前沿 | scatter | 展示目标权衡 |
| 平行坐标 | parallel_coordinates | 比较候选配方结构 |
| 收敛历史 | line | 检查 DE / NSGA-II 收敛 |
| BO acquisition 表 | table | 检查下一轮实验推荐依据 |
| 目标误差分布 | chart_with_table | 判断是否满足用户目标 |

精确数值类内容优先表格；趋势、分布和权衡关系才使用图。

## 14. 错误与警告

| 错误 / 警告 | 条件 | 处理 |
|---|---|---|
| `REQUEST_VALIDATION_FAILED` | 请求结构不合法 | 拒绝执行 |
| `OBJECTIVE_CONFLICT` | 同一目标约束冲突 | 拒绝执行 |
| `SEARCH_SPACE_EMPTY` | 边界交集为空 | 拒绝执行 |
| `HARD_CONSTRAINT_INFEASIBLE` | 预检查不可满足 | 拒绝执行 |
| `MODEL_ARTIFACT_NOT_FOUND` | 无可用模型 | 拒绝执行 |
| `MODEL_TARGET_NOT_FOUND` | 目标无模型 | 拒绝执行 |
| `MODEL_SCHEMA_MISMATCH` | 变量无法映射 | 拒绝执行 |
| `NO_MODEL_METRICS` | 缺少目标尺度 | 拒绝高可信优化 |
| `NO_APPLICABILITY_DOMAIN` | 缺少 AD | 拒绝高可信优化 |
| `LOW_MODEL_QUALITY` | 指标偏低 | 降级为实验输出 |
| `MODEL_QUALITY_BLOCKED` | 质量门禁为 block 且低于阈值 | 拒绝输出推荐 |
| `INSUFFICIENT_HISTORY` | 有效实测历史完全为空 | 拒绝 next-experiments |
| `BO_COLD_START_FALLBACK` | 有效历史数少于 GP 最小样本数 | 回退冷启动设计，不训练 GP |
| `HISTORY_SUPPORT_WEAK` | DE-RAG 支撑不足 | 回退其他策略 |
| `INSUFFICIENT_TRUSTED_CANDIDATES` | AD 可信候选不足 | 返回不足数量 |
| `INSUFFICIENT_PARETO_FRONT_CANDIDATES` | rank 1 前沿少于 top_n | 按 rank 补充并降级 |
| `NO_FEASIBLE_CANDIDATES` | 全部候选违反硬约束 | 返回空推荐和诊断信息 |
| `OPTIMIZATION_TIMEOUT` | 超时但已有可行评估结果 | 返回 `PARTIAL` |
| `OPTIMIZATION_TIMEOUT_NO_CANDIDATES` | 超时且没有已评估候选 | 返回 `FAILED` |
| `SOFT_CONSTRAINT_CONFLICT` | 软约束方向或偏好冲突 | 输出 warning，不自动消解 |
| `EXPLORATION_BOUND_USED` | 使用模型边界外探索 | 只进入探索集合 |
| `ALGORITHM_OVERRIDE_USED` | 用户覆盖策略 | 记录并正常执行 |

## 15. 交叉验证测试矩阵

### 15.1 契约测试

```text
合法请求解析
非法 operator
目标冲突
变量缺失
类别缺失
边界为空
固定变量校验
模型目标缺失
Registry 缺失
BO observed_values 缺失
软约束归一化和冲突检测
质量门禁 warn / block
超时参数校验
```

### 15.2 策略选择测试

| 输入 | 期望策略 |
|---|---|
| 低维单目标 equal，候选数可枚举 | Candidate Rank |
| 连续高维 equal，无历史 | DE 或 Active Set DE |
| 连续 equal，历史覆盖足够 | DE-RAG |
| 连续多阈值 | NSGA-II |
| 混合变量多阈值且候选规模不可枚举 | Mixed NSGA-II |
| 下一轮实验请求且有效历史足够 | BO |
| 分类组合超过 Candidate Rank 上限 | Mixed NSGA-II 分层随机初始化 |
| BO 有效历史少于最小 GP 样本数 | cold_start_design |
| 用户显式覆盖 | override，并保留同一约束流水线 |

### 15.3 约束测试

```text
linear_sum 等式修复
linear_sum 不等式过滤
bound 修复
mutex 修复
categorical_incompatibility 过滤
target_threshold 预测后复核
整数取整后总量复核
不可行约束预检查
硬约束候选不得进入最终列表
```

### 15.4 算法测试

```text
DE 固定 seed 可复现
DE-RAG 覆盖率阈值和回退
Active Set DE 变量剔除记录
NSGA-II 非支配排序正确性
NSGA-II 拥挤距离正确性
Mixed NSGA-II 整数和类别保留原始值
分类组合超限时不物化完整笛卡尔积
Mixed NSGA-II 分层随机初始化和类别变异
BO EI / PI / UCB 数值正确性
BO batch 多样性
BO 拒绝缺实测值历史
BO 冷启动回退且不输出伪 acquisition
软约束 tie_breaker / additional_objective / prefilter
软约束冲突 warning
```

### 15.5 手册验收映射

| 手册场景 | 通用化测试 | 通过标准 |
|---|---|---|
| 多个性能阈值推荐 5 组 | 任意两个 `target_1 / target_2` 阈值，`top_n=5` | 5 组均满足硬约束并输出模型版本 |
| 总量和上限约束 | `linear_sum` + `bound` | 所有最终候选逐项满足 |
| Pareto 候选 | 任意三个合成目标 | 非支配关系正确，输出 rank |
| 外推区域 | 构造模型分布外候选 | 不进入高可信推荐 |
| 35 组历史推荐下一轮 5 组 | 任意 35 条含实测值历史，BO `top_n=5` | 真实 GP + acquisition，输出 5 个多样实验 |

### 15.6 异常与中断测试

```text
全部候选违反硬约束
Pareto rank 1 少于 top_n
高可信候选不足
time_limit 触发且已有可行候选
time_limit 触发且没有可行候选
模型质量 warn 降级
模型质量 block 熔断
软约束 prefilter 和冲突
分类组合爆炸降级
```

### 15.7 集成测试

使用已注册真实模型，不重新训练：

```text
Model Registry
→ ModelArtifact
→ OptimizationRequest
→ OptimizationResult
→ OptimizationArtifact
→ visualization_datasets
```

断言：

1. 未触发数据治理；
2. 未触发模型训练；
3. 输出引用正确 `model_id / model_version / dataset_artifact_id`；
4. 固定 seed 可复现；
5. 硬约束违反者未进入最终候选；
6. `OUT_OF_DOMAIN` 未进入高可信候选；
7. 可视化数据足以生成候选表、Pareto 图和收敛图。

## 16. 交叉验证检查清单

交叉验证者应重点回答：

1. 场景表是否遗漏真实用户请求形态；
2. 策略选择是否存在不可达或互相矛盾的分支；
3. 精确目标、单边阈值、混合阈值、区间目标是否都有明确算法；
4. 智能体识别约束与引擎执行显式约束的边界是否清晰；
5. 硬约束是否可能在最终输出前绕过；
6. `OUT_OF_DOMAIN` 是否可能混入高可信推荐；
7. BO 是否存在用预测值冒充实测值的风险；
8. 输出字段是否足以解释每个候选为什么被推荐；
9. 测试矩阵是否覆盖所有策略和错误分支；
10. 是否存在业务字段或业务阈值写死风险。

## 17. 明确不承诺

1. 不保证全局最优，只保证真实计算、可复现和可追溯；
2. 不在引擎内自动理解自然语言；
3. 不自动发明未声明的业务约束；
4. 不把模型边界外候选包装成高可信结果；
5. 不在正式引擎中渲染图片；
6. 不绕过 Registry 直接加载临时模型作为默认路径。
