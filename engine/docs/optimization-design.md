# 配方优化引擎设计方案

## 1. 目标与边界

本模块面向材料研发场景中的模型驱动逆向设计，负责根据用户提出的目标性能、变量范围和约束条件，推荐候选配方或工艺方案。引擎独立于 Demo3 Agent、FastAPI 和前端运行，先以离线 CLI 与 Python API 交付，后续通过适配层接入 Demo3 或其他系统。

本设计对齐手册第十章 `Inverse Design + Optimization + BO` 的要求：

- 支持 `Continuous / Integer / Categorical` 变量；
- 支持 `HARD / SOFT` 约束；
- 支持材料特有约束，如总量约束、用量上限、互斥使用；
- 执行顺序为候选生成、硬约束过滤、正向模型预测、适用域判断、多目标排序和候选多样性选择；
- BO 使用 Gaussian Process 代理模型，支持 `EI / PI / UCB`；
- `OUT_OF_DOMAIN` 候选不得进入高可信推荐。

商业与来源边界：

- MatDesign 仅作为功能对照，不复制其代码、模块名、接口形态或文件结构；
- 不引入 MatDesign 依赖；
- 算法实现基于 `numpy / pandas / scikit-learn / scipy / DEAP` 等通用库；
- 所有目标名、变量名、单位和阈值均来自请求契约或模型元数据，不在实现中写死。

## 2. 已评估的设计决策

### 2.1 Projection DE 调整

Demo3 当前的 `optimize_target_de_projection` 是一种两阶段稀疏启发式：

```text
第一阶段：全维度 DE 搜索
第二阶段：剔除低用量变量后，在活跃变量子集内再次 DE 搜索
```

科学性评估：

- 优点：能减少高维稀疏配方中的无效搜索维度，计算成本低于直接高维优化；
- 局限：它不是严格的投影优化，也没有全局最优保证；
- 风险：固定阈值裁剪可能误删微量但重要的添加剂；
- 结论：保留为高维连续目标匹配场景的备选启发式，不作为默认主策略。

调整点：

1. 名称改为 `active_set_de`，避免与数学意义上的投影优化混淆；
2. 裁剪阈值不使用固定 `0.5`，由以下优先级决定：
   - 用户显式配置；
   - 变量单位对应的业务最小有效量；
   - 训练数据中该变量的非零分布分位数；
   - 引擎默认兜底值；
3. 被剔除变量必须写入 diagnostics，不允许静默删除；
4. 仅用于连续变量和明确目标值匹配；
5. 输出必须标记 `sparse_method`、活跃变量集合和被剔除变量集合。

### 2.2 NSGA-II / Pareto 调整

Demo3 当前已有基于 DEAP 的 NSGA-II 骨架，具备连续变量边界、Pareto 前沿、收敛曲线和基础多样性选择。但核查后发现以下不足：

- 未系统覆盖分类变量和整数变量；
- 约束表达主要依赖 DataFrame 列名和边界，缺少通用契约；
- 总量约束、互斥约束、软约束未形成独立约束层；
- 目标阈值多以惩罚项进入适应度，最终候选缺少逐项硬约束报告；
- 多目标多样性评分部分使用单目标分数，不能充分代表多目标分布；
- 缺少多目标基准测试和 Pareto 正确性测试。

调整结论：

- 保留 NSGA-II 的非支配排序思想和 DEAP 的成熟选择算子；
- 不直接沿用当前 `FormulaOptimizer` 的紧耦合实现；
- 重新实现通用 `mixed_nsga2` 策略层，覆盖混合变量、通用约束、目标归一化、Pareto 排序、拥挤距离和多样性选择；
- 所有候选输出逐项约束满足情况，硬约束违反者不得进入最终推荐。

### 2.3 通用性约束

`impact / mfr / vicat / A / B / C / 35 / 18 / 125 / A+B+C=100` 等只允许出现在：

- 示例说明；
- 验收测试夹具；
- 用户请求输入；
- 模型元数据。

引擎实现不得写死这些名称或数值。所有业务语义必须来自 `OptimizationRequest` 与 `ModelBundle`。

## 3. 输入输出契约

### 3.1 OptimizationRequest

```text
OptimizationRequest
  request_id
  mode
    recommend_recipe
    recommend_next_experiments
  objectives
  variables
  hard_constraints
  soft_constraints
  model_registry_path
  model_selection
  model_bundle_ids
  historical_candidates
  top_n
  random_seed
  max_evaluations
  time_limit
  execution_options
  strategy_thresholds
  model_quality_gate
  algorithm_override
```

`model_selection`：

```text
latest_valid
explicit_model_id
test_fixture
```

优化阶段必须优先通过 `model_registry_path` 和 `model_selection` 加载已持久化的 `ModelArtifact`。`model_bundle_ids` 仅用于显式指定版本，不允许成为绕过 Registry 的默认路径。

`strategy_thresholds` 管理策略切换阈值，默认值：

```text
candidate_rank_max_count = 5000
candidate_rank_max_free_dimensions = 3
candidate_rank_max_points_per_continuous_dimension = 100
de_rag_min_history_count = 5
de_rag_min_active_dimension = 3
de_rag_min_coverage_ratio = 0.20
active_set_min_search_dimension = 20
active_set_stage1_top_ratio = 0.20
mixed_nsga2_categorical_pool_multiplier = 3
soft_conflict_zero_discrimination_tolerance = 1e-6
```

所有阈值可配置，但必须通过正数和上下界校验，并写入 diagnostics。

### 3.2 ObjectiveSpec

```text
ObjectiveSpec
  target_name
  operator
    equal
    greater_or_equal
    less_or_equal
    maximize
    minimize
  value
  weight
  model_id
  unit
```

`target_name` 必须能匹配 `ModelBundle.target_schema` 中的目标名。引擎不假设目标数量，也不假设目标方向。

### 3.3 VariableSpec

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
```

变量类型决定生成和进化算子：

| 类型 | 生成方式 | 进化方式 |
|---|---|---|
| continuous | 边界内均匀采样或 LHS | SBX / 多项式变异 |
| integer | 边界内整数采样 | 变异后取整并修复 |
| categorical | 类别白名单采样 | 类别均匀变异或子树交换 |

### 3.4 HardConstraintSpec

统一使用结构化约束，不依赖中文列名或固定 DataFrame 格式。

```text
HardConstraintSpec
  kind
    linear_sum
    bound
    mutex
    target_threshold
    categorical_incompatibility
  variables
  coefficients
  constant
  lower
  upper
  target_name
  operator
  tolerance
```

示例语义：

```text
A+B+C=100
→ linear_sum(variables=[A,B,C], coefficients=[1,1,1], constant=100)

A≤15
→ bound(variable=A, upper=15)

A与B不能同时使用
→ mutex(variables=[A,B])
```

上述 `A / B / C / 100 / 15` 只是说明，不是实现常量。

### 3.5 SoftConstraintSpec

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

软约束默认不淘汰候选，只计算归一化惩罚并影响排序。用户显式声明 `ranking_policy` 时：

| 策略 | 行为 |
|---|---|
| `tie_breaker` | 作为排序或 Pareto 同 rank 内的附加排序项 |
| `additional_objective` | 作为附加最小化目标参与 Pareto 排序 |
| `prefilter` | 惩罚分数超过 `filter_threshold` 时删除候选 |

不允许因为权重高而自动切换策略。`prefilter` 只能用于成本、历史距离等非安全性偏好，不能替代硬约束。

### 3.6 CandidateResult

```text
CandidateResult
  candidate_id
  values
  predicted_values
  prediction_uncertainty
  objective_values
  hard_constraint_report
  soft_constraint_score
  applicability_domain
  pareto_rank
  crowding_distance
  diversity_score
  trust_level
```

### 3.7 OptimizationResult

```text
OptimizationResult
  request_id
  status
  selected_candidates
  exploratory_candidates
  diagnostic_candidates
  diagnostics
  warnings
  artifact_ids
```

diagnostics 至少包含：

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

## 4. 统一执行流水线

所有策略共用同一条流水线：

```text
OptimizationRequest
  → 加载 model-registry.json
  → 解析 ModelArtifact
  → Schema 校验
  → 模型与目标匹配校验
  → 构建 SearchSpace
  → 策略选择
  → 候选生成或优化搜索
  → 硬约束修复与过滤
  → 模型预测
  → 适用域判断
  → 目标归一化
  → 软约束评分
  → Pareto 或单目标排序
  → 候选多样性选择
  → Top-N 输出
```

该顺序与手册第十章一致，保证硬约束候选不会仅因适应度较好而进入最终列表。

执行状态：

```text
COMPLETE
PARTIAL
EMPTY
FAILED
```

`time_limit` 触发时：

```text
已有硬约束可行的已评估候选
→ status = PARTIAL
→ 返回当前最优可行结果

尚无已评估可行候选
→ status = FAILED
→ 返回 OPTIMIZATION_TIMEOUT_NO_CANDIDATES
```

中断结果必须记录已完成评估数、当前代数、停止原因，并持久化 Artifact。

模型复用规则：

```text
1. 优化测试以已建好的 ModelArtifact 为输入；
2. 不重复执行数据治理；
3. 不重复训练模型；
4. 目标名和特征 Schema 必须来自 ModelArtifact；
5. Registry 中找不到匹配模型时立即失败；
6. 结果 artifact 记录 model_id、model_version 和 dataset_version。
```

## 5. 策略选择规则

策略由输入结构、模型元数据和历史候选质量自动决定。

```python
def select_strategy(request, model_bundle, historical_candidates):
    variables = request.variables
    objectives = request.objectives
    thresholds = request.strategy_thresholds

    d_free = count_free_variables(variables)
    d_search = count_independent_search_dimensions(variables)
    d_history = count_active_variable_union(historical_candidates)
    estimated_candidate_count = estimate_candidate_count(variables)
    has_mixed_variable = any(
        variable.type in {"integer", "categorical"}
        for variable in variables
        if variable.fixed_value is None
    )
    objective_count = len(objectives)
    exact_target_count = count_exact_targets(objectives)
    can_candidate_rank = (
        d_free <= thresholds.candidate_rank_max_free_dimensions
        and estimated_candidate_count <= thresholds.candidate_rank_max_count
    )

    if request.mode == "recommend_next_experiments":
        valid_history_count = count_complete_observed_experiments(
            historical_candidates,
            objectives,
        )
        min_gp_samples = max(10, 2 * d_search + 1)
        if valid_history_count == 0:
            raise INSUFFICIENT_HISTORY
        if valid_history_count < min_gp_samples:
            return "cold_start_design"
        return "bo"

    if can_candidate_rank:
        return "candidate_rank"

    if has_mixed_variable and (objective_count >= 2 or has_hard_target_constraint(request)):
        return "mixed_nsga2"

    if all_objectives_are_exact_targets(objectives):
        if (
            d_history >= thresholds.de_rag_min_active_dimension
            and d_free > 0
            and d_history / d_free >= thresholds.de_rag_min_coverage_ratio
            and len(historical_candidates) >= thresholds.de_rag_min_history_count
        ):
            return "de_rag"

        if (
            d_free > thresholds.active_set_min_search_dimension
            and model_bundle.summary.sample_feature_ratio < 1
        ):
            return "active_set_de"

        return "de"

    if objective_count >= 2:
        return "nsga2"

    return "nsga2"
```

### 策略分层表

| 输入条件 | 输出策略 | 原因 |
|---|---|---|
| 用户要求下一批实验且有效实测历史足够 | `bo` | 需要信息价值和提升潜力，不只是目标匹配 |
| 存在整数或分类自由变量，候选规模不可枚举，且为多目标或有目标约束 | `mixed_nsga2` | 混合变量需要专门进化、分层初始化和修复算子 |
| 全部目标为明确值，RAG 活跃变量覆盖至少 20% 且历史候选不少于 5 条 | `de_rag` | 历史配方可约束搜索空间，减少无效探索 |
| 高维小样本，RAG 支撑不足，目标为明确值 | `active_set_de` | 两阶段稀疏搜索降低维度 |
| 普通连续目标值匹配 | `de` | 目标匹配问题适合差分进化 |
| 多目标或多约束阈值 | `nsga2` / `mixed_nsga2` | 需要 Pareto 前沿 |
| 低维小规模可枚举 | `candidate_rank` | 直接生成全候选并排序，避免不必要进化 |
| 分类 / 整数组合超过枚举上限 | `mixed_nsga2` + 分层随机初始化 | 避免物化超大笛卡尔积 |
| BO 有效实测历史少于最小 GP 样本数 | `cold_start_design` | 不训练病态 GP，先做多样性探索采样 |

策略选择必须输出 `strategy_reason`，便于审计和调试。

离散组合必须先估算数量：

```text
estimated_discrete_count =
product(len(categories_i) or legal_integer_count_i)
```

超过 `candidate_rank_max_count` 时不允许生成完整笛卡尔积。分类变量不使用 PMX；PMX 面向排列问题，不适合普通类别白名单变量。

## 6. 搜索空间构建

### 6.1 边界合并

变量有效边界按以下优先级合并：

```text
VariableSpec 边界
∩ HardConstraintSpec 边界
∩ ModelBundle.feature_bounds
```

若交集为空，直接返回 `SEARCH_SPACE_EMPTY`，不得静默扩大边界。

### 6.2 固定变量

如果 `fixed_value` 非空：

```text
1. 校验类型和边界；
2. 从自由搜索维度移除；
3. 预测时按模型特征顺序补齐；
4. 输出时保留该变量和值。
```

### 6.3 类别变量

类别变量必须满足：

```text
1. categories 非空且无重复；
2. 模型元数据可编码该类别；
3. 约束层可识别互斥或不兼容组合；
4. 输出保留原始类别值，而不是内部编码。
```

## 7. 目标函数

目标函数使用模型 RMSE 归一化，避免不同量纲的目标互相压倒。

```text
rmse_i = ModelBundle 中目标 i 的 RMSE
scale_i = max(rmse_i, epsilon)
```

### 7.1 equal

```text
error_i = abs(predicted_i - target_i) / scale_i
```

### 7.2 greater_or_equal

```text
shortage_i = max(0, threshold_i - predicted_i)
error_i = shortage_i / scale_i
```

### 7.3 less_or_equal

```text
excess_i = max(0, predicted_i - threshold_i)
error_i = excess_i / scale_i
```

### 7.4 maximize / minimize

```text
maximize:
    objective_i = -normalized_predicted_i

minimize:
    objective_i = normalized_predicted_i
```

### 7.5 多目标聚合

DE 与 BO 可使用加权聚合：

```text
weighted_objective = sum(weight_i * error_i) / sum(weight_i)
```

NSGA-II 不聚合为单值，而是输出多目标向量，由非支配排序处理。

## 8. 硬约束处理

## 8.1 校验优先级

```text
1. 请求 Schema 校验；
2. 变量存在性校验；
3. 边界一致性校验；
4. 约束可满足性预检查；
5. 模型目标匹配校验。
```

若总量下界之和大于目标总量，或总量上界之和小于目标总量，直接返回不可行，不启动优化。

## 8.2 linear_sum 修复

适用于总量、配比、工艺加和等通用线性约束：

```text
Σ coefficient_j * variable_j = constant
```

修复流程：

```text
1. 只选取参与该约束的自由连续或整数变量；
2. 计算当前加和；
3. 按比例缩放差额；
4. 若变量越过边界，固定到边界；
5. 将剩余差额分配给未触界变量；
6. 整数变量在修复后取整并重新校验；
7. 若超出 tolerance，判定不可行。
```

## 8.3 bound

```text
if value < lower:
    value = lower
elif value > upper:
    value = upper
```

修复后必须重新执行 `linear_sum` 修复，避免边界裁剪破坏总量约束。

## 8.4 mutex

互斥集合内最多允许一个变量取非零或非空类别：

```text
1. 找出当前活跃变量；
2. 若活跃数量 ≤ 1，通过；
3. 若活跃数量 > 1：
   a. 用户配置了保留优先级时，保留优先级最高者；
   b. 否则保留数值最大者；
   c. 其余连续变量置 0，类别变量置空类别；
4. 重新执行相关 linear_sum 修复。
```

## 8.5 target_threshold

目标约束先作为目标函数或惩罚参与搜索，最终输出前必须再次校验：

```text
greater_or_equal:
    predicted_value >= threshold - tolerance

less_or_equal:
    predicted_value <= threshold + tolerance
```

不满足者写入 `hard_constraint_report`，并从最终推荐中移除。

## 9. 软约束评分

软约束输出统一为 `[0, 1]` 惩罚分数，越大越差。

```text
soft_score = weighted_mean(soft_penalty_i)
weight_i >= 0
```

归一化基准：

```text
1. 用户显式 normalization.lower / upper；
2. 与请求一起确定的搜索空间理论边界或历史距离 q95；
3. 仅当前候选池可用的有限分位数，并输出非确定性归一化警告；
4. 归一化后截断到 [0,1]，并记录截断比例。
```

当两个非零权重软约束对同一变量给出完全相反方向，或聚合后候选分数极差小于 `soft_conflict_zero_discrimination_tolerance`，输出 `SOFT_CONSTRAINT_CONFLICT`，引擎不自动选择一方。

常见类型：

| 类型 | 评分 |
|---|---|
| 成本最小化 | 按候选成本在搜索空间中的归一化位置 |
| 历史距离最小化 | 标准化变量空间中到最近历史候选的距离 |
| 工艺稳定性 | 与中心工艺窗口的标准化偏差 |
| 自定义表达式 | 由安全表达式求值器计算后归一化 |

最终排序：

```text
final_score = objective_score + soft_weight * soft_score
```

NSGA-II 中软约束作为附加目标或 tie-breaker，不改变 Pareto 支配关系的核心定义。
只有用户显式声明 `ranking_policy=additional_objective` 时，软约束才进入 Pareto 目标向量；只有用户显式声明 `prefilter` 和 `filter_threshold` 时，软约束才前置过滤候选。禁止按权重大小自动切换。

## 10. 策略实现

## 10.1 DE

适用：

```text
连续变量
目标为明确匹配值
无复杂分类约束
```

参数默认规则：

```text
pop_size = clamp(5 * d_search, 30, 120)
if d_search < 3:
    pop_size = 100

generations = 50
```

用户显式配置优先。

输出：

```text
best_candidate
all_feasible_candidates
convergence_history
evaluation_count
converged
```

## 10.2 DE-RAG

适用：

```text
连续变量
目标为明确匹配值
历史候选活跃变量覆盖率 ≥ 20%
历史候选数量 ≥ 5
活跃变量维度 ≥ 3
```

搜索维度：

```text
search_variables = model_features ∩ historical_active_variable_union
```

非搜索变量按以下规则补齐：

```text
1. 历史候选中位数；
2. 训练数据中位数；
3. 业务默认值；
4. 0，仅当变量语义允许。
```

输出必须记录：

```text
historical_candidate_count
historical_active_dimension
search_dimension
coverage_ratio
fixed_variable_strategy
```

## 10.3 Active Set DE

适用：

```text
连续变量
目标为明确匹配值
d_search > 20
sample_feature_ratio < 1
DE-RAG 条件不满足
```

流程：

```text
Stage 1: 全维度 DE
→ 依据变量级 active_threshold 选择活跃集合
→ Stage 2: 活跃集合内 DE
→ 历史候选可注入 Stage 2 初始种群
→ 输出活跃变量、剔除变量和阈值来源
```

变量级 `active_threshold` 优先级：

```text
1. 用户请求配置；
2. VariableSpec.min_effective_value；
3. 训练数据非零分布分位数；
4. 引擎保守默认值。
```

活跃集合选择规则：

```text
1. 取 Stage 1 硬约束可行且目标排序前
   active_set_stage1_top_ratio 的候选；
2. 对每个变量计算：
   usage_ratio = median(top_candidate_value / variable_effective_range)；
3. usage_ratio ≥ normalized_active_threshold 的变量进入活跃集合；
4. 活跃集合为空时，保留 Stage 1 目标误差最小的前 10% 变量；
5. 输出每个变量的 usage_ratio、阈值、阈值来源和是否保留。
```

`variable_effective_range = max(upper - lower, epsilon)`；当下界为 0 时可直接使用有效上界。若 `active_threshold` 来源是绝对用量，必须先除以 `variable_effective_range` 转成 `normalized_active_threshold`；若来源本身是比例，则直接使用并记录来源类型。该规则只裁剪搜索变量，不改变模型特征输入。

该策略必须输出：

```text
sparse_method = active_set_de
active_variables
removed_variables
threshold_source
stage1_metrics
stage2_metrics
```

## 10.4 NSGA-II

适用：

```text
连续变量为主
多目标或多约束阈值
需要 Pareto 前沿
```

实现要点：

```text
1. 使用 DEAP selNSGA2 作为非支配选择器；
2. 连续变量使用 SBX 和多项式变异；
3. 整数变量变异后取整并修复；
4. 每代候选先修复硬约束，再计算模型预测和目标；
5. OUT_OF_DOMAIN 不参与高可信支配排序，仅进入探索集合；
6. 输出 Pareto rank 和 crowding distance。
```

收敛判断：

```text
patience = 20
relative_tolerance = 1e-3
```

当 Pareto 前沿目标均值在连续 `patience` 代内最大相对变化小于 `relative_tolerance`，允许提前停止。

## 10.5 Mixed NSGA-II

适用：

```text
存在整数或分类自由变量
且存在多目标或目标硬约束
```

个体表示：

```text
candidate = {
  continuous_values,
  integer_values,
  categorical_values
}
```

算子：

| 变量类型 | 交叉 | 变异 |
|---|---|---|
| continuous | SBX | 多项式变异 |
| integer | 均匀交叉后取整 | 边界内整数均匀变异 |
| categorical | 按字段交换 | 类别白名单均匀变异 |

硬约束修复顺序：

```text
bound
→ mutex / categorical_incompatibility
→ linear_sum
→ integer rounding
→ linear_sum
```

当分类 / 整数组合规模超过 `candidate_rank_max_count` 时：

```text
1. 不生成完整笛卡尔积；
2. 初始种群使用分层随机采样，覆盖高频类别、边界类别和随机类别；
3. 连续和整数变量使用 LHS 或边界均匀采样；
4. 后续仍使用字段交换、类别均匀变异和硬约束修复；
5. 输出分类组合规模和初始采样方法。
```

不使用 PMX。PMX 面向排列问题，不适合普通多字段类别白名单。

## 10.6 Candidate Rank

适用：

```text
d_free ≤ 3
估算候选数 ≤ 5000
```

生成方式：

```text
continuous: 均匀网格或 LHS
integer: 枚举合法整数
categorical: 类别笛卡尔积
```

候选总数必须在生成前估算：

```text
effective_points_per_dimension =
min(
  candidate_rank_max_points_per_continuous_dimension,
  floor(candidate_rank_max_count ** (1 / continuous_dimension_count))
)

estimated_candidate_count =
discrete_combination_count × effective_points_per_dimension ^ continuous_dimension_count
```

网格必须包含边界点；若连续维度为 0，则只估算离散组合数。

若总数超过 `candidate_rank_max_count`，不得物化完整候选集，自动改用 NSGA-II 或 Mixed NSGA-II。

流程：

```text
生成候选
→ 硬约束过滤
→ 模型预测
→ 适用域过滤
→ 目标与软约束评分
→ 排序与多样性选择
```

## 10.7 Bayesian Optimization

触发条件：

```text
request.mode = recommend_next_experiments
```

历史实验输入：

```text
HistoricalExperiment
  experiment_id
  values
  observed_values
  constraints_report
```

硬性规则：

```text
1. observed_values 必须覆盖全部 BO 目标；
2. 缺实测值的历史样本不得参与 GP 训练；
3. 不得用模型预测值冒充实测值；
4. 与历史样本重复的候选不得作为新实验推荐。
```

最小 GP 样本数：

```text
min_gp_samples = max(10, 2 * d_search + 1)
```

`d_search` 是标准化后仍有自由变化的数值维度；分类维度按独立类别字段计数，不按 one-hot 展开重复计数。

冷启动规则：

```text
有效历史实验数 < min_gp_samples
→ selected_strategy = cold_start_design
→ 生成硬约束可行的 LHS / 分层混合采样候选池
→ 用 max-min 多样性选择 top_n
→ 输出 BO_COLD_START_FALLBACK
```

冷启动结果不得输出 EI / PI / UCB acquisition 值，不得标记为 GP BO；模型预测和 AD 只能作为参考。

### Surrogate

```text
GaussianProcessRegressor
```

训练输入：

```text
X = 标准化后的历史实验变量
y = 标准化后的 reward
```

reward 由通用 ObjectiveSpec 计算得到，不绑定特定性能名。

### Acquisition 自动选择

```python
def select_acquisition(request, observed_rewards):
    if request.acquisition is specified:
        return request.acquisition

    if not has_feasible_reward(observed_rewards):
        return "UCB"

    if request.preference == "exploit":
        return "PI"

    return "EI"
```

| 条件 | 采集函数 | 目的 |
|---|---|---|
| 默认 | EI | 平衡期望提升与不确定性 |
| 已有可行较优结果且用户要求利用 | PI | 更偏开发当前最优区域 |
| 无可行结果或信息不足 | UCB | 更偏探索 |

### Batch Diversity

```text
1. 在 SearchSpace 中生成 candidate_pool；
2. GP 预测 mean 与 std；
3. 计算 acquisition；
4. 硬约束违反者 acquisition = -inf；
5. 选择当前最大 acquisition 候选；
6. 移除该候选邻域内样本；
7. 重复直到达到 top_n。
```

默认邻域半径：

```text
standardized Euclidean distance = 0.1
```

## 11. 适用域判断

训练阶段：

```text
1. 使用与模型一致的标准化器处理训练特征；
2. 计算每个训练样本到其他训练样本的 kNN 距离；
3. k = min(5, n_train - 1)；
4. 取训练距离分布 q75 与 q95。
```

推理阶段：

```text
distance = 候选到训练集的第 k 近邻平均距离

distance ≤ q75:
    IN_DOMAIN

q75 < distance ≤ q95:
    EDGE

distance > q95:
    OUT_OF_DOMAIN
```

分类变量补充规则：

```text
训练集中未见过的类别或类别组合 → OUT_OF_DOMAIN
```

推荐规则：

| AD 状态 | 推荐行为 |
|---|---|
| IN_DOMAIN | 可进入高可信推荐 |
| EDGE | 可推荐，但必须输出边界风险 |
| OUT_OF_DOMAIN | 不进入高可信推荐，只能进入显式探索集合 |

## 12. Pareto 与候选排序

### 12.1 非支配排序

所有目标先转换为最小化方向：

```text
maximize objective → -objective
minimize / error → 原值
```

候选 A 支配候选 B 当且仅当：

```text
1. A 在所有目标上不差于 B；
2. A 至少在一个目标上严格优于 B。
```

### 12.2 拥挤距离

对同一 Pareto rank 内的候选：

```text
1. 按每个归一化目标排序；
2. 边界候选拥挤距离为 inf；
3. 中间候选累计相邻目标区间长度；
4. 拥挤距离越大，优先保留。
```

### 12.3 多样性选择

多样性在标准化混合变量空间中计算：

```text
continuous: 标准化欧氏距离
integer: 标准化距离
categorical: mismatch penalty
```

选择规则：

```text
1. hard_constraint_violation = 0；
2. AD ≠ OUT_OF_DOMAIN；
3. Pareto rank 升序；
4. crowding distance 降序；
5. 在同 rank 内执行 max-min 多样性选择；
6. 输出 top_n。
```

若可信候选不足：

```text
1. 返回实际可得数量；
2. 输出 INSUFFICIENT_TRUSTED_CANDIDATES；
3. 不用 OUT_OF_DOMAIN 候选补齐高可信列表。
```

若 Pareto rank 1 数量少于 `top_n`，可按 rank 升序补充硬约束可行且 AD 可信的候选，但必须：

```text
1. 输出 INSUFFICIENT_PARETO_FRONT_CANDIDATES；
2. 标注补充候选的 pareto_rank；
3. 将补充候选 trust_level 降为 MEDIUM；
4. 不使用 OUT_OF_DOMAIN 候选补齐。
```

若所有已评估候选均违反硬约束：

```text
selected_candidates = []
status = EMPTY
warning = NO_FEASIBLE_CANDIDATES
```

`diagnostic_candidates` 可返回少量约束违反量最小的候选用于诊断，但 trust_level 必须为 `REJECTED`，不得进入推荐列表。

统一混合变量多样性距离：

```text
numeric_difference_i = abs(a_i - b_i) / max(upper_i - lower_i, epsilon)
categorical_difference_i = 1 if category differs else 0
distance = sqrt(mean(all component differences squared))
```

Top-N 选择使用 max-min 多样性距离，避免候选聚簇在单个局部最优附近。

## 13. 代码模块设计

```text
engine/optimization/
  contracts.py
  search_space.py
  strategy_selector.py
  constraints/
    validator.py
    repair.py
    filter.py
    soft_score.py
  evaluation/
    predictor.py
    objectives.py
    applicability_domain.py
  ranking/
    pareto.py
    crowding.py
    diversity.py
    trust.py
  strategies/
    de.py
    de_rag.py
    active_set_de.py
    nsga2.py
    mixed_nsga2.py
    candidate_rank.py
    bo.py
  candidate_generator.py
  service.py
```

核心入口：

```python
def optimize(request: OptimizationRequest) -> OptimizationResult:
    validate_request(request)
    model_bundle = load_model_bundle(request.model_bundle_ids)
    space = build_search_space(request.variables, request.hard_constraints, model_bundle)
    strategy = select_strategy(request, model_bundle, request.historical_candidates)
    candidates = run_strategy(strategy, request, model_bundle, space)
    candidates = repair_and_filter(candidates, request)
    candidates = predict(candidates, model_bundle)
    candidates = classify_applicability_domain(candidates, model_bundle)
    candidates = rank_candidates(candidates, request)
    selected = select_diverse_trusted(candidates, request.top_n)
    return build_optimization_result(selected, strategy, diagnostics)
```

BO 入口：

```python
def recommend_next_experiments(request: OptimizationRequest) -> OptimizationResult:
    validate_request(request)
    model_bundle = load_model_bundle(request.model_bundle_ids)
    return BayesianOptimizer(request, model_bundle).optimize()
```

## 14. 错误与警告

| 错误 | 条件 | 处理 |
|---|---|---|
| `MODEL_TARGET_NOT_FOUND` | 目标名不在模型目标 Schema | 拒绝执行 |
| `MODEL_ARTIFACT_NOT_FOUND` | Registry 中没有可用模型或指定版本不存在 | 拒绝执行，并提示先运行建模阶段 |
| `SEARCH_SPACE_EMPTY` | 边界交集为空 | 拒绝执行 |
| `HARD_CONSTRAINT_INFEASIBLE` | 预检查不可满足 | 拒绝执行 |
| `NO_MODEL_METRICS` | 缺少 RMSE / MAE 等评估指标 | 拒绝高可信优化 |
| `NO_APPLICABILITY_DOMAIN` | 模型缺少 AD 元数据 | 拒绝高可信优化 |
| `LOW_MODEL_QUALITY` | 指标低于配置阈值且质量门禁为 `warn` | 可运行实验优化，但输出低可信警告 |
| `MODEL_QUALITY_BLOCKED` | 指标低于配置阈值且质量门禁为 `block` | 拒绝输出推荐 |
| `INSUFFICIENT_TRUSTED_CANDIDATES` | IN_DOMAIN / EDGE 候选不足 | 返回不足数量和警告 |
| `INSUFFICIENT_PARETO_FRONT_CANDIDATES` | Pareto rank 1 少于 `top_n` | 按 rank 补充并降低可信度 |
| `NO_FEASIBLE_CANDIDATES` | 全部候选违反硬约束 | 返回空推荐和诊断候选 |
| `INSUFFICIENT_HISTORY` | BO 有效实测历史为空 | 拒绝 next-experiments |
| `BO_COLD_START_FALLBACK` | BO 有效历史少于最小 GP 样本数 | 回退冷启动设计，不训练 GP |
| `OPTIMIZATION_TIMEOUT` | 超时且已有可行评估候选 | 返回 `PARTIAL` |
| `OPTIMIZATION_TIMEOUT_NO_CANDIDATES` | 超时且没有已评估候选 | 返回 `FAILED` |
| `SOFT_CONSTRAINT_CONFLICT` | 软约束方向或偏好冲突 | 输出 warning，不自动消解 |
| `HISTORY_SUPPORT_WEAK` | DE-RAG 支撑不足 | 回退 DE 或 active_set_de |

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

`warn` 用于离线或探索任务；`block` 用于生产或高可信推荐。阈值来自部署配置或请求，不在实现中写死。

质量门禁优先级：

```text
部署配置 block
> 部署配置 warn

请求可以在部署 warn 时请求 block；
请求不得把部署 block 放宽为 warn。
```

若引擎无法获得部署级配置，则使用请求配置，但适配层必须记录配置来源。

## 15. 测试计划

### 15.1 单元测试

```text
SearchSpace 边界合并
变量类型编码
linear_sum 修复
bound 修复
mutex 修复
目标归一化
软约束评分
软约束冲突
模型质量门禁 warn / block
Pareto 非支配排序
crowding distance
多样性选择
分类组合规模估算
AD 三分类
策略选择规则
BO 冷启动样本数
```

### 15.2 策略测试

| 策略 | 测试点 |
|---|---|
| DE | 收敛、边界、重复种子可复现 |
| DE-RAG | 历史维度裁剪、覆盖率、弱支撑回退 |
| Active Set DE | 阈值来源、剔除变量记录、两阶段指标 |
| NSGA-II | Pareto 正确性、多目标收敛、约束过滤 |
| Mixed NSGA-II | 整数、分类、互斥和总量约束 |
| Candidate Rank | 网格规模、候选总数估算、排序完整性 |
| BO | EI / PI / UCB 数值正确性和 batch 多样性 |
| Cold Start Design | LHS / 分层采样、max-min 多样性、无伪 acquisition |

### 15.2.1 异常与中断测试

```text
全部候选违反硬约束
Pareto rank 1 少于 top_n
time_limit 触发且已有可行候选
time_limit 触发且没有可行候选
模型质量 warn 降级
模型质量 block 熔断
软约束 prefilter 和冲突
分类组合爆炸自动改用 Mixed NSGA-II
```

### 15.3 手册验收映射

手册示例只作为测试输入，不进入实现常量。

| 手册测试 | 通用化测试 |
|---|---|
| 两个性能阈值，推荐 5 组 | 任意两个模型目标阈值，`top_n=5`，使用混合或多目标策略 |
| 总量和上限约束 | 通用 `linear_sum` 与 `bound` 约束 |
| 多性能 Pareto | 任意三个合成目标验证非支配排序 |
| 外推区域 | 构造训练分布外候选，验证 OUT_OF_DOMAIN 不进入高可信推荐 |
| 35 组实验推荐下一轮 5 组 | 任意 35 个历史样本，BO 输出 5 个满足硬约束的多样候选 |

集成测试必须使用已注册模型：

```text
engine/artifacts/models/model-registry.json
→ load ModelArtifact
→ OptimizationRequest
→ OptimizationResult
→ engine/artifacts/optimizations/{run_id}/
```

测试断言：

```text
1. 未触发数据治理；
2. 未触发模型训练；
3. 输出引用正确 model_id / version；
4. 优化结果可追溯到 dataset_version；
5. 删除或屏蔽 Registry 时得到 MODEL_ARTIFACT_NOT_FOUND。
```

### 15.4 回归测试

```text
固定随机种子
固定请求数据
固定模型元数据
校验 selected_candidates
校验 diagnostics
校验 objective 排序
校验约束报告
```

## 16. Demo3 接入方式

第一阶段不改变 Demo3 Agent 流程，新增薄适配层：

```text
Demo3 tools/formula_optimizer.py
  → engine.adapters.demo3.OptimizationAdapter
```

适配流程：

```text
1. 将 Demo3 target dict 转换为 ObjectiveSpec；
2. 将模型 feature_bounds 转换为 VariableSpec；
3. 将 RAG recipes 转换为 historical_candidates；
4. 调用 engine.optimization.optimize；
5. 将 OptimizationResult 转回 Demo3 当前返回结构；
6. 对比新旧输出；
7. 通过后替换内部实现。
```

## 17. 交付物

```text
engine/optimization/ 源码
engine/tests/optimization/ 测试
OptimizationRequest JSON Schema
OptimizationResult JSON Schema
离线 CLI
策略选择说明
约束修复说明
Demo3 适配层
回归测试报告
```
