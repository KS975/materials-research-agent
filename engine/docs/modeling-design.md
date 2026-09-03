# 机器学习建模详细设计

## 1. 目标与边界

本模块负责基于通过建模门禁的 `DatasetArtifact` 训练、评价、注册和加载机器学习模型。

模块边界：

- 输入：`DatasetArtifact`、`ModelingGateResult`、`TrainingConfig`；
- 输出：`ModelArtifact`、`EvaluationReport`、`PredictionResult`；
- 不负责：数据清洗、约束优化和实验推荐；
- 不假设目标名、变量名或材料体系；
- 模型指标必须来自真实训练和评估，不允许由 LLM 生成。

## 2. 输入输出契约

### 2.1 TrainingConfig

```text
TrainingConfig
  target_names
  feature_names
  reserved_test_ratio
  cv_mode
    auto
    loocv
    kfold
    repeated_kfold
  random_seed
  algorithms
  disabled_algorithms
  optuna_enabled
  optuna_trials
  transform_config
  cache_enabled
  cache_dir
  metric_primary
```

### 2.2 TrainingInput

```text
TrainingInput
  dataset_artifact
  gate_result
  training_config
```

### 2.3 ModelArtifact

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
  training_config
  metrics
  applicability_domain
  status
  file_path
  created_at
```

### 2.4 EvaluationReport

```text
EvaluationReport
  model_artifact_id
  train_metrics
  test_metrics
  cv_metrics
  residual_summary
  model_comparison
  warnings
```

指标：

```text
R²
MAE
RMSE
CV Mean
CV Std
```

## 3. 建模前置条件

```text
1. DatasetArtifact 存在且可加载；
2. ModelingGateResult != FAIL；
3. target_names 均存在于 target_schema；
4. feature_names 均存在于 feature_schema；
5. 每个目标有效样本数满足配置；
6. 特征和目标无交集；
7. random_seed 已显式配置；
8. 目标变换配置合法。
```

若 gate 为 `CONDITIONAL_PASS`：

```text
允许训练 EXPERIMENTAL 模型；
禁止注册为 APPROVED / ACTIVE；
输出 warning；
```

## 4. 数据拆分与交叉验证策略

### 4.1 有效样本计算

每个目标独立计算：

```text
n_target = target 非缺失样本数
p = feature_count
ratio = n_target / p
```

多目标时不使用全局 `n/p` 替代每个目标的 `n/p`。

### 4.2 自动 CV 策略

```text
if ratio < 1:
    cv = LOOCV
elif ratio < 3:
    cv = 5-Fold KFold
else:
    cv = 5-Fold KFold
```

可选：

```text
n_target > 500:
    可自动切换 5-Fold，避免 LOOCV 过高成本
```

切换规则由配置决定，必须输出 `strategy_reason`。

### 4.3 数据拆分

```text
reserved_test_ratio 默认 0.2
random_seed 固定
shuffle = true
```

要求：

```text
1. 每个目标单独拆分；
+2. 目标缺失样本不得进入该目标训练；
3. 标准化器必须在训练折内拟合；
4. 缓存命中时仍需保留拆分元数据；
5. 测试集不得参与模型选择。
```

## 5. 策略分层

### Tier 1：高维小样本

触发：

```text
ratio < 1
```

算法：

```text
Linear Regression
Ridge
Lasso
ElasticNet
BayesianRidge
PLS
```

目标：

```text
控制方差；
避免复杂模型过拟合；
优先可解释和稳定模型。
```

### Tier 2：中等样本特征比

触发：

```text
1 <= ratio < 3
```

算法：

```text
Tier 1
+ Gaussian Process
+ Random Forest
+ XGBoost
```

### Tier 3：相对充足样本

触发：

```text
ratio >= 3
```

算法：

```text
Tier 2
+ LightGBM
+ Gradient Boosting
+ SVR
```

## 6. 算法实现细节

### 6.1 Linear Regression

```text
LinearRegression
```

定位：

```text
可解释基线模型；
用于衡量非线性模型和正则化模型是否带来真实增益。
```

要求：

```text
1. 数值特征先标准化；
2. 记录系数和截距；
3. 输出 train / test prediction；
4. 输出完整指标；
5. 不做超参数搜索。
```

### 6.2 Ridge

```text
RidgeCV
alphas = logspace(-3, 3, 50)
```

输出：

```text
alpha
train / test prediction
metrics
```

### 6.3 Lasso

```text
LassoCV
alphas = logspace(-3, 2, 100)
max_iter = 10000
```

要求：

```text
数值特征先标准化；
记录收敛状态；
未收敛输出 warning。
```

### 6.4 ElasticNet

```text
ElasticNetCV
l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
alphas = logspace(-3, 2, 30)
max_iter = 10000
```

输出：

```text
alpha
l1_ratio
```

### 6.5 BayesianRidge

```text
BayesianRidge
```

适合小样本线性基线，不启用 Optuna。

### 6.6 PLS

```text
PLSRegression
n_components = 1 .. min(n_train - 1, p, 20)
```

选择规则：

```text
在训练折内按 CV RMSE 选择成分数；
禁止使用测试集选择成分。
```

### 6.7 Gaussian Process

```text
GaussianProcessRegressor
kernel = RBF() + WhiteKernel()
normalize_y = true
random_state = configured seed
```

输出：

```text
预测均值
预测标准差
kernel 参数
```

限制：

```text
大样本或高维场景需显式开启；
默认只在策略允许时执行。
```

### 6.8 Random Forest

使用 Optuna：

```text
n_estimators: int [50, 200]
max_depth: int [2, 12]
min_samples_leaf: int [3, 15]
```

目标：

```text
minimize CV RMSE
```

### 6.9 XGBoost

使用 Optuna：

```text
n_estimators: int [50, 200]
max_depth: int [2, 6]
learning_rate: float [0.01, 0.2]
subsample: float [0.6, 1.0]
```

要求：

```text
random_state 固定；
verbosity = 0；
输出 best_params。
```

### 6.10 LightGBM

使用 Optuna：

```text
n_estimators: int [50, 200]
max_depth: int [2, 6]
learning_rate: float [0.01, 0.2]
num_leaves: int [15, 63]
```

要求：

```text
random_state 固定；
verbose = -1；
仅 Tier 3 默认启用。
```

### 6.11 Gradient Boosting

使用 Optuna：

```text
n_estimators: int [50, 200]
max_depth: int [2, 5]
learning_rate: float [0.01, 0.2]
subsample: float [0.7, 1.0]
```

### 6.12 SVR

使用标准化 Pipeline：

```text
StandardScaler + SVR
```

Optuna 搜索：

```text
C: loguniform [0.01, 100]
epsilon: uniform [0.001, 0.5]
gamma: ["scale", "auto"] 或 loguniform
kernel: configured
```

仅 Tier 3 默认启用。

## 7. Optuna 策略

### 7.1 试验数

```text
p < 20:
    trials = 30
20 <= p <= 50:
    trials = 50
p > 50:
    trials = 50
```

用户显式配置优先。

### 7.2 采样器

```text
TPESampler(seed=random_seed)
```

### 7.3 目标函数

```text
minimize mean CV RMSE
```

### 7.4 禁止事项

```text
1. 禁止使用 reserved test set 参与 Optuna；
2. 禁止不同目标混用同一标签向量；
3. 禁止折外预拟合标准化器；
4. 禁止无 seed 随机搜索；
5. 禁止只依据 R² 选择模型。
```

## 8. 目标变换

配置：

```text
none
log1p
sqrt
box_cox
custom
```

规则：

```text
1. 变换只作用于训练目标；
2. 评估前必须逆变换回原始空间；
3. 负值或零值必须先通过合法性检查；
4. 变换参数只由训练折拟合；
5. ModelArtifact 保存变换配置。
```

## 9. 指标计算

全部在原始目标空间计算。

```text
R² = 1 - SS_res / SS_tot
MAE = mean(abs(y_true - y_pred))
RMSE = sqrt(mean((y_true - y_pred)^2))
CV Mean = mean(fold_metric)
CV Std = std(fold_metric)
```

输出层级：

```text
train
test
cv
per_fold
```

主选择指标默认：

```text
CV RMSE Mean
```

并列时依次比较：

```text
1. CV RMSE Mean；
2. CV RMSE Std；
3. test RMSE；
4. test MAE；
5. 模型复杂度惩罚；
6. 可解释性优先级。
```

## 10. 模型缓存

缓存键：

```text
dataset_artifact_id
dataset_data_hash
target_name
feature_schema_hash
training_config_hash
engine_version
source_code_hash
```

命中条件：

```text
所有键完全一致
```

缓存输出必须保留：

```text
model artifact
evaluation report
cache path
strategy reason
```

模型缓存与持久化 ModelArtifact 是两类概念：

```text
cache：
  可按缓存键失效和重建；

ModelArtifact：
  正式版本，不可删除、不可覆盖；
  只能通过新版本或 DEPRECATED 状态演进。
```

### 10.1 模型持久化目录

模型统一保存到：

```text
engine/artifacts/models/{model_id}/{version}/
```

每个版本包含：

```text
model.joblib
metadata.json
metrics.json
feature_schema.json
```

注册表：

```text
engine/artifacts/models/model-registry.json
```

Registry 记录：

```text
model_id
version
target_name
model_type
dataset_artifact_id
dataset_data_hash
metrics
artifact_path
status
created_at
```

写入规则：

```text
1. 每个目标独立注册；
2. 同一 model_id + version 已存在时不覆盖；
3. 重复训练生成新版本；
4. 训练输出先写入 ModelArtifact，再进入 Registry；
5. 供优化测试使用的模型不得被测试清理函数删除；
6. 降级只修改 Registry 中的新状态版本，不删除旧模型文件。
```

### 10.2 真实数据建模测试

建模集成测试必须基于治理好的基准数据生成 DatasetArtifact，并将训练出的模型保留在 `engine/artifacts/models/`。

测试链路：

```text
治理好的 Parquet
→ DatasetArtifact
→ train_models
→ ModelArtifact
→ model-registry.json
→ 供 prediction / optimization 测试复用
```

要求：

```text
1. 固定 random_seed；
2. 保存完整指标；
3. 不删除模型；
4. 后续优化测试通过 Registry 加载模型；
5. 禁止在优化测试中重新训练模型。
```

## 11. 适用域设计

### 11.1 训练阶段

```text
1. 使用训练折或最终训练集特征；
2. 应用与模型一致的标准化；
3. 计算 kNN 距离；
4. k = min(5, n_train - 1)；
5. 记录距离分布 q75 和 q95；
6. 保存标准化器和阈值。
```

### 11.2 推理阶段

```text
distance = 候选到训练集第 k 近邻平均距离

distance <= q75:
    IN_DOMAIN

q75 < distance <= q95:
    EDGE

distance > q95:
    OUT_OF_DOMAIN
```

类别变量：

```text
未见类别或类别组合 → OUT_OF_DOMAIN
```

## 12. 预测 API

```python
predict(model_artifact: ModelArtifact, inputs: DataFrame) -> list[PredictionResult]
```

输入校验：

```text
必填特征
数值范围
类别白名单
缺失值
单位一致性
固定值补齐
```

输出：

```text
predicted_value
prediction_uncertainty
applicability_domain
model_id
model_version
dataset_version
warnings
```

## 13. 模块设计

```text
engine/modeling/
  trainer.py
  strategy.py
  algorithms/
    linear.py
    kernel.py
    tree.py
    ensemble.py
  tuning.py
  transform.py
  artifact.py
  cache.py
  registry.py

engine/evaluation/
  metrics.py
  residuals.py
  calibration.py
  applicability_domain.py

engine/prediction/
  validator.py
  predictor.py
```

核心 API：

```python
select_strategy(dataset, config) -> ModelingStrategy

train_models(
    dataset_artifact: DatasetArtifact,
    gate_result: ModelingGateResult,
    config: TrainingConfig,
) -> list[ModelArtifact]

evaluate_model(
    model_artifact: ModelArtifact,
    dataset_artifact: DatasetArtifact,
) -> EvaluationReport

predict(
    model_artifact: ModelArtifact,
    inputs: DataFrame,
) -> list[PredictionResult]

register_model(
    model_artifact: ModelArtifact,
    registry_path: Path,
) -> None
```

## 14. 测试计划

### 14.1 策略测试

```text
ratio < 1 → Tier 1 + LOOCV
1 <= ratio < 3 → Tier 2 + 5-Fold
ratio >= 3 → Tier 3 + 5-Fold
用户禁用算法后候选集正确
用户指定算法后策略 override 正确
```

### 14.2 算法测试

每个算法使用小型确定性数据验证：

```text
可训练
可预测
指标有限
参数可序列化
模型可加载
seed 可复现
```

### 14.3 指标测试

构造已知预测误差：

```text
完美预测
常数预测
固定偏差
异常偏差
```

验证：

```text
R²
MAE
RMSE
CV Mean
CV Std
```

### 14.4 泄漏测试

```text
1. 标准化器不得在全量数据拟合；
2. Optuna 不得使用 reserved test；
3. 目标派生特征必须被拒绝；
4. 后实验字段必须被拒绝；
5. 缓存不得跨数据版本复用。
```

### 14.5 AD 测试

```text
训练中心样本 → IN_DOMAIN
边界样本 → EDGE
远离训练分布样本 → OUT_OF_DOMAIN
未见类别 → OUT_OF_DOMAIN
```

## 15. 验收标准

```text
1. 三种 Tier 自动选择正确；
2. 每个目标独立训练；
3. 所有候选模型输出完整指标；
4. 主模型选择规则可解释；
5. ModelArtifact 可保存和加载；
6. 同输入同 seed 可复现；
7. 预测包含模型版本和 AD；
8. FAIL 数据不能正式建模；
9. CONDITIONAL_PASS 模型不能晋级正式模型；
10. 无业务字段硬编码。
```
