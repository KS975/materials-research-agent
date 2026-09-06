# 分阶段交付开发方案 V0.2

状态：待确认方案。确认后先执行第一阶段，不并行启动试点期和企业期改造。

## 1. 结论与开发主线

本方案将此前三部分内容合并为一条开发主线：

1. 差距根因：后端架构和引擎能力已成型，但真实企业库配置、前端引擎适配、任务态展示、Registry 治理和部署运维未达交付要求。
2. 架构迁移映射：以《方案 V0.2》为目标架构，从现有过渡架构逐层演进，不推倒重写。
3. 分阶段交付：按“内部可演示 → 试点可使用 → 企业交付”推进；每个阶段保留 V0.2 的分层、权限、审计和数据血缘边界。

后续开发以本文档为主计划。引擎专项细节引用 `engine/docs/development-plan.md`、`data-preprocessing-design.md`、`modeling-design.md`、`optimization-design.md`，不在本文重复展开。

## 2. 目标架构

```text
React / SSE 前端
  对话、任务状态、阶段图表、Engine Workflow 结果卡、
  任务取消 / 恢复 / 审批、最终完整报告
        ↓
FastAPI Gateway
  请求校验 / 身份权限 / SSE / API 版本
        ↓
Agent Runtime
  Intent Router / Scenario Workflow Composer /
  Workflow Orchestrator / Skill Registry /
  Tool Registry / Evidence and Result Builder
        ↓
领域与引擎能力
  知识问答 / 数据治理 / AutoML / 预测 /
  配方优化 / 下一轮实验 / 实验回流 / 模型治理
        ↓
数据与运行时层
  业务 MySQL 只读 / Qdrant / Dataset Registry /
  Model Registry / Experiment Registry / Runtime DB / 对象存储
        ↓
横切能力
  权限 / 审计 / 血缘 / 日志监控 / 配置密钥 /
  人机审批 / 安全护栏
```

## 3. 架构演进原则

1. 架构面向企业交付，能力分阶段启用；接口和数据边界先按 V0.2 设计，运行时可用模块化单体和 JSON 检查点过渡。
2. 业务 MySQL 永远只读；LLM 不得自由生成 SQL 访问生产库，必须走 Permission Adapter → Tool / Repository → 参数化只读查询。
3. 平台写入只进入 Runtime DB / 对象存储，不写业务 MySQL，不覆盖旧 Dataset。
4. 预测和优化不得隐式训练；无模型时返回 `MODEL_REQUIRED`。
5. 执行中只输出结构化状态和图表数据，不输出阶段性文字报告；终态一次性输出完整报告。
6. 所有结果可追溯 Dataset Version、Model Version、Tool Call、Evidence、Approval 和 Audit。
7. 实验回流先保留接口；正式闭环放在试点阶段。

## 4. 当前过渡态与 V0.2 映射

| 当前状态 | V0.2 目标 | 迁移策略 |
|---|---|---|
| Scenario Workflow 支持多 Skill DAG | 固定场景 DAG | 保留并固化模板 |
| Engine Task 有 JSON 检查点、Worker、取消、恢复、审批 | 异步任务体系 | 第一阶段接入前端和 Chat SSE |
| Tool Registry 有 Schema、权限、审计 | 工具治理 | 试点期审计迁入 Runtime DB |
| Engine Tools 按 JSON Tool 暴露 | Skill 调用领域服务 | 保留工具级封装，不做引擎级大包装 |
| 引擎快照走 `list_samples_for_analysis` | 企业 MySQL 只读数据源 | 第一阶段接入真实配置和健康检查 |
| 模型使用项目目录和 registry 文件 | Model Registry | 试点期补状态、版本、血缘和审批 |
| 前端仅识别旧版建模 / 优化卡片 | Engine Workflow 通用结果卡 | 第一阶段完成结果、图表、任务态适配 |
| legacy / shadow / engine 优化路由并存 | engine 正式路由 | 内部演示 shadow 验收，试点期切换 |
| 实验回流仅接口 | Experiment Registry + 回流闭环 | 试点期正式开发 |

## 5. 第一阶段：内部可演示版本

目标：打通“真实企业只读数据 → 引擎工具 → 异步任务 → 前端结构化展示 → 最终报告”。

### 5.1 真实只读数据库接入

工作项：

1. 从本地基线 `.env` 迁移 `BUSINESS_DB_*` 到 Demo3 本地 `.env`；`.env` 保持 gitignore，不提交凭据。
2. 保留 `BusinessMySQLClient` 的 SQL 白名单、单语句限制和 `SET SESSION TRANSACTION READ ONLY`。
3. 新增 `GET /api/v1/health/business-db`，只返回 `configured`、`connected`、`database`、`session_read_only`、`error_class`，不返回主机、账号、密码、SQL 或连接串。
4. 为引擎快照增加内部演示验收：必须来自 `business_mysql` 授权项目，可断言样品数、字段目录、扫描完整性和只读状态。
5. 未配置真实库时集成测试跳过，不得用本地数据伪装真实库；离线单测继续保留。
6. 测试期前端身份只允许来自本地 `frontend/.env.local` 的 `VITE_DEV_USER_ID / VITE_DEV_COMPANY_ID / VITE_DEV_PROJECT_IDS`；源码不得硬编码公司、用户或项目范围。

验收：

```text
业务库连通
session_read_only = true
授权项目可生成 Engine Source Snapshot
快照血缘包含 business_mysql / company / project / hash
无数据库凭据泄露
前端无硬编码授权范围
EXP-128 授权公司上下文可查询
```

### 5.2 前端 Engine Workflow 适配

新增统一 Engine Workflow 结果卡，输入契约：

```text
schema_version / workflow / status / scope /
steps / result / answer / evidence / warnings
```

| Workflow | 展示 |
|---|---|
| engine_prepare_dataset | Gate、Dataset 版本、清洗规则、技术摘要、警告 |
| automl_training | 候选模型、CV 指标、选定模型、训练警告 |
| predict_performance | 模型版本、预测值、不确定性、适用域 |
| optimize_formula | 目标、约束、候选方案、Pareto、适用域、排序依据 |
| recommend_next_experiments | 下一批实验、acquisition、不确定性、实验条件 |

图表适配：

1. 前端消费 `visualization_datasets`，不解析本地文件路径。
2. 首期实现 `table`、`bar`、`horizontal_bar`、`scatter` 四类通用渲染。
3. 图表展示标题、来源 artifact、字段名和记录数。
4. 后端负责暴露数据，前端负责作图。

验收：

```text
执行中：状态 + 阶段 + 图表数据
终态：完整报告 + Engine Workflow 结果卡 + 图表 + 警告
失败：结构化错误 + 可恢复操作
不出现阶段性文字报告
```

### 5.3 异步任务前端接入

工作项：

1. 前端接入 `POST /api/v1/engine-tasks`、`GET /api/v1/engine-tasks/{task_id}`、cancel / approve / resume。
2. SSE progress 携带 `engine_task_id` 时，前端轮询任务状态。
3. 展示 `QUEUED / RUNNING / AWAITING_APPROVAL / CANCEL_REQUESTED / SUCCEEDED / FAILED / CANCELLED / INTERRUPTED`。
4. 审批、取消、恢复只对允许状态开放。
5. Worker 中断后刷新可见 `INTERRUPTED`，用户可恢复。

Chat 入口策略：

```text
ENGINE_TASK_MODE=sync
  保持现有兼容链路

ENGINE_TASK_MODE=async
  engine intent 创建 Engine Task
  SSE 返回 task_id 和阶段事件
  前端轮询任务终态
  终态一次性展示完整报告
```

第一阶段默认 `sync`，内部演示环境切换 `async`；不得影响普通问答、RAG 和数据库查询。

### 5.4 优化路由 shadow 验收

1. 默认保持 `legacy`。
2. shadow 对比可行候选数、硬约束满足率、目标命中率、Pareto 多样性、适用域分布、耗时和异常率。
3. 制定 engine 切换阈值和回退策略。
4. shadow 未验收前，不把 engine 设为默认优化路由。

### 5.5 第一阶段测试与演示

| 类型 | 内容 |
|---|---|
| 单元测试 | Engine 结果转换、任务状态转换、图表数据归一化 |
| API 测试 | health / engine-tasks / SSE 事件契约 |
| 集成测试 | 真实只读库快照，未配置则跳过 |
| 引擎测试 | 预处理、建模、预测、优化既有测试 |
| 前端构建 | Vite build 通过 |
| 端到端演示 | 对话发起 → 任务状态 → 图表 → 最终报告 |

演示场景：

1. 基于真实只读数据准备建模 Dataset。
2. 训练并注册候选模型。
3. 使用已注册模型预测样品或配方。
4. 基于已注册模型推荐配方。
5. 无模型请求预测 / 优化时返回 `MODEL_REQUIRED`。
6. 长任务取消、恢复和审批状态展示。

第一阶段完成定义：真实 MySQL 只读链路可用；前端可展示核心 Engine Workflow 结果；异步任务可观测、可取消、可恢复；图表可渲染；最终报告只在终态出现；核心测试和构建通过。

## 6. 第二阶段：试点可使用版本

### 6.1 Registry 与血缘

1. Dataset Registry：dataset_id、version、source、schema、cleaning_rules、lineage。
2. Model Registry：model_id、version、metrics、dataset_version、applicability_domain、status。
3. Experiment Registry：recommendation_id、prediction_id、measured_values、merge_status。
4. Runtime DB：session、workflow、task、tool_call、approval、audit。

### 6.2 模型治理

```text
EXPERIMENTAL
→ CANDIDATE
→ VALIDATED
→ APPROVED
→ ACTIVE
→ DEPRECATED
```

默认预测 / 优化使用 `ACTIVE` 模型；用户显式指定时允许 `APPROVED` 或验收白名单模型。晋级、废弃、回滚需要审批，并记录模型与 Dataset 血缘。

### 6.3 实验回流闭环

```text
上传实验数据
→ Schema 校验
→ 识别新增实验 / 推荐验证 / 历史新实验
→ 匹配 recommendation_id / prediction_id / recipe_hash
→ 生成 Dataset 新版本
→ 按用户意图选择仅补充 / 重训 / 优化
→ Challenger 评估
→ 审批晋级
→ 下一轮推荐
```

回流不默认重训，不覆盖旧 Dataset，不写业务 MySQL。

### 6.4 优化路由切换与前端试点能力

1. shadow 达标后切 `engine`，保留 `legacy` 快速回退。
2. 建立优化基准集、回归测试、切换审批和版本记录。
3. 增加 Dataset / Model / Experiment 管理页、任务历史、模型治理审批、审计查询。

### 6.5 真实登录权限接入

1. 前端部署在同源单位平台内，禁止携带开发 `X-User-Id / X-Company-Id / X-Project-Ids`。
2. 后端切换 `PERMISSION_MODE=platform`，只信任已鉴权网关转发的 Authorization、company-id、organization-id、organization-level。
3. Permission Adapter 根据登录态解析稳定用户 ID、公司 ID 和项目范围；项目映射规则必须来自平台权限服务或明确配置，不由 LLM 或前端推断。
4. 所有 MySQL Repository、Engine Scope、Task、附件、知识索引和审计继续强制公司/项目过滤。
5. 使用真实登录用户完成跨公司隔离、越权访问和 `EXP-128` 所属数据集回归验收。

## 7. 第三阶段：企业交付版本

部署架构：

```text
React Static Build
→ Nginx / Gateway
→ FastAPI Application
→ Workflow Worker
→ Runtime DB
→ Qdrant
→ Object Storage
→ Business MySQL READ ONLY
```

企业能力：

1. 多公司、多项目、多角色权限。
2. Tool 级和数据行级权限，且权限来源只能是可信登录态或权限服务。
3. 审计落库和查询。
4. 高风险动作审批。
5. 配置中心与密钥管理。
6. API、Workflow、Tool、模型质量监控。
7. 备份、恢复、告警和发布回滚。
8. 企业只读账号网络策略和最小权限。

企业验收覆盖功能、数据、模型、安全、运维、协作和质量；未通过不得正式发布。

## 8. 阶段门槛与风险

| 阶段切换 | 门槛 |
|---|---|
| 内部演示 → 试点 | 真实只读库、前端引擎适配、异步任务、图表和最终报告链路验收 |
| 试点 → 企业 | Registry、模型治理、实验回流、engine 默认路由、真实登录权限和审计持久化验收 |
| 企业发布 | 部署、安全、监控、备份、多角色验收完成 |

| 风险 | 处理 |
|---|---|
| 数据库凭据泄露 | `.env` 不提交；健康接口脱敏；测试输出禁止连接串 |
| 真实库不可用 | 集成测试跳过；前端显示 degraded，不伪装真实数据 |
| 长任务 HTTP 超时 | async task、task_id、前端轮询 / SSE |
| 图表类型扩散 | 首期仅通用 table / bar / horizontal_bar / scatter |
| 优化结果不可信 | shadow 对比、适用域过滤、质量熔断、legacy 回退 |
| 架构过度重构 | 每阶段只替换一层实现，接口契约保持稳定 |

## 9. 自核查结论

| 检查项 | 结论 |
|---|---|
| 差距根因、架构迁移、分阶段方案是否合并 | 已合并，主线唯一 |
| 前端引擎适配是否在第一阶段 | 已在第一阶段 |
| 真实只读数据库是否在内部演示阶段 | 已在第一阶段 |
| 真实登录权限是否纳入阶段路径 | 已纳入试点期，企业期完成多角色治理 |
| 是否与 V0.2 架构冲突 | 无冲突，采用过渡实现向 Registry / Worker / Runtime DB 演进 |
| 是否重复引擎专项方案 | 无重复，本文只管平台集成 |
| 是否满足执行过程报告约束 | 执行中仅结构化状态和图表，终态完整报告 |
| 是否扩大当前范围 | 未扩大企业期功能，实验回流留在试点期 |
