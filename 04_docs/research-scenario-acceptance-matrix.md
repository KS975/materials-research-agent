# 展示版 20 场景验收矩阵

更新日期：2026-09-08  
基线分支：`codex/structure-adaptation`  
当前实现：阶段三只读研究场景；阶段四、阶段五仅保留固定 Workflow 分类和边界，不伪装成已执行能力。

## 1. 执行状态定义

| 状态 | 含义 |
|---|---|
| `UNIT_PASS` | 已有单元测试证明路由、Tool 调用或 EvidenceFrame 行为正确 |
| `READY_FOR_REAL_DATA` | 代码链路已开放，等待用当前公司真实 MySQL / 向量数据做人工验收 |
| `PLANNED_STAGE_4` | 属于分析与报告场景，本轮不执行，不输出伪结论 |
| `PLANNED_STAGE_5` | 涉及写入、审批、回流或跨项目资产，本轮不执行 |

外部向量结果依赖 `VECTOR_SEARCH_PROVIDER=external_api`、平台请求凭证和该公司已入库数据。向量源不可用时系统保留证据缺口，不会用推测补齐。

## 2. 场景矩阵

| 序号 | 场景 | Workflow | 状态 | 验收问题示例 | 主要 Tool / 能力 | 写入 |
|---:|---|---|---|---|---|---|
| 1 | 相似配方检索 | `hybrid_search_rank` | `UNIT_PASS / READY_FOR_REAL_DATA` | 查与 EXP-128 相似的配方；按配方找相似历史样品；找相近组分但性能更好的方案 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 2 | 相似样品 / 相似实验检索 | `hybrid_search_rank` | `UNIT_PASS / READY_FOR_REAL_DATA` | 找与 EXP-128 工艺最像的样品；找相似实验系列；找配方和性能都接近的样品 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 3 | 按目标性能反查历史方案 | `hybrid_search_rank` | `READY_FOR_REAL_DATA` | 冲击强度大于 35 的样品；MFR 在 10 到 18 的方案；成本低于 30 且冲击不低于 32 的历史样品 | 结构化 filters、`search_vector_knowledge` | 否 |
| 4 | 配方-工艺-性能联查 | `evidence_profile` | `UNIT_PASS / READY_FOR_REAL_DATA` | 查 EXP-128 完整研发上下文；查某样品配方工艺和性能；查某实验对应样品与测试结果 | `get_sample_context`、`search_vector_knowledge` | 否 |
| 5 | 原料使用效果查询 | `hybrid_search_rank` | `UNIT_PASS / READY_FOR_REAL_DATA` | 查 PC 用在哪些样品；PC 用量多少；使用 PC 的样品性能如何 | 授权配方扫描、`search_vector_knowledge` | 否 |
| 6 | 原料替代历史检索 | `hybrid_search_rank` | `UNIT_PASS / READY_FOR_REAL_DATA` | PC 替代 ABS 的历史；某牌号替换记录；替代后性能变化 | 授权配方扫描、`search_vector_knowledge` | 否 |
| 7 | 失败配方 / 失败实验检索 | `hybrid_search_rank` | `READY_FOR_REAL_DATA` | 查失败配方；以前类似路线为什么失败；失败后调整了什么 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 8 | 关键变量识别 | `historical_analysis` | `PLANNED_STAGE_4` | 哪些变量影响冲击强度；影响 MFR 的关键因素；缩小下一步实验变量 | Evidence Dataset 与分析器 | 否 |
| 9 | 配方 / 工艺窗口发现 | `process_window` | `PLANNED_STAGE_4` | 找稳定工艺窗口；满足多目标的组分区间；哪些参数组合更稳 | Evidence Dataset、窗口分析 | 否 |
| 10 | 多性能冲突分析 | `historical_analysis` | `PLANNED_STAGE_4` | 冲击与 MFR 怎么冲突；强度和成本权衡；提高 A 为什么 B 下降 | Evidence Dataset、相关性 / 冲突分析 | 否 |
| 11 | 批次差异分析 | `historical_analysis` | `PLANNED_STAGE_4` | 正常批次和异常批次差什么；批次间性能波动来源；原料批次是否影响结果 | Evidence Dataset、批次对比 | 否 |
| 12 | 异常与失效案例检索 | `hybrid_search_rank` | `READY_FOR_REAL_DATA` | 查开裂案例；析出或变色历史；粘接失效类似案例 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 13 | 竞品对标 | `hybrid_search_rank` | `READY_FOR_REAL_DATA` | 与竞品性能差距；历史上哪些路线最接近竞品；竞品关键性能对比 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 14 | 新项目冷启动 | `research_cold_start` | `READY_FOR_REAL_DATA` | 新项目目标性能给首轮方案；结合历史和失败记录冷启动；有原料限制时从哪开始 | 多条件筛选、相似历史、失败案例、向量证据 | 否 |
| 15 | 测试结果自动关联样品 | `result_feature_ingestion` | `PLANNED_STAGE_5` | LIMS 结果回样品；检测结果错配发现；自动关联实验与配方 | 结果匹配、特征登记、人工审核 | 是，需审核 |
| 16 | 图谱 / 曲线结果复用 | `result_feature_ingestion` | `PLANNED_STAGE_5` | DSC 特征参与分析；粒径曲线复用；谱图特征关联样品 | 特征抽取、待审核登记 | 是，需审核 |
| 17 | 项目知识快速问答 | `evidence_profile` | `READY_FOR_REAL_DATA` | 去年做过哪些方案；某原料为什么停用；项目结论和风险是什么 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 18 | 自动生成阶段总结 | `stage_report` | `PLANNED_STAGE_4` | 生成阶段报告；汇总项目进展；形成实验结论和证据链 | 聚合、引用校验、报告生成 | 否 |
| 19 | 模型版本与实验回流 | `closed_loop_asset` | `PLANNED_STAGE_5` | 新实验回流；比较 Challenger；模型是否晋级 | Dataset 版本、模型治理、审批 | 是，需审批 |
| 20 | 跨项目复用 | `closed_loop_asset` | `PLANNED_STAGE_5` | 其他项目能否复用模型；复用历史经验；资产授权范围是什么 | Dataset / Model / 报告资产索引 | 是，需审批 |

## 3. 阶段三补充验收口径

每个 `READY_FOR_REAL_DATA` 场景至少使用 3 个真实问题验收，并记录：

```text
场景编号
入口问题
命中的 Intent / Workflow
调用的 Tool
MySQL 返回范围
向量返回条数与 score 范围
EvidenceFrame 来源计数
是否出现冲突 / 警告
最终报告是否引用证据
```

原料替代场景必须输出 `CO_OCCURRENCE_ONLY` 边界：同一样品含两种原料只能证明共存，不能直接证明替代关系。所有阶段三场景均为只读，不生成 Dataset、Model 或待审核登记。

## 4. 当前测试记录

```text
tests/unit/test_research_scenarios.py
tests/unit/test_hybrid_research_qa.py
tests/unit/test_hybrid_research_routing.py
```

2026-09-08 定向结果：10 passed。覆盖 20 场景到 8 Workflow 的完整性、自然语言相似表述、原料使用效果、原料替代边界、目标性能筛选优先级、跨源相似路由不被单源意图降级，以及原有 EvidenceFrame 行为。

## 5. 2026-09-08 真实只读数据冒烟

环境：本地 FastAPI + 真实业务 MySQL + DeepSeek 语义路由。业务库健康检查为 `connected=true`、`session_read_only=true`。

| 用例 | 结果 | 记录 |
|---|---|---|
| 查 EXP-128 的完整研发上下文，并结合历史资料综合判断 | 通过 | 命中场景 4 / `evidence_profile`；MySQL `ok`，生成 16 条结构化证据 |
| 查找与 EXP-128 相似的配方，并结合历史案例 | 通过 | 命中场景 1 / `hybrid_search_rank`；自然语言“相似的配方”已接入结构化相似策略 |
| 查“水”的原料使用效果，并结合历史案例 | 通过 | 命中场景 5；授权扫描命中 383 条，默认截断展示 50 条并保留 warning |
| 查找密度差大于 100 且持液量小于 0.1 的历史样品，并结合历史资料 | 通过 | 命中场景 3 / `structured_multi_condition_filter`；返回 50 条候选、200 条 MySQL 证据 |

向量侧当前仍为 legacy Qdrant 配置，缺少 `EMBEDDING_BASE_URL`，因此所有真实冒烟均显式输出 `vector_unavailable` 和证据缺口警告，未用推测补齐。这不属于 MySQL 链路失败；启用外部向量 API 需要部署配置 `VECTOR_SEARCH_PROVIDER=external_api`、正式 endpoint 和平台请求凭证。

## 6. 2026-09-09 外部向量真实链路补充冒烟

本地运行时已切换为 `VECTOR_SEARCH_PROVIDER=external_api`。测试数据范围通过 `EXTERNAL_VECTOR_QUERY_COMPANY_ID=test-company-id` 显式配置：请求头 `Company-Id` 保持当前登录公司，向量查询参数 `companyId` 指向授权测试数据公司。生产环境该覆盖项必须留空。

| 用例 | 结果 | 记录 |
|---|---|---|
| 直接调用受管 `search_vector_knowledge` | 通过 | `provider=external_vector_api`，命中 5 条，score 约 0.543-0.664 |
| Hybrid Research QA 证据链 | 通过 | MySQL `ok`，外部向量 `ok`；EvidenceFrame 来源计数 `mysql=200`、`vector_api=5`、`dialog=1` |
| 平台登录公司校验 | 通过 | Token 所属公司与 Header 公司一致，项目模式为 `company_all_projects` |
| Hybrid QA LLM 稳定化 | 通过 | 真实链路 `SUCCEEDED`；MySQL `ok`、向量 `ok=5`、LLM 首次成功；压缩上下文 8,005 字符 / 22 条记录，最终报告 1,236 字符 |

此前真实 LLM 综述曾出现一次 `ReadTimeout`；使用固定摘要器复跑证明 MySQL、外部向量与 EvidenceFrame 链路本身正常。2026-09-09 切片 1 已增加有界证据上下文、一次降容重试和确定性降级报告，真实端到端复测不再超时。生产配置禁止非空 `EXTERNAL_VECTOR_QUERY_COMPANY_ID`。
