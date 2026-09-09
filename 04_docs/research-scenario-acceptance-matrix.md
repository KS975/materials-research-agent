# 展示版 20 场景验收矩阵

更新日期：2026-09-09
基线分支：`codex/structure-adaptation`  
当前实现：阶段三只读研究场景；阶段四、阶段五仅保留固定 Workflow 分类和边界，不伪装成已执行能力。

## 1. 执行状态定义

| 状态 | 含义 |
|---|---|
| `UNIT_PASS` | 已有单元测试证明路由、Tool 调用或 EvidenceFrame 行为正确 |
| `READY_FOR_REAL_DATA` | 代码链路已开放，等待用当前公司真实 MySQL / 向量数据做人工验收 |
| `REAL_DATA_PASS` | 已完成本切片 3 个真实问题验收；MySQL、外部向量、EvidenceFrame 与 LLM 综述均通过 |
| `REAL_DATA_INFRA_GAP` | 路由、槽位、MySQL、外部向量与 EvidenceFrame 已通过，但部分问题被外部 LLM 服务权限阻断 |
| `PLANNED_STAGE_4` | 属于分析与报告场景，本轮不执行，不输出伪结论 |
| `PLANNED_STAGE_5` | 涉及写入、审批、回流或跨项目资产，本轮不执行 |

外部向量结果依赖 `VECTOR_SEARCH_PROVIDER=external_api`、平台请求凭证和该公司已入库数据。向量源不可用时系统保留证据缺口，不会用推测补齐。

## 2. 场景矩阵

| 序号 | 场景 | Workflow | 状态 | 验收问题示例 | 主要 Tool / 能力 | 写入 |
|---:|---|---|---|---|---|---|
| 1 | 相似配方检索 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 查与 EXP-128 相似的配方；按配方找相似历史样品；找相近组分但性能更好的方案 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 2 | 相似样品 / 相似实验检索 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 找与 EXP-128 工艺最像的样品；找相似实验系列；找配方和性能都接近的样品 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 3 | 按目标性能反查历史方案 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 冲击强度大于 35 的样品；MFR 在 10 到 18 的方案；成本低于 30 且冲击不低于 32 的历史样品 | 结构化 filters、`search_vector_knowledge` | 否 |
| 4 | 配方-工艺-性能联查 | `evidence_profile` | `REAL_DATA_PASS 3/3` | 查 EXP-128 完整研发上下文；查某样品配方工艺和性能；查某实验对应样品与测试结果 | `get_sample_context`、`search_vector_knowledge` | 否 |
| 5 | 原料使用效果查询 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 查 PC 用在哪些样品；PC 用量多少；使用 PC 的样品性能如何 | 授权配方扫描、`search_vector_knowledge` | 否 |
| 6 | 原料替代历史检索 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | PC 替代 ABS 的历史；某牌号替换记录；替代后性能变化 | 授权配方扫描、`search_vector_knowledge` | 否 |
| 7 | 失败配方 / 失败实验检索 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 查失败配方；以前类似路线为什么失败；失败后调整了什么 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 8 | 关键变量识别 | `historical_analysis` | `PLANNED_STAGE_4` | 哪些变量影响冲击强度；影响 MFR 的关键因素；缩小下一步实验变量 | Evidence Dataset 与分析器 | 否 |
| 9 | 配方 / 工艺窗口发现 | `process_window` | `PLANNED_STAGE_4` | 找稳定工艺窗口；满足多目标的组分区间；哪些参数组合更稳 | Evidence Dataset、窗口分析 | 否 |
| 10 | 多性能冲突分析 | `historical_analysis` | `PLANNED_STAGE_4` | 冲击与 MFR 怎么冲突；强度和成本权衡；提高 A 为什么 B 下降 | Evidence Dataset、相关性 / 冲突分析 | 否 |
| 11 | 批次差异分析 | `historical_analysis` | `PLANNED_STAGE_4` | 正常批次和异常批次差什么；批次间性能波动来源；原料批次是否影响结果 | Evidence Dataset、批次对比 | 否 |
| 12 | 异常与失效案例检索 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 查开裂案例；析出或变色历史；粘接失效类似案例 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 13 | 竞品对标 | `hybrid_search_rank` | `REAL_DATA_PASS 3/3` | 与竞品性能差距；历史上哪些路线最接近竞品；竞品关键性能对比 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
| 14 | 新项目冷启动 | `research_cold_start` | `REAL_DATA_PASS 3/3` | 新项目目标性能给首轮方案；结合历史和失败记录冷启动；有原料限制时从哪开始 | 多条件筛选、相似历史、失败案例、向量证据 | 否 |
| 15 | 测试结果自动关联样品 | `result_feature_ingestion` | `PLANNED_STAGE_5` | LIMS 结果回样品；检测结果错配发现；自动关联实验与配方 | 结果匹配、特征登记、人工审核 | 是，需审核 |
| 16 | 图谱 / 曲线结果复用 | `result_feature_ingestion` | `PLANNED_STAGE_5` | DSC 特征参与分析；粒径曲线复用；谱图特征关联样品 | 特征抽取、待审核登记 | 是，需审核 |
| 17 | 项目知识快速问答 | `evidence_profile` | `REAL_DATA_PASS 3/3` | 去年做过哪些方案；某原料为什么停用；项目结论和风险是什么 | `list_samples_for_analysis`、`search_vector_knowledge` | 否 |
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
tests/unit/test_research_slots.py
tests/unit/test_evidence_relation.py
tests/unit/test_hybrid_router_override.py
```

2026-09-08 定向结果：10 passed。覆盖 20 场景到 8 Workflow 的完整性、自然语言相似表述、原料使用效果、原料替代边界、目标性能筛选优先级、跨源相似路由不被单源意图降级，以及原有 EvidenceFrame 行为。

2026-09-09 切片 2 定向结果：26 passed。补充覆盖数值阈值不被误判为样品 ID、槽位兜底解析、来源关联分级、显式跨源研究问题的确定性路由，以及证据比较差异。

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

## 7. 2026-09-09 切片 2：阶段三只读场景真实验收

执行方式：真实业务 MySQL + 受管外部向量 API + 真实 LLM；33 个问题、串行执行；每个场景 3 问。测试数据范围仍显式使用 `EXTERNAL_VECTOR_QUERY_COMPANY_ID=test-company-id`，生产必须留空。

### 7.1 汇总

| 场景 | 通过 | 路由 / 槽位 | MySQL | 外部向量 | EvidenceFrame | 来源关联 | 结论 |
|---:|---:|---|---|---|---|---|---|
| 1 相似配方检索 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 2 相似样品 / 相似实验检索 | 3/3 | 通过 | ok / 1 例参考字段不足警告 | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 3 按目标性能反查历史方案 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 4 配方-工艺-性能联查 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 5 原料使用效果查询 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 6 原料替代历史检索 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过；输出共存边界 |
| 7 失败配方 / 失败实验检索 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 12 异常与失效案例检索 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 13 竞品对标 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 14 新项目冷启动 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |
| 17 项目知识快速问答 | 3/3 | 通过 | ok | ok | 通过 | `TOPIC_ONLY` | 通过 |

模型权限恢复初期，最终 LLM 综述曾出现间歇性拒绝；最小直连探针一度返回：

```text
HTTP 403
AccessDenied.Unpurchased
Access to model denied. Please make sure you are eligible for using the model.
```

权限生效后按串行冷却策略复跑 `S12-03、S14-02、S14-03`，三例均通过；最终结果为 33/33。并发 3 曾触发更多 `403 Forbidden` 和偶发 `ReadTimeout`，本矩阵仅把 `synthesis.status=ok` 的真实 LLM 报告记为通过，不把降级报告记为通过。

### 7.2 33 个真实问题

| 用例 | 问题 | 结果 |
|---|---|---|
| S1-01 | 查找与 EXP-128 相似的配方，并结合历史案例。 | 通过 |
| S1-02 | 找与 EXP-128 配方相近的历史样品，并结合历史资料。 | 通过 |
| S1-03 | 查组分和 EXP-128 最接近的配方，并结合历史资料。 | 通过 |
| S2-01 | 找与 EXP-128 工艺最像的样品，并结合历史资料。 | 通过；参考字段不足已显式警告 |
| S2-02 | 查与 EXP-128 综合条件相近的实验，并结合历史案例。 | 通过 |
| S2-03 | 查找和 EXP-128 配方性能都接近的历史样品，并结合资料。 | 通过 |
| S3-01 | 查找密度差大于 100 且持液量小于 0.1 的历史样品，并结合历史资料。 | 通过 |
| S3-02 | 查找密度差大于 150 的样品，并结合历史案例。 | 通过 |
| S3-03 | 查找持液量在 0.05 到 0.1 之间的样品，并结合历史资料。 | 通过 |
| S4-01 | 查 EXP-128 的完整研发上下文，并结合历史资料综合判断。 | 通过 |
| S4-02 | 查看 EXP-127 的配方、工艺和性能，并结合历史资料。 | 通过 |
| S4-03 | 查 EXP-129 对应样品和测试结果，并结合历史案例。 | 通过 |
| S5-01 | 查水用在哪些样品、用量多少以及性能如何，并结合历史案例。 | 通过 |
| S5-02 | 查询 P507+煤油 的原料使用效果，并结合历史资料。 | 通过 |
| S5-03 | 使用水的样品性能怎么样？结合历史资料判断。 | 通过 |
| S6-01 | 查找 P507+煤油 替代 水的历史记录，并结合历史资料。 | 通过；仅证明共存/历史上下文 |
| S6-02 | 有没有用水替换 P507+煤油 的配方记录？结合历史案例说明。 | 通过 |
| S6-03 | 查这两种原料的替代后性能变化：P507+煤油 和 水，并结合资料。 | 通过 |
| S7-01 | 查找失败配方和失败实验，并结合历史资料。 | 通过 |
| S7-02 | 以前类似萃取配方路线为什么失败？结合历史案例。 | 通过 |
| S7-03 | 查失败后调整过的实验记录，并结合内部历史资料。 | 通过 |
| S12-01 | 查找开裂异常案例，并结合历史资料。 | 通过 |
| S12-02 | 查析出或变色的历史失效案例，并结合资料。 | 通过 |
| S12-03 | 查粘接失效类似案例，并结合内部资料。 | 通过；权限恢复后降容重试成功 |
| S13-01 | 与竞品 ABS 的性能差距是多少？结合历史数据判断。 | 通过 |
| S13-02 | 历史上哪些路线最接近竞品 PC/ABS？结合资料。 | 通过 |
| S13-03 | 查竞品对标的内部样品和资料。 | 通过 |
| S14-01 | 新项目要求密度差大于 100 且持液量小于 0.1，请结合历史和失败记录给首轮方案。 | 通过 |
| S14-02 | 新项目冷启动：目标密度差大于 150，请结合历史资料给第一轮方案。 | 通过；权限恢复后串行复跑成功 |
| S14-03 | 新项目从零开始，请结合历史项目、失败记录和资料给首轮实验建议。 | 通过；权限恢复后降容重试成功 |
| S17-01 | 去年做过哪些方案？结合数据库和历史资料。 | 通过 |
| S17-02 | 这个项目里水为什么后来停用？结合历史资料。 | 通过 |
| S17-03 | 项目知识问答：当前授权项目的结论和风险有哪些？结合资料。 | 通过 |

### 7.3 验收结论

阶段三的固定 Workflow 分类、确定性路由兜底、槽位解析、真实 MySQL 检索、受管外部向量检索、证据分型、冲突保留和引用输出已经达到切片 2 目标。`TOPIC_ONLY` 是本轮真实数据的实际关联等级：向量文档缺少可证明同一样品/实验/项目的显式 ID，系统没有把文本相似度伪装成实体级关联。

`qwen3.7-plus` 权限恢复后，`S12-03、S14-02、S14-03` 已全部串行复跑通过；无需调整业务数据链路或降低证据阈值。

此前真实 LLM 综述曾出现一次 `ReadTimeout`；使用固定摘要器复跑证明 MySQL、外部向量与 EvidenceFrame 链路本身正常。2026-09-09 切片 1 已增加有界证据上下文、一次降容重试和确定性降级报告，真实端到端复测不再超时。生产配置禁止非空 `EXTERNAL_VECTOR_QUERY_COMPANY_ID`。
