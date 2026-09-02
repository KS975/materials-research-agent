# Cross-section Field Analysis V0.2

- 保留 `performance_rank` / `performance_statistics` 兼容意图名，内部扩展为配方、工艺、性能三类动态字段的统一排序与平均值计算。
- “增韧剂含量”“PC含量”等自然语言会先经授权字段目录绑定为 `formula.增韧剂`、`formula.PC`，不再被误当作性能字段。
- 字段绑定采用“数据库真实名称精确匹配优先、保守去除口语前后缀其次”的顺序；真实字段本身包含“含量”时不会被破坏。
- DeepSeek 只能建议字段名和类别；用户原句及后端字段目录会二次校正，模型臆造的类别不会直接控制查询。
- 同名字段跨配方/工艺/性能出现时停止计算并要求明确类别，不自动任选。
- 数据库浏览对配方、工艺、性能统一展示出现、有效、空值、非数值和重复字段计数；快捷提问显式携带字段类别。
- 保持当前公司与授权项目边界、只读查询、单位一致性检查和完整分页扫描；不修改 `.env` 或数据库结构。

# Performance Field Binding Fix V0.1

- 修复数据库浏览按钮生成“找拉伸强度最高的样品”时，规则路由错误保留“找”并形成 `找拉伸强度` 的问题。
- DeepSeek 返回的排名指标会由用户原句确定性纠正，再由后端授权字段目录绑定为标准字段名。
- 性能排名和平均值统一使用规范化字段匹配，不再要求自然语言参数与数据库字段逐字完全相同。
- 性能排名新增字段未记录、空值、非数值、重复字段、有效数值五类确定性统计。
- 数据库字段浏览区分“字段出现数”和“有效数值数”，避免将 JSON 中存在字段误解为可直接计算。
- 保留当前公司和项目权限边界；未修改 `.env`、数据库结构或写入权限。

# Database Navigator V0.1.1 - 历史导入项目

- 数据库项目目录不再只依赖 `mat_project`，同时纳入当前公司样品实际引用的 `project_id`。
- `project_id < 0` 作为合法的历史导入项目展示，不再被项目浏览页遗漏。
- 历史导入项目没有独立项目主表记录时，生成只读的项目卡片和名称，不伪造项目档案。
- 概览增加历史导入项目计数，项目、样品和详情页统一显示“历史导入”标识。
- 项目目录、样品筛选和详情仍强制继承 `UserContext.company_id` 与项目权限范围。

# Skill Architecture V1

- 新增 `SkillRegistry`：operation 唯一归属、输入/输出 Schema、Tool 白名单、证据规则、护栏、审批点和异常策略。
- 新增 `ScenarioWorkflowComposer`：把旧 intent 兼容字段编排为原子 Skill 执行计划。
- 注册 Knowledge QA、Data Governance、AutoML、Ensure Model、Prediction、Optimization、Experiment Ingestion、Model Governance 八类业务 Skill。
- LangGraph state、JSON checkpoint、SSE 和最终响应 routing 均记录 `skill_name` 与 `scenario_plan`。
- AgentCore 不再全局遍历所有旧 Skill；先按注册表选择 Skill，再在该 Skill 的确定性处理器内执行。
- 新增 `GET /api/v1/skills` 只读目录接口。
- 旧 `intent/tool_name/tool_args`、前端接口、数据库结构和 `.env` 保持兼容。

# Round 2B-2.1.3

- 分析面板默认收起，运行中标题展示当前真实动作。
- 增加处理计划、查询对象、授权范围、检索词、命中证据和脱敏错误组件。
- 联合 MySQL + Qdrant 分析增加完整的查询和证据合成事件。
- 历史 RAG、单样品历史分析增加检索词、命中文件、Project、位置和 score。
- Database Explorer 展示通过校验的 authorized_* SQL 及分 attempt 重试记录。
- 常用确定性意图展示数据库对象、扫描数量和计算内容。
- 延续安全边界：展示可审计摘要，不展示隐藏 Chain-of-Thought 原文。
