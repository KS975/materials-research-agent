const SCENARIO_LABELS = {
  1: "相似配方检索", 2: "相似样品/实验检索", 3: "按目标性能反查",
  4: "配方-工艺-性能联查", 5: "原料使用效果", 6: "原料替代历史",
  7: "失败配方/实验检索", 8: "关键变量识别", 9: "配方/工艺窗口发现",
  10: "多性能冲突分析", 11: "批次差异分析", 12: "异常与失效案例",
  13: "竞品对标", 14: "新项目冷启动", 15: "测试结果自动关联",
  16: "图谱/曲线结果复用", 17: "项目知识快速问答", 18: "自动生成阶段总结",
  19: "模型版本与实验回流", 20: "跨项目复用",
};

function statusClass(status) {
  const value = String(status || "").toUpperCase();
  if (["OK", "SUCCEEDED", "COMPLETED"].includes(value)) return "good";
  if (["NO_EVIDENCE", "PARTIAL", "NO_STRUCTURED_FEATURES", "MISSING_IDENTIFIER", "INTERFACE_RESERVED"].includes(value)) return "warn";
  if (["FAILED", "ERROR"].includes(value)) return "bad";
  return "neutral";
}

function isResearchPayload(data) {
  return Boolean(
    data
    && typeof data === "object"
    && data.analysis_type === "hybrid_research_qa"
    && data.research_workflow
  );
}

function ResultAssociationTable({ result }) {
  const matches = result?.matches || [];
  if (!matches.length) return null;
  return <div className="researchTableWrap">
    <table>
      <thead>
        <tr><th>输入编号</th><th>匹配状态</th><th>匹配样品</th><th>确认</th></tr>
      </thead>
      <tbody>
        {matches.map((item, i) => {
          const sample = item.matched_sample || {};
          return <tr key={i}>
            <td>{item.identifier}</td>
            <td>{item.match_status === "UNIQUE_MATCH" ? "唯一匹配" : item.match_status === "AMBIGUOUS" ? `歧义(${item.candidate_count}候选)` : "未匹配"}</td>
            <td>{sample.name || sample.sample_id || "-"}</td>
            <td>{item.confirmation_required ? "待确认" : "-"}</td>
          </tr>;
        })}
      </tbody>
    </table>
  </div>;
}

function SpectrumFeatureTable({ result }) {
  const features = result?.features || [];
  if (!features.length) return null;
  return <div className="researchTableWrap">
    <table>
      <thead>
        <tr><th>样品</th><th>特征字段</th><th>值</th></tr>
      </thead>
      <tbody>
        {features.slice(0, 20).map((item, i) => <tr key={i}>
          <td>{item.sample_id || "-"}</td>
          <td>{item.field}</td>
          <td>{String(item.value)}</td>
        </tr>)}
      </tbody>
    </table>
    <small>{features.length} 项特征</small>
  </div>;
}

function CrossProjectTable({ result }) {
  const projects = result?.projects || [];
  if (!projects.length) return null;
  return <div className="researchTableWrap">
    <table>
      <thead>
        <tr><th>项目</th><th>样品数</th><th>资产</th><th>权限</th></tr>
      </thead>
      <tbody>
        {projects.map((project, i) => <tr key={i}>
          <td>{project.project_id}</td>
          <td>{project.sample_count}</td>
          <td>{(project.assets || []).map(a => a.asset_type).join(", ") || "-"}</td>
          <td>只读</td>
        </tr>)}
      </tbody>
    </table>
  </div>;
}

export default function ResearchScenarioCard({ data, warnings }) {
  if (!isResearchPayload(data)) return null;
  const workflow = data.research_workflow;
  const scenarioId = workflow?.scenario_id;
  const scenarioName = workflow?.scenario_name || SCENARIO_LABELS[scenarioId] || "研究场景";
  const mysqlResult = data.mysql_result;
  const analysisResult = data.analysis_result;
  const warningRows = Array.isArray(warnings) ? warnings.filter(Boolean).slice(0, 4) : [];

  return <section className={`researchScenarioCard ${statusClass(data.status)}`}>
    <header>
      <div>
        <small>RESEARCH SCENARIO</small>
        <b>{scenarioName}</b>
      </div>
      <span className={`statusPill ${statusClass(data.status)}`}>{String(data.status || "-").toUpperCase()}</span>
    </header>
    <div className="researchScenarioMeta">
      <span>Workflow: {workflow?.workflow_name || "-"}</span>
      <span>写入策略: {workflow?.write_policy === "NO_WRITE" ? "只读" : workflow?.write_policy || "-"}</span>
    </div>
    {scenarioId === 15 && mysqlResult && <ResultAssociationTable result={mysqlResult} />}
    {scenarioId === 16 && mysqlResult && <SpectrumFeatureTable result={mysqlResult} />}
    {scenarioId === 20 && mysqlResult && <CrossProjectTable result={mysqlResult} />}
    {scenarioId === 19 && <div className="researchReserved">
      <p>实验回流正式闭环暂未开放，当前仅预留接口位置。</p>
    </div>}
    {analysisResult && analysisResult.status === "ok" && scenarioId !== 19 && <div className="researchAnalysis">
      <small>确定性分析已完成，详见报告。</small>
    </div>}
    {!!warningRows.length && <div className="researchWarnings">
      {warningRows.map((item, i) => <span key={i}>{String(item)}</span>)}
    </div>}
  </section>;
}
