import {useState} from "react";
import {
  ENGINE_WORKFLOW_LABELS,
  engineTaskActions,
  engineTaskIsTerminal,
  engineTaskProgress,
  isEngineWorkflowData,
  visualizationDatasets,
} from "./engineWorkflow";

const ACTION_LABELS={approve:"批准",cancel:"取消",resume:"恢复"};
const TERMINAL_STATES=["SUCCEEDED","FAILED","CANCELLED"];

function statusClass(value){
  const status=String(value||"").toUpperCase();
  if(["OK","SUCCEEDED","COMPLETED"].includes(status))return "good";
  if(["MODEL_REQUIRED","BLOCKED","AWAITING_APPROVAL","QUEUED","RUNNING","CANCEL_REQUESTED","INTERRUPTED"].includes(status))return "warn";
  if(["ERROR","FAILED","CANCELLED"].includes(status))return "bad";
  return "neutral";
}

function formatValue(value){
  if(value===null||value===undefined||value==="")return "-";
  if(typeof value==="number"&&Number.isFinite(value)){
    return Number.isInteger(value)?String(value):value.toFixed(3);
  }
  if(typeof value==="object"){
    if(Array.isArray(value))return `${value.length} 项`;
    return Object.keys(value).length?`${Object.keys(value).length} 字段`:"-";
  }
  return String(value);
}

function Metric({label,value,sub}){
  return <div className="engineMetric"><span>{label}</span><b>{formatValue(value)}</b>{sub&&<small>{sub}</small>}</div>;
}

function ResultMetrics({workflow,result}){
  const training=result?.training?.training_run||result?.training||{};
  const models=training.model_artifacts||result?.model_artifacts||[];
  const artifact=result?.dataset_artifact||result?.preprocessing?.dataset_artifact||{};
  const predictions=result?.predictions||result?.prediction_preview||[];
  const candidates=result?.selected_candidates||result?.next_experiments||[];
  const selectedModels=result?.selected_models||[];
  const metrics=[];

  if(artifact.dataset_id)metrics.push(<Metric key="dataset" label="Dataset" value={`${artifact.dataset_id||"-"} / ${artifact.version||"-"}`} />);
  if(workflow==="automl_training")metrics.push(<Metric key="models" label="候选模型" value={models.length} />);
  if(selectedModels.length)metrics.push(<Metric key="selected" label="已选模型" value={selectedModels.length} />);
  if(workflow==="predict_performance")metrics.push(<Metric key="predictions" label="预测条数" value={predictions.length} />);
  if(["optimize_formula","recommend_next_experiments"].includes(workflow)){
    metrics.push(<Metric key="candidates" label="推荐候选" value={candidates.length} />);
  }
  if(Array.isArray(result?.objectives))metrics.push(<Metric key="objectives" label="优化目标" value={result.objectives.length} />);
  if(Array.isArray(result?.hard_constraints))metrics.push(<Metric key="constraints" label="硬约束" value={result.hard_constraints.length} />);
  if(!metrics.length&&result?.target_metric)metrics.push(<Metric key="target" label="目标字段" value={result.target_metric} />);
  return metrics.length?<div className="engineMetricGrid">{metrics}</div>:null;
}

function DataTable({dataset}){
  const records=Array.isArray(dataset.records)?dataset.records:[];
  const declared=Array.isArray(dataset.columns)?dataset.columns.map(item=>item.name||item):[];
  const columns=declared.length?declared:Object.keys(records[0]||{});
  const visible=columns.slice(0,12);
  const rows=records.slice(0,20);
  return <div className="engineTableWrap">
    <table>
      <thead><tr>{visible.map(column=><th key={String(column)}>{String(column)}</th>)}</tr></thead>
      <tbody>{rows.map((row,index)=><tr key={index}>{visible.map(column=><td key={String(column)}>{formatValue(row?.[column])}</td>)}</tr>)}</tbody>
    </table>
    <small>{records.length>20?`显示前 20 / ${records.length} 行`:`共 ${records.length} 行`}{columns.length>12?` · ${columns.length-visible.length} 列已省略`:""}</small>
  </div>;
}

function numeric(value){
  const parsed=Number(value);
  return Number.isFinite(parsed)?parsed:null;
}

function ChartBars({dataset,horizontal}){
  const records=Array.isArray(dataset.records)?dataset.records:[];
  const category=horizontal?String(dataset.y_fields?.[0]||""):"";
  const valueFields=horizontal?[dataset.x_field]:(dataset.y_fields||[]);
  const rows=records.slice(0,horizontal?15:25);
  const max=Math.max(1,...rows.flatMap(row=>valueFields.map(field=>Math.abs(numeric(row[field])??0))));
  if(!rows.length)return <p className="engineEmpty">暂无图表数据</p>;
  return <div className={horizontal?"engineHBarList":"engineBarList"}>
    {rows.map((row,index)=>{
      const label=String(row[horizontal?category:dataset.x_field]?? "");
      return <div className="engineBarRow" key={`${label}-${index}`}>
        <span title={label}>{label||`#${index+1}`}</span>
        <div>
          {valueFields.map(field=>{
            const value=numeric(row[field]);
            const size=Math.round((Math.abs(value??0)/max)*100);
            return <b key={String(field)} style={horizontal?{width:`${Math.max(2,size)}%`}:{height:`${Math.max(2,size)}%`}} title={`${field}: ${formatValue(value)}`} />;
          })}
        </div>
      </div>;
    })}
  </div>;
}

function ScatterChart({dataset}){
  const records=Array.isArray(dataset.records)?dataset.records:[];
  const x=dataset.x_field;
  const firstY=dataset.y_fields?.[0];
  const points=records.map(record=>({
    x:numeric(record?.[x]),
    y:numeric(record?.[firstY]),
    label:`${x}=${formatValue(record?.[x])}; ${firstY}=${formatValue(record?.[firstY])}`,
  })).filter(point=>point.x!==null&&point.y!==null).slice(0,300);
  if(!points.length)return <p className="engineEmpty">暂无可绘制散点</p>;
  const values=points.flatMap(point=>[point.x,point.y]);
  const min=Math.min(...values),max=Math.max(...values);
  const range=(max-min)||1;
  const position=value=>6+((value-min)/range)*88;
  return <svg className="engineScatter" viewBox="0 0 100 58" role="img" aria-label={dataset.title||"scatter chart"}>
    <line x1="6" y1="52" x2="96" y2="52" />
    <line x1="6" y1="4" x2="6" y2="52" />
    {points.map((point,index)=><circle key={index} cx={position(point.x)} cy={50-((point.y-min)/range)*44} r="1.05"><title>{point.label}</title></circle>)}
  </svg>;
}

function VisualizationDatasetCard({dataset}){
  const type=String(dataset.chart_type||"table");
  const supported=["table","bar","horizontal_bar","scatter"].includes(type);
  return <details className="engineChart" open>
    <summary>
      <span>{dataset.title||dataset.dataset_id}</span>
      <small>{type} · {Array.isArray(dataset.records)?dataset.records.length:0} rows</small>
    </summary>
    <div className="engineChartBody">
      {!supported||type==="table"
        ?<DataTable dataset={dataset}/>
        :type==="scatter"
          ?<ScatterChart dataset={dataset}/>
          :<ChartBars dataset={dataset} horizontal={type==="horizontal_bar"}/>}
      {!!dataset.source_artifact?.uri&&<small>source artifact: {dataset.source_artifact.uri}</small>}
    </div>
  </details>;
}

function ShadowComparison({shadow}){
  if(!shadow)return null;
  const legacy=shadow.metrics?.legacy||{};
  const engine=shadow.metrics?.engine||{};
  const rows=[
    ["selected_count","推荐候选"],
    ["hard_feasible_rate","硬约束满足率"],
    ["target_hit_rate","目标命中率"],
    ["pareto_diversity","Pareto 多样性"],
    ["in_domain_rate","适用域内比例"],
    ["duration_ms","耗时 ms"],
  ];
  return <div className="engineShadow">
    <header>
      <div><small>SHADOW ACCEPTANCE</small><b>legacy → engine 验收</b></div>
      <span className={`statusPill ${shadow.decision==="READY"?"good":"warn"}`}>{shadow.decision||"NOT_READY"}</span>
    </header>
    <div className="engineTableWrap"><table>
      <thead><tr><th>指标</th><th>Legacy</th><th>Engine</th></tr></thead>
      <tbody>{rows.map(([key,label])=><tr key={key}><td>{label}</td><td>{formatValue(legacy[key])}</td><td>{formatValue(engine[key])}</td></tr>)}</tbody>
    </table></div>
    <p>回退路由：{shadow.rollback_route||"legacy"}{shadow.error?` · ${shadow.error.type}: ${shadow.error.message}`:""}</p>
  </div>;
}

export function EngineWorkflowCard({data}){
  if(!isEngineWorkflowData(data))return null;
  const workflow=String(data.workflow||"");
  const datasets=visualizationDatasets(data);
  const warnings=Array.isArray(data.warnings)?data.warnings:[];
  const shadow=data.optimization_shadow||data.result?.optimization_shadow;
  return <section className="engineWorkflowCard">
    <header>
      <div><small>ENGINE WORKFLOW</small><b>{ENGINE_WORKFLOW_LABELS[workflow]||workflow}</b></div>
      <span className={`statusPill ${statusClass(data.status)}`}>{String(data.status).toUpperCase()}</span>
    </header>
    {!!data.scope?.project_id&&<p className="engineScope">Project {data.scope.project_id}{data.scope.conversation_id?` · ${data.scope.conversation_id}`:""}</p>}
    <ResultMetrics workflow={workflow} result={data.result}/>
    {!!data.steps?.length&&<div className="engineStepChips">{data.steps.map((step,index)=><span key={`${step.name}-${index}`} className={String(step.status).toUpperCase()==="COMPLETED"?"done":""}>{step.name}</span>)}</div>}
    {!!warnings.length&&<div className="engineWarnings"><b>警告</b>{warnings.slice(0,5).map((warning,index)=><span key={index}>{typeof warning==="string"?warning:formatValue(warning)}</span>)}</div>}
    <ShadowComparison shadow={shadow}/>
    {!!datasets.length&&<div className="engineCharts">{datasets.map(dataset=><VisualizationDatasetCard key={dataset.dataset_id||dataset.title} dataset={dataset}/>)}</div>}
  </section>;
}

export function EngineTaskCard({task,onAction,disabled}){
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  if(!task?.task_id)return null;
  const status=String(task.status||"").toUpperCase();
  const actions=engineTaskActions(status);
  const progress=engineTaskProgress(task);
  async function run(action){
    if(busy||!onAction)return;
    setBusy(true);setError("");
    try{await onAction(action)}
    catch(exc){setError(String(exc?.message||exc))}
    finally{setBusy(false)}
  }
  return <section className={`engineTaskCard ${statusClass(status)}`}>
    <header>
      <div><small>ENGINE TASK</small><b>{ENGINE_WORKFLOW_LABELS[task.intent]||task.intent||"引擎任务"}</b></div>
      <span className={`statusPill ${statusClass(status)}`}>{status||"UNKNOWN"}</span>
    </header>
    <div className="engineTaskMeta">
      <span>任务 {task.task_id}</span>
      <span>阶段 {task.current_stage||"-"}</span>
      <span>恢复 {task.resume_count??0} 次</span>
      {TERMINAL_STATES.includes(status)&&task.finished_at&&<span>完成时间 {task.finished_at}</span>}
    </div>
    <div className="engineTaskProgress"><i style={{width:`${progress}%`}}/><small>{progress}%{engineTaskIsTerminal(status)?"":" · 执行中"}</small></div>
    {!!task.error?.message&&<p className="engineTaskError">{task.error.type}: {task.error.message}</p>}
    {!!actions.length&&<div className="engineTaskActions">{actions.map(action=><button key={action} disabled={disabled||busy} onClick={()=>run(action)}>{ACTION_LABELS[action]}</button>)}</div>}
    {error&&<p className="engineTaskError">{error}</p>}
  </section>;
}
