export function createInitialAnalysisStep(){
  return {
    schema_version:"1.1",
    source:"client",
    stage:"stream_transport",
    status:"running",
    title:"连接实时分析通道",
    message:"请求已提交，正在连接后端 SSE 流式接口。",
    elapsed_ms:0,
  };
}

export function mergeProgressStep(steps,next,maxSteps=30){
  const current=Array.isArray(steps)?steps:[];
  const eventKey=item=>{
    const attempt=item?.attempt!=null?`:attempt-${item.attempt}`:"";
    return `${item?.source||"backend"}:${item?.stage||"unknown"}${attempt}`;
  };
  const key=eventKey(next);
  const index=current.findIndex(item=>
    eventKey(item)===key
  );
  const merged=index<0
    ? [...current,next]
    : current.map((item,itemIndex)=>itemIndex===index?{...item,...next}:item);
  return merged.slice(-Math.max(1,Number(maxSteps)||30));
}

const PROGRESS_PHASES=[
  {id:"understand",title:"理解问题与权限范围",stages:["stream_transport","hybrid_research_plan","intent_router","intent"]},
  {id:"retrieve",title:"读取并对齐证据",stages:["mysql_evidence_recall","vector_evidence_recall","evidence_dataset_build","evidence_alignment"]},
  {id:"analyze",title:"执行确定性分析",stages:["deterministic_analysis","engine_workflow","engine_task"]},
  {id:"synthesize",title:"生成综合结论",stages:["llm_synthesis"]},
  {id:"complete",title:"完成最终报告",stages:["hybrid_final_report","final_report","complete"]},
];

function phaseStatus(steps){
  if(steps.some(step=>step?.status==="failed"))return "failed";
  if(steps.some(step=>step?.status==="running"||step?.status==="retrying"))return "running";
  if(steps.length&&steps.every(step=>step?.status==="completed"))return "completed";
  return steps[0]?.status||"running";
}

export function mergeProgressPhases(steps){
  const source=Array.isArray(steps)?steps:[];
  const assigned=new Set();
  const result=[];
  for(const phase of PROGRESS_PHASES){
    const rows=source.filter(step=>{
      if(assigned.has(step))return false;
      return phase.stages.includes(step?.stage);
    });
    rows.forEach(step=>assigned.add(step));
    if(!rows.length)continue;
    const active=[...rows].reverse().find(step=>
      step?.status==="running"||step?.status==="retrying"
    )||rows[rows.length-1];
    result.push({
      phase_id:phase.id,
      title:phase.title,
      status:phaseStatus(rows),
      message:active?.message||"",
      elapsed_ms:rows.reduce((total,step)=>total+(Number(step?.elapsed_ms)||0),0),
      source_count:rows.length,
      children:rows,
    });
  }
  const other=source.filter(step=>!assigned.has(step));
  if(other.length){
    const active=[...other].reverse().find(step=>
      step?.status==="running"||step?.status==="retrying"
    )||other[other.length-1];
    result.push({
      phase_id:"other",
      title:"其他执行步骤",
      status:phaseStatus(other),
      message:active?.message||"",
      elapsed_ms:other.reduce((total,step)=>total+(Number(step?.elapsed_ms)||0),0),
      source_count:other.length,
      children:other,
    });
  }
  return result;
}
