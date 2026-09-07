export const ENGINE_WORKFLOW_LABELS={
  engine_prepare_dataset:"数据预处理",
  automl_training:"自动建模",
  predict_performance:"性能预测",
  optimize_formula:"配方优化",
  recommend_next_experiments:"下一批实验",
  ensure_model:"模型确认",
};

export function isEngineWorkflowData(data){
  return Boolean(
    data
    && typeof data==="object"
    && data.workflow
    && data.status
    && data.result
    && typeof data.result==="object"
  );
}

export function visualizationDatasets(data){
  const payload=data?.result?.visualization_datasets||data?.visualization_datasets;
  if(Array.isArray(payload))return payload.filter(Boolean);
  if(Array.isArray(payload?.datasets))return payload.datasets.filter(Boolean);
  return [];
}

export function engineTaskActions(status){
  const value=String(status||"").toUpperCase();
  if(value==="AWAITING_APPROVAL")return ["approve","cancel"];
  if(["QUEUED","RUNNING","CANCEL_REQUESTED"].includes(value))return ["cancel"];
  if(["FAILED","CANCELLED","INTERRUPTED"].includes(value))return ["resume"];
  return [];
}

export function engineTaskProgress(status){
  const steps=Array.isArray(status?.scenario_plan?.steps)?status.scenario_plan.steps:[];
  const completed=Array.isArray(status?.completed_steps)?status.completed_steps:[];
  if(!steps.length)return status?.status==="SUCCEEDED"?100:0;
  return Math.min(100,Math.round(completed.length/steps.length*100));
}

export function engineTaskIsTerminal(status){
  return ["SUCCEEDED","FAILED","CANCELLED"].includes(String(status||"").toUpperCase());
}
