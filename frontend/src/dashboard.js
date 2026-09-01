export const DASHBOARD_TABS=[
  {id:"overview",label:"概览"},
  {id:"projects",label:"项目"},
  {id:"samples",label:"样品"},
  {id:"fields",label:"可查询字段"},
];

export const FIELD_SECTION_LABELS={
  sample:"样品基础",
  formula:"配方",
  process:"工艺",
  performance:"性能",
  conditions:"测试条件",
};

function sampleId(sample){
  const value=Number(sample?.id);
  return Number.isInteger(value)?String(value):"";
}

export function buildSamplePrompt(action,samples){
  const selected=(samples||[]).map(sampleId).filter(Boolean);
  if(!selected.length)return "";
  if(selected.length===1){
    const value=selected[0];
    if(action==="similar")return `找和${value}最像的5个样品`;
    if(action==="history")return `以前有没有和${value}类似的情况`;
    return `查看样品${value}的完整信息`;
  }
  const [left,right]=selected;
  if(action==="formula")return `比较样品${left}和${right}的配方差异`;
  if(action==="process")return `比较样品${left}和${right}的工艺差异`;
  if(action==="performance")return `比较样品${left}和${right}的性能差异`;
  return `比较样品${left}和${right}`;
}

export function buildProjectPrompt(project){
  const value=Number(project?.id);
  return Number.isInteger(value)?`分析项目${value}的样品数据`:"";
}

export function buildFieldPrompt(section,field){
  const name=String(field?.name||"").trim();
  if(!name)return "";
  if(section==="formula")return `找${name}含量最高的样品`;
  if(section==="performance")return `找${name}最高的样品`;
  if(section==="process")return `找${name}最高的样品`;
  if(section==="conditions")return `找测试条件中包含${name}的样品`;
  return `查找与${name}有关的样品`;
}
