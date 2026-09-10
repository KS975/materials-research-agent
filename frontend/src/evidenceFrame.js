export const EVIDENCE_SOURCE_LABELS={
  mysql:"MySQL",
  vector_api:"向量库",
  upload:"上传",
  dialog:"对话",
  derived:"派生",
};

export const EVIDENCE_AUTHORITY_LABELS={
  AUTHORITATIVE:"权威",
  REVIEWED:"已审核",
  DOCUMENT:"文档",
  EXTRACTED:"抽取",
  USER_INPUT:"用户输入",
};

export function isHybridEvidencePayload({frame,evidence,intent}={}){
  return Boolean(
    (frame&&typeof frame==="object"&&Array.isArray(frame.records))
    ||(intent==="hybrid_research_qa"&&Array.isArray(evidence)&&evidence.length)
  );
}

export function evidenceMetricRows(frame){
  const summary=frame?.source_summary||{};
  return [
    {label:"MySQL",value:summary.mysql??0},
    {label:"向量库",value:summary.vector_api??0},
    {label:"上传",value:summary.upload??0},
    {label:"对话",value:summary.dialog??0},
    {label:"冲突",value:frame?.conflicts?.length??0},
  ];
}

export function normalizedEvidenceRecords({frame,evidence}={}){
  const fromFrame=Array.isArray(frame?.records)?frame.records:[];
  const source=fromFrame.length?fromFrame:(Array.isArray(evidence)?evidence:[]);
  return source.map(record=>({
    record_id:record.record_id,
    source_type:record.source_type,
    subject_id:record.subject_id,
    entity_type:record.entity_type,
    attribute:record.attribute,
    value:record.value,
    unit:record.unit,
    confidence:record.confidence,
    authority_level:record.authority_level,
    conflict_group:record.conflict_group,
    source_uri:record.source_uri,
    metadata:record.metadata||{},
  }));
}

export function structuredEvidenceGroups(records){
  const grouped=new Map();
  for(const record of Array.isArray(records)?records:[]){
    if(record.source_type!=="mysql")continue;
    if(["record_reference","status"].includes(record.attribute))continue;
    const key=record.subject_id||record.entity_type||"unknown";
    if(!grouped.has(key)){
      grouped.set(key,{
        key,
        subject_id:record.subject_id,
        entity_type:record.entity_type,
        rows:[],
      });
    }
    const group=grouped.get(key);
    const rowKey=`${record.attribute}|${record.value}`;
    if(!group.rows.some(row=>row._key===rowKey)){
      group.rows.push({_key:rowKey,attribute:record.attribute,value:record.value,unit:record.unit});
    }
  }
  return [...grouped.values()];
}

export function documentEvidenceRecords(records){
  return (Array.isArray(records)?records:[]).filter(record=>
    ["vector_api","upload"].includes(record.source_type)
  );
}
