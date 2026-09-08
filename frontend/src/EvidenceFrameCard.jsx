import {
  EVIDENCE_AUTHORITY_LABELS as AUTHORITY_LABELS,
  EVIDENCE_SOURCE_LABELS as SOURCE_LABELS,
  evidenceMetricRows,
  isHybridEvidencePayload,
  normalizedEvidenceRecords,
} from "./evidenceFrame";

function formatValue(value){
  if(value===null||value===undefined||value==="")return "-";
  if(typeof value==="number"&&Number.isFinite(value)){
    return Number.isInteger(value)?String(value):value.toFixed(3);
  }
  if(typeof value==="object")return Array.isArray(value)?`${value.length} 项`:`${Object.keys(value).length} 字段`;
  return String(value);
}

function statusClass(status){
  const value=String(status||"").toUpperCase();
  if(["OK","SUCCEEDED","COMPLETED"].includes(value))return "good";
  if(["NO_EVIDENCE","PARTIAL","VECTOR_UNAVAILABLE"].includes(value))return "warn";
  if(["FAILED","ERROR"].includes(value))return "bad";
  return "neutral";
}

export default function EvidenceFrameCard({status,frame,evidence,intent,warnings}){
  if(!isHybridEvidencePayload({frame,evidence,intent}))return null;
  const records=normalizedEvidenceRecords({frame,evidence});
  const metrics=evidenceMetricRows(frame);
  const conflictRows=Array.isArray(frame?.conflicts)?frame.conflicts.slice(0,8):[];
  const warningRows=Array.isArray(warnings)?warnings.filter(Boolean).slice(0,5):[];
  const total=records.length+(frame?.records?.length>records.length?frame.records.length-records.length:0);

  return <section className={`evidenceFrameCard ${statusClass(status)}`}>
    <header>
      <div>
        <small>HYBRID RESEARCH EVIDENCE</small>
        <b>混合研究证据</b>
      </div>
      <span className={`statusPill ${statusClass(status)}`}>{String(status||"EVIDENCE").toUpperCase()}</span>
    </header>
    {!!metrics.length&&<div className="evidenceMetrics">
      {metrics.map(item=><div key={item.label}>
        <span>{item.label}</span>
        <b>{formatValue(item.value)}</b>
      </div>)}
    </div>}
    <div className="evidenceTableWrap">
      <table>
        <thead>
          <tr><th>来源</th><th>对象</th><th>属性</th><th>置信度</th><th>级别</th><th>冲突</th></tr>
        </thead>
        <tbody>
          {records.map((record,index)=><tr key={record.record_id||index} title={record.source_uri||""}>
            <td>{SOURCE_LABELS[record.source_type]||record.source_type||"-"}</td>
            <td>{record.subject_id||"-"}</td>
            <td>{record.attribute||"-"}</td>
            <td>{formatValue(record.confidence)}</td>
            <td>{AUTHORITY_LABELS[record.authority_level]||record.authority_level||"-"}</td>
            <td>{record.conflict_group?"CONFLICT":"-"}</td>
          </tr>)}
        </tbody>
      </table>
      <small>{records.length} / {total||records.length} 条证据</small>
    </div>
    {!!conflictRows.length&&<details className="evidenceConflicts">
      <summary>冲突保留 {conflictRows.length} 组</summary>
      <div>{conflictRows.map((item,index)=><p key={item.conflict_group||index}>
        <b>{item.subject_id} · {item.attribute}</b>
        <span>{item.reason} · {item.record_ids?.length||0} 条记录 · {item.resolution}</span>
      </p>)}</div>
    </details>}
    {!!warningRows.length&&<div className="evidenceWarnings">
      {warningRows.map((item,index)=><span key={index}>{String(item)}</span>)}
    </div>}
  </section>;
}
