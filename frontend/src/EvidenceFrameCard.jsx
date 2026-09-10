import { useEffect, useState } from "react";
import {
  documentEvidenceRecords,
  evidenceMetricRows,
  isHybridEvidencePayload,
  normalizedEvidenceRecords,
  structuredEvidenceGroups,
} from "./evidenceFrame";

const ENTITY_LABELS = {
  sample: "样品",
  project: "项目",
  experiment: "实验",
  material: "原料",
  knowledge_chunk: "知识资料",
  upload_chunk: "上传资料",
  user_request: "本轮问题",
  scenario_summary: "结构化汇总",
};

const ATTRIBUTE_LABELS = {
  name: "名称",
  project_id: "所属项目",
  sample_type: "样品类型",
  describe: "说明",
  record_reference: "关联记录",
  text: "资料内容",
};

function formatValue(value, maxLength = 180) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  if (Array.isArray(value)) {
    const text = value.map(item => formatValue(item, 60)).join("、");
    return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
  }
  if (typeof value === "object") {
    const keys = Object.keys(value);
    return keys.length ? `已记录 ${keys.length} 项字段` : "-";
  }
  const text = String(value);
  return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
}

function businessAttribute(value) {
  const text = String(value || "-");
  if (ATTRIBUTE_LABELS[text]) return ATTRIBUTE_LABELS[text];
  return text
    .replace(/^conditions\./, "测试条件 · ")
    .replace(/^formula\./, "配方 · ")
    .replace(/^process\./, "工艺 · ")
    .replace(/^performance\./, "性能 · ")
    .replace(/^service_performance\./, "服役性能 · ");
}

function displaySubject(recordOrGroup) {
  const subject = String(recordOrGroup.subject_id || "");
  const entityLabel = ENTITY_LABELS[recordOrGroup.entity_type] || "结构化对象";
  const sample = subject.match(/^sample:(\d+)$/);
  if (sample) return `样品 ${sample[1]}`;
  const project = subject.match(/^project(?:_id)?:(-?\d+)$/);
  if (project) return `项目 ${project[1]}`;
  if (subject.startsWith("vector_chunk:")) return "知识资料";
  if (subject.startsWith("upload:")) return "上传资料";
  if (subject.startsWith("dialog:")) return "本轮问题";
  return entityLabel;
}

function documentTitle(record) {
  const metadata = record.metadata || {};
  return String(
    metadata.title
    || metadata.filename
    || metadata.file_name
    || record.source_uri
    || "知识资料"
  ).replace(/^vector:\/\//, "");
}

function EvidenceGroup({ group }) {
  return <article className="evidenceGroup">
    <header>
      <b>{displaySubject(group)}</b>
      <span>{ENTITY_LABELS[group.entity_type] || "结构化数据"}</span>
    </header>
    <dl>
      {group.rows.map((row, index) => <div key={row._key || index}>
        <dt>{businessAttribute(row.attribute)}</dt>
        <dd>{formatValue(row.value)}{row.unit ? ` ${row.unit}` : ""}</dd>
      </div>)}
    </dl>
  </article>;
}

function DocumentItem({ record }) {
  const text = formatValue(record.value, 240);
  const score = Number(record.confidence);
  return <details className="evidenceDocument">
    <summary>
      <b>{documentTitle(record)}</b>
      <span>{Number.isFinite(score) ? `${formatValue(score * 100, 1)}% 相关度` : "文档证据"}</span>
    </summary>
    <p>{text}</p>
  </details>;
}

export default function EvidenceFrameCard({frame,evidence,intent}){
  if(!isHybridEvidencePayload({frame,evidence,intent}))return null;
  const records=normalizedEvidenceRecords({frame,evidence});
  const metrics=evidenceMetricRows(frame);
  const structuredGroups=structuredEvidenceGroups(records);
  const documents=documentEvidenceRecords(records);
  const conflictRows=Array.isArray(frame?.conflicts)?frame.conflicts:[];
  const [visibleGroups,setVisibleGroups]=useState(5);
  const [visibleDocuments,setVisibleDocuments]=useState(3);
  useEffect(()=>{setVisibleGroups(5);setVisibleDocuments(3)},[frame]);
  const source=frame?.source_summary||{};
  const structuredCount=source.mysql??0;
  const documentCount=(source.vector_api??0)+(source.vector??0)+(source.upload??0);

  return <section className="evidenceFrameCard">
    <details className="evidencePanel">
      <summary>依据：结构化数据 {structuredCount} 条 · 资料 {documentCount} 条</summary>
      <div className="evidencePanelBody">
        {!!metrics.length&&<div className="evidenceMetrics">
          {metrics.map(item=><div key={item.label}>
            <span>{item.label}</span>
            <b>{formatValue(item.value)}</b>
          </div>)}
        </div>}

        {!!structuredGroups.length&&<section className="evidenceSection">
          <header><b>结构化数据</b><span>按样品、项目或实验聚合</span></header>
          <div className="evidenceGroups">
            {structuredGroups.slice(0,visibleGroups).map(group=><EvidenceGroup group={group} key={group.key}/>)}
          </div>
          {structuredGroups.length>visibleGroups&&<button
            className="evidenceLoadMore"
            type="button"
            onClick={()=>setVisibleGroups(count=>count+5)}
          >加载更多<span>剩余 {structuredGroups.length-visibleGroups} 组</span></button>}
        </section>}

        {!!documents.length&&<section className="evidenceSection">
          <header><b>资料证据</b><span>仅表示主题相关，关联等级以结论提示为准</span></header>
          <div className="evidenceDocuments">
            {documents.slice(0,visibleDocuments).map((record,index)=><DocumentItem record={record} key={record.record_id||index}/>)}
          </div>
          {documents.length>visibleDocuments&&<button
            className="evidenceLoadMore"
            type="button"
            onClick={()=>setVisibleDocuments(count=>count+3)}
          >加载更多<span>剩余 {documents.length-visibleDocuments} 条</span></button>}
        </section>}

        {!!conflictRows.length&&<details className="evidenceConflicts">
          <summary>存在 {conflictRows.length} 组不一致记录，已保留供核对</summary>
          <div>{conflictRows.map((item,index)=><p key={item.conflict_group||index}>
            <b>{displaySubject(item)} · {businessAttribute(item.attribute)}</b>
            <span>{item.reason}</span>
          </p>)}</div>
        </details>}
      </div>
    </details>
  </section>;
}
