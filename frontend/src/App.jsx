import { useEffect, useMemo, useRef, useState } from "react";
import {
  chat,
  deleteChatFile,
  getModelingStatus,
  getScope,
  health,
  saveScope,
  uploadChatFile,
} from "./api";

const versions=[
  ["V0.1.1","Agent + MySQL","查询 · 对比 · 分析","done"],
  ["V0.1.2","File + Knowledge + RAG","附件 · 历史检索 · MySQL+历史联合分析","done"],
  ["V0.1.3","Dataset + ML","准入 · 训练 · 评估 · 适用域","on"],
  ["V0.1.4","Optimization + BO","反向设计 · 优化","off"]
];
const quick=[
  {label:"完整研发上下文",text:"查 3811 的完整研发上下文",type:"chat"},
  {label:"历史知识检索",text:"历史有没有类似的冲击强度下降问题？",type:"chat"},
  {label:"联合分析",text:"3811 的冲击强度比 3809 低很多，历史上有没有类似问题？结合数据库数据和历史报告分析一下。",type:"chat"},
  {label:"V0.1.3 建模状态",text:"检查当前项目的冲击强度是否可以建模",type:"ml"},
];
const welcome={
  id:"welcome",
  role:"assistant",
  content:`你好，我是材数智能体。

V0.1.3 UI 已接入：除 MySQL、当前附件、历史 RAG 与联合分析外，现在还能在聊天区查看 Modeling Gate、模型评估和 Applicability Domain 状态。`
};
const id=()=>Date.now()+"-"+Math.random().toString(16).slice(2);

function Logo(){return <div className="logo">◇</div>}

function StatusPill({value}){
  const normalized=String(value||"UNKNOWN").toUpperCase();
  const cls=normalized==="PASS"||normalized==="IN_DOMAIN"?"good":normalized==="CONDITIONAL_PASS"||normalized==="BORDERLINE"?"warn":normalized==="FAIL"||normalized==="OUT_OF_DOMAIN"?"bad":"neutral";
  return <span className={`statusPill ${cls}`}>{normalized}</span>;
}

function Metric({label,value,sub}){
  return <div className="metric"><span>{label}</span><b>{value??"-"}</b>{sub&&<small>{sub}</small>}</div>
}

function ModelingCards({data}){
  if(!data||data.kind!=="v013_modeling_status") return null;
  const gate=data.gate;
  const cv=data.cross_validation;
  const comparison=data.model_comparison;
  const ad=data.latest_applicability_domain;
  const gateSummary=gate?.dataset_summary||{};
  const cvBest=cv?.best_cv_model||{};
  const bestSingle=comparison?.best_model||{};
  const adInfo=ad?.applicability_domain||{};
  const prediction=ad?.prediction;

  return <div className="mlCards">
    <div className="mlCard gateCard">
      <div className="mlCardHead"><div><small>MODELING GATE</small><b>建模准入</b></div><StatusPill value={gate?.decision||"NO_REPORT"}/></div>
      {gate?<>
        <div className="metricGrid">
          <Metric label="闭环样本" value={gateSummary.core_closed_samples}/>
          <Metric label="严格闭环" value={gateSummary.strict_closed_samples}/>
          <Metric label="目标数值" value={gateSummary.target_numeric_count}/>
          <Metric label="测试条件覆盖" value={typeof gateSummary.condition_coverage_on_core==="number"?`${(gateSummary.condition_coverage_on_core*100).toFixed(0)}%`:"-"}/>
        </div>
        <div className="permitLine"><span>训练允许</span><b className={gate.training_allowed?"yes":"no"}>{gate.training_allowed?"是":"否"}</b><span>正式模型</span><b className={gate.official_model_allowed?"yes":"no"}>{gate.official_model_allowed?"是":"否"}</b></div>
        {!!gate.fail_reasons?.length&&<div className="reasonList"><strong>阻断原因</strong>{gate.fail_reasons.slice(0,4).map((x,i)=><span key={i}>• {x}</span>)}</div>}
        {!!gate.warnings?.length&&<div className="reasonList warning"><strong>风险提示</strong>{gate.warnings.slice(0,4).map((x,i)=><span key={i}>• {x}</span>)}</div>}
      </>:<p className="emptyCard">当前项目还没有 V0.1.3 Modeling Gate 报告。</p>}
    </div>

    {(cv||comparison)&&<div className="mlCard">
      <div className="mlCardHead"><div><small>MODEL EVALUATION</small><b>模型评估</b></div><StatusPill value={cv?"PASS":"SINGLE_SPLIT"}/></div>
      <div className="modelName">{cvBest.model_name||bestSingle.model_name||"-"}</div>
      {cv?<div className="metricGrid three">
        <Metric label="CV R²" value={typeof cvBest.r2_mean==="number"?cvBest.r2_mean.toFixed(3):"-"} sub={typeof cvBest.r2_std==="number"?`± ${cvBest.r2_std.toFixed(3)}`:null}/>
        <Metric label="CV MAE" value={typeof cvBest.mae_mean==="number"?cvBest.mae_mean.toFixed(3):"-"} sub={typeof cvBest.mae_std==="number"?`± ${cvBest.mae_std.toFixed(3)}`:null}/>
        <Metric label="CV RMSE" value={typeof cvBest.rmse_mean==="number"?cvBest.rmse_mean.toFixed(3):"-"} sub={typeof cvBest.rmse_std==="number"?`± ${cvBest.rmse_std.toFixed(3)}`:null}/>
      </div>:<div className="metricGrid three">
        <Metric label="R²" value={typeof bestSingle.r2==="number"?bestSingle.r2.toFixed(3):"-"}/>
        <Metric label="MAE" value={typeof bestSingle.mae==="number"?bestSingle.mae.toFixed(3):"-"}/>
        <Metric label="RMSE" value={typeof bestSingle.rmse==="number"?bestSingle.rmse.toFixed(3):"-"}/>
      </div>}
      <p className="cardFoot">{cv?`${cv.cv?.folds||5}-fold 交叉验证 · 结果来自真实 sklearn 执行`:`当前仅发现单次 train/test 模型比较结果`}</p>
    </div>}

    {ad&&<div className="mlCard adCard">
      <div className="mlCardHead"><div><small>APPLICABILITY DOMAIN</small><b>预测适用域</b></div><StatusPill value={adInfo.status}/></div>
      <div className="adMain"><div><span>风险等级</span><b>{adInfo.risk||"-"}</b></div>{prediction&&<div><span>最近预测</span><b>{typeof prediction.value==="number"?prediction.value.toFixed(3):"-"}</b><small>{prediction.target_metric||data.target_metric}</small></div>}</div>
      {!!adInfo.reasons?.length&&<div className="reasonList"><strong>判断原因</strong>{adInfo.reasons.slice(0,3).map((x,i)=><span key={i}>• {x}</span>)}</div>}
    </div>}

    <div className="mlDisclaimer">V0.1.3 卡片只展示已落盘的真实运行报告；若数据来自 synthetic fixture，指标仅用于验证 ML 链路，不代表真实材料结论。</div>
  </div>
}

function Detail({m}){
  const [open,setOpen]=useState(false);
  if(m.role!=="assistant"||!m.meta) return null;
  return <div className="detail">
    <button onClick={()=>setOpen(!open)}>执行详情 <span>{open?"⌃":"⌄"}</span></button>
    {open&&<div className="detailBody">
      <div className="meta">
        <code>intent: {m.meta.intent||"-"}</code>
        <code>tool: {m.meta.tool||"-"}</code>
        <code>router: {m.meta.router||"-"}</code>
      </div>
      {m.meta.summary&&<p>{m.meta.summary}</p>}
      {!!m.evidence?.length&&<div className="chips">{m.evidence.map((e,i)=><span key={i}>
        {e.source === "chat_attachment"
          ? `附件 · ${e.filename}${e.page ? ` · 第${e.page}页` : ""} · chunk ${e.chunk_index}`
          : e.source === "knowledge_index" || e.evidence_type === "knowledge_index"
            ? `历史 · ${e.filename || "-"}${e.page ? ` · 第${e.page}页` : e.paragraph_start != null ? ` · 段落${e.paragraph_start}-${e.paragraph_end ?? e.paragraph_start}` : ""} · score ${typeof e.score === "number" ? e.score.toFixed(3) : "-"}`
            : `MySQL · ${e.source || "-"} · ${e.record_id ?? "-"}`
        }
      </span>)}</div>}
      {m.data&&<details><summary>结构化结果</summary><pre>{JSON.stringify(m.data,null,2)}</pre></details>}
    </div>}
  </div>
}

function Message({m}){
  return <div className={`msg ${m.role}`}>
    <div className="avatar">{m.role==="assistant"?<Logo/>:"你"}</div>
    <div className="msgcol">
      <small>{m.role==="assistant"?"材数智能体":"你"}</small>
      <div className="bubble"><div className="content">{m.content}</div><ModelingCards data={m.data}/><Detail m={m}/></div>
    </div>
  </div>
}

export default function App(){
  const [messages,setMessages]=useState([welcome]);
  const [text,setText]=useState("");
  const [busy,setBusy]=useState(false);
  const [uploading,setUploading]=useState(false);
  const [attachments,setAttachments]=useState([]);
  const [err,setErr]=useState("");
  const [scope,setScope]=useState(getScope);
  const [online,setOnline]=useState(null);
  const [showScope,setShowScope]=useState(false);
  const end=useRef(null);
  const fileInput=useRef(null);

  useEffect(()=>{health().then(()=>setOnline(true)).catch(()=>setOnline(false));},[]);
  useEffect(()=>{end.current?.scrollIntoView({behavior:"smooth"});},[messages,busy]);

  const history=useMemo(()=>messages.filter(x=>x.id!=="welcome").map(x=>({role:x.role,content:x.content})),[messages]);
  const updateScope=(k,v)=>{const n={...scope,[k]:v};setScope(n);saveScope(n)};
  const currentProjectId=()=>{
    const raw=String(scope.projectIds||"").split(",").map(x=>x.trim()).find(Boolean);
    const value=Number(raw);
    return Number.isInteger(value)&&value>0?value:null;
  };

  async function onFilesSelected(event){
    const files=[...(event.target.files||[])]; event.target.value=""; if(!files.length)return;
    setErr(""); setUploading(true);
    try{
      for(const file of files){
        const result=await uploadChatFile(file,scope);
        setAttachments(prev=>[...prev,{attachmentId:result.attachment_id,filename:result.filename,parser:result.parser,pageCount:result.page_count,charCount:result.char_count,chunkCount:result.chunk_count}]);
      }
    }catch(e){setErr(e.message)}finally{setUploading(false)}
  }

  async function removeAttachment(item){
    try{await deleteChatFile(item.attachmentId,scope)}catch(e){console.warn(e)}
    setAttachments(prev=>prev.filter(x=>x.attachmentId!==item.attachmentId));
  }

  async function sendModelStatus(targetMetric="冲击强度"){
    if(busy||uploading)return;
    const projectId=currentProjectId();
    if(!projectId){setErr("当前 Project IDs 无法解析，请先在左侧权限范围中填写项目号。 ");return;}
    const q=`检查 Project ${projectId} 的${targetMetric}建模状态`;
    setErr("");
    setMessages(x=>[...x,{id:id(),role:"user",content:q}]);
    setBusy(true);
    try{
      const r=await getModelingStatus(projectId,targetMetric,scope);
      setMessages(x=>[...x,{id:id(),role:"assistant",content:r.answer,meta:{intent:"get_modeling_status",tool:"v013_runtime_reports",router:"ui_action",summary:"只读加载 V0.1.3 已落盘运行报告，不触发训练。"},data:r.data,evidence:[]}]);
    }catch(e){setErr(e.message)}finally{setBusy(false)}
  }

  async function send(value){
    const q=(value??text).trim(); if(!q||busy||uploading)return;
    setText(""); setErr(""); setMessages(x=>[...x,{id:id(),role:"user",content:q}]); setBusy(true);
    try{
      const r=await chat(q,history,scope,attachments.map(x=>x.attachmentId));
      setMessages(x=>[...x,{id:id(),role:"assistant",content:r.answer,meta:{intent:r.intent,tool:r.tool_name,router:r.router,summary:r.reasoning_summary},data:r.data,evidence:r.evidence||[]}]);
    }catch(e){setErr(e.message)}finally{setBusy(false)}
  }

  function newChat(){setMessages([welcome]);setText("");setErr("");setAttachments([])}
  function runQuick(item){return item.type==="ml"?sendModelStatus():send(item.text)}

  return <div className="shell">
    <aside>
      <div className="brand"><Logo/><div><b>材数智能体</b><span>Materials Research Agent</span></div></div>
      <label className="section">能力版本</label>
      {versions.map(v=><div className={`version ${v[3]}`} key={v[0]}><div className="dot">{v[3]==="on"?"●":v[3]==="done"?"✓":"○"}</div><div><b>{v[0]} {v[3]==="on"&&<em>当前</em>}{v[3]==="done"&&<em>已通过</em>}</b><strong>{v[1]}</strong><span>{v[2]}</span></div></div>)}
      <div className="spacer"/>
      <div className="scope"><button onClick={()=>setShowScope(!showScope)}><span>开发权限范围<br/><b>Project {scope.projectIds}</b></span><i>{showScope?"⌃":"⌄"}</i></button>{showScope&&<div className="scopeFields"><label>User ID<input value={scope.userId} onChange={e=>updateScope("userId",e.target.value)}/></label><label>Company ID<input value={scope.companyId} onChange={e=>updateScope("companyId",e.target.value)}/></label><label>Project IDs<input value={scope.projectIds} onChange={e=>updateScope("projectIds",e.target.value)}/></label><p>仅用于 development_header，正式版替换为平台登录态。</p></div>}</div>
      <div className="status"><i className={online?"ok":"bad"}/><span>{online===null?"检查后端中":online?"后端已连接":"后端未连接"}</span></div>
    </aside>

    <main>
      <header><div><h1>研发对话</h1><p>V0.1.3 · Dataset Reality + Modeling Gate + ML Evaluation + Applicability Domain</p></div><div><span className="badge">V0.1.3</span><button className="modelStatusBtn" disabled={busy} onClick={()=>sendModelStatus()}>模型状态</button><button onClick={newChat}>新对话</button></div></header>
      <section className="scroll"><div className="inner">
        {messages.length===1&&<div className="quick"><small>可以试试</small><div>{quick.map(q=><button key={q.text} onClick={()=>runQuick(q)}><span>{q.label}</span><b>{q.text}</b></button>)}</div></div>}
        <div className="messages">{messages.map(m=><Message key={m.id} m={m}/>)}{busy&&<div className="msg assistant"><div className="avatar"><Logo/></div><div className="msgcol"><small>材数智能体</small><div className="bubble loading">● ● ● <span>正在读取研发证据 / 建模报告</span></div></div></div>}</div>
        {err&&<div className="error"><b>请求失败</b>{err}</div>}<div ref={end}/>
      </div></section>

      <footer>
        {!!attachments.length&&<div className="attachments">{attachments.map(item=><div className="attachmentChip" key={item.attachmentId}><span>附件</span><b>{item.filename}</b><span>{item.pageCount?`${item.pageCount}页`:item.parser} · {item.chunkCount} chunks</span><button title="移除附件" onClick={()=>removeAttachment(item)}>×</button></div>)}</div>}
        {uploading&&<div className="uploadState">正在上传并解析附件…</div>}
        <div className="composer"><input ref={fileInput} type="file" accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" multiple hidden onChange={onFilesSelected}/><button className="upload uploadReady" disabled={uploading||busy} onClick={()=>fileInput.current?.click()}>＋ 上传文件 <span>PDF/DOCX</span></button><textarea value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}}} placeholder={attachments.length?"问当前附件，例如：分析这份报告":"输入研发问题，或点击上方“模型状态”查看 V0.1.3"}/><button className="send" disabled={!text.trim()||busy||uploading} onClick={()=>send()}>➤</button></div>
        <p>业务 MySQL = READ ONLY · V0.1.3 UI 只读运行报告，不会从前台直接触发训练</p>
      </footer>
    </main>
  </div>
}
