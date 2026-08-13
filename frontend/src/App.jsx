import { useEffect, useMemo, useRef, useState } from "react";
import {
  chat,
  deleteChatFile,
  getScope,
  health,
  saveScope,
  uploadChatFile,
} from "./api";

const versions=[
  ["V0.1.1","Agent + MySQL","查询 · 对比 · 分析","done"],
  ["V0.1.2","File + Knowledge + RAG","附件 · 历史检索 · MySQL+历史联合分析","on"],
  ["V0.1.3","Dataset + ML","数据集 · 机器学习","off"],
  ["V0.1.4","Optimization + BO","反向设计 · 优化","off"]
];
const quick=[
  {label:"完整研发上下文",text:"查 3811 的完整研发上下文"},
  {label:"历史知识检索",text:"历史有没有类似的冲击强度下降问题？"},
  {label:"联合分析",text:"3811 的冲击强度比 3809 低很多，历史上有没有类似问题？结合数据库数据和历史报告分析一下。"}
];
const welcome={
  id:"welcome",
  role:"assistant",
  content:`你好，我是材数智能体。

V0.1.2 T07 已开放：MySQL 只读查询、当前 PDF/DOCX 附件分析、历史 Qdrant RAG，以及 MySQL + 历史报告联合分析。`
};
const id=()=>Date.now()+"-"+Math.random().toString(16).slice(2);

function Logo(){return <div className="logo">◇</div>}

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
      <div className="bubble"><div className="content">{m.content}</div><Detail m={m}/></div>
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

  useEffect(()=>{
    health().then(()=>setOnline(true)).catch(()=>setOnline(false));
  },[]);
  useEffect(()=>{
    end.current?.scrollIntoView({behavior:"smooth"});
  },[messages,busy]);

  const history=useMemo(
    ()=>messages.filter(x=>x.id!=="welcome").map(x=>({role:x.role,content:x.content})),
    [messages]
  );
  const updateScope=(k,v)=>{const n={...scope,[k]:v};setScope(n);saveScope(n)};

  async function onFilesSelected(event){
    const files=[...(event.target.files||[])];
    event.target.value="";
    if(!files.length) return;
    setErr("");
    setUploading(true);
    try{
      for(const file of files){
        const result=await uploadChatFile(file,scope);
        setAttachments(prev=>[
          ...prev,
          {
            attachmentId:result.attachment_id,
            filename:result.filename,
            parser:result.parser,
            pageCount:result.page_count,
            charCount:result.char_count,
            chunkCount:result.chunk_count,
          }
        ]);
      }
    }catch(e){
      setErr(e.message);
    }finally{
      setUploading(false);
    }
  }

  async function removeAttachment(item){
    try{
      await deleteChatFile(item.attachmentId,scope);
    }catch(e){
      // Even if server-side temp file already expired, remove stale UI chip.
      console.warn(e);
    }
    setAttachments(prev=>prev.filter(x=>x.attachmentId!==item.attachmentId));
  }

  async function send(value){
    const q=(value??text).trim();
    if(!q||busy||uploading)return;
    setText("");
    setErr("");
    setMessages(x=>[...x,{id:id(),role:"user",content:q}]);
    setBusy(true);
    try{
      const r=await chat(
        q,
        history,
        scope,
        attachments.map(x=>x.attachmentId),
      );
      setMessages(x=>[...x,{
        id:id(),
        role:"assistant",
        content:r.answer,
        meta:{intent:r.intent,tool:r.tool_name,router:r.router,summary:r.reasoning_summary},
        data:r.data,
        evidence:r.evidence||[]
      }]);
    }catch(e){
      setErr(e.message);
    }finally{
      setBusy(false);
    }
  }

  function newChat(){
    setMessages([welcome]);
    setText("");
    setErr("");
    setAttachments([]);
  }

  return <div className="shell">
    <aside>
      <div className="brand"><Logo/><div><b>材数智能体</b><span>Materials Research Agent</span></div></div>
      <label className="section">能力版本</label>
      {versions.map(v=><div className={`version ${v[3]}`} key={v[0]}>
        <div className="dot">{v[3]==="on"?"●":v[3]==="done"?"✓":"○"}</div>
        <div>
          <b>{v[0]} {v[3]==="on"&&<em>当前</em>}{v[3]==="done"&&<em>已通过</em>}</b>
          <strong>{v[1]}</strong><span>{v[2]}</span>
        </div>
      </div>)}
      <div className="spacer"/>
      <div className="scope">
        <button onClick={()=>setShowScope(!showScope)}><span>开发权限范围<br/><b>Project {scope.projectIds}</b></span><i>{showScope?"⌃":"⌄"}</i></button>
        {showScope&&<div className="scopeFields">
          <label>User ID<input value={scope.userId} onChange={e=>updateScope("userId",e.target.value)}/></label>
          <label>Company ID<input value={scope.companyId} onChange={e=>updateScope("companyId",e.target.value)}/></label>
          <label>Project IDs<input value={scope.projectIds} onChange={e=>updateScope("projectIds",e.target.value)}/></label>
          <p>仅用于 development_header，正式版替换为平台登录态。</p>
        </div>}
      </div>
      <div className="status"><i className={online?"ok":"bad"}/><span>{online===null?"检查后端中":online?"后端已连接":"后端未连接"}</span></div>
    </aside>

    <main>
      <header><div><h1>研发对话</h1><p>V0.1.2 T07 · MySQL 事实 + 当前附件 + 历史 Qdrant RAG</p></div><div><span className="badge">V0.1.2-T07</span><button onClick={newChat}>新对话</button></div></header>
      <section className="scroll"><div className="inner">
        {messages.length===1&&<div className="quick"><small>可以试试</small><div>{quick.map(q=><button key={q.text} onClick={()=>send(q.text)}><span>{q.label}</span><b>{q.text}</b></button>)}</div></div>}
        <div className="messages">
          {messages.map(m=><Message key={m.id} m={m}/>)}
          {busy&&<div className="msg assistant"><div className="avatar"><Logo/></div><div className="msgcol"><small>材数智能体</small><div className="bubble loading">● ● ● <span>正在识别意图并读取证据</span></div></div></div>}
        </div>
        {err&&<div className="error"><b>请求失败</b>{err}</div>}
        <div ref={end}/>
      </div></section>

      <footer>
        {!!attachments.length&&<div className="attachments">{attachments.map(item=><div className="attachmentChip" key={item.attachmentId}>
          <span>附件</span><b>{item.filename}</b><span>{item.pageCount?`${item.pageCount}页`:item.parser} · {item.chunkCount} chunks</span>
          <button title="移除附件" onClick={()=>removeAttachment(item)}>×</button>
        </div>)}</div>}
        {uploading&&<div className="uploadState">正在上传并解析附件…</div>}
        <div className="composer">
          <input ref={fileInput} type="file" accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" multiple hidden onChange={onFilesSelected}/>
          <button className="upload uploadReady" disabled={uploading||busy} onClick={()=>fileInput.current?.click()}>＋ 上传文件 <span>PDF/DOCX</span></button>
          <textarea value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}}} placeholder={attachments.length?"问当前附件，例如：分析这份报告":"输入研发问题，或先上传 PDF / DOCX"}/>
          <button className="send" disabled={!text.trim()||busy||uploading} onClick={()=>send()}>➤</button>
        </div>
        <p>当前附件 = Chat 临时上下文 · 历史资料 = Qdrant Knowledge Index · 业务 MySQL = READ ONLY</p>
      </footer>
    </main>
  </div>
}
