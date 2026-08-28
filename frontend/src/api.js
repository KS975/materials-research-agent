const KEY = "materials-agent-dev-scope";
const DEFAULT_SCOPE = {
  userId:"local-test",
  companyId:"6a4b19f62d0e000027001eb8",
  projectIds:"*",
};
const LEGACY_PROJECT_SCOPES = new Set([
  "115",
  "115,9010,9018,9026,9036,930066",
]);

function normalizeProjectScope(value){
  const raw=String(value??"").trim();
  const compact=raw.replace(/\s+/g,"");
  if(!compact || LEGACY_PROJECT_SCOPES.has(compact)) return "*";
  if(compact.toLowerCase()==="all") return "*";
  return raw;
}

export function getScope(){
  try {
    const stored=JSON.parse(localStorage.getItem(KEY)||"{}");
    const scope={...DEFAULT_SCOPE,...stored};
    scope.projectIds=normalizeProjectScope(scope.projectIds);
    // One-time migration from the old demo whitelist to current-company-all.
    localStorage.setItem(KEY,JSON.stringify(scope));
    return scope;
  } catch {
    return {...DEFAULT_SCOPE};
  }
}

export function saveScope(v){
  localStorage.setItem(KEY,JSON.stringify({...v,projectIds:normalizeProjectScope(v?.projectIds)}));
}

function scopeHeaders(scope){
  return {
    "X-User-Id":scope.userId,
    "X-Company-Id":scope.companyId,
    "X-Project-Ids":normalizeProjectScope(scope.projectIds),
  };
}

async function dashboardGet(path, params, scope){
  const query=new URLSearchParams();
  Object.entries(params||{}).forEach(([key,value])=>{
    if(value!==undefined&&value!==null&&value!=="")query.set(key,String(value));
  });
  const suffix=query.size?`?${query.toString()}`:"";
  const r=await fetch(`/agent-api/api/v1/dashboard/${path}${suffix}`,{
    headers:scopeHeaders(scope),
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok)throw new Error(data?.detail||`数据库浏览失败 HTTP ${r.status}`);
  return data;
}

export function getDashboardSummary(scope){return dashboardGet("summary",{},scope)}
export function getDashboardProjects(params,scope){return dashboardGet("projects",params,scope)}
export function getDashboardSamples({q="",projectId=null,limit=20,offset=0},scope){
  return dashboardGet("samples",{q,project_id:projectId,limit,offset},scope);
}
export function getDashboardSampleDetail(sampleId,scope){return dashboardGet(`samples/${sampleId}`,{},scope)}
export function getDashboardFields({q="",section="all"},scope){return dashboardGet("fields",{q,section},scope)}

export async function health(){
  const r=await fetch("/agent-api/health");
  if(!r.ok) throw new Error();
  return r.json();
}

export async function uploadChatFile(file, scope){
  const form = new FormData();
  form.append("file", file);
  const r = await fetch("/agent-api/api/v1/files/chat-upload", {
    method:"POST",
    headers:scopeHeaders(scope),
    body:form,
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok) throw new Error(data?.detail||`上传失败 HTTP ${r.status}`);
  return data;
}

export async function deleteChatFile(attachmentId, scope){
  const r = await fetch(`/agent-api/api/v1/files/chat-attachments/${attachmentId}`, {
    method:"DELETE",
    headers:scopeHeaders(scope),
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok) throw new Error(data?.detail||`删除失败 HTTP ${r.status}`);
  return data;
}

export async function chat(
  message,
  history,
  scope,
  attachmentIds=[],
  attachmentReferenceMode=false,
){
  const r=await fetch("/agent-api/api/v1/chat-ui",{
    method:"POST",
    headers:{"Content-Type":"application/json",...scopeHeaders(scope)},
    body:JSON.stringify({
      message,
      history:history.slice(-12),
      attachment_ids:attachmentIds,
      attachment_reference_mode:Boolean(attachmentReferenceMode),
    })
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok) throw new Error(data?.detail||`HTTP ${r.status}`);
  return data;
}

export async function chatWithProgress(
  message,
  history,
  scope,
  attachmentIds=[],
  attachmentReferenceMode=false,
  onProgress=()=>{},
){
  const startedAt=Date.now();
  const reportClientProgress=(stage,status,title,message,details={})=>onProgress({
    schema_version:"1.1",
    source:"client",
    stage,
    status,
    title,
    message,
    elapsed_ms:Math.max(0,Date.now()-startedAt),
    ...details,
  });
  const payload={
    message,
    history:history.slice(-12),
    attachment_ids:attachmentIds,
    attachment_reference_mode:Boolean(attachmentReferenceMode),
  };
  reportClientProgress(
    "stream_transport",
    "running",
    "连接实时分析通道",
    "正在连接后端 SSE 流式接口。",
  );
  let r;
  try{
    r=await fetch("/agent-api/api/v1/chat-ui/stream",{
      method:"POST",
      headers:{
        "Content-Type":"application/json",
        "Accept":"text/event-stream",
        ...scopeHeaders(scope),
      },
      body:JSON.stringify(payload),
    });
  }catch(error){
    reportClientProgress(
      "stream_transport",
      "failed",
      "实时分析通道连接失败",
      String(error?.message||error||"网络连接失败"),
    );
    throw error;
  }
  // Compatibility with a backend that has not yet received the SSE endpoint.
  if(r.status===404||r.status===405){
    reportClientProgress(
      "stream_transport",
      "retrying",
      "后端不支持实时分析",
      "已切换到兼容接口；本轮只能展示前端请求与最终结果记录。",
      {transport:"sync_fallback"},
    );
    const result=await chat(
      message,
      history,
      scope,
      attachmentIds,
      attachmentReferenceMode,
    );
    reportClientProgress(
      "result_received",
      "completed",
      "兼容模式结果已返回",
      "最终答案已返回，但本轮没有后端实时执行事件。",
      {transport:"sync_fallback"},
    );
    return result;
  }
  if(!r.ok){
    const data=await r.json().catch(()=>null);
    reportClientProgress(
      "stream_transport",
      "failed",
      "实时分析请求失败",
      data?.detail||`HTTP ${r.status}`,
      {http_status:r.status},
    );
    throw new Error(data?.detail||`HTTP ${r.status}`);
  }
  const contentType=String(r.headers.get("content-type")||"").toLowerCase();
  if(!contentType.includes("text/event-stream")){
    reportClientProgress(
      "stream_transport",
      "failed",
      "代理未返回 SSE 数据流",
      `收到 ${contentType||"未知 Content-Type"}；请检查 /agent-api 反向代理配置。`,
      {content_type:contentType||null},
    );
    throw new Error("实时分析接口没有返回 text/event-stream，请检查后端版本或 Nginx 代理配置");
  }
  reportClientProgress(
    "stream_transport",
    "completed",
    "实时分析通道已连接",
    "浏览器已连接后端 SSE，正在接收可核验执行步骤。",
    {transport:"sse"},
  );
  if(!r.body){
    reportClientProgress(
      "stream_transport",
      "failed",
      "浏览器无法读取实时数据流",
      "当前浏览器没有提供可读取的响应流。",
    );
    throw new Error("浏览器不支持流式响应");
  }

  const reader=r.body.getReader();
  const decoder=new TextDecoder("utf-8");
  let buffer="";
  let finalResult=null;
  let streamError=null;

  function consume(block){
    const lines=block.split(/\r?\n/);
    let eventName="message";
    const dataLines=[];
    for(const line of lines){
      if(line.startsWith("event:")) eventName=line.slice(6).trim();
      if(line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if(!dataLines.length)return;
    let data;
    try{
      data=JSON.parse(dataLines.join("\n"));
    }catch(error){
      streamError={detail:`SSE 事件解析失败：${error?.message||error}`};
      return;
    }
    if(eventName==="progress") onProgress(data);
    if(eventName==="result") finalResult=data;
    if(eventName==="error") streamError=data;
  }

  while(true){
    const {value,done}=await reader.read();
    buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
    const blocks=buffer.split(/\r?\n\r?\n/);
    buffer=blocks.pop()||"";
    for(const block of blocks){if(block.trim())consume(block)}
    if(done)break;
  }
  if(buffer.trim())consume(buffer);
  if(streamError){
    reportClientProgress(
      "stream_result",
      "failed",
      "后端分析执行失败",
      streamError.detail||"流式请求失败",
      {transport:"sse"},
    );
    throw new Error(streamError.detail||"流式请求失败");
  }
  if(!finalResult){
    reportClientProgress(
      "stream_result",
      "failed",
      "实时响应提前结束",
      "数据流已结束，但没有收到最终答案事件。",
      {transport:"sse"},
    );
    throw new Error("流式响应结束，但未收到最终答案");
  }
  reportClientProgress(
    "result_received",
    "completed",
    "最终结果已接收",
    "后端实时执行记录与最终答案均已接收。",
    {transport:"sse"},
  );
  return finalResult;
}

export async function getModelingStatus(projectId, targetMetric, scope){
  const params = new URLSearchParams({
    project_id:String(projectId),
    target_metric:targetMetric,
  });
  const r=await fetch(`/agent-api/api/v1/ml-ui/status?${params.toString()}`,{
    headers:scopeHeaders(scope),
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok) throw new Error(data?.detail||`建模状态读取失败 HTTP ${r.status}`);
  return data;
}
