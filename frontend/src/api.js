const KEY = "materials-agent-dev-scope";

export function getScope(){
  try {
    return {
      ...{userId:"local-test",companyId:"6a4b19f62d0e000027001eb8",projectIds:"115"},
      ...JSON.parse(localStorage.getItem(KEY)||"{}")
    };
  } catch {
    return {userId:"local-test",companyId:"6a4b19f62d0e000027001eb8",projectIds:"115"};
  }
}

export function saveScope(v){ localStorage.setItem(KEY,JSON.stringify(v)); }

function scopeHeaders(scope){
  return {
    "X-User-Id":scope.userId,
    "X-Company-Id":scope.companyId,
    "X-Project-Ids":scope.projectIds,
  };
}

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

export async function chat(message, history, scope, attachmentIds=[]){
  const r=await fetch("/agent-api/api/v1/chat-ui",{
    method:"POST",
    headers:{"Content-Type":"application/json",...scopeHeaders(scope)},
    body:JSON.stringify({
      message,
      history:history.slice(-12),
      attachment_ids:attachmentIds,
    })
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok) throw new Error(data?.detail||`HTTP ${r.status}`);
  return data;
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
