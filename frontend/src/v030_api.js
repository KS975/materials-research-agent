import { apiFetch } from "./api";

async function jsonRequest(url, scope, options={}){
  const r=await apiFetch(url,{
    ...options,
    headers:{
      ...(options.body?{"Content-Type":"application/json"}:{}),
      ...(options.headers||{}),
    },
  });
  const data=await r.json().catch(()=>null);
  if(!r.ok) throw new Error(data?.detail||`V0.3 请求失败 HTTP ${r.status}`);
  return data;
}

export function getAutonomyStatus({campaignId,projectId},scope){
  const params=new URLSearchParams();
  if(campaignId) params.set("campaign_id",campaignId);
  if(projectId) params.set("project_id",String(projectId));
  return jsonRequest(`/agent-api/api/v1/autonomy-ui/status?${params.toString()}`,scope);
}

export function operatorOverride(payload,scope){
  return jsonRequest("/agent-api/api/v1/autonomy-ui/operator",scope,{
    method:"POST",
    body:JSON.stringify(payload),
  });
}
