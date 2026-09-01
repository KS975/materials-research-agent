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
  if(!r.ok) throw new Error(data?.detail||`V0.2 请求失败 HTTP ${r.status}`);
  return data;
}

export function getFeedbackStatus({campaignId,projectId},scope){
  const params=new URLSearchParams();
  if(campaignId) params.set("campaign_id",campaignId);
  if(projectId) params.set("project_id",String(projectId));
  return jsonRequest(`/agent-api/api/v1/feedback-ui/status?${params.toString()}`,scope);
}

export function startFeedbackRound(campaignId,roundId,scope){
  return jsonRequest("/agent-api/api/v1/feedback-ui/start-round",scope,{
    method:"POST",body:JSON.stringify({campaign_id:campaignId,round_id:roundId}),
  });
}

export function submitFeedbackResult(payload,scope){
  return jsonRequest("/agent-api/api/v1/feedback-ui/result",scope,{
    method:"POST",body:JSON.stringify(payload),
  });
}

export function closeFeedbackRound(campaignId,roundId,scope){
  return jsonRequest("/agent-api/api/v1/feedback-ui/close-round",scope,{
    method:"POST",body:JSON.stringify({campaign_id:campaignId,round_id:roundId}),
  });
}

export function advanceFeedbackCampaign(campaignId,scope){
  return jsonRequest("/agent-api/api/v1/feedback-ui/advance",scope,{
    method:"POST",body:JSON.stringify({campaign_id:campaignId}),
  });
}

export function approveFeedbackModel(campaignId,scope){
  return jsonRequest("/agent-api/api/v1/feedback-ui/approve-model",scope,{
    method:"POST",body:JSON.stringify({campaign_id:campaignId}),
  });
}
