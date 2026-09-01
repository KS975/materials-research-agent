// Keep the default request same-origin with the Vite frontend.
// frontend/vite.config.js proxies /api -> http://127.0.0.1:8000,
// matching the existing V0.2/V0.3 API clients and avoiding browser CORS.
import { apiFetch } from "./api";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/agent-api";

export async function getMondayDemoStatus(){
  const r=await apiFetch(`${API_BASE}/api/v1/demo-ui/status`);
  const body=await r.json().catch(()=>({}));
  if(!r.ok){
    if(r.status===404){
      throw new Error("演示模式 API 尚未加载（HTTP 404）。请重启 FastAPI 后端后再点“演示模式”。");
    }
    throw new Error(body.detail||body.error||`HTTP ${r.status}`);
  }
  return body;
}
