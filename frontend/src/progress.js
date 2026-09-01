export function createInitialAnalysisStep(){
  return {
    schema_version:"1.1",
    source:"client",
    stage:"stream_transport",
    status:"running",
    title:"连接实时分析通道",
    message:"请求已提交，正在连接后端 SSE 流式接口。",
    elapsed_ms:0,
  };
}

export function mergeProgressStep(steps,next,maxSteps=30){
  const current=Array.isArray(steps)?steps:[];
  const eventKey=item=>{
    const attempt=item?.attempt!=null?`:attempt-${item.attempt}`:"";
    return `${item?.source||"backend"}:${item?.stage||"unknown"}${attempt}`;
  };
  const key=eventKey(next);
  const index=current.findIndex(item=>
    eventKey(item)===key
  );
  const merged=index<0
    ? [...current,next]
    : current.map((item,itemIndex)=>itemIndex===index?{...item,...next}:item);
  return merged.slice(-Math.max(1,Number(maxSteps)||30));
}
