function formatTime(value){
  if(!value)return "-";
  const date=new Date(value);
  return Number.isNaN(date.getTime())?String(value):date.toLocaleString("zh-CN",{
    month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false,
  });
}

export default function ChatHistoryPanel({
  conversations=[],
  currentId=null,
  loading=false,
  error="",
  disabled=false,
  onNew,
  onOpen,
  onRename,
  onDelete,
  onRefresh,
}){
  return <section className="conversationSidebar" aria-label="历史会话">
    <button className="sidebarNewChat" onClick={onNew} disabled={disabled}><span>＋</span><b>新建会话</b></button>
    <div className="conversationSidebarHead">
      <div><b>历史会话</b><span>{conversations.length} 个</span></div>
      <button title="刷新会话" aria-label="刷新会话" onClick={onRefresh} disabled={loading}>{loading?"…":"↻"}</button>
    </div>
    {error&&<div className="conversationSidebarError">{error}</div>}
    <div className="conversationSidebarList">
      {loading&&!conversations.length
        ? <div className="conversationSidebarEmpty">正在读取历史会话…</div>
        : !conversations.length
          ? <div className="conversationSidebarEmpty">还没有历史会话。<br/>发送第一条消息后会自动保存。</div>
          : conversations.map(item=><article className={item.conversation_id===currentId?"active":""} key={item.conversation_id}>
            <button className="conversationSidebarMain" onClick={()=>onOpen(item)} disabled={disabled} title={item.title||"未命名会话"}>
              <b>{item.title||"未命名会话"}</b>
              <span>{item.last_message_preview||"暂无内容"}</span>
              <small>{formatTime(item.updated_at)} · {item.message_count||0} 条</small>
            </button>
            <div className="conversationSidebarActions">
              <button title="重命名" aria-label={`重命名 ${item.title||"未命名会话"}`} onClick={()=>onRename(item)} disabled={disabled}>✎</button>
              <button className="danger" title="删除" aria-label={`删除 ${item.title||"未命名会话"}`} onClick={()=>onDelete(item)} disabled={disabled}>×</button>
            </div>
          </article>)}
    </div>
    <p className="conversationSidebarBoundary">会话按当前用户和公司隔离保存</p>
  </section>;
}
