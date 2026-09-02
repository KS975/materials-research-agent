import { useEffect, useMemo, useState } from "react";
import {
  getDashboardFields,
  getDashboardProjects,
  getDashboardSampleDetail,
  getDashboardSamples,
  getDashboardSummary,
} from "./api";
import {
  buildFieldPrompt,
  buildProjectPrompt,
  buildSamplePrompt,
  DASHBOARD_TABS,
  FIELD_SECTION_LABELS,
  isHistoricalImportProject,
  projectDisplayName,
} from "./dashboard";

const PAGE_SIZE=20;

function formatDate(value){
  if(!value)return "-";
  const date=new Date(value);
  return Number.isNaN(date.getTime())?String(value):date.toLocaleString("zh-CN",{hour12:false});
}

function countLabel(value){return Number(value||0).toLocaleString("zh-CN")}

function Empty({children}){return <div className="dbEmpty">{children}</div>}
function Loading(){return <div className="dbLoading">正在读取当前公司授权数据…</div>}

function ScopeBanner({scope}){
  const all=scope?.project_mode==="company_all_projects";
  return <div className="dbScopeBanner">
    <span>当前公司</span><b>{scope?.company_id||"-"}</b>
    <em>{all?"公司内全部项目":`授权项目：${(scope?.project_ids||[]).join("、")||"无"}`}</em>
  </div>
}

function Overview({summary,onTab}){
  if(!summary)return <Loading/>;
  return <div className="dbOverview">
    <ScopeBanner scope={summary.scope}/>
    <div className="dbMetricGrid">
      <div><span>授权项目 · 历史导入 {countLabel(summary.historical_import_project_count)}</span><b>{countLabel(summary.project_count)}</b></div>
      <div><span>授权样品</span><b>{countLabel(summary.sample_count)}</b></div>
      <div className="wide"><span>样品最近更新</span><b>{formatDate(summary.latest_sample_update)}</b></div>
    </div>
    <div className="dbStartCards">
      <button onClick={()=>onTab("projects")}><b>从项目开始</b><span>查看项目及其样品数量</span></button>
      <button onClick={()=>onTab("samples")}><b>查找样品</b><span>按名称或ID搜索并选择</span></button>
      <button onClick={()=>onTab("fields")}><b>看看能查什么</b><span>浏览配方、工艺和性能字段</span></button>
    </div>
    <p className="dbBoundary">这里只浏览当前公司及当前项目权限范围内的业务 MySQL 数据，所有操作均为只读。</p>
  </div>
}

function ProjectBrowser({data,loading,query,setQuery,onSearch,onOpenSamples,onUsePrompt}){
  return <div className="dbBrowser">
    <div className="dbSearch"><input value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==="Enter"&&onSearch()} placeholder="搜索项目名称或项目ID"/><button onClick={onSearch}>搜索</button></div>
    <div className="dbResultLine"><b>{countLabel(data?.total)} 个项目</b><span>仅当前公司授权范围</span></div>
    {loading?<Loading/>:!data?.projects?.length?<Empty>没有找到符合条件的项目</Empty>:<div className="dbProjectList">
      {data.projects.map(project=>{
        const historical=project.project_origin==="history_import"||isHistoricalImportProject(project.id);
        return <div className="dbProjectCard" key={project.id}>
          <div><small>PROJECT {project.id}{historical?" · 历史导入":""}</small><b>{projectDisplayName(project)}</b><span>{project.describe||(historical?"由历史记录导入，未建立独立项目档案":"暂无项目描述")}</span></div>
          <div className="dbProjectStats"><b>{countLabel(project.sample_count)}</b><span>样品</span></div>
          <div className="dbRowActions"><button onClick={()=>onOpenSamples(project)}>查看样品</button><button onClick={()=>onUsePrompt(buildProjectPrompt(project))}>带入对话</button></div>
        </div>;
      })}
    </div>}
  </div>
}

function FieldGroup({section,items,totalSamples,onUsePrompt}){
  if(!items?.length)return null;
  return <div className="dbFieldGroup">
    <h3>{FIELD_SECTION_LABELS[section]||section}<span>{items.length} 个字段</span></h3>
    <div>{items.map(field=>{
      const observed=Number(field.observed_sample_count||0);
      const coverage=totalSamples?Math.round(observed*100/totalSamples):0;
      return <button className="dbFieldRow" key={`${section}-${field.name}`} onClick={()=>onUsePrompt(buildFieldPrompt(section,field))}>
        <span><b>{field.name}</b><small>{(field.units||[]).join(" / ")||"未记录单位"}</small></span>
        <em>{observed} 条 · {coverage}%</em>
      </button>;
    })}</div>
  </div>
}

function FieldBrowser({data,loading,query,setQuery,section,setSection,onSearch,onUsePrompt}){
  return <div className="dbBrowser">
    <div className="dbSearch dbFieldSearch"><select value={section} onChange={e=>setSection(e.target.value)}><option value="all">全部类别</option>{Object.entries(FIELD_SECTION_LABELS).map(([key,label])=><option value={key} key={key}>{label}</option>)}</select><input value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==="Enter"&&onSearch()} placeholder="搜索PC、注塑温度、冲击强度…"/><button onClick={onSearch}>搜索</button></div>
    <div className="dbResultLine"><b>来源于 {countLabel(data?.source_sample_count)} 条授权样品</b><span>{data?.scan_complete?"完整扫描":"扫描未完成"}</span></div>
    {loading?<Loading/>:!data?<Empty>点击搜索读取字段目录</Empty>:<div className="dbFieldList">
      {Object.entries(data.sections||{}).map(([key,items])=><FieldGroup key={key} section={key} items={items} totalSamples={data.source_sample_count} onUsePrompt={onUsePrompt}/>)}
      {!Object.values(data.sections||{}).some(items=>items?.length)&&<Empty>没有找到符合条件的字段</Empty>}
    </div>}
  </div>
}

function DetailFields({title,items}){
  return <section className="dbDetailSection"><h4>{title}<span>{items?.length||0}</span></h4>{!items?.length?<p>没有结构化记录</p>:<div>{items.slice(0,30).map((item,index)=><div key={`${item.raw_key||item.name}-${index}`}><span>{item.name||item.raw_key}</span><b>{String(item.value??"-")} {item.unit||""}</b></div>)}</div>}</section>
}

function SampleDetail({detail,onClose,onUsePrompt}){
  if(!detail)return null;
  const data=detail.data||{};
  const sample=data.sample||{};
  const historical=isHistoricalImportProject(sample.project_id);
  return <div className="dbDetail">
    <div className="dbDetailHead"><div><small>SAMPLE {sample.id}</small><h3>{sample.name||"未命名样品"}</h3><p>Project {sample.project_id}{historical?" · 历史导入":""} · {sample.sample_type||"类型未记录"}</p></div><button onClick={onClose}>返回列表</button></div>
    <div className="dbDetailActions"><button onClick={()=>onUsePrompt(buildSamplePrompt("profile",[sample]))}>查看完整信息</button><button onClick={()=>onUsePrompt(buildSamplePrompt("similar",[sample]))}>找相似样品</button><button onClick={()=>onUsePrompt(buildSamplePrompt("history",[sample]))}>查历史案例</button></div>
    <DetailFields title="配方" items={data.formula}/><DetailFields title="工艺" items={data.process}/><DetailFields title="性能" items={data.performance}/>
    <section className="dbDetailSection"><h4>测试条件<span>{Object.keys(data.conditions||{}).length}</span></h4>{!Object.keys(data.conditions||{}).length?<p>没有结构化记录</p>:<div>{Object.entries(data.conditions||{}).map(([key,value])=><div key={key}><span>{key}</span><b>{String(value)}</b></div>)}</div>}</section>
  </div>
}

function SampleActions({selected,onUsePrompt,onClear}){
  if(!selected.length)return null;
  return <div className="dbSelectionBar"><div><b>已选择 {selected.length} 个样品</b><span>{selected.map(x=>`${x.id}·${x.name||"未命名"}`).join("、")}</span></div><div>
    {selected.length===1?<><button onClick={()=>onUsePrompt(buildSamplePrompt("profile",selected))}>查看</button><button onClick={()=>onUsePrompt(buildSamplePrompt("similar",selected))}>找相似</button><button onClick={()=>onUsePrompt(buildSamplePrompt("history",selected))}>历史案例</button></>:<><button onClick={()=>onUsePrompt(buildSamplePrompt("compare",selected))}>比较</button><button onClick={()=>onUsePrompt(buildSamplePrompt("formula",selected))}>配方差异</button><button onClick={()=>onUsePrompt(buildSamplePrompt("process",selected))}>工艺差异</button></>}
    <button className="quiet" onClick={onClear}>清空</button>
  </div></div>
}

function SampleBrowser({data,loading,query,setQuery,onSearch,page,setPage,projectFilter,onClearProject,selected,onToggle,onDetail,onUsePrompt}){
  const total=Number(data?.total||0);
  const pages=Math.max(1,Math.ceil(total/PAGE_SIZE));
  return <div className="dbBrowser dbSamplesBrowser">
    {projectFilter&&<div className="dbProjectFilter"><span>当前项目</span><b>{projectFilter.id} · {projectDisplayName(projectFilter)}</b><button onClick={onClearProject}>查看全部</button></div>}
    <div className="dbSearch"><input value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==="Enter"&&onSearch()} placeholder="搜索样品名称或样品ID"/><button onClick={onSearch}>搜索</button></div>
    <div className="dbResultLine"><b>{countLabel(total)} 个样品</b><span>第 {page}/{pages} 页</span></div>
    {loading?<Loading/>:!data?.samples?.length?<Empty>没有找到符合条件的样品</Empty>:<div className="dbSampleList">{data.samples.map(sample=>{
      const checked=selected.some(x=>x.id===sample.id);
      const counts=sample.field_counts||{};
      const historical=sample.project_origin==="history_import"||isHistoricalImportProject(sample.project_id);
      return <div className={`dbSampleCard ${checked?"selected":""}`} key={sample.id}>
        <button className="dbSelect" onClick={()=>onToggle(sample)} aria-label={checked?"取消选择":"选择样品"}>{checked?"✓":"＋"}</button>
        <button className="dbSampleMain" onClick={()=>onDetail(sample)}><small>{sample.id} · PROJECT {sample.project_id??"-"}{historical?" · 历史导入":""}</small><b>{sample.name||"未命名样品"}</b><span>{sample.project_name||"项目名称未记录"} · {sample.sample_type||"类型未记录"}</span><em>配方 {counts.formula||0} · 工艺 {counts.process||0} · 性能 {counts.performance||0} · 条件 {counts.conditions||0}</em></button>
      </div>;
    })}</div>}
    <div className="dbPager"><button disabled={page<=1} onClick={()=>setPage(page-1)}>上一页</button><span>{page} / {pages}</span><button disabled={page>=pages} onClick={()=>setPage(page+1)}>下一页</button></div>
    <SampleActions selected={selected} onUsePrompt={onUsePrompt} onClear={()=>selected.forEach(onToggle)}/>
  </div>
}

export default function DatabaseNavigator({open,scope,onClose,onUsePrompt,onSummary}){
  const [tab,setTab]=useState("overview");
  const [summary,setSummary]=useState(null);
  const [projects,setProjects]=useState(null);
  const [samples,setSamples]=useState(null);
  const [fields,setFields]=useState(null);
  const [projectQuery,setProjectQuery]=useState("");
  const [sampleQuery,setSampleQuery]=useState("");
  const [fieldQuery,setFieldQuery]=useState("");
  const [fieldSection,setFieldSection]=useState("all");
  const [projectFilter,setProjectFilter]=useState(null);
  const [selected,setSelected]=useState([]);
  const [page,setPage]=useState(1);
  const [detail,setDetail]=useState(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState("");
  const scopeKey=useMemo(()=>JSON.stringify([scope?.companyId,scope?.projectIds]),[scope]);

  async function loadSummary(){
    setLoading(true);setError("");
    try{const data=await getDashboardSummary(scope);setSummary(data);onSummary?.(data)}catch(e){setError(e.message)}finally{setLoading(false)}
  }
  async function loadProjects(){
    setLoading(true);setError("");
    try{setProjects(await getDashboardProjects({q:projectQuery,limit:50,offset:0},scope))}catch(e){setError(e.message)}finally{setLoading(false)}
  }
  async function loadSamples(nextPage=page){
    setLoading(true);setError("");
    try{setSamples(await getDashboardSamples({q:sampleQuery,projectId:projectFilter?.id,limit:PAGE_SIZE,offset:(nextPage-1)*PAGE_SIZE},scope))}catch(e){setError(e.message)}finally{setLoading(false)}
  }
  async function loadFields(){
    setLoading(true);setError("");
    try{setFields(await getDashboardFields({q:fieldQuery,section:fieldSection},scope))}catch(e){setError(e.message)}finally{setLoading(false)}
  }
  async function openDetail(sample){
    setLoading(true);setError("");
    try{setDetail(await getDashboardSampleDetail(sample.id,scope))}catch(e){setError(e.message)}finally{setLoading(false)}
  }

  useEffect(()=>{
    setSummary(null);setProjects(null);setSamples(null);setFields(null);setDetail(null);setSelected([]);setPage(1);
    if(open)loadSummary();
  },[scopeKey]);
  useEffect(()=>{if(open&&!summary)loadSummary()},[open]);
  useEffect(()=>{if(!open||detail)return;if(tab==="projects"&&!projects)loadProjects();if(tab==="samples"&&!samples)loadSamples();if(tab==="fields"&&!fields)loadFields()},[open,tab,detail]);
  useEffect(()=>{if(open&&tab==="samples"&&!detail)loadSamples(page)},[page,projectFilter]);

  function changeTab(value){setDetail(null);setTab(value)}
  function openProjectSamples(project){setProjectFilter(project);setPage(1);setSamples(null);setTab("samples")}
  function toggleSample(sample){setSelected(current=>current.some(x=>x.id===sample.id)?current.filter(x=>x.id!==sample.id):current.length>=2?[current[1],sample]:[...current,sample])}
  function usePrompt(value){if(!value)return;onUsePrompt(value)}

  if(!open)return null;
  return <div className="dbBackdrop" onMouseDown={event=>event.target===event.currentTarget&&onClose()}>
    <div className="dbNavigator" role="dialog" aria-modal="true" aria-label="数据库导航器">
      <div className="dbNavHead"><div><span>DATABASE NAVIGATOR</span><h2>数据库浏览</h2><p>当前公司授权数据 · 只读</p></div><button onClick={onClose}>×</button></div>
      <nav>{DASHBOARD_TABS.map(item=><button className={tab===item.id?"active":""} key={item.id} onClick={()=>changeTab(item.id)}>{item.label}</button>)}</nav>
      <div className="dbNavBody">
        {error&&<div className="dbError">{error}</div>}
        {detail?<SampleDetail detail={detail} onClose={()=>setDetail(null)} onUsePrompt={usePrompt}/>:<>
          {tab==="overview"&&(loading&&!summary?<Loading/>:<Overview summary={summary} onTab={changeTab}/>)}
          {tab==="projects"&&<ProjectBrowser data={projects} loading={loading} query={projectQuery} setQuery={setProjectQuery} onSearch={loadProjects} onOpenSamples={openProjectSamples} onUsePrompt={usePrompt}/>} 
          {tab==="samples"&&<SampleBrowser data={samples} loading={loading} query={sampleQuery} setQuery={setSampleQuery} onSearch={()=>{setPage(1);loadSamples(1)}} page={page} setPage={setPage} projectFilter={projectFilter} onClearProject={()=>{setProjectFilter(null);setPage(1);setSamples(null)}} selected={selected} onToggle={toggleSample} onDetail={openDetail} onUsePrompt={usePrompt}/>} 
          {tab==="fields"&&<FieldBrowser data={fields} loading={loading} query={fieldQuery} setQuery={setFieldQuery} section={fieldSection} setSection={setFieldSection} onSearch={loadFields} onUsePrompt={usePrompt}/>} 
        </>}
      </div>
    </div>
  </div>;
}
