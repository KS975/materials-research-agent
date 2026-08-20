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
import {
  advanceFeedbackCampaign,
  approveFeedbackModel,
  closeFeedbackRound,
  getFeedbackStatus,
  startFeedbackRound,
  submitFeedbackResult,
} from "./v020_api";
import { getAutonomyStatus, operatorOverride } from "./v030_api";
import { getMondayDemoStatus } from "./demo_api";

const versions=[
  ["V0.1.1","Agent + MySQL","查询 · 对比 · 分析","done"],
  ["V0.1.2","File + Knowledge + RAG","附件 · 历史检索 · 联合分析","done"],
  ["V0.1.3","Dataset + ML","准入 · 训练 · 评估 · 适用域","done"],
  ["V0.1.4","Optimization + BO","逆向设计 · Pareto · 下一轮实验","done"],
  ["V0.2","Experiment Feedback Loop","Campaign · 实测回流 · 重训 · 多轮闭环","done"],
  ["V0.3","Autonomous Orchestration","设备 · Safety · Telemetry · Crash/Resume","on"]
];
const quick=[
  {label:"单位真实数据",text:"查看单位真实数据概况",type:"chat"},
  {label:"V0.3 自主闭环",text:"Project 9036：查看 V0.3 自主实验状态",type:"chat"},
  {label:"V0.2 闭环状态",text:"Project 9026：查看 V0.2 闭环状态",type:"chat"},
  {label:"V0.2 实验回填",text:"Project 9024：查看 V0.2 实验回填状态",type:"chat"},
  {label:"逆向设计 · T17",text:"Project 9016：冲击强度 >= 43、MFR >= 8.5，推荐5组方案",type:"chat"},
  {label:"联合分析",text:"3811 的冲击强度比 3809 低很多，历史上有没有类似问题？结合数据库数据和历史报告分析一下。",type:"chat"},
];
const welcome={
  id:"welcome",
  role:"assistant",
  content:`你好，我是材数智能体。

V0.3 UI 已接入自主实验运行态：Protocol、Scheduler、Telemetry、Safety、自动结果回流、多轮闭环与 Crash/Resume；当前仍只连接 Simulator，不代表真实设备验证。`
};
const id=()=>Date.now()+"-"+Math.random().toString(16).slice(2);

function Logo(){return <div className="logo">◇</div>}

function StatusPill({value}){
  const normalized=String(value||"UNKNOWN").toUpperCase();
  const good=["PASS","SUCCESS","IN_DOMAIN","COMPLETED","PROMOTE","ACTIVE","RECOVERED","SAFE"].includes(normalized);
  const warn=["CONDITIONAL_PASS","PARTIAL_FEASIBLE_DESIGN","BORDERLINE","PLANNED","RUNNING","PARTIALLY_COMPLETED","REVIEW_REQUIRED","PAUSED","CANDIDATE","KEEP_INCUMBENT","PROCESSING","HEATING","COOLING","MEASURING","MATERIAL_LOADING","RECOVERY_REQUIRED","WAITING"].includes(normalized);
  const bad=["FAIL","NO_FEASIBLE_DESIGN","OUT_OF_DOMAIN","BLOCKED","FAILED","INVALID","CANCELLED","SAFETY_STOP","ERROR","TIMEOUT"].includes(normalized);
  const cls=good?"good":warn?"warn":bad?"bad":"neutral";
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


function normalizedBOConditions(row){
  const explicit=Array.isArray(row?.experiment_conditions)?row.experiment_conditions.filter(Boolean):[];
  if(explicit.length){
    return explicit.map(item=>({
      name:item.name||item.label||"设计变量",
      group:item.group||"设计变量",
      label:item.label||item.name||"设计变量",
      value:item.value,
      unit:item.unit||"",
    }));
  }
  return Object.entries(row?.features||{}).map(([name,value])=>{
    const parts=String(name).split("::");
    const prefix=parts.length>1?parts[0]:"";
    const label=parts.length>1?parts.slice(1).join("::"):name;
    const group=prefix==="formula"?"配方":prefix==="process"?"工艺":prefix==="condition"?"测试条件":"设计变量";
    return {name,group,label,value,unit:""};
  });
}


function OptimizationCards({data}){
  if(!data) return null;

  if(data.kind==="v014_inverse_design"){
    const counts=data.counts||{};
    const designs=data.design_cards||[];
    const misses=data.near_miss_candidates||[];
    return <div className="optCards">
      <div className="optCard optSummary">
        <div className="mlCardHead"><div><small>INVERSE DESIGN · T17</small><b>逆向设计</b></div><StatusPill value={data.status}/></div>
        <div className="metricGrid">
          <Metric label="HARD-valid" value={counts.generated_hard_valid}/>
          <Metric label="IN_DOMAIN" value={counts.trusted_in_domain}/>
          <Metric label="全部达标" value={counts.qualified_all_targets}/>
          <Metric label="Pareto Front" value={counts.pareto_front}/>
        </div>
        <p className="cardFoot">正式推荐仅允许 IN_DOMAIN；NO_FEASIBLE_DESIGN 时不会补造方案。</p>
      </div>

      {!!designs.length&&<div className="designList">
        {designs.map(card=><div className="designCard" key={card.candidate_id}>
          <div className="designHead"><b>#{card.recommendation_rank} · {card.candidate_id}</b><span>Pareto {card.pareto_rank}</span></div>
          <div className="predictionGrid">{Object.entries(card.predictions||{}).map(([metric,value])=><div key={metric}><span>{metric}</span><b>{typeof value==="number"?value.toFixed(3):value}</b><small>margin {typeof card.target_margins?.[metric]==="number"?`${card.target_margins[metric]>=0?"+":""}${card.target_margins[metric].toFixed(3)}`:"-"}</small></div>)}</div>
          <div className="designMeta"><StatusPill value={card.applicability_domain?.status}/><span>风险 {card.applicability_domain?.risk||"-"}</span><span>soft penalty {Number(card.soft_penalty||0).toFixed(3)}</span></div>
          <details><summary>配方 / 工艺</summary><pre>{JSON.stringify(card.features,null,2)}</pre></details>
        </div>)}
      </div>}

      {data.status==="NO_FEASIBLE_DESIGN"&&<div className="optCard noFeasible"><b>没有可信可行设计</b><p>{data.answer}</p>{!!misses.length&&<div className="nearMisses">{misses.slice(0,3).map(x=><span key={x.candidate_id}>{x.candidate_id} · shortfall {Number(x.total_normalized_threshold_shortfall||0).toFixed(3)}</span>)}</div>}</div>}
      <div className="mlDisclaimer">T17 推荐值来自已落盘 sklearn 模型；fixture 结果只用于工程验收，不代表真实材料科学结论。</div>
    </div>;
  }

  if(data.kind==="v014_bayesian_optimization"){
    const filtering=data.candidate_filtering||{};
    const bo=data.bayesian_optimization||{};
    const obs=data.observations||{};
    const next=data.next_experiments||[];
    return <div className="optCards">
      <div className="optCard optSummary">
        <div className="mlCardHead"><div><small>BAYESIAN OPTIMIZATION · T18</small><b>下一轮实验</b></div><StatusPill value={data.status}/></div>
        <div className="metricGrid">
          <Metric label="历史实验" value={obs.rows}/>
          <Metric label="历史最好" value={typeof obs.best_observed==="number"?obs.best_observed.toFixed(3):"-"}/>
          <Metric label="BO 可选" value={filtering.eligible_for_bo}/>
          <Metric label="OOD 排除" value={filtering.out_of_domain_excluded}/>
        </div>
        <div className="boPolicy"><span>{bo.acquisition||"EI"}</span><span>{bo.batch_strategy||"-"}</span><span>adjusted acquisition</span></div>
      </div>

      <div className="experimentList">{next.map(row=>{
        const conditions=normalizedBOConditions(row);
        return <div className="experimentCard" key={row.candidate_id}>
          <div className="designHead"><b>第 {row.round} 组 · {row.candidate_id}</b><StatusPill value={row.applicability_domain?.status}/></div>
          <div className="predictionGrid threeOpt"><div><span>GP mean</span><b>{Number(row.posterior_mean).toFixed(3)}</b></div><div><span>GP std</span><b>{Number(row.posterior_std).toFixed(3)}</b></div><div><span>{bo.acquisition||"EI"}</span><b>{Number(row.adjusted_acquisition).toFixed(3)}</b><small>raw {Number(row.acquisition_value).toFixed(3)}</small></div></div>
          <div className="designMeta"><span>风险 {row.applicability_domain?.risk||"-"}</span><span>soft penalty {Number(row.soft_penalty||0).toFixed(3)}</span></div>
          <div className="boConditions">
            <div className="boConditionsHead">
              <b>推荐配方 / 工艺条件</b>
              <span>{conditions.length ? `${conditions.length} 个变量` : "条件数据缺失"}</span>
            </div>
            {conditions.length?
              <div className="boConditionGrid">{conditions.map((item,index)=><div className="boConditionItem" key={`${row.candidate_id}-${item.name}-${index}`}>
                <small>{item.group}</small>
                <span>{item.label}</span>
                <b>{typeof item.value==="number"&&Number.isFinite(item.value)?item.value.toFixed(Number.isInteger(item.value)?0:3):String(item.value??"-")}{item.unit?` ${item.unit}`:""}</b>
              </div>)}</div>
              :<div className="boConditionMissing">本组推荐缺少设计变量，请重新运行 T18；不会用空条件代替实验方案。</div>}
          </div>
          <details className="boRawConditions"><summary>查看原始 features JSON</summary><pre>{JSON.stringify(row.features||{},null,2)}</pre></details>
        </div>
      })}</div>
      <div className="mlDisclaimer">posterior mean / std 是 Gaussian Process 模型估计与不确定性，不是未来真实实验测量值。</div>
    </div>;
  }

  return null;
}


function formatNum(value,digits=3){
  return typeof value==="number"&&Number.isFinite(value)?value.toFixed(digits):"-";
}

function ExperimentFeedbackRow({campaignId,roundId,roundStatus,item,scope,onUpdated}){
  const [status,setStatus]=useState("COMPLETED");
  const [values,setValues]=useState({});
  const [reason,setReason]=useState("");
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  const metrics=item.required_metrics||[];
  const canSubmit=["RUNNING","PARTIALLY_COMPLETED"].includes(roundStatus);

  async function submit(){
    if(!canSubmit||busy)return;
    const measurements={};
    if(status==="COMPLETED"){
      for(const metric of metrics){
        const raw=values[metric];
        if(raw===""||raw==null){setError(`请填写 ${metric} 实测值`);return;}
        const value=Number(raw);
        if(!Number.isFinite(value)){setError(`${metric} 必须是数值`);return;}
        measurements[metric]=value;
      }
    }
    if((status==="FAILED"||status==="INVALID")&&!reason.trim()){
      setError("FAILED / INVALID 必须填写原因");return;
    }
    setBusy(true);setError("");
    try{
      const r=await submitFeedbackResult({
        campaign_id:campaignId,round_id:roundId,candidate_id:item.candidate_id,
        status,test_condition_signature:item.expected_test_condition_signature,
        measurements,units:status==="COMPLETED"?(item.units||{}):{},
        failure_reason:reason,notes:"V0.2 UI feedback",
      },scope);
      onUpdated(r.data);
    }catch(e){setError(e.message)}finally{setBusy(false)}
  }

  return <div className="feedbackExperiment">
    <div className="designHead"><b>{item.candidate_id}</b><StatusPill value={item.status}/></div>
    <div className="feedbackPrediction">{Object.entries(item.prediction||{}).map(([metric,p])=><span key={metric}>{metric} 预测 <b>{formatNum(p?.value)}</b>{typeof p?.posterior_std==="number"&&<small> ± {formatNum(p.posterior_std)}</small>}</span>)}</div>
    <details><summary>配方 / 工艺</summary><pre>{JSON.stringify(item.features||{},null,2)}</pre></details>
    <div className="feedbackForm">
      <select value={status} onChange={e=>setStatus(e.target.value)} disabled={!canSubmit||busy}>
        <option value="COMPLETED">COMPLETED · 已完成</option>
        <option value="FAILED">FAILED · 实验失败</option>
        <option value="INVALID">INVALID · 数据无效</option>
        <option value="NOT_TESTED">NOT_TESTED · 未测试</option>
      </select>
      {status==="COMPLETED"&&metrics.map(metric=><label key={metric}><span>{metric} · {item.units?.[metric]||""}</span><input type="number" step="any" value={values[metric]??""} onChange={e=>setValues(v=>({...v,[metric]:e.target.value}))} placeholder="输入真实实测值" disabled={!canSubmit||busy}/></label>)}
      {(status==="FAILED"||status==="INVALID")&&<label><span>原因</span><input value={reason} onChange={e=>setReason(e.target.value)} placeholder="填写失败 / 无效原因" disabled={!canSubmit||busy}/></label>}
      <div className="feedbackCondition">测试条件：{item.expected_test_condition_signature||"-"}</div>
      <button className="feedbackAction" disabled={!canSubmit||busy} onClick={submit}>{busy?"提交中…":"提交实验结果"}</button>
      {error&&<div className="feedbackInlineError">{error}</div>}
    </div>
  </div>;
}

function FeedbackCards({data,scope}){
  const isFeedback=data?.kind==="v020_feedback_loop";
  const [view,setView]=useState(isFeedback?data:null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  useEffect(()=>{if(isFeedback)setView(data)},[data,isFeedback]);
  if(!isFeedback||!view) return null;
  const campaign=view.campaign||{};
  const round=view.latest_round||{};
  const datasets=view.datasets||[];
  const evaluation=view.evaluation;
  const promotion=view.model_promotion;
  const registry=view.model_registry;
  const checkpoint=view.checkpoint;
  const e2e=view.end_to_end;
  const pending=round.pending_experiments||[];

  async function action(fn){
    setBusy(true);setError("");
    try{const r=await fn();setView(r.data)}catch(e){setError(e.message)}finally{setBusy(false)}
  }

  return <div className="feedbackCards">
    <div className="feedbackLiveAnswer">{view.answer}</div>
    <div className="feedbackCard feedbackHero">
      <div className="mlCardHead"><div><small>EXPERIMENT FEEDBACK LOOP · V0.2</small><b>{campaign.name||"研发闭环"}</b></div><StatusPill value={campaign.status}/></div>
      <div className="metricGrid">
        <Metric label="当前轮次" value={campaign.current_round_no}/>
        <Metric label="Round 状态" value={round.status||"-"}/>
        <Metric label="最新 Dataset" value={view.latest_dataset?.dataset_version||"-"} sub={view.latest_dataset?.row_count!=null?`${view.latest_dataset.row_count} rows`:null}/>
        <Metric label="Active Model" value={view.summary?.active_model_version||"-"}/>
      </div>
      <div className="feedbackToolbar">
        {round.can_start&&<button disabled={busy} onClick={()=>action(()=>startFeedbackRound(campaign.campaign_id,round.round_id,scope))}>开始本轮</button>}
        {round.can_close_round&&<button disabled={busy} onClick={()=>action(()=>closeFeedbackRound(campaign.campaign_id,round.round_id,scope))}>关闭本轮</button>}
        {round.can_advance&&<button disabled={busy||!view.advance_inputs?.ready} title={view.advance_inputs?.ready?"通过 T25 checkpoint 生成下一轮":"缺少 candidate_pool.csv / gate.json"} onClick={()=>action(()=>advanceFeedbackCampaign(campaign.campaign_id,scope))}>生成下一轮</button>}
        {promotion?.decision==="PROMOTE"&&registry?.models?.[promotion.challenger_model_version]?.status==="CANDIDATE"&&<button className="approveBtn" disabled={busy} onClick={()=>action(()=>approveFeedbackModel(campaign.campaign_id,scope))}>批准模型晋级</button>}
        <button disabled={busy} onClick={()=>action(()=>getFeedbackStatus({campaignId:campaign.campaign_id},scope))}>刷新</button>
      </div>
      {round.can_advance&&!view.advance_inputs?.ready&&<p className="cardFoot">生成下一轮前需准备 .runtime/v020/ui_inputs/project_{campaign.project_id}/candidate_pool.csv + gate.json。</p>}
      {view.advance_inputs?.source==="fixture_fallback"&&<p className="fixtureFlag">当前 advance 输入来自 fixture fallback，仅用于工程验收。</p>}
      {error&&<div className="feedbackInlineError">{error}</div>}
    </div>

    <div className="feedbackCard">
      <div className="mlCardHead"><div><small>CAMPAIGN / ROUND</small><b>实验轮次</b></div><StatusPill value={round.status||campaign.status}/></div>
      <div className="roundTimeline">{(view.rounds||[]).map(r=><div className="roundItem" key={r.round_id}><div><b>R{String(r.round_no).padStart(2,"0")}</b><span>{r.dataset_version}</span></div><StatusPill value={r.status}/><small>{r.progress?.completed??0}/{r.planned_experiments??0} completed</small></div>)}</div>
    </div>

    {evaluation&&<div className="feedbackCard">
      <div className="mlCardHead"><div><small>PREDICTION VS MEASUREMENT · T21</small><b>预测 vs 实测</b></div><StatusPill value={evaluation.round_status}/></div>
      <div className="metricGrid">
        <Metric label="MAE" value={formatNum(evaluation.aggregate?.mae)}/>
        <Metric label="RMSE" value={formatNum(evaluation.aggregate?.rmse)}/>
        <Metric label="R²" value={formatNum(evaluation.aggregate?.r2)}/>
        <Metric label="Bias" value={formatNum(evaluation.aggregate?.bias_actual_minus_predicted)}/>
      </div>
      <p className="cardFoot">{evaluation.round_id} · 2σ miss {evaluation.uncertainty?.overconfident_2sigma_miss_count??0}</p>
    </div>}

    <div className="feedbackCard">
      <div className="mlCardHead"><div><small>DATASET LINEAGE · T22</small><b>数据集版本</b></div><StatusPill value={datasets.length?"PASS":"NO_DATA"}/></div>
      <div className="datasetLineage">{datasets.map((d,i)=><div key={d.dataset_version}><b>{d.dataset_version}</b><span>{d.row_count} rows</span>{i>0&&<small>+{d.added_row_count??0}</small>}</div>)}</div>
    </div>

    {promotion&&<div className="feedbackCard">
      <div className="mlCardHead"><div><small>MODEL GOVERNANCE · T23</small><b>模型晋级</b></div><StatusPill value={promotion.decision}/></div>
      <div className="modelCompare">
        <div><span>Incumbent</span><b>{promotion.incumbent_model_version}</b><small>RMSE {formatNum(promotion.incumbent?.holdout?.rmse)}</small></div>
        <div><span>Challenger</span><b>{promotion.challenger_model_version}</b><small>RMSE {formatNum(promotion.challenger?.holdout?.rmse)}</small></div>
        <div><span>Active</span><b>{registry?.active_model_version||"-"}</b><small>不会自动替换</small></div>
      </div>
      {!!promotion.reasons?.length&&<p className="cardFoot">{promotion.reasons[0]}</p>}
    </div>}

    {checkpoint&&<div className="feedbackCard">
      <div className="mlCardHead"><div><small>CHECKPOINT · T25</small><b>断点恢复</b></div><StatusPill value={checkpoint.status}/></div>
      <div className="checkpointSteps">{(checkpoint.completed_steps||[]).map(step=><span key={step}>✓ {step}</span>)}</div>
      <p className="cardFoot">resume count {checkpoint.resume_count??0}</p>
    </div>}

    {e2e&&<div className="feedbackCard endToEndCard">
      <div className="mlCardHead"><div><small>END-TO-END · T26</small><b>闭环总验收</b></div><StatusPill value={e2e.decision}/></div>
      <div className="metricGrid">
        <Metric label="Rounds" value={e2e.round_count}/>
        <Metric label="总实验" value={e2e.experiment_integrity?.experiment_count}/>
        <Metric label="初始 Best" value={formatNum(e2e.best_so_far?.initial)}/>
        <Metric label="最终 Best" value={formatNum(e2e.best_so_far?.final)} sub={typeof e2e.best_so_far?.net_improvement==="number"?`Δ +${e2e.best_so_far.net_improvement.toFixed(3)}`:null}/>
      </div>
    </div>}

    {!!pending.length&&<div className="feedbackCard feedbackEntryCard">
      <div className="mlCardHead"><div><small>EXPERIMENT RESULT INGESTION · T20</small><b>实验结果回填</b></div><StatusPill value={round.status}/></div>
      <p className="cardFoot">只有真实实验结果才能提交。FAILED / INVALID / NOT_TESTED 不会被写成 0，也不会进入训练数据。</p>
      <div className="feedbackExperimentList">{pending.map(item=><ExperimentFeedbackRow key={item.candidate_id} campaignId={campaign.campaign_id} roundId={round.round_id} roundStatus={round.status} item={item} scope={scope} onUpdated={setView}/>)}</div>
    </div>}

    <div className="mlDisclaimer">V0.2 UI 展示并操作 T19-T26 的确定性状态层；模型建议、GP 预测与 synthetic fixture 都不能当作真实实验测量值。</div>
  </div>;
}



function AutonomyCards({data,scope}){
  const isAutonomy=data?.kind==="v030_autonomy";
  const [view,setView]=useState(isAutonomy?data:null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  useEffect(()=>{if(isAutonomy)setView(data)},[data,isAutonomy]);
  if(!isAutonomy||!view)return null;

  const campaign=view.campaign||{};
  const round=view.latest_round||{};
  const scheduler=round.scheduler||{};
  const telemetry=round.telemetry||{};
  const latestTel=telemetry.latest||{};
  const safety=round.safety||{};
  const operator=view.operator||{};
  const loop=view.autonomous_loop;
  const datasets=view.datasets||[];
  const promotion=view.model_promotion;
  const activeJob=scheduler.active_job;

  async function refresh(){
    setBusy(true);setError("");
    try{
      const r=await getAutonomyStatus({campaignId:campaign.campaign_id},scope);
      setView(r.data);
    }catch(e){setError(e.message)}finally{setBusy(false)}
  }

  async function doOperator(action){
    const defaults={
      RESUME:"确认恢复该自主实验任务",
      CANCEL_JOB:"人工取消当前设备任务",
      ABORT_ROUND:"人工终止当前 Round",
    };
    const reason=window.prompt(`${action} 原因`,defaults[action]||"operator action");
    if(!reason)return;
    setBusy(true);setError("");
    try{
      const r=await operatorOverride({
        campaign_id:campaign.campaign_id,
        round_id:round.round_id,
        action,
        reason,
      },scope);
      setView(r.data);
    }catch(e){setError(e.message)}finally{setBusy(false)}
  }

  return <div className="autonomyCards">
    <div className="autonomyLiveAnswer">{view.answer}</div>
    <div className="autonomyCard autonomyHero">
      <div className="mlCardHead">
        <div><small>AUTONOMOUS ORCHESTRATION · V0.3</small><b>{campaign.name||"自主实验闭环"}</b></div>
        <StatusPill value={view.status}/>
      </div>
      <div className="metricGrid">
        <Metric label="Round" value={round.round_no??campaign.current_round_no}/>
        <Metric label="Round 状态" value={round.status||"-"}/>
        <Metric label="自动回流" value={view.summary?.automatic_capture_count??0}/>
        <Metric label="Safety Stops" value={view.summary?.safety_stop_count??0}/>
      </div>
      <div className="autonomyToolbar">
        <button disabled={busy} onClick={refresh}>刷新运行态</button>
        {operator.available&&operator.actions?.includes("RESUME")&&<button disabled={busy||safety.state==="SAFETY_STOP"} onClick={()=>doOperator("RESUME")}>RESUME</button>}
        {operator.available&&operator.actions?.includes("CANCEL_JOB")&&<button disabled={busy} onClick={()=>doOperator("CANCEL_JOB")}>CANCEL JOB</button>}
        {operator.available&&operator.actions?.includes("ABORT_ROUND")&&<button className="dangerAction" disabled={busy} onClick={()=>doOperator("ABORT_ROUND")}>ABORT ROUND</button>}
      </div>
      {safety.state==="SAFETY_STOP"&&<p className="safetyWarning">SAFETY_STOP 已锁存。UI 的 RESUME 不允许绕过 T31；必须先按安全流程处理并重新确认。</p>}
      {!operator.inputs?.base_ready&&round.crash_checkpoint&&<p className="fixtureFlag">Operator Override 尚缺 trusted UI inputs：{(operator.inputs?.missing_base||[]).join(", ")}</p>}
      {error&&<div className="feedbackInlineError">{error}</div>}
    </div>

    <div className="autonomyGrid">
      <div className="autonomyCard">
        <div className="mlCardHead"><div><small>SCHEDULER · T29</small><b>任务调度</b></div><StatusPill value={activeJob?.status||round.status}/></div>
        <div className="metricGrid">
          <Metric label="Queued" value={scheduler.counts?.QUEUED??0}/>
          <Metric label="Running" value={scheduler.counts?.RUNNING??0}/>
          <Metric label="Completed" value={scheduler.counts?.COMPLETED??0}/>
          <Metric label="Failed/Timeout" value={(scheduler.counts?.FAILED??0)+(scheduler.counts?.TIMEOUT??0)}/>
        </div>
        {activeJob&&<div className="activeJob"><b>{activeJob.candidate_id}</b><span>{activeJob.device_id||"-"}</span><small>{activeJob.elapsed_ticks??0}/{activeJob.timeout_ticks??"-"} ticks</small></div>}
      </div>

      <div className="autonomyCard">
        <div className="mlCardHead"><div><small>DEVICE + TELEMETRY · T28/T30</small><b>设备运行态</b></div><StatusPill value={latestTel.phase||"NO_TELEMETRY"}/></div>
        <div className="metricGrid">
          <Metric label="Progress" value={latestTel.progress_percent!=null?`${Number(latestTel.progress_percent).toFixed(0)}%`:"-"}/>
          <Metric label="Temperature" value={latestTel.temperature_c!=null?`${Number(latestTel.temperature_c).toFixed(1)} °C`:"-"}/>
          <Metric label="Pressure" value={latestTel.pressure_mpa!=null?`${Number(latestTel.pressure_mpa).toFixed(2)} MPa`:"-"}/>
          <Metric label="RPM" value={latestTel.rpm!=null?Number(latestTel.rpm).toFixed(0):"-"}/>
        </div>
        <p className="cardFoot">{latestTel.experiment_id||"-"} · {telemetry.session_count||0} telemetry sessions · simulator={String(telemetry.all_simulator??true)}</p>
      </div>

      <div className="autonomyCard safetyCard">
        <div className="mlCardHead"><div><small>SAFETY INTERLOCK · T31</small><b>安全联锁</b></div><StatusPill value={safety.state||"NO_STATE"}/></div>
        <div className="metricGrid">
          <Metric label="Interlocks" value={safety.interlock_count??0}/>
          <Metric label="Stops" value={safety.safety_stop_count??0}/>
          <Metric label="Trip Code" value={safety.current_trip?.code||"-"}/>
          <Metric label="Ack" value={safety.current_trip?.acknowledged?"YES":"NO"}/>
        </div>
        <p className="cardFoot">automatic_resume_allowed = false</p>
      </div>

      <div className="autonomyCard">
        <div className="mlCardHead"><div><small>AUTO RESULT · T32</small><b>自动结果回流</b></div><StatusPill value={round.capture_receipts>0?"PASS":"WAITING"}/></div>
        <div className="metricGrid">
          <Metric label="本轮 receipts" value={round.capture_receipts??0}/>
          <Metric label="总 receipts" value={view.summary?.automatic_capture_count??0}/>
          <Metric label="人工提交" value="0"/>
          <Metric label="Real Measurement" value="NO"/>
        </div>
      </div>
    </div>

    <div className="autonomyCard">
      <div className="mlCardHead"><div><small>AUTONOMOUS ROUND / LOOP · T33/T34</small><b>自主轮次</b></div><StatusPill value={loop?"PASS":round.status}/></div>
      <div className="roundTimeline">{(view.rounds||[]).map(r=><div className="roundItem" key={r.round_id}><div><b>R{String(r.round_no).padStart(2,"0")}</b><span>{r.dataset_version}</span></div><StatusPill value={r.status}/><small>{r.capture_receipts||0} auto captures</small></div>)}</div>
      {loop&&<p className="cardFoot">{loop.round_count} rounds · {loop.total_experiments} experiments · replay={String(loop.idempotent_replay??false)}</p>}
    </div>

    {(round.crash_checkpoint||round.recovery_report)&&<div className="autonomyCard crashCard">
      <div className="mlCardHead"><div><small>CRASH / RESUME · T35</small><b>异常恢复</b></div><StatusPill value={round.recovery_report?"RECOVERED":"RECOVERY_REQUIRED"}/></div>
      {round.crash_checkpoint&&<div className="metricGrid">
        <Metric label="崩溃前完成" value={round.crash_checkpoint.completed_results_before_crash}/>
        <Metric label="崩溃前待处理" value={round.crash_checkpoint.pending_results_before_crash}/>
        <Metric label="Active Progress" value={round.crash_checkpoint.device_progress_percent!=null?`${Number(round.crash_checkpoint.device_progress_percent).toFixed(0)}%`:"-"}/>
        <Metric label="Active Candidate" value={round.crash_checkpoint.active_candidate_id||"-"}/>
      </div>}
      {round.recovery_report&&<p className="cardFoot">recovery audit={String(round.recovery_report.recovery_audit_valid)} · idempotent replay supported</p>}
    </div>}

    <div className="autonomyCard">
      <div className="mlCardHead"><div><small>DATASET + MODEL GOVERNANCE</small><b>数据与模型治理</b></div><StatusPill value={promotion?.decision||"NO_DECISION"}/></div>
      <div className="datasetLineage">{datasets.map((d,i)=><div key={d.dataset_version}><b>{d.dataset_version}</b><span>{d.row_count} rows</span>{i>0&&<small>+{d.added_row_count??0}</small>}</div>)}</div>
      <p className="cardFoot">Active Model {view.summary?.active_model_version||"-"} · automatic activation = false</p>
    </div>

    <div className="mlDisclaimer">V0.3 当前 UI 只展示/操作 deterministic Simulator runtime。Synthetic telemetry 和 result 不是单位真实设备或真实材料测量；Operator Override 不能绕过 Safety，模型也不会自动批准晋级。</div>
  </div>;
}



function DemoCards({data}){
  if(!data||data.kind!=="monday_demo") return null;
  const versions=data.versions||[];
  const readyInternal=data.prepared_internal_versions||"0/4";
  return <div className="demoCards">
    <div className="demoHero">
      <div>
        <small>MONDAY DEMO MODE</small>
        <h2>V0.1.1 → V0.3 全能力演示</h2>
        <p>数据访问 → 知识/RAG → 建模 → BO → 实验回流 → 自动重训 → V0.3 自主实验</p>
      </div>
      <div className="demoHeroStatus">
        <StatusPill value={data.status}/>
        <b>{readyInternal}</b>
        <span>内部闭环 Runtime</span>
      </div>
    </div>

    <div className="demoStoryline">
      {(data.storyline||[]).map((x,i)=><div key={x}><span>{i+1}</span><b>{x}</b>{i<data.storyline.length-1&&<i>→</i>}</div>)}
    </div>

    <div className="demoVersionMatrix">
      {versions.map(v=><div className="demoVersionRow" key={v.version}>
        <div className="demoVersionName"><b>{v.version}</b><span>{v.title}</span></div>
        <StatusPill value={v.status}/>
        <div className="demoVersionCaps">{(v.capabilities||[]).map(x=><span key={x}>{x}</span>)}</div>
        <div className="demoVersionAction">
          {v.project_id&&<code>Project {v.project_id}</code>}
          <p>{v.demo_prompt}</p>
        </div>
      </div>)}
    </div>

    <div className="demoLearning">
      <div className="mlCardHead"><div><small>AUTO LEARNING STORY</small><b>实验反馈后系统如何“学习”</b></div><StatusPill value="PASS"/></div>
      <div className="demoLearningFlow">
        <span>实验结果</span><i>→</i>
        <span>Prediction vs Measurement</span><i>→</i>
        <span>Dataset v+1</span><i>→</i>
        <span>Challenger 重训</span><i>→</i>
        <span>PROMOTE / KEEP</span><i>→</i>
        <span>下一轮 BO</span>
      </div>
      <p>自动的是：结果回流、Dataset 版本升级、Challenger 重训与比较、下一轮 BO。正式模型 PROMOTE 仍需人工批准。</p>
    </div>

    <div className="demoBoundary">
      <b>演示边界</b>
      <span>业务 MySQL = READ ONLY</span>
      <span>V0.1.2 RAG = live 外部能力</span>
      <span>V0.1.3 / V0.1.4 = synthetic engineering fixture</span>
      <span>V0.2 / V0.3 measurement = deterministic synthetic fixture</span>
      <span>V0.3 = Simulator only，不冒充真实设备</span>
      <span>Safety 不可绕过 · 模型不自动批准晋级</span>
    </div>

    <details className="demoScope">
      <summary>演示权限建议</summary>
      <p>Development Header 的 Project IDs 建议包含：</p>
      <code>{(data.demo_scope_project_ids||[]).join(",")}</code>
    </details>
  </div>;
}

function CompanyDataCards({data}){
  if(!data||data.kind!=="company_real_data") return null;
  const summary=data.summary||{};
  const p=data.presentation;
  const selected=data.selected_product;
  if(!p) return null;
  const inherited=Boolean(p.scope_inherited);
  const metrics=p.metrics||[];
  const details=p.details||[];
  const globalMetrics=[
    {label:"真实样品",value:summary.samples},
    {label:"产品类型",value:summary.products},
    {label:"原料字段",value:summary.materials},
    {label:"性能指标",value:summary.performance_metrics},
  ];

  return <div className="companyDataCards focusedCompanyCards">
    <div className="currentDataScope">
      <span>当前分析对象</span>
      <b>{p.scope_label||"全部真实数据"}</b>
      {inherited&&<em>沿用上一轮</em>}
    </div>

    <div className={`companyDataCard focusedDataCard ${p.card_type||"overview"}`}>
      <div className="mlCardHead">
        <div>
          <small>{p.eyebrow||"REAL DATA"}</small>
          <b>{p.headline||"单位真实数据"}</b>
        </div>
        <StatusPill value={p.status||data.status}/>
      </div>

      {!!metrics.length&&<div className={`metricGrid ${metrics.length===5?"five":""}`}>
        {metrics.map((x,i)=><Metric key={`${x.label}-${i}`} label={x.label} value={x.value} sub={x.sub}/>)}
      </div>}

      {!!p.highlights?.length&&<div className="companyHighlights">
        {p.highlights.map((x,i)=><span key={i}>{x}</span>)}
      </div>}

      {!!details.length&&<div className="companyFocusList">
        {details.map((x,i)=><div key={`${x.label}-${i}`}>
          <span>{x.label}</span>
          <b>{x.value}</b>
          {x.sub&&<small>{x.sub}</small>}
        </div>)}
      </div>}

      {selected&&<p className="scopeHint">
        后续直接继续问“有没有异常值 / 冲击强度缺失多少 / 哪些字段可以建模”，系统会继续沿用 {selected.product_type}。
      </p>}
    </div>

    {p.show_global_details&&<details className="companyGlobalDetails">
      <summary>查看全库信息</summary>
      <div className="metricGrid">
        {globalMetrics.map((x,i)=><Metric key={i} label={x.label} value={x.value}/>)}
      </div>
      <p>canonical：{data.source?.canonical_source||"-"} · source SHA256 {String(data.source?.sha256||"").slice(0,12)}…</p>
    </details>}

    {!p.show_global_details&&<div className="companySourceLine">
      canonical：{data.source?.canonical_source||"-"} · source SHA256 {String(data.source?.sha256||"").slice(0,12)}…
    </div>}
  </div>;
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

function Message({m,scope}){
  const isV020Feedback=m.data?.kind==="v020_feedback_loop";
  const isV030Autonomy=m.data?.kind==="v030_autonomy";
  return <div className={`msg ${m.role}`}>
    <div className="avatar">{m.role==="assistant"?<Logo/>:"你"}</div>
    <div className="msgcol">
      <small>{m.role==="assistant"?"材数智能体":"你"}</small>
      <div className="bubble">{!isV020Feedback&&!isV030Autonomy&&<div className="content">{m.content}</div>}<ModelingCards data={m.data}/><OptimizationCards data={m.data}/><FeedbackCards data={m.data} scope={scope}/><AutonomyCards data={m.data} scope={scope}/><DemoCards data={m.data}/><CompanyDataCards data={m.data}/><Detail m={m}/></div>
    </div>
  </div>
}

const MONDAY_DEMO_PROJECTS={
  mysql:115,
  modeling:9010,
  optimization:9018,
  feedback:9026,
  autonomy:9036,
  companyRealData:930066,
};

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

  function appendUiFailure(label,error){
    const message=String(error?.message||error||"未知错误");
    setErr(message);
    setMessages(x=>[...x,{
      id:id(),
      role:"assistant",
      content:`${label}加载失败：${message}`,
      meta:{
        intent:"ui_action_error",
        tool:"frontend_demo_router",
        router:"ui_action",
        summary:"UI action failed; error surfaced in chat."
      },
      data:{kind:"ui_action_error",label,error:message},
      evidence:[]
    }]);
  }

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

  async function sendModelStatus(
    targetMetric="冲击强度",
    projectIdOverride=MONDAY_DEMO_PROJECTS.modeling
  ){
    if(busy||uploading)return;
    const projectId=projectIdOverride||currentProjectId();
    if(!projectId){setErr("当前 Project IDs 无法解析，请先在左侧权限范围中填写项目号。 ");return;}
    const q=`检查 Project ${projectId} 的${targetMetric}建模状态`;
    setErr("");
    setMessages(x=>[...x,{id:id(),role:"user",content:q}]);
    setBusy(true);
    try{
      const r=await getModelingStatus(projectId,targetMetric,scope);
      setMessages(x=>[...x,{id:id(),role:"assistant",content:r.answer,meta:{intent:"get_modeling_status",tool:"v013_runtime_reports",router:"ui_action",summary:"只读加载 V0.1.3 已落盘运行报告，不触发训练。"},data:r.data,evidence:[]}]);
    }catch(e){appendUiFailure(`Project ${projectId} 模型状态`,e)}finally{setBusy(false)}
  }


  async function sendFeedbackStatus(
    projectIdOverride=MONDAY_DEMO_PROJECTS.feedback
  ){
    if(busy||uploading)return;
    const projectId=projectIdOverride||currentProjectId();
    if(!projectId){setErr("当前 Project IDs 无法解析，请先在左侧权限范围中填写项目号。");return;}
    const q=`查看 Project ${projectId} 的 V0.2 闭环状态`;
    setErr("");setMessages(x=>[...x,{id:id(),role:"user",content:q}]);setBusy(true);
    try{
      const r=await getFeedbackStatus({projectId},scope);
      setMessages(x=>[...x,{id:id(),role:"assistant",content:r.answer,meta:{intent:"v020_feedback_loop_status",tool:"v020_campaign_runtime",router:"ui_action",summary:"读取 T19-T26 Campaign、Round、实验反馈、Dataset、模型治理、Checkpoint 与 BO 状态。"},data:r.data,evidence:[]}]);
    }catch(e){appendUiFailure(`Project ${projectId} V0.2 闭环状态`,e)}finally{setBusy(false)}
  }


  async function sendAutonomyStatus(
    projectIdOverride=MONDAY_DEMO_PROJECTS.autonomy
  ){
    if(busy||uploading)return;
    const projectId=projectIdOverride||currentProjectId();
    if(!projectId){setErr("当前 Project IDs 无法解析，请先在左侧权限范围中填写项目号。");return;}
    const q=`查看 Project ${projectId} 的 V0.3 自主实验状态`;
    setErr("");setMessages(x=>[...x,{id:id(),role:"user",content:q}]);setBusy(true);
    try{
      const r=await getAutonomyStatus({projectId},scope);
      setMessages(x=>[...x,{id:id(),role:"assistant",content:r.answer,meta:{intent:"v030_autonomy_status",tool:"v030_autonomy_runtime",router:"ui_action",summary:"读取 T27-T36 Protocol、Scheduler、Telemetry、Safety、自动回流、多轮闭环与 Crash/Resume 状态。"},data:r.data,evidence:[]}]);
    }catch(e){appendUiFailure(`Project ${projectId} V0.3 自主实验状态`,e)}finally{setBusy(false)}
  }

  async function sendDemoStatus(){
    if(busy||uploading)return;
    const q="打开周一 V0.1.1 → V0.3 演示模式";
    setErr("");
    setMessages(x=>[...x,{id:id(),role:"user",content:q}]);
    setBusy(true);
    try{
      const r=await getMondayDemoStatus();
      setMessages(x=>[...x,{
        id:id(),
        role:"assistant",
        content:r.answer,
        meta:{
          intent:"monday_demo_status",
          tool:"monday_demo_runtime",
          router:"ui_action",
          summary:"读取周一 Demo Report：V0.1.1/V0.1.2 live readiness + V0.1.3/V0.1.4/V0.2/V0.3 deterministic runtime。"
        },
        data:r.data,
        evidence:[]
      }]);
    }catch(e){appendUiFailure("周一演示模式",e)}finally{setBusy(false)}
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
      <header><div><h1>研发对话</h1><p>V0.3 · Autonomous Experiment Orchestration</p></div><div><button className="versionTopBtn active" disabled={busy||uploading} onClick={()=>sendAutonomyStatus(MONDAY_DEMO_PROJECTS.autonomy)}>V0.3 · 9036</button><button className="demoModeBtn" disabled={busy||uploading} onClick={sendDemoStatus}>演示模式</button><button className="modelStatusBtn autonomyStatusBtn" disabled={busy||uploading} onClick={()=>sendAutonomyStatus(MONDAY_DEMO_PROJECTS.autonomy)}>自主状态</button><button className="modelStatusBtn feedbackStatusBtn" disabled={busy||uploading} onClick={()=>sendFeedbackStatus(MONDAY_DEMO_PROJECTS.feedback)}>V0.2 · 9026</button><button className="modelStatusBtn" disabled={busy||uploading} onClick={()=>sendModelStatus("冲击强度",MONDAY_DEMO_PROJECTS.modeling)}>模型 · 9010</button><button disabled={busy||uploading} onClick={newChat}>新对话</button></div></header>
      <section className="scroll"><div className="inner">
        {messages.length===1&&<div className="quick"><small>可以试试</small><div>{quick.map(q=><button key={q.text} onClick={()=>runQuick(q)}><span>{q.label}</span><b>{q.text}</b></button>)}</div></div>}
        <div className="messages">{messages.map(m=><Message key={m.id} m={m} scope={scope}/>)}{busy&&<div className="msg assistant"><div className="avatar"><Logo/></div><div className="msgcol"><small>材数智能体</small><div className="bubble loading">● ● ● <span>正在读取研发证据 / 运行优化算法</span></div></div></div>}</div>
        {err&&<div className="error"><b>请求失败</b>{err}</div>}<div ref={end}/>
      </div></section>

      <footer>
        {!!attachments.length&&<div className="attachments">{attachments.map(item=><div className="attachmentChip" key={item.attachmentId}><span>附件</span><b>{item.filename}</b><span>{item.pageCount?`${item.pageCount}页`:item.parser} · {item.chunkCount} chunks</span><button title="移除附件" onClick={()=>removeAttachment(item)}>×</button></div>)}</div>}
        {uploading&&<div className="uploadState">正在上传并解析附件…</div>}
        <div className="composer"><input ref={fileInput} type="file" accept=".pdf,.docx,.xlsx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" multiple hidden onChange={onFilesSelected}/><button className="upload uploadReady" disabled={uploading||busy} onClick={()=>fileInput.current?.click()}>＋ 上传文件 <span>PDF/DOCX/XLSX</span></button><textarea value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}}} placeholder={attachments.length?"问当前附件，例如：分析这份报告":"输入研发问题，例如：查看 V0.3 自主实验状态 / Safety / Crash Resume"}/><button className="send" disabled={!text.trim()||busy||uploading} onClick={()=>send()}>➤</button></div>
        <p>业务 MySQL = READ ONLY · V0.3 Simulator only · Safety 不可绕过 · Dataset 不覆盖旧版本 · 模型不自动晋级</p>
      </footer>
    </main>
  </div>
}
