import assert from "node:assert/strict";
import test from "node:test";

import {
  createInitialAnalysisStep,
  mergeProgressStep,
} from "../src/progress.js";

test("analysis panel starts before the network response arrives",()=>{
  const step=createInitialAnalysisStep();
  assert.equal(step.source,"client");
  assert.equal(step.stage,"stream_transport");
  assert.equal(step.status,"running");
});

test("running and completed events update one auditable stage",()=>{
  const running={
    source:"backend",
    stage:"candidate_generation",
    status:"running",
    title:"生成候选设计",
  };
  const completed={
    source:"backend",
    stage:"candidate_generation",
    status:"completed",
    title:"候选设计生成完成",
  };
  const steps=mergeProgressStep(mergeProgressStep([],running),completed);
  assert.equal(steps.length,1);
  assert.equal(steps[0].status,"completed");
});

test("frontend and backend transport events remain independently visible",()=>{
  const client=createInitialAnalysisStep();
  const backend={
    source:"backend",
    stage:"stream_connected",
    status:"completed",
  };
  const steps=mergeProgressStep([client],backend);
  assert.equal(steps.length,2);
});

test("structured query details survive running-to-completed merge",()=>{
  const running={source:"backend",stage:"knowledge_search",status:"running"};
  const completed={
    source:"backend",
    stage:"knowledge_search",
    status:"completed",
    query_preview:"冲击强度下降 历史异常",
    detail_items:[{label:"可靠命中",value:"2 个"}],
    evidence_preview:[{filename:"历史报告.docx",project_id:115,score:0.61}],
  };
  const steps=mergeProgressStep(mergeProgressStep([],running),completed);
  assert.equal(steps.length,1);
  assert.equal(steps[0].query_preview,"冲击强度下降 历史异常");
  assert.equal(steps[0].evidence_preview[0].filename,"历史报告.docx");
});

test("database explorer retry attempts remain separately auditable",()=>{
  const first={source:"backend",stage:"sql_retry",status:"retrying",attempt:1};
  const second={source:"backend",stage:"sql_retry",status:"retrying",attempt:2};
  const steps=mergeProgressStep(mergeProgressStep([],first),second);
  assert.equal(steps.length,2);
  assert.deepEqual(steps.map(item=>item.attempt),[1,2]);
});
