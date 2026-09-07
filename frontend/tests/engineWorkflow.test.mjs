import test from "node:test";
import assert from "node:assert/strict";

import {
  engineTaskActions,
  engineTaskIsTerminal,
  engineTaskProgress,
  isEngineWorkflowData,
  visualizationDatasets,
} from "../src/engineWorkflow.js";

test("isEngineWorkflowData recognizes the public engine envelope",()=>{
  assert.equal(isEngineWorkflowData({workflow:"optimize_formula",status:"OK",result:{}}),true);
  assert.equal(isEngineWorkflowData({kind:"v014_inverse_design",design_cards:[]}),false);
});

test("visualizationDatasets accepts bundle and inline array forms",()=>{
  const bundle={datasets:[{dataset_id:"a"},null,{dataset_id:"c"}]};
  assert.equal(visualizationDatasets({result:{visualization_datasets:bundle}}).length,2);
  assert.equal(visualizationDatasets({visualization_datasets:[{dataset_id:"b"}]})[0].dataset_id,"b");
});

test("engine task helpers expose progress, actions, and terminal state",()=>{
  const task={
    status:"AWAITING_APPROVAL",
    scenario_plan:{steps:[{},{}]},
    completed_steps:["skill-1"],
  };
  assert.deepEqual(engineTaskActions(task.status),["approve","cancel"]);
  assert.equal(engineTaskProgress(task),50);
  assert.equal(engineTaskIsTerminal("SUCCEEDED"),true);
  assert.equal(engineTaskIsTerminal("INTERRUPTED"),false);
});
