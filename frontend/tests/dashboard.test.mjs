import assert from "node:assert/strict";
import test from "node:test";

import {
  buildFieldPrompt,
  buildProjectPrompt,
  buildSamplePrompt,
} from "../src/dashboard.js";


test("one selected sample can be brought into deterministic actions",()=>{
  const sample={id:3811,name:"trial_6"};
  assert.equal(buildSamplePrompt("profile",[sample]),"查看样品3811的完整信息");
  assert.equal(buildSamplePrompt("similar",[sample]),"找和3811最像的5个样品");
  assert.equal(buildSamplePrompt("history",[sample]),"以前有没有和3811类似的情况");
});

test("two selected samples produce explicit identifiers",()=>{
  const selected=[{id:3811},{id:3809}];
  assert.equal(buildSamplePrompt("compare",selected),"比较样品3811和3809");
  assert.equal(buildSamplePrompt("formula",selected),"比较样品3811和3809的配方差异");
  assert.equal(buildSamplePrompt("process",selected),"比较样品3811和3809的工艺差异");
});

test("project and field selections become useful questions",()=>{
  assert.equal(buildProjectPrompt({id:115}),"分析项目115的样品数据");
  assert.equal(buildFieldPrompt("formula",{name:"PC"}),"找PC含量最高的样品");
  assert.equal(buildFieldPrompt("performance",{name:"冲击强度"}),"找冲击强度最高的样品");
});
