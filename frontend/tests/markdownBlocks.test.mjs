import test from "node:test";
import assert from "node:assert/strict";

import {classifyLine, splitBlocks} from "../src/markdownBlocks.js";

test("heading is isolated from following body",()=>{
  const blocks=splitBlocks("### 结论\n正文内容\n### 证据依据\n- 列表项");
  assert.deepEqual(blocks,["### 结论","正文内容","### 证据依据","- 列表项"]);
});

test("multi-line paragraph stays together but is separable by line kind",()=>{
  const blocks=splitBlocks("第一段\n第二段\n### 标题");
  assert.deepEqual(blocks,["第一段\n第二段","### 标题"]);
});

test("closed code block keeps fences together and recovers following text",()=>{
  const blocks=splitBlocks("```js\nconst a=1;\n```\n正文");
  assert.deepEqual(blocks,["```js\nconst a=1;\n```","正文"]);
});

test("classifyLine recognizes common block kinds",()=>{
  assert.equal(classifyLine("### 标题"),"heading");
  assert.equal(classifyLine("- item"),"ul");
  assert.equal(classifyLine("1. item"),"ol");
  assert.equal(classifyLine("> quote"),"quote");
  assert.equal(classifyLine("```"),"code");
  assert.equal(classifyLine("普通文本"),"text");
  assert.equal(classifyLine(""),"blank");
});
