import test from "node:test";
import assert from "node:assert/strict";

import {
  documentEvidenceRecords,
  evidenceMetricRows,
  isHybridEvidencePayload,
  normalizedEvidenceRecords,
  structuredEvidenceGroups,
} from "../src/evidenceFrame.js";

test("hybrid evidence helpers expose source counts and bounded records",()=>{
  const frame={
    source_summary:{mysql:5,vector_api:2,upload:1,dialog:1},
    records:Array.from({length:35},(_,index)=>({
      record_id:`record-${index}`,
      source_type:"mysql",
      subject_id:"sample:128",
      attribute:`field.${index}`,
      confidence:1,
      authority_level:"AUTHORITATIVE",
    })),
    conflicts:[{conflict_group:"c1",subject_id:"sample:128",attribute:"impact",reason:"VALUE_MISMATCH",record_ids:["a","b"],resolution:"PRESERVE_ALL"}],
  };

  assert.equal(isHybridEvidencePayload({frame}),true);
  assert.deepEqual(evidenceMetricRows(frame).map(x=>x.value),[5,2,1,1,1]);
  const records=normalizedEvidenceRecords({frame});
  assert.equal(records.length,35);
  assert.equal(structuredEvidenceGroups(records).length,1);
  assert.equal(structuredEvidenceGroups(records)[0].rows.length,35);
  assert.equal(documentEvidenceRecords(records).length,0);
  assert.equal(isHybridEvidencePayload({evidence:[{}],intent:"other"}),false);
  assert.equal(isHybridEvidencePayload({evidence:[{}],intent:"hybrid_research_qa"}),true);
});
