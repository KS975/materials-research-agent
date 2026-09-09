import assert from "node:assert/strict";
import test from "node:test";

function fakeStorage(entries) {
  const keys = Object.keys(entries);
  return {
    length: keys.length,
    key(index) {
      return keys[index] ?? null;
    },
    getItem(name) {
      return entries[name] ?? null;
    },
  };
}

async function headersFor(entries) {
  globalThis.window = {
    localStorage: fakeStorage(entries),
    sessionStorage: fakeStorage({}),
  };
  const {platformRequestHeaders} = await import("../src/platformIdentity.js");
  return platformRequestHeaders();
}

test("plain token and company_Id cache keys become platform headers", async () => {
  const headers = await headersFor({
    token: "jwt-token",
    company_Id: "company-1",
  });

  assert.deepEqual(headers, {
    authorization: "Bearer jwt-token",
    "company-id": "company-1",
  });
});

test("a bearer token is not double-prefixed", async () => {
  const headers = await headersFor({
    token: "Bearer jwt-token",
    companyId: "company-1",
  });

  assert.equal(headers.authorization, "Bearer jwt-token");
  assert.equal(headers["company-id"], "company-1");
});

test("explicit platform keys take precedence over structured values", async () => {
  const headers = await headersFor({
    token: "explicit-jwt",
    company_Id: "explicit-company",
    loginProfile: JSON.stringify({
      accessToken: "structured-jwt",
      selectCompany: "structured-company",
    }),
  });

  assert.equal(headers.authorization, "Bearer explicit-jwt");
  assert.equal(headers["company-id"], "explicit-company");
});
