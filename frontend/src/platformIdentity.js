const SAME_ORIGIN_API_PREFIX = "/agent-api/";

function safeUrl(url) {
  try {
    return new URL(String(url), window.location.href);
  } catch {
    return null;
  }
}

export function isSameOriginAgentApi(url) {
  if (typeof window === "undefined") return false;
  const target = safeUrl(url);
  return Boolean(
    target
      && target.origin === window.location.origin
      && target.pathname.startsWith(SAME_ORIGIN_API_PREFIX)
  );
}

function normalizeHeaderName(name) {
  return String(name || "").trim().toLowerCase();
}

function normalizeHeaders(value) {
  if (!value || typeof value !== "object") return {};
  const output = {};
  for (const [key, rawValue] of Object.entries(value)) {
    const name = normalizeHeaderName(key);
    if (!name || rawValue == null) continue;
    const item = String(rawValue).trim();
    if (item) output[name] = item;
  }
  return output;
}

function findLoginValue(storage, preferredNames) {
  const preferred = new Set(preferredNames.map(normalizeHeaderName));
  let direct = null;
  let structuredToken = null;
  let structuredCompany = null;

  for (let index = 0; index < storage.length; index += 1) {
    const storageKey = storage.key(index);
    if (!storageKey) continue;
    const key = normalizeHeaderName(storageKey);
    let value = null;
    try {
      value = storage.getItem(storageKey);
    } catch {
      continue;
    }
    if (!value) continue;

    if (preferred.has(key)) direct = value;

    let parsed = null;
    try {
      parsed = JSON.parse(value);
    } catch {
      parsed = null;
    }
    if (!parsed || typeof parsed !== "object") continue;
    for (const [innerKey, innerValue] of Object.entries(parsed)) {
      const name = normalizeHeaderName(innerKey);
      const item = innerValue && typeof innerValue === "object"
        ? findInObject(innerValue)
        : { [name]: innerValue };
      for (const [candidateKey, candidateValue] of Object.entries(item)) {
        const candidateName = normalizeHeaderName(candidateKey);
        const text = candidateValue == null ? "" : String(candidateValue);
        if (!text) continue;
        if (candidateName.includes("accesstoken") || candidateName === "token") {
          structuredToken = text;
        }
        if (
          candidateName.includes("companyid")
          || candidateName === "company-id"
          || candidateName.includes("selectcompany")
        ) {
          structuredCompany = text;
        }
      }
    }
  }

  return {
    authorization: structuredToken ? `Bearer ${structuredToken}` : "",
    "company-id": structuredCompany || "",
    direct,
  };
}

function findInObject(value) {
  const output = {};
  const stack = [value];
  const seen = new Set();
  while (stack.length) {
    const item = stack.pop();
    if (!item || typeof item !== "object" || seen.has(item)) continue;
    seen.add(item);
    for (const [key, rawValue] of Object.entries(item)) {
      if (rawValue && typeof rawValue === "object") {
        stack.push(rawValue);
      } else if (rawValue != null) {
        output[key] = rawValue;
      }
    }
  }
  return output;
}

export function platformRequestHeaders() {
  if (typeof window === "undefined") return {};

  const explicit = normalizeHeaders(window.__MATCLOUD_REQUEST_HEADERS__);
  if (explicit.authorization && explicit["company-id"]) {
    return {
      authorization: explicit.authorization,
      "company-id": explicit["company-id"],
      ...(explicit["organization-id"] ? {"organization-id": explicit["organization-id"]} : {}),
      ...(explicit["organization-level"] ? {"organization-level": explicit["organization-level"]} : {}),
    };
  }

  const local = findLoginValue(window.localStorage, [
    "accessToken",
    "access_token",
    "matcloud_token",
    "token",
  ]);
  const session = findLoginValue(window.sessionStorage, [
    "accessToken",
    "access_token",
    "matcloud_token",
    "token",
  ]);
  const authorization = local.authorization || session.authorization;
  const companyId = local["company-id"] || session["company-id"];
  if (!authorization || !companyId) return {};
  return {authorization, "company-id": companyId};
}
