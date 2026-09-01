# KNOWN_GAPS — V0.1.1-dev1

## 1. Production Permission Adapter — BLOCKING FOR RELEASE

The actual MatCloud/材数智元 login-state transfer contract has not yet been provided.

Current local integration mode:

```text
PERMISSION_MODE=development_header
Local development still supports X-User-Id / X-Company-Id / X-Project-Ids.
The unit deployment uses authorization / company-id / organization-id /
organization-level through PlatformPermissionAdapter.
```

This is explicitly a development compatibility adapter, not a replacement permission system.

Before V0.1.1 release:
- confirm the real Token/User Context transfer;
- connect the unit gateway's real JWT signature validation before enabling
  `PLATFORM_TRUST_FORWARDED_HEADERS=true`;
- verify project visibility with real users;
- disable `development_header` in production.

## 2. Agent Runtime schema not provisioned

Runtime code and DDL are included, but the runtime DB is disabled by default because the team has
not yet approved/created a separate MySQL schema.

The code refuses to use `materials` as the runtime DB.

## 3. Dynamic column semantics

Verified:
- `R3-xxx -> sample_materials.id`
- `Pxxxxx / SPxxxxx -> data_column.id`

For `Sxxxxx`, the resolver attempts `data_column.id` and reports unresolved if no definition exists.
No unknown field is silently renamed by the LLM.

Condition keys such as `15955-0` are currently preserved as raw condition keys until their exact
business dictionary is confirmed.

## 4. Real acceptance sample names

The plan examples `S128` and `S125` do not exist in the current database.

Real test samples must be selected from the current company/project. `trial_10` is useful for
formula/performance mapping, while project 140 contains newer records with formula, process and
performance values.

## 5. No production DB writes

Intentional V0.1.1 limitation:
- no INSERT
- no UPDATE
- no DELETE

This includes business data. Runtime writes, when enabled later, target only a separate runtime
schema.
