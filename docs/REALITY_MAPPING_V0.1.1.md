# REALITY_MAPPING_V0.1.1

## Scope

Reality mapping for Code V0.1.1: Agent + read-only MySQL.

## Confirmed business database

- MySQL
- business schema/database: `materials`
- core tables do not declare explicit MySQL foreign keys; relations are maintained by business IDs.

## Confirmed tables and relations

### Project

`mat_project`

Verified real relation:

```text
eln_sample.project_id -> mat_project.id
```

Real example:
- sample `trial_10`: `project_id=120`
- project `id=120`: exists, project name `推荐`

### Sample

`eln_sample`

Key fields:
- `id`
- `name`
- `company`
- `project_id`
- `recipes`
- `recipe_batches`
- `craft_detail`
- `craft_param`
- `performances`
- `service_performances`
- `conditions`
- `delete`

Real sample:
- id: `3713`
- name: `trial_10`
- project_id: `120`
- recipes: `{"R3-401":"81.1064","R3-402":"16.0889"}`
- performances: `{"P14598":"41.2052"}`
- service_performances: `{"SP14741":"11"}`

### Archive

`archive_data`

`data_type`:
- 1 sample
- 2 simulation
- 3 test report

Verified with 30 recent real records:

```text
archive_data.data_id -> eln_sample.id
```

for `data_type=1`, with matching names.

`archive_data.project_id` may be NULL while `eln_sample.project_id` is present, therefore the
primary Sample -> Project path is `eln_sample.project_id`.

### Formula/material dynamic keys

Verified:

```text
R3-401 -> sample_materials.id=401 -> 水
R3-402 -> sample_materials.id=402 -> P507+煤油
R3-603 -> sample_materials.id=603 -> Si  (unit=%)
R3-604 -> sample_materials.id=604 -> Fe  (unit=%)
```

Therefore:

```text
R3-xxx -> sample_materials.id
```

### Performance dynamic keys

`data_column` is a dynamic-column dictionary.

Verified:
- `P14598` -> `data_column.id=14598` -> `密度差`
- `SP14741` -> `data_column.id=14741` -> `服役性能`

Therefore P/SP numeric suffixes can be resolved through `data_column.id` when a matching row exists.

### Optional synthesis/test records

Tables exist:
- `eln_synthesis_exp`
- `eln_verify_item`
- `eln_verify_item_sample`
- `eln_verify_exp`

`trial_10` has no matching synthesis or verify records.

V0.1.1 must therefore treat these as optional evidence, not mandatory joins.

## V0.1.1 data model

```text
mat_project
    ^
    | project_id
eln_sample
    ^
    | data_id  (archive_data.data_type=1)
archive_data
```

Sample snapshot:

```text
eln_sample
├── recipes
├── craft_param
├── performances
├── service_performances
└── conditions
```

## Repository rules

- Fixed repository methods only.
- Parameters bound through DB-API placeholders.
- Business DB is read-only.
- Every sample query is scoped by `company` and allowed `project_id`.
- Unknown dynamic fields remain unresolved.
- Duplicate sample names return ambiguity instead of choosing silently.

## Remaining release blockers

- real MatCloud Permission Adapter contract;
- real T01-T03 execution;
- separate Agent Runtime DB approval/provisioning.
