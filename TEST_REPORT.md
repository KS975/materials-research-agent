# TEST_REPORT — V0.1.1-dev1

## Generation-environment test result

Executed:

```text
pytest
```

Actual result:

```text
23 passed, 1 skipped
```

The skipped test is the opt-in real MySQL integration test.

Also executed:

```text
python -m compileall -q .
```

Result: PASS.

## Covered by executed unit/regression tests

- business SQL read-only guard;
- rejection of INSERT / UPDATE / DELETE / DDL / multi-statement SQL;
- recipe mapping `R3-xxx -> sample_materials`;
- performance mapping through `data_column`;
- unresolved-field behavior (no invented names);
- MySQL JSON string decoding;
- deterministic V0.1.1 intent routing;
- sample comparison diff;
- development permission scope parsing and fail-closed behavior;
- FastAPI health endpoint;
- runtime DB safety guard.

## Real MySQL integration

Not claimed as executed in the generation environment because the generated package does not
receive the user's database password.

Run locally:

```text
RUN_REAL_MYSQL_TESTS=1
TEST_USER_ID=...
TEST_COMPANY_ID=...
TEST_PROJECT_ID=120
TEST_SAMPLE_NAME=trial_10
pytest -m integration -s
```

## Acceptance status

- T01 real sample context: PENDING local real-DB execution
- T02 real sample comparison: PENDING selecting two real same-project samples
- T03 performance-drop analysis: PENDING selecting a real pair/target property
- Production permission regression: PENDING real permission interface

The version must not be frozen/released until these are executed and recorded.
