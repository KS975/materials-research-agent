# CHANGELOG

## 0.1.1-dev1

### Added
- V0.1.1 project skeleton.
- FastAPI health/chat endpoints.
- Read-only MySQL client with SQL guard.
- Repositories for samples, projects, archive data, sample materials, dynamic columns and experiment/test records.
- Dynamic field resolver:
  - `R3-xxx -> sample_materials.id`
  - `Pxxxxx / SPxxxxx / Sxxxxx -> data_column.id` when a definition exists.
- First six V0.1.1 tools.
- Core + three Skills.
- Lightweight LangGraph workflow.
- Generic LLM provider interface.
- Temporary development permission adapter boundary.
- Separate Agent Runtime MySQL schema boundary.
- Unit tests and opt-in real MySQL integration test.

### Not implemented
- Production MatCloud Permission Adapter.
- Qdrant/RAG.
- ML.
- Optimization/BO.
- Production Runtime DB provisioning.
