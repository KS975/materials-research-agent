from api.chat_ui import initial_stream_progress_event


def test_stream_contract_starts_with_immediate_backend_event():
    event = initial_stream_progress_event()
    assert event == {
        "schema_version": "1.1",
        "source": "backend",
        "stage": "stream_connected",
        "status": "completed",
        "title": "实时分析通道已建立",
        "message": "后端已接受请求，后续执行阶段将通过当前通道持续返回。",
        "elapsed_ms": 0,
    }
