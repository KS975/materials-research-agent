# Chat History V0.1 JSON 测试报告

日期：2026-09-01

## 已完成验证

- Python 静态编译：PASS
- JSON 存储冒烟测试：PASS
  - 新建与读取
  - 列表查询
  - 重复消息去重
  - 跨用户不可见
  - 重命名
  - 删除
- 前端 Vite 生产构建：PASS
- 前端现有测试：8 PASS / 0 FAIL
- dist 中包含历史会话 API 和历史面板：PASS

## 自动测试文件

新增 `tests/unit/test_chat_history_json.py`，覆盖：

1. JSON 往返与所有权元数据；
2. 跨用户/跨公司隔离；
3. `client_message_id` 幂等去重；
4. 重命名、删除和路径穿越拦截；
5. 消息数量上限。

当前 Codex Python 环境未安装项目完整 FastAPI/Pytest 依赖，所以本轮没有重新运行
完整 Python pytest；新增测试已经放入交付包，单位或本地 `.venv` 中可执行：

```bash
python -m pytest tests/unit/test_chat_history_json.py tests/unit/test_permission.py -q
```
