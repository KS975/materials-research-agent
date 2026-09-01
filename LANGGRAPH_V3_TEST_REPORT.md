# LangGraph Migration V3 验证报告

## 已完成验证

- 全项目318个 Python 源文件完成 AST 语法扫描，0 个错误。
- 正常工作流生成公司/用户作用域检查点并进入 `SUCCEEDED`。
- 已成功工作流显式恢复时复用缓存响应，生产执行器调用次数不增加。
- `classify_primary` 安全暂停点通过：暂停前生产执行器调用次数为0。
- 暂停工作流使用相同问题和 `workflow_id` 后继续成功。
- 修改问题后恢复被请求指纹校验拒绝。
- 不同用户作用域无法读取其他用户的工作流状态。
- RUNNING工作流租约未过期时拒绝并发恢复。
- 40个并发工作流和相同ID跨用户隔离烟测通过。
- `resume_workflow=true` 但未提供 `workflow_id` 时拒绝执行。
- 生产执行失败只记录一次FAILED，不进行后台自动重试。
- 检查点原子写入的临时I/O错误按上限重试，测试为第3次成功。
- NaN等非有限数值可以安全投影到JSON检查点。
- 检查点配置72小时默认保留期，清理范围限制为合法作用域/工作流文件。
- V2三个一级分支和五个语义证据分支继续通过。
- Database Explorer内部SQL纠错上限未修改。
- 迁移前生产路由函数体AST保持一致。
- ZIP生成后执行压缩完整性和文件清单检查。

## 新增测试

`tests/unit/test_chat_ui_workflow_v3.py` 覆盖：

- 成功检查点和缓存响应幂等恢复
- 安全暂停与继续
- 请求指纹和用户作用域保护
- 缺失workflow ID保护
- FAILED状态和禁止自动业务重试
- 检查点I/O有限重试

V1、V2、附件直问和现有Chat UI测试继续随本地累计增量包交付。

## 环境说明

当前构建工作区未安装项目锁定的FastAPI、LangGraph和pytest，因此未执行
完整pytest。V3条件边使用等价内存图执行器完成契约烟测，检查点存储使用真实
文件系统实现完成暂停、恢复、冲突、幂等和I/O重试测试。

在安装 `requirements.txt` 的环境可执行：

```bash
python -m pytest -q \
  tests/unit/test_chat_ui_langgraph_v1.py \
  tests/unit/test_chat_ui_langgraph_v2.py \
  tests/unit/test_chat_ui_workflow_v3.py \
  tests/unit/test_attachment_reference_mode.py \
  tests/unit/test_chat_ui_stream_progress_v212.py \
  tests/unit/test_round2a2_1_chat_ui_mysql_precedence.py \
  tests/unit/test_database_explorer_v01.py
```
