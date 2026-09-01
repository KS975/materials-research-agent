# LangGraph Migration V3

## 目标

在V2条件图上增加安全检查点、显式暂停/恢复、结果幂等复用和有限I/O重试，
同时避免自动重复执行DeepSeek、数据库查询或T17/T18。

## 核心行为

### 自动检查点

每次Chat UI请求会生成一个 `workflow_id`，并在Agent运行目录记录：

- 当前用户和公司作用域
- 请求指纹（问题、历史、附件ID、权限范围）
- 一级分支和语义证据分支
- 当前阶段和成功/暂停/失败状态
- 已完成执行的响应，用于安全恢复
- 错误摘要和恢复次数

不会保存数据库密码、模型密钥、物理连接信息或附件原文件内容。

V3使用专门的JSON安全投影，而不是把LangGraph内存状态整体序列化；因此
`ApplicationContainer`、数据库连接、LLM客户端等运行对象不会进入检查点。

默认目录：

```text
.runtime/chat_ui_workflows/
```

### 显式暂停

请求可携带：

```json
{
  "message": "找和3811最像的5个样品",
  "pause_after": "classify_primary"
}
```

系统只完成一级路由，不查询数据库、不调用回答模型，并在响应的
`routing.workflow.workflow_id` 返回工作流ID。

### 显式恢复

使用相同问题和权限范围：

```json
{
  "message": "找和3811最像的5个样品",
  "workflow_id": "上一步返回的ID",
  "resume_workflow": true
}
```

- 暂停点没有执行结果：继续执行一次。
- 已有完整执行结果：直接复用检查点，不重复查询或调用模型。
- 问题、附件或权限范围变化：返回409并拒绝恢复。

### 状态查询

```text
GET /api/v1/chat-ui/workflows/{workflow_id}
```

只返回脱敏运行状态，不返回完整答案或数据库行；仍受当前用户和公司范围约束。

### 重试边界

- 检查点原子写入发生临时I/O错误时，默认最多重试3次。
- 生产执行器不会自动重试，防止重复模型调用或重复业务执行。
- Database Explorer原有SQL错误反馈/修正循环保持不变，仍由它自己控制上限。
- 用户可以对FAILED工作流显式恢复；这是新的、可审计的执行，不是后台死循环。

## 新配置（均有默认值）

```env
CHAT_UI_WORKFLOW_DIR=.runtime/chat_ui_workflows
CHAT_UI_WORKFLOW_MAX_RESPONSE_CHARS=2000000
CHAT_UI_WORKFLOW_CHECKPOINT_RETRIES=3
CHAT_UI_WORKFLOW_LEASE_SECONDS=120
CHAT_UI_WORKFLOW_TTL_HOURS=72
```

不修改 `.env` 也可以运行。若希望容器重建后仍可恢复，需要把
`.runtime/chat_ui_workflows` 放入持久化卷或宿主机挂载目录。

同一工作流处于RUNNING且租约未过期时，系统拒绝并发恢复，防止重复执行。
进程硬崩溃且无法写入FAILED时，默认等待120秒租约到期后可显式恢复。
检查点默认保留72小时，过期文件由工作流存储定期清理；清理只匹配公司/用户
作用域目录中的合法工作流JSON文件。

## 部署

```bash
docker build --progress=plain -t materials-agent:1.0 .
docker-compose up -d --no-deps --force-recreate mat-agent
docker-compose restart nginx
```

本版没有前端改动，不需要替换dist。
