# Chat History V0.1 JSON 存储与 Docker 部署

## 功能

- 对话回答成功后自动保存；
- 按 `user_id + company_id` 隔离；
- 历史会话列表、打开、重命名、删除；
- 使用 `client_message_id` 去重；
- 原子写入 JSON，避免半文件；
- 不保存 Authorization/Bearer Token；
- 业务 MySQL 继续只读，不需要启用 Runtime DB。

默认配置：

```env
CHAT_HISTORY_DIR=.runtime/chat_history
CHAT_HISTORY_MAX_MESSAGES=400
```

每个会话最多保留 400 条消息。超过上限时保留最近的消息，并在 JSON 中标记
`messages_trimmed=true`。

## 本地运行

本地记录保存在项目目录：

```text
.runtime/chat_history/<用户与公司哈希>/<conversation_id>.json
```

本地继续使用：

```env
PERMISSION_MODE=development_header
PLATFORM_TRUST_FORWARDED_HEADERS=false
```

## Docker 临时模式

不增加 Volume 时，记录位于容器内：

```text
/app/.runtime/chat_history
```

这种模式可以验证功能，但重新构建或 `--force-recreate mat-agent` 后记录可能丢失。

## Docker 推荐持久化模式

在宿主机创建专用目录：

```bash
mkdir -p /opt/matfusion/data/mat-agent-runtime
```

在 `docker-compose.yml` 的 `mat-agent` 服务下增加：

```yaml
services:
  mat-agent:
    volumes:
      - /opt/matfusion/data/mat-agent-runtime:/app/.runtime
```

然后重建服务：

```bash
docker-compose up -d --no-deps --force-recreate mat-agent
```

这样 JSON 实际保存在宿主机，容器重建后仍可恢复。该目录包含用户对话和研发
内容，应限制为服务器管理员访问，不应通过 Nginx 暴露。

## 验收

1. 使用用户 A 登录并发送两轮对话；
2. 打开页面顶部“历史”，应看到一个会话和 4 条消息；
3. 刷新页面并重新打开该会话，内容应恢复；
4. 使用用户 B 登录，不应看到用户 A 的会话；
5. 切换 Company ID，不应看到原公司的会话；
6. 重命名与删除应立即更新列表；
7. 挂载 Volume 后重建 `mat-agent`，历史记录仍应存在。

服务器当前应保持单个 Uvicorn worker。JSON 版本使用线程锁和原子替换，尚未为
多个独立后端容器/多 worker 并发写同一会话设计；未来横向扩容时应迁移到 Runtime
MySQL。
