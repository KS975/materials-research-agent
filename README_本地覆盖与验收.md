# Chat History V0.1 本地完整增量

本包已经合并“单位平台四请求头身份接入 V0.1”和“JSON 历史会话 V0.1”。
将包内文件按相同相对路径覆盖到当前项目，不要删除其他目录，也不要用
`.env.example` 覆盖你自己的 `.env`。

本地 `.env`：

```env
PERMISSION_MODE=development_header
PLATFORM_TRUST_FORWARDED_HEADERS=false
CHAT_HISTORY_DIR=.runtime/chat_history
CHAT_HISTORY_MAX_MESSAGES=400
```

覆盖后重启后端和前端。发送一轮对话后，点击页面顶部“历史”进行验收。

