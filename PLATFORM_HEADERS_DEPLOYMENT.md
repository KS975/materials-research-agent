# 单位平台四请求头接入说明

## 1. 本版接入的请求头

后端在 `PERMISSION_MODE=platform` 时读取：

```text
authorization
company-id
organization-id
organization-level
```

请求头名称不区分大小写。`authorization` 必须为 `Bearer <JWT>`。原始
Token 不会写入运行文件、日志、接口响应或前端页面。

后端从已经通过单位身份网关验证的 JWT 中提取稳定用户 ID，默认依次查找：

```text
userId, user_id, sub, id
```

同时兼容 `user.userId` 和 `data.user.userId` 这类嵌套结构。

## 2. 单位服务器 `.env`

只有在上游网关已经验证 Token 的前提下，才可使用：

```env
PERMISSION_MODE=platform
PLATFORM_TRUST_FORWARDED_HEADERS=true
PLATFORM_JWT_USER_CLAIMS=userId,user_id,sub,id
```

如果没有可信身份网关，保持：

```env
PLATFORM_TRUST_FORWARDED_HEADERS=false
```

此时后端会返回 503，不会把浏览器任意伪造的 Header 当成真实身份。

## 3. Nginx / 网关要求

`/agent-api/` 反向代理必须保留四个请求头。已有 location 中建议明确加入：

```nginx
proxy_set_header Authorization $http_authorization;
proxy_set_header company-id $http_company_id;
proxy_set_header organization-id $http_organization_id;
proxy_set_header organization-level $http_organization_level;
```

注意：上述配置只负责转发，不负责验证 Token。当前若仍是公网 Nginx 直接
`proxy_pass http://mat-agent:8000/`，必须先接入单位现有的鉴权网关或
`auth_request`，再开启信任开关。

## 4. 部署后检查

使用当前登录请求中的真实 Header 访问：

```text
GET /agent-api/api/v1/session-context
```

成功时只返回脱敏后的身份上下文，例如：

```json
{
  "user_id": "2090369875129171970",
  "company_id": "6a4b19f62d0e000027001eb8",
  "organization_id": "6a6b2090f1a6586e94f958bb",
  "organization_level": "1",
  "permission_source": "platform_forwarded_headers",
  "project_mode": "company_all_projects",
  "project_ids": []
}
```

前端左下角会显示“后端与平台身份已连接”，展开后可核对以上四项身份结果，
不会显示 Authorization。

## 5. 当前权限与聊天记录边界

- 业务 MySQL 仍只读，并始终以 `company-id` 限定当前公司。
- 本版按当前公司浏览全部项目；`organization-id/level` 原样进入用户上下文，
  暂不猜测它们与项目的映射关系。
- 临时附件和 LangGraph 工作流检查点已经按 `user_id + company_id` 校验所有权。
- 本版尚未增加“历史会话列表/重命名/删除/长期保存”页面；稳定用户身份已经为
  后续实现该功能提供了可靠的归属键。

## 6. 本地开发

本地后端可继续使用：

```env
PERMISSION_MODE=development_header
```

`npm run dev` 会在开发构建中使用本地测试 Header；生产 `dist` 不包含这些旧
Header，也不读取浏览器 localStorage 中的开发权限。
