# 单位平台身份接入 V0.1 测试报告

日期：2026-09-01

## 验证结果

- 前端生产构建：PASS（Vite 8.2.1）
- 前端现有测试：8 PASS / 0 FAIL
- Python 静态编译：PASS
- PlatformPermissionAdapter JWT 身份解析冒烟测试：PASS
- 生产 dist 旧开发 Header 检查：PASS
  - `X-User-Id`：不存在
  - `X-Company-Id`：不存在
  - `X-Project-Ids`：不存在
  - `materials-agent-dev-scope`：不存在
- 生产 dist 新身份链路检查：PASS
  - `/api/v1/session-context`：存在
  - 只读“平台登录上下文”面板：存在

## 新增自动测试覆盖

`tests/unit/test_permission.py` 已增加以下测试：

1. 四个请求头与顶层 `userId` 解析；
2. `data.user.userId` 嵌套解析；
3. 缺失任一平台 Header 时拒绝；
4. 未显式信任身份网关时拒绝；
5. Token 缺少稳定用户标识时拒绝；
6. `/api/v1/session-context` 不返回 Token。

## 当前测试环境说明

本次 Codex 容器没有安装项目所需的 FastAPI/Pytest Python 依赖，因此没有在
此容器重新执行完整 Python pytest 套件；已完成不依赖第三方包的适配器冒烟
验证和全量 Python 静态编译。单位容器安装 `requirements.txt` 后可执行：

```bash
python -m pytest tests/unit/test_permission.py -q
```

## 安全边界

本版代码不会验证 JWT 签名。只有上游单位身份网关完成 Token 验证并注入/转发
四个请求头后，才能设置 `PLATFORM_TRUST_FORWARDED_HEADERS=true`。否则保持
false，后端会失败关闭。
