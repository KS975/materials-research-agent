# Database Navigator V0.1.1 本地覆盖说明

本增量修复数据库浏览遗漏负数项目编号的问题。

## 覆盖

将压缩包内容按原目录覆盖到项目根目录。不会修改 `.env`、依赖版本或数据库结构。

本地后端重启：

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

如果使用前端开发服务器，重新启动 `npm run dev`。包内也包含已重新构建的 `frontend/dist`。

## 验收

打开“数据库浏览 → 项目”，确认正数项目和负数历史导入项目同时出现，例如：

```text
PROJECT 115
PROJECT -1539 · 历史导入
PROJECT -1540 · 历史导入
```

点击负数项目“查看样品”，应只返回当前公司中对应 `project_id` 的样品。

## 定向测试

```powershell
python -m pytest tests/unit/test_dashboard_v01.py -q
node --test frontend/tests/dashboard.test.mjs frontend/tests/progress.test.mjs
```
