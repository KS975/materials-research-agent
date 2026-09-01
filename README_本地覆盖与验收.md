# Skill Architecture V1 本地增量

## 覆盖

在项目根目录解压并覆盖同名文件。不要删除原目录中的其它文件。

本包不包含也不会覆盖：

- `.env`
- `.runtime/chat_history`
- 业务数据库配置
- 前端源码或 `dist`

## 启动前验证

```powershell
python -m pytest -q tests/unit/test_skill_registry_v1.py
python -m pytest -q tests/unit/test_chat_ui_langgraph_v4.py
python -m pytest -q tests/unit/test_material_intelligence_round2a1.py
```

启动后检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/skills
```

平台身份请求头模式下，该接口需要携带现有的 Authorization、company-id、
organization-id、organization-level；本地开发模式继续使用现有开发身份配置。

## 验收问题

```text
查看3811的完整信息
所有样品的冲击强度平均值是多少
找项目115里PC含量大于50%、注塑温度高于70℃的样品
3811历史上有没有类似情况
找和3811最像的5个样品
```

展开“查询与分析详情”后，应看到 `Skill 编排完成`，并显示 Skill、Operation、
执行节点和固定 Workflow。
