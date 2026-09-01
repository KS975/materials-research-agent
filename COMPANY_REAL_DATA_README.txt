V0.2 单位真实数据接入补丁
========================

本补丁把单位提供的真实数据 ZIP 接到 materials-research-agent 的本地 runtime，不写业务 MySQL，不把原始 ZIP 复制进源码。

Canonical 数据源：海科数据整理/总库。专项目录只记录来源，不重复合并。

本次真实数据检查：496 个样品、101 个产品类型、473 个原料字段、36 个测试性能指标；测试条件表 0 行；材料工艺参数 0 行（LOGINCATEGORY/TASKCATEGORY 按工作流元数据处理）。

因此当前可以：查询、统计、产品覆盖分析、Reality Check；不能绕过 Modeling Gate 直接做正式建模/逆向设计/BO。

安装依赖：
pip install -r requirements.txt

导入：
python -m scripts.import_company_real_data --source-zip "C:\path\单位数据.zip"

真实数据 smoke：
python -m scripts.run_company_real_data_smoke --source-zip "C:\path\单位数据.zip" --reset

聊天：
查看单位真实数据概况
单位真实数据：PC/ABS FR303 数据情况

导出 Reality/Gate：
python -m scripts.export_company_model_dataset --product "PC/ABS FR303" --target "悬臂梁冲击强度"

然后按输出的 local_project_id 运行：
python -m scripts.run_v013_modeling_gate --project-id <id> --target "悬臂梁冲击强度"

按当前文件内容 Gate 应 FAIL（缺材料工艺参数和显式测试条件），这是正确安全行为。

单元测试：
pytest tests/unit/test_company_data_ingestion.py -q
预期：6 passed
