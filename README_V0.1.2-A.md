# V0.1.2-A — Current Chat Attachment / T04

这一批只做实施方案中 V0.1.2 的第一项验收：

```text
Chat 上传 PDF / DOCX
→ 临时解析
→ 当前 Chat 可直接分析/提问
→ 返回附件 Evidence
```

**本批明确不做：** Qdrant、长期 Knowledge Index、历史 RAG、正式 Data Governance。

当前附件保存在本机 `.runtime/chat_uploads`，默认 180 分钟过期；不会写入业务 MySQL，也不会写入 Qdrant。

## 1. 应用补丁

先停止 FastAPI 和 Vite。

把本补丁 ZIP 解压，例如到：

```text
C:\Users\sunke\Downloads\materials-research-agent-v0.1.2-A
```

然后：

```powershell
cd C:\Users\sunke\Downloads\materials-research-agent-v0.1.2-A
python .\apply_v012a.py C:\Users\sunke\materials-research-agent
```

脚本会先自动备份被修改文件到项目中的：

```text
_backup_before_v012a_YYYYMMDD_HHMMSS
```

如果脚本说“找不到预期锚点”，不要手改整个项目；把对应文件发给 GPT Sol 定点适配。

## 2. 安装依赖

```powershell
cd C:\Users\sunke\materials-research-agent
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

新增依赖只有：

```text
python-multipart
pypdf
python-docx
```

## 3. 跑新增单测

```powershell
pytest -q tests/unit/test_chat_file_parser.py tests/unit/test_chat_attachment_store.py
```

预期：

```text
2 passed
```

## 4. 回归 V0.1.1

```powershell
$env:DEV_USER_ID="local-test"
$env:DEV_COMPANY_ID="6a4b19f62d0e000027001eb8"
$env:DEV_PROJECT_IDS="115"

python -m scripts.run_acceptance_v011 `
  --sample-a "3811" `
  --sample-b "3809" `
  --why "为什么 3811 的冲击强度比 3809 低？"
```

T01/T02/T03 必须继续通过。

## 5. 启动后端

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

应能看到 FastAPI 正常启动。

## 6. 启动前端

另开 PowerShell：

```powershell
$env:Path += ";C:\Program Files\nodejs"
cd C:\Users\sunke\materials-research-agent\frontend
npm run dev
```

打开：

```text
http://localhost:5173
```

## 7. T04 验收

页面上“上传文件”现在会启用。

上传一份 `.docx` 或文本型 `.pdf`，解析成功后输入：

```text
分析这份报告
```

预期：

```text
intent = analyze_current_attachment
Tool = -
Router = deepseek
Evidence = chat_attachment + 文件名 + 页码/段落 + chunk
```

再问一个文件内明确问题，例如：

```text
这份报告的核心目标是什么？
```

预期：

```text
intent = ask_current_attachment
```

回答只允许依据当前附件。

## 8. 当前限制

### PDF

这一批 PDF 使用 `pypdf` 读取文本型 PDF。

如果 PDF 是纯扫描图片，接口会明确返回：

```text
PDF 未提取到可读文本。若这是扫描件，后续需要接 MinerU/OCR 解析器。
```

这不是静默失败。下一小步可以把现有 MinerU 接入 `ChatFileParser`。

### 当前附件 != 长期知识库

```text
当前 Chat 临时附件       V0.1.2-A 已做
Knowledge Index/Qdrant   下一阶段
历史 RAG                 下一阶段
MySQL + RAG              再下一阶段
```
