from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"无法自动修改 {label}：没有找到预期锚点。请不要继续覆盖，把文件发给 GPT Sol 定点适配。")
    return text.replace(old, new, 1)


def append_unique_lines(path: Path, lines: list[str]) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    additions = [line for line in lines if line not in current]
    if additions:
        with path.open("a", encoding="utf-8") as f:
            if current and not current.endswith("\n"):
                f.write("\n")
            f.write("\n".join(additions) + "\n")


def main() -> None:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (project / "app" / "main.py").exists():
        raise SystemExit(f"不是 materials-research-agent 项目根目录：{project}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = project / f"_backup_before_v012a_{stamp}"
    backup.mkdir(parents=True)

    replace_files = [
        "agent/deepseek_intent_router.py",
        "api/chat_ui.py",
        "api/files.py",
        "file_processing/__init__.py",
        "file_processing/models.py",
        "file_processing/parser.py",
        "runtime/chat_attachments.py",
        "skills/current_attachment.py",
        "tests/unit/test_chat_file_parser.py",
        "tests/unit/test_chat_attachment_store.py",
        "frontend/src/App.jsx",
        "frontend/src/api.js",
        "frontend/src/styles.css",
    ]
    modify_files = [
        "app/config.py",
        "app/container.py",
        "app/main.py",
        "requirements.txt",
        ".env.example",
    ]

    for rel in replace_files + modify_files:
        dst = project / rel
        if dst.exists():
            b = backup / rel
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, b)

    # Copy additive/replacement files.
    for rel in replace_files:
        src = PACKAGE_ROOT / rel
        if not src.exists():
            raise RuntimeError(f"补丁包缺少文件：{src}")
        dst = project / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Settings.
    p = project / "app/config.py"
    s = p.read_text(encoding="utf-8")
    if "chat_upload_dir:" not in s:
        s = replace_once(
            s,
            "    runtime_enabled: bool = False\n",
            "    # V0.1.2-A: current Chat temporary attachments (NOT Qdrant)\n"
            "    chat_upload_dir: str = \".runtime/chat_uploads\"\n"
            "    chat_upload_max_mb: int = 25\n"
            "    chat_upload_ttl_minutes: int = 180\n\n"
            "    runtime_enabled: bool = False\n",
            "app/config.py",
        )
    p.write_text(s, encoding="utf-8")

    # Container.
    p = project / "app/container.py"
    s = p.read_text(encoding="utf-8")
    if "from file_processing import ChatFileParser" not in s:
        s = replace_once(
            s,
            "from llm.factory import create_llm_provider\n",
            "from llm.factory import create_llm_provider\n"
            "from file_processing import ChatFileParser\n"
            "from runtime.chat_attachments import ChatAttachmentStore\n"
            "from skills.current_attachment import CurrentAttachmentSkill\n",
            "app/container.py imports",
        )
    if "self.chat_file_parser = ChatFileParser()" not in s:
        s = replace_once(
            s,
            "        self.llm = create_llm_provider(settings)\n",
            "        self.llm = create_llm_provider(settings)\n"
            "        self.chat_file_parser = ChatFileParser()\n"
            "        self.chat_attachment_store = ChatAttachmentStore(\n"
            "            settings.chat_upload_dir,\n"
            "            settings.chat_upload_ttl_minutes,\n"
            "        )\n"
            "        self.current_attachment_skill = CurrentAttachmentSkill(\n"
            "            self.chat_attachment_store,\n"
            "            self.llm,\n"
            "        )\n",
            "app/container.py init",
        )
    p.write_text(s, encoding="utf-8")

    # FastAPI main. Preserve existing chat-ui router if already installed.
    p = project / "app/main.py"
    s = p.read_text(encoding="utf-8")
    if "from api.files import router as files_router" not in s:
        anchor = "from api.health import router as health_router\n"
        s = replace_once(
            s,
            anchor,
            anchor + "from api.files import router as files_router\n",
            "app/main.py import",
        )
    if "app.include_router(files_router)" not in s:
        # Put it after chat/chat-ui registrations, before EOF.
        s = s.rstrip() + "\napp.include_router(files_router)\n"
    s = s.replace('version="0.1.1-dev1"', 'version="0.1.2-dev1"')
    s = s.replace('version="0.1.1"', 'version="0.1.2-dev1"')
    p.write_text(s, encoding="utf-8")

    append_unique_lines(
        project / "requirements.txt",
        [
            "python-multipart==0.0.22",
            "pypdf==6.7.5",
            "python-docx==1.2.0",
        ],
    )
    append_unique_lines(
        project / ".env.example",
        [
            "CHAT_UPLOAD_DIR=.runtime/chat_uploads",
            "CHAT_UPLOAD_MAX_MB=25",
            "CHAT_UPLOAD_TTL_MINUTES=180",
        ],
    )

    print("V0.1.2-A 补丁已应用。")
    print(f"项目：{project}")
    print(f"备份：{backup}")
    print("下一步：pip install -r requirements.txt")
    print("然后：pytest -q tests/unit/test_chat_file_parser.py tests/unit/test_chat_attachment_store.py")


if __name__ == "__main__":
    main()
