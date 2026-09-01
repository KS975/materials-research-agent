from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.chat_ui import ChatUIRequest, chat_ui
from schemas.user_context import UserContext
from skills.current_attachment import CurrentAttachmentSkill


def _ctx() -> UserContext:
    return UserContext(
        user_id="user-a",
        company_id="company-a",
        project_ids=(115,),
        permission_source="test",
    )


class FakeLLM:
    def __init__(self):
        self.system = ""
        self.user = ""

    def complete(self, system: str, user: str) -> str:
        self.system = system
        self.user = user
        return "DeepSeek 直接回答"


class FakeStore:
    def __init__(self, attachment):
        self.attachment = attachment

    def get(self, attachment_id, ctx):
        assert attachment_id == "attachment-a"
        assert ctx.company_id == "company-a"
        return self.attachment


def test_reference_mode_sends_full_attachment_content_directly_to_deepseek():
    chunks = [
        {"index": 0, "text": "温度不得高于 230 ℃", "page": 1},
        {"index": 1, "text": "历史数组 220, 225, 228", "page": 2},
    ]
    chunks.extend(
        {"index": index, "text": f"附件尾部信息 {index}", "page": index + 1}
        for index in range(2, 15)
    )
    attachment = SimpleNamespace(
        attachment_id="attachment-a",
        filename="requirements.pdf",
        parser="pypdf",
        page_count=None,
        char_count=500,
        chunk_count=15,
        chunks=tuple(chunks),
    )
    llm = FakeLLM()
    skill = CurrentAttachmentSkill(FakeStore(attachment), llm)

    result = skill.answer(
        message="按附件推荐5组实验",
        attachment_ids=["attachment-a"],
        ctx=_ctx(),
        reference_mode=True,
    )

    assert result["kind"] == "deepseek_attachment_answer"
    assert result["reference_mode"] is True
    assert result["answer_basis"] == "current_chat_attachments_only"
    assert result["answer"] == "DeepSeek 直接回答"
    assert len(result["evidence"]) == 15
    assert "直接根据所提供的附件正文回答用户问题" in llm.system
    assert "不要套用固定的实验推荐格式" in llm.system
    assert "硬性规定" not in llm.system
    assert "历史数组只能用来" not in llm.system
    assert "温度不得高于 230 ℃" in llm.user
    assert "历史数组 220, 225, 228" in llm.user
    assert "附件尾部信息 14" in llm.user


class FakeReferenceSkill:
    def __init__(self):
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "ok",
            "answer": "attachment answer",
            "evidence": [{"source": "chat_attachment"}],
            "warnings": ["attachment-only"],
        }


def test_reference_mode_bypasses_inverse_design_and_returns_raw_llm_answer():
    skill = FakeReferenceSkill()
    container = SimpleNamespace(
        settings=SimpleNamespace(llm_enabled=True),
        current_attachment_skill=skill,
    )
    response = chat_ui(
        body=ChatUIRequest(
            message="Project 9016：冲击强度 >= 43，推荐5组方案",
            attachment_ids=["attachment-a"],
            attachment_reference_mode=True,
        ),
        ctx=_ctx(),
        container=container,
    )

    assert response.answer == "attachment answer"
    assert response.intent == "deepseek_attachment_answer"
    assert response.router == "deepseek_attachment_passthrough"
    assert response.routing["constraints"]["business_intent_bypassed"] is True
    assert response.routing["constraints"]["database_not_queried"] is True
    assert response.routing["constraints"]["optimization_engine_not_run"] is True
    assert response.routing["constraints"]["answer_postprocessed"] is False
    assert skill.calls[0]["reference_mode"] is True


def test_reference_mode_requires_an_attachment():
    container = SimpleNamespace(
        settings=SimpleNamespace(llm_enabled=True),
        current_attachment_skill=FakeReferenceSkill(),
    )
    with pytest.raises(HTTPException) as exc_info:
        chat_ui(
            body=ChatUIRequest(
                message="按附件推荐5组实验",
                attachment_reference_mode=True,
            ),
            ctx=_ctx(),
            container=container,
        )

    assert exc_info.value.status_code == 400
    assert "请先上传" in str(exc_info.value.detail)
