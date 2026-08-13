from knowledge.redaction import SecretRedactor


def test_env_json_and_bearer_redaction():
    text = """
BUSINESS_DB_PASSWORD=matcloud
EMBEDDING_API_KEY=sk-abcdef123456
{"password": "hello-world", "name": "demo"}
Authorization: Bearer abcdefghijklmnop
"""
    result = SecretRedactor().redact(text)

    assert result.count >= 4
    assert "matcloud" not in result.text
    assert "sk-abcdef123456" not in result.text
    assert "hello-world" not in result.text
    assert "abcdefghijklmnop" not in result.text
    assert result.text.count("[REDACTED]") >= 4
