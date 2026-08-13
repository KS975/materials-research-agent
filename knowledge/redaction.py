from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    count: int


class SecretRedactor:
    """Best-effort secret redaction before text is sent to embeddings/Qdrant.

    This is deliberately applied before semantic embedding and before the raw
    chunk text is stored in Qdrant payload.

    It is not a DLP product, but it covers the common credential formats that
    may appear in .env/config/report text:
      PASSWORD / PASSWD
      API_KEY / APIKEY
      SECRET
      ACCESS_TOKEN / REFRESH_TOKEN
      PRIVATE_KEY
      AUTHORIZATION / BEARER_TOKEN
    """

    _ENV_PATTERN = re.compile(
        r"""(?im)
        \b(
            [A-Z0-9_]*(?:
                PASSWORD|PASSWD|API_KEY|APIKEY|SECRET|
                ACCESS_TOKEN|REFRESH_TOKEN|PRIVATE_KEY|
                AUTHORIZATION|BEARER_TOKEN
            )[A-Z0-9_]*
            \s*=\s*
        )
        ([^\r\n#]+)
        """,
        re.VERBOSE,
    )

    _KV_PATTERN = re.compile(
        r"""(?im)
        (
            ["']?
            (?:
                password|passwd|api_key|apikey|secret|
                access_token|refresh_token|private_key|
                authorization|bearer_token
            )
            ["']?
            \s*:\s*
        )
        (
            "(?:\\.|[^"])*"
            |
            '(?:\\.|[^'])*'
            |
            [^,\r\n}]+
        )
        """,
        re.VERBOSE,
    )

    _BEARER_PATTERN = re.compile(
        r"(?i)\b(Bearer\s+)([A-Za-z0-9._~+/=-]{8,})"
    )

    def redact(self, text: str) -> RedactionResult:
        count = 0

        def env_repl(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return f"{match.group(1)}[REDACTED]"

        def kv_repl(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return f"{match.group(1)}[REDACTED]"

        def bearer_repl(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return f"{match.group(1)}[REDACTED]"

        output = self._ENV_PATTERN.sub(env_repl, text)
        output = self._KV_PATTERN.sub(kv_repl, output)
        output = self._BEARER_PATTERN.sub(bearer_repl, output)

        return RedactionResult(text=output, count=count)
