from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from pipeline_v2.shared.io_utils import read_api_key

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, object] | None:
    candidate = text.strip()
    fenced_match = _JSON_BLOCK_PATTERN.search(candidate)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None
    try:
        parsed = json.loads(candidate[first_brace : last_brace + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True)
class OpenRouterJSONClient:
    base_url: str
    timeout_seconds: float
    model_fallbacks: tuple[str, ...]

    @property
    def api_key(self) -> str:
        return read_api_key()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, role: str, system_prompt: str, payload: dict[str, object]) -> dict[str, object]:
        if not self.available:
            return {
                "role": role,
                "invoked": False,
                "status": "missing_api_key",
                "model": "",
                "parsed": {},
                "text": "",
                "error": "Set OPENROUTER_API_KEY to enable model requests.",
            }
        last_error = ""
        for model in self.model_fallbacks:
            request_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": 0.0,
            }
            request = urllib.request.Request(
                url=self.base_url,
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://local-v2-pipeline",
                    "X-Title": "HTE Pipeline V2",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload_data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
                continue
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = str(exc)
                continue

            choices = payload_data.get("choices") or []
            if not choices:
                last_error = "OpenRouter returned no completion choices."
                continue
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
            else:
                text = ""
            parsed = _extract_json_object(text)
            if isinstance(parsed, dict):
                return {
                    "role": role,
                    "invoked": True,
                    "status": "success",
                    "model": model,
                    "parsed": parsed,
                    "text": text,
                    "error": "",
                }
            last_error = "Model response was not valid JSON."
        return {
            "role": role,
            "invoked": True,
            "status": "request_failed",
            "model": "",
            "parsed": {},
            "text": "",
            "error": last_error,
        }
