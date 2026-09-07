from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from hte_agent.shared.io_utils import read_api_key

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
        result = {"role": role, "invoked": False, "status": "missing_api_key", "model": "",
                  "response_model": "", "parsed": {}, "text": "", "error": "",
                  "configured_models": list(self.model_fallbacks), "attempts": []}
        if not self.available:
            result["error"] = "Set OPENROUTER_API_KEY to enable model requests."
            return result
        if not self.model_fallbacks:
            result["status"] = "no_models_configured"
            return result
        for model in self.model_fallbacks:
            request_payload = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "temperature": 0.0,
            }
            request = urllib.request.Request(
                url=self.base_url, data=json.dumps(request_payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                         "X-Title": "HTE-Agent"}, method="POST",
            )
            attempt = {"model": model, "status": "request_failed", "response_model": ""}
            result["attempts"].append(attempt)
            result["invoked"] = True
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload_data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # Provider error bodies can echo credentials or prompts: never persist them.
                result["error"] = f"HTTP {exc.code}"
                exc.close()
                continue
            except (urllib.error.URLError, OSError, ValueError) as exc:
                result["error"] = type(exc).__name__
                continue

            if not isinstance(payload_data, dict):
                attempt["status"] = "invalid_response"
                result["error"] = "Invalid provider response schema."
                continue
            returned_model = payload_data.get("model")
            attempt["response_model"] = returned_model if isinstance(returned_model, str) else ""
            choices = payload_data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                attempt["status"] = "invalid_response"
                result["error"] = "No valid completion choices."
                continue
            message = choices[0].get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
            else:
                text = ""
            parsed = _extract_json_object(text)
            if parsed is not None:
                attempt["status"] = "success"
                result.update(status="success", model=model, response_model=attempt["response_model"],
                              parsed=parsed, text=text, error="")
                return result
            attempt["status"] = "invalid_json"
            result["error"] = "Model response was not valid JSON."
        result["status"] = "request_failed"
        return result
