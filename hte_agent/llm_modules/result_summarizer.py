from __future__ import annotations

from hte_agent.shared.candidate_schema import LLMS2Summary, RoundAggregate
from hte_agent.shared.task_representation import TaskRepresentation


def summarize_results(task: TaskRepresentation, aggregate: RoundAggregate, client) -> LLMS2Summary:
    payload = {
        "task_profile": task.profile_name,
        "substrate_features": list(task.substrate_features),
        "round_index": aggregate.round_index,
        "observation_count": aggregate.observation_count,
        "mean_yield": aggregate.mean_yield,
        "best_yield": aggregate.best_yield,
        "factor_tables": aggregate.factor_tables,
        "output_schema": {
            "summary": "short string",
            "promising_patterns": ["axis:value"],
            "risky_patterns": ["axis:value"],
            "exploration_axes": ["short instruction"],
        },
    }
    response = client.complete_json(
        "result_summarizer",
        (
            "You are the LLM-S2 result summarizer. Return JSON only. "
            "Summarize first-round statistics into promising motifs, risky motifs, and one exploration suggestion."
        ),
        payload,
    )
    status = str(response["status"])
    parsed = response.get("parsed", {})
    if status == "success" and (not isinstance(parsed, dict) or any(
        key in parsed and not isinstance(parsed[key], list)
        for key in ("promising_patterns", "risky_patterns", "exploration_axes")
    )):
        status = "invalid_response"
    if status != "success" or not response["invoked"]:
        return LLMS2Summary(bool(response["invoked"]), status, str(response["model"]),
                            "", raw_response=response)
    return LLMS2Summary(
        invoked=bool(response["invoked"]), status="success", model=str(response["model"]),
        summary=str(parsed.get("summary", "")),
        promising_patterns=[item for item in parsed.get("promising_patterns", []) if isinstance(item, str)],
        risky_patterns=[item for item in parsed.get("risky_patterns", []) if isinstance(item, str)],
        exploration_axes=[item for item in parsed.get("exploration_axes", []) if isinstance(item, str)],
        raw_response=response,
    )
