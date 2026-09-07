from __future__ import annotations

from pipeline_v2.shared.candidate_schema import LLMS2Summary, RoundAggregate
from pipeline_v2.shared.task_representation import TaskRepresentation


def _fallback_summary(task: TaskRepresentation, aggregate: RoundAggregate) -> LLMS2Summary:
    promising: list[str] = []
    risky: list[str] = []
    exploration: list[str] = []
    for axis, table in aggregate.factor_tables.items():
        if table:
            promising.append(f"{axis}:{table[0]['label']}")
            risky.append(f"{axis}:{table[-1]['label']}")
    if task.profile_name == "aza_heteroaryl_chloride_guarded":
        exploration.append("probe non-leading solvent while staying in guarded temperature band")
    elif task.profile_name == "hindered_aryl_chloride_activation":
        exploration.append("keep one activation-heavy ligand off the current anchor base")
    else:
        exploration.append("preserve one solvent-diversifying exploratory slot")
    return LLMS2Summary(
        invoked=False,
        status="fallback",
        model="deterministic_backstop",
        summary=f"Fallback summarizer flagged {', '.join(promising[:2]) or 'no clear positive motif'} as promising and {', '.join(risky[:2]) or 'no clear risk motif'} as risky.",
        promising_patterns=promising[:4],
        risky_patterns=risky[:4],
        exploration_axes=exploration[:3],
        raw_response={},
    )


def summarize_results(task: TaskRepresentation, aggregate: RoundAggregate, client) -> LLMS2Summary:
    fallback = _fallback_summary(task, aggregate)
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
    if response["status"] != "success":
        return LLMS2Summary(
            invoked=bool(response["invoked"]),
            status=str(response["status"]),
            model=str(response["model"]),
            summary=fallback.summary,
            promising_patterns=fallback.promising_patterns,
            risky_patterns=fallback.risky_patterns,
            exploration_axes=fallback.exploration_axes,
            raw_response=response,
        )
    parsed = response["parsed"]
    return LLMS2Summary(
        invoked=True,
        status="success",
        model=str(response["model"]),
        summary=str(parsed.get("summary", "")).strip() or fallback.summary,
        promising_patterns=[str(item) for item in parsed.get("promising_patterns", []) if str(item).strip()] or fallback.promising_patterns,
        risky_patterns=[str(item) for item in parsed.get("risky_patterns", []) if str(item).strip()] or fallback.risky_patterns,
        exploration_axes=[str(item) for item in parsed.get("exploration_axes", []) if str(item).strip()] or fallback.exploration_axes,
        raw_response=response,
    )
