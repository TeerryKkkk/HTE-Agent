from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pipeline_v2.shared.candidate_schema import PrecedentRecord
from pipeline_v2.shared.io_utils import read_csv_rows
from pipeline_v2.shared.task_representation import TaskRepresentation

_TEMP_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)C(?:_(\d+(?:\.\d+)?)h)?\s*$", re.IGNORECASE)
_HALIDE_PATTERN = re.compile(r"(Cl|Br|I)")


@dataclass(frozen=True)
class HistoricalDataset:
    precedents: tuple[PrecedentRecord, ...]
    success_cases: tuple[PrecedentRecord, ...]
    failure_cases: tuple[PrecedentRecord, ...]
    dataset_stats: dict[str, object]
    family_lookup: dict[str, dict[str, str]]


@dataclass(frozen=True)
class RetrievedSupportBundle:
    precedents: tuple[PrecedentRecord, ...]
    success_cases: tuple[PrecedentRecord, ...]
    failure_cases: tuple[PrecedentRecord, ...]
    retrieval_table: tuple[dict[str, object], ...]


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").replace("_", " ").split()).strip()


def _strip_numeric_prefix(value: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)?\s*", "", value).strip()


def _normalize_solvent_label(raw_label: str) -> tuple[str, str]:
    label = _clean_text(raw_label).lower()
    mapping = (
        ("dioxane", ("Dioxane", "ethereal_aprotic")),
        ("tetrahydrofuran", ("THF", "ethereal_aprotic")),
        (" thf", ("THF", "ethereal_aprotic")),
        ("dmac", ("DMAc", "polar_aprotic")),
        ("dimethylacetamide", ("DMAc", "polar_aprotic")),
        ("dmf", ("DMF", "polar_aprotic")),
        ("acetonitrile", ("MeCN", "polar_aprotic")),
        ("mecn", ("MeCN", "polar_aprotic")),
        ("toluene", ("Toluene", "nonpolar_aromatic")),
        ("xylene", ("Xylene", "nonpolar_aromatic")),
        ("methanol", ("MeOH", "protic")),
        ("water", ("Water", "aqueous")),
    )
    for token, normalized in mapping:
        if token in label:
            return normalized
    compact = _clean_text(raw_label).replace("1ml ", "").replace("1ml", "").strip()
    return compact or "unknown", "unknown"


def _normalize_base_label(raw_label: str) -> tuple[str, str]:
    label = _clean_text(raw_label).lower()
    mapping = (
        ("cesium carbonate", ("Cs2CO3", "carbonate")),
        ("caesium carbonate", ("Cs2CO3", "carbonate")),
        ("cs2co3", ("Cs2CO3", "carbonate")),
        ("potassium carbonate", ("K2CO3", "carbonate")),
        ("k2co3", ("K2CO3", "carbonate")),
        ("tripotassium phosphate", ("K3PO4", "phosphate")),
        ("potassium phosphate", ("K3PO4", "phosphate")),
        ("k3po4", ("K3PO4", "phosphate")),
        ("lithium bis(trimethylsilyl)amide", ("LiHMDS", "silylamide")),
        ("hexamethyldisilazide", ("LiHMDS", "silylamide")),
        ("lihmds", ("LiHMDS", "silylamide")),
        ("sodium tert-butoxide", ("NaOtBu", "alkoxide")),
        ("t-buona", ("NaOtBu", "alkoxide")),
        ("potassium tert-butoxide", ("KOtBu", "alkoxide")),
        ("t-buok", ("KOtBu", "alkoxide")),
        ("dbu", ("DBU", "organic_amine")),
        ("diisopropylethylamine", ("DIPEA", "organic_amine")),
        ("dipea", ("DIPEA", "organic_amine")),
        ("triethylamine", ("Et3N", "organic_amine")),
        ("naotms", ("NaOTMS", "silanolate")),
        ("trimethylsilanolate", ("NaOTMS", "silanolate")),
    )
    for token, normalized in mapping:
        if token in label:
            return normalized
    return _clean_text(raw_label) or "unknown", "unknown"


def _normalize_ligand_label(raw_label: str) -> tuple[str, str]:
    label = _clean_text(raw_label).lower()
    mapping = (
        ("brettphos", ("BrettPhos", "biaryl_phosphine")),
        ("tbubrettphos", ("tBuBrettPhos", "biaryl_phosphine")),
        ("t-bubrettphos", ("tBuBrettPhos", "biaryl_phosphine")),
        ("xphos", ("XPhos", "biaryl_phosphine")),
        ("sphos", ("SPhos", "biaryl_phosphine")),
        ("johnphos", ("JohnPhos", "biaryl_phosphine")),
        ("davephos", ("DavePhos", "biaryl_phosphine")),
        ("gphos", ("GPhos", "biaryl_phosphine")),
        ("alphos", ("AlPhos", "biaryl_phosphine")),
        ("xantphos", ("XantPhos", "chelating_phosphine")),
        ("binap", ("BINAP", "chelating_phosphine")),
        ("bippyphos", ("BippyPhos", "chelating_phosphine")),
    )
    for token, normalized in mapping:
        if token in label:
            return normalized
    if "dimethoxy" in label and "dicyclohexylphosphino" in label:
        return "BrettPhos", "biaryl_phosphine"
    if "triisopropyl" in label and "dicyclohexylphosphino" in label:
        return "XPhos", "biaryl_phosphine"
    if not label:
        return "unknown", "unknown"
    return _clean_text(raw_label), "other_phosphine"


def _extract_halide_type(substrate: str) -> str:
    match = _HALIDE_PATTERN.search(substrate)
    if not match:
        return "aryl_halide"
    halide = match.group(1)
    return {"Cl": "aryl_chloride", "Br": "aryl_bromide", "I": "aryl_iodide"}.get(halide, "aryl_halide")


def _extract_reaction_features(reaction_smiles: str) -> tuple[str, str, list[str], list[str]]:
    reactants_raw, _, product = reaction_smiles.partition(">>")
    reactants = [part for part in reactants_raw.split(".") if part]
    halide_candidates = [part for part in reactants if _HALIDE_PATTERN.search(part)]
    substrate = max(halide_candidates or reactants, key=len, default=reactants_raw)

    substrate_features: set[str] = {_extract_halide_type(substrate)}
    product_features: set[str] = set()
    if any(token in substrate for token in ("n", "[n", "s", "o")):
        substrate_features.add("heteroaryl")
    if substrate.count("n") >= 2:
        substrate_features.add("azaheteroaryl")
    if "s" in substrate:
        substrate_features.add("sulfur_heteroaryl")
    if "#N" in substrate or "C(=O)" in substrate:
        substrate_features.add("electron_poor")
    if substrate.count("c") + substrate.count("n") >= 12:
        substrate_features.add("sterically_hindered")
    if "C(=O)" in substrate:
        substrate_features.add("amide_bearing")
    if "N" in product:
        product_features.add("arylamine")
    if any(token in product for token in ("n", "[n", "s", "o")):
        product_features.add("heteroaryl_product")

    halide_type = _extract_halide_type(substrate)
    if "azaheteroaryl" in substrate_features and halide_type == "aryl_chloride":
        substrate_archetype = "aza_heteroaryl_chloride"
    elif "sterically_hindered" in substrate_features and halide_type == "aryl_chloride":
        substrate_archetype = "hindered_aryl_chloride"
    elif "heteroaryl" in substrate_features and halide_type == "aryl_bromide":
        substrate_archetype = "heteroaryl_bromide"
    else:
        substrate_archetype = halide_type
    product_archetype = "heteroaryl_arylamine" if "heteroaryl_product" in product_features else "arylamine"
    return substrate_archetype, product_archetype, sorted(substrate_features), sorted(product_features)


def _parse_percent(value: str | None) -> float | None:
    raw = (value or "").strip().replace("%", "")
    if not raw:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group()) / 100.0))


def _parse_condition_list(raw_condition: str) -> list[dict[str, object]]:
    text = (raw_condition or "").strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _condition_label(item: dict[str, object]) -> str:
    for key in ("iupac_name", "property", "smiles"):
        value = _clean_text(str(item.get(key, "")))
        if value:
            return value
    return "unknown"


def _extract_precedent_conditions(items: list[dict[str, object]]) -> tuple[str, str, str, str, str, str, str]:
    catalyst_item = next((item for item in items if str(item.get("role", "")).lower() in {"catalyst", "catalysts"}), None)
    ligand_item = next((item for item in items if str(item.get("role", "")).lower() in {"ligand", "ligands"}), None)
    base_item = next((item for item in items if str(item.get("role", "")).lower() == "base"), None)
    solvent_item = next((item for item in items if str(item.get("role", "")).lower() in {"solvent", "solvents"}), None)
    catalyst_label = _clean_text(_condition_label(catalyst_item or {})) or "Pd"
    ligand_label, ligand_family = _normalize_ligand_label(_condition_label(ligand_item or {}))
    base_label, base_family = _normalize_base_label(_condition_label(base_item or {}))
    solvent_label, solvent_family = _normalize_solvent_label(_condition_label(solvent_item or {}))
    return catalyst_label, ligand_label, ligand_family, base_label, base_family, solvent_label, solvent_family


def _make_precedent_record(row: dict[str, str], index: int) -> PrecedentRecord | None:
    reaction_smiles = (row.get("rsmi") or "").strip()
    if not reaction_smiles:
        return None
    condition_items = _parse_condition_list(row.get("condition", ""))
    _, ligand_label, ligand_family, base_label, base_family, solvent_label, solvent_family = _extract_precedent_conditions(condition_items)
    present_labels = sum(
        1 for value in (ligand_label, base_label, solvent_label) if value and value != "unknown"
    )
    if present_labels < 2:
        return None
    substrate_archetype, product_archetype, substrate_features, product_features = _extract_reaction_features(reaction_smiles)
    outcome_yield = _parse_percent(row.get("yield")) or 0.0
    outcome_label = "success" if outcome_yield >= 0.7 else "mixed" if outcome_yield >= 0.4 else "failure"
    confidence = min(0.93, 0.55 + 0.12 * present_labels)
    return PrecedentRecord(
        record_id=f"precedent-{index:05d}",
        source_kind="precedent_corpus",
        reaction_smiles=reaction_smiles,
        substrate_archetype=substrate_archetype,
        product_archetype=product_archetype,
        substrate_features=substrate_features,
        product_features=product_features,
        ligand_label=ligand_label,
        ligand_family=ligand_family,
        base_label=base_label,
        base_family=base_family,
        solvent_label=solvent_label,
        solvent_family=solvent_family,
        temperature_c=80.0,
        time_h=10.0,
        outcome_label=outcome_label,
        outcome_yield=round(outcome_yield, 4),
        confidence=round(confidence, 4),
        provenance=row.get("patent_name") or "precedent_corpus",
        raw_record={key: str(value) for key, value in row.items()},
    )


def _parse_exp_name(exp_name: str) -> tuple[str, str, str, float, float]:
    parts = [part.strip() for part in exp_name.split(",") if part.strip()]
    if len(parts) < 4:
        raise ValueError(f"Unexpected exp_name format: {exp_name}")
    temp_idx = next((index for index, part in enumerate(parts) if _TEMP_PATTERN.match(part)), len(parts) - 1)
    solvent_idx = next(
        (
            index
            for index, part in enumerate(parts)
            if any(token in part.lower() for token in ("dioxane", "thf", "toluene", "dmac", "dmf", "mecn", "water", "methanol"))
        ),
        max(1, temp_idx - 1),
    )
    base_idx = max(0, solvent_idx - 1)
    catalyst_text = _strip_numeric_prefix(" / ".join(parts[:base_idx]))
    base_text = _strip_numeric_prefix(parts[base_idx])
    solvent_text = _strip_numeric_prefix(parts[solvent_idx])
    temp_match = _TEMP_PATTERN.match(parts[temp_idx])
    if temp_match is None:
        raise ValueError(f"Unable to parse temperature block: {exp_name}")
    return catalyst_text, base_text, solvent_text, float(temp_match.group(1)), float(temp_match.group(2) or 10.0)


def _extract_ligand_from_catalyst(catalyst_text: str) -> tuple[str, str]:
    lowered = catalyst_text.lower()
    if " pd " in lowered:
        ligand_text = catalyst_text[: lowered.index(" pd ")].strip()
    elif "/" in catalyst_text:
        ligand_text = catalyst_text.split("/", 1)[1].strip()
    else:
        ligand_text = catalyst_text
    return _normalize_ligand_label(ligand_text)


def _make_case_record(row: dict[str, str], index: int, source_kind: str) -> PrecedentRecord | None:
    reaction_smiles = (row.get("SMILES") or "").strip()
    if not reaction_smiles:
        return None
    catalyst_text, base_text, solvent_text, temperature_c, time_h = _parse_exp_name(row.get("exp_name", ""))
    ligand_label, ligand_family = _extract_ligand_from_catalyst(catalyst_text)
    base_label, base_family = _normalize_base_label(base_text)
    solvent_label, solvent_family = _normalize_solvent_label(solvent_text)
    substrate_archetype, product_archetype, substrate_features, product_features = _extract_reaction_features(reaction_smiles)
    outcome_yield = 0.84 if source_kind == "success_bank" else 0.18
    return PrecedentRecord(
        record_id=f"{source_kind}-{index:04d}",
        source_kind=source_kind,
        reaction_smiles=reaction_smiles,
        substrate_archetype=substrate_archetype,
        product_archetype=product_archetype,
        substrate_features=substrate_features,
        product_features=product_features,
        ligand_label=ligand_label,
        ligand_family=ligand_family,
        base_label=base_label,
        base_family=base_family,
        solvent_label=solvent_label,
        solvent_family=solvent_family,
        temperature_c=temperature_c,
        time_h=time_h,
        outcome_label="success" if source_kind == "success_bank" else "failure",
        outcome_yield=outcome_yield,
        confidence=0.9,
        provenance=row.get("Rxn_ID") or source_kind,
        raw_record={key: str(value) for key, value in row.items()},
    )


def _top_counts(records: tuple[PrecedentRecord, ...], attribute: str, limit: int = 8) -> list[dict[str, object]]:
    counter = Counter(getattr(record, attribute) for record in records if getattr(record, attribute) != "unknown")
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


@lru_cache(maxsize=1)
def load_historical_dataset(dataset_root: str) -> HistoricalDataset:
    base = Path(dataset_root)
    precedent_rows = read_csv_rows(base / "DataAssets_rscore1_Buchwald.csv")
    success_rows = read_csv_rows(base / "success_buchwald.csv")
    failure_rows = read_csv_rows(base / "failed_buchwald.csv")

    precedents = tuple(
        record for index, row in enumerate(precedent_rows, start=1) if (record := _make_precedent_record(row, index)) is not None
    )
    success_cases = tuple(
        record for index, row in enumerate(success_rows, start=1) if (record := _make_case_record(row, index, "success_bank")) is not None
    )
    failure_cases = tuple(
        record for index, row in enumerate(failure_rows, start=1) if (record := _make_case_record(row, index, "failure_bank")) is not None
    )

    family_lookup = {"ligand_label": {}, "base_label": {}, "solvent_label": {}}
    for record in (*precedents, *success_cases, *failure_cases):
        family_lookup["ligand_label"][record.ligand_label] = record.ligand_family
        family_lookup["base_label"][record.base_label] = record.base_family
        family_lookup["solvent_label"][record.solvent_label] = record.solvent_family

    return HistoricalDataset(
        precedents=precedents,
        success_cases=success_cases,
        failure_cases=failure_cases,
        dataset_stats={
            "precedent_count": len(precedents),
            "success_bank_count": len(success_cases),
            "failure_bank_count": len(failure_cases),
            "top_ligands": _top_counts(precedents, "ligand_label"),
            "top_bases": _top_counts(precedents, "base_label"),
            "top_solvents": _top_counts(precedents, "solvent_label"),
        },
        family_lookup=family_lookup,
    )


def _retrieval_score(task: TaskRepresentation, record: PrecedentRecord) -> tuple[float, list[str]]:
    notes: list[str] = []
    feature_overlap = len(set(task.substrate_features) & set(record.substrate_features))
    feature_score = 0.18 * feature_overlap
    if feature_overlap:
        notes.append(f"substrate_feature_overlap={feature_overlap}")
    archetype_bonus = 0.0
    if task.profile_name.startswith("aza_heteroaryl") and "aza_heteroaryl" in record.substrate_archetype:
        archetype_bonus = 0.22
        notes.append("matched_aza_heteroaryl_archetype")
    elif task.profile_name.startswith("hindered_aryl") and "hindered_aryl_chloride" in record.substrate_archetype:
        archetype_bonus = 0.22
        notes.append("matched_hindered_aryl_archetype")
    elif task.profile_name == "bromothiophene_screen" and "heteroaryl_bromide" in record.substrate_archetype:
        archetype_bonus = 0.22
        notes.append("matched_heteroaryl_bromide_archetype")
    source_kind_bonus = {"success_bank": 0.12, "precedent_corpus": 0.08, "failure_bank": -0.08}.get(record.source_kind, 0.0)
    if source_kind_bonus:
        notes.append(f"source_kind={record.source_kind}")
    score = feature_score + archetype_bonus + 0.22 * record.outcome_yield + 0.12 * record.confidence + source_kind_bonus
    return round(score, 4), notes


def retrieve_support_bundle(
    task: TaskRepresentation,
    dataset: HistoricalDataset,
    precedent_pool_size: int,
    support_case_pool_size: int,
) -> RetrievedSupportBundle:
    def rank(records: tuple[PrecedentRecord, ...]) -> list[PrecedentRecord]:
        ranked: list[PrecedentRecord] = []
        for record in records:
            score, notes = _retrieval_score(task, record)
            ranked.append(PrecedentRecord(**{**record.__dict__, "retrieval_score": score, "retrieval_notes": notes}))
        ranked.sort(key=lambda item: item.retrieval_score, reverse=True)
        return ranked

    ranked_precedents = rank(dataset.precedents)[:precedent_pool_size]
    ranked_success = rank(dataset.success_cases)[:support_case_pool_size]
    ranked_failure = rank(dataset.failure_cases)[:support_case_pool_size]
    retrieval_rows: list[dict[str, object]] = []
    for bucket_name, records in (
        ("precedent_pool", ranked_precedents),
        ("success_support", ranked_success),
        ("failure_support", ranked_failure),
    ):
        for rank_index, record in enumerate(records, start=1):
            retrieval_rows.append(
                {
                    "bucket": bucket_name,
                    "rank": rank_index,
                    "record_id": record.record_id,
                    "source_kind": record.source_kind,
                    "ligand_label": record.ligand_label,
                    "base_label": record.base_label,
                    "solvent_label": record.solvent_label,
                    "temperature_c": record.temperature_c,
                    "outcome_label": record.outcome_label,
                    "outcome_yield": record.outcome_yield,
                    "retrieval_score": record.retrieval_score,
                    "retrieval_notes": "; ".join(record.retrieval_notes),
                }
            )
    return RetrievedSupportBundle(
        precedents=tuple(ranked_precedents),
        success_cases=tuple(ranked_success),
        failure_cases=tuple(ranked_failure),
        retrieval_table=tuple(retrieval_rows),
    )
