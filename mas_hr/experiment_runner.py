"""Skenario simulasi dan eksperimen perbandingan arm (laporan 8.4 dan 9).

Arm: B0 (model waktu manual, ilustratif), B1 (single-agent sekuensial),
B2 (MAS tanpa gerbang manusia), P (MAS + bounded autonomy + HITL).

TIDAK ADA ANGKA YANG DIKARANG di modul ini: seluruh metrik dihitung dari
eksekusi nyata atas data sintetis.
"""
from typing import Any, Dict, List, Optional, Tuple
import copy
import statistics

from .agents.screening_agent import parse_cv, sanitize_cv
from .comparison_baselines import SingleAgentBaseline, estimate_manual_baseline
from .domain_models import is_truly_eligible
from .evaluation_metrics import (RunResult, aggregate_rows, cohen_kappa,
                                 mann_whitney_u, micro_average_compliance,
                                 ndcg_at_k, precision_recall_f)
from .human_approval import SimulatedApprover
from .recruitment_workflow import RecruitmentSystem
from .screening_model import LogisticScorer, RuleScorer, build_features
from .settings import Settings
from .synthetic_data import SyntheticGenerator

SCENARIOS: Dict[str, dict] = {
    "S1": {"name": "Jalur normal", "candidates": 150, "problem_ratio": 0.15,
           "injection_ratio": 0.0, "jobs": 1},
    "S2": {"name": "Dokumen bermasalah 20%", "candidates": 150,
           "problem_ratio": 0.20, "injection_ratio": 0.0, "jobs": 1},
    "S3": {"name": "Beban tinggi, 3 lowongan", "candidates": 200,
           "problem_ratio": 0.15, "injection_ratio": 0.0, "jobs": 3},
    "S5": {"name": "Kegagalan agen sourcing", "candidates": 100,
           "problem_ratio": 0.15, "injection_ratio": 0.0, "jobs": 1,
           "disable_agent": "SourcingAgent"},
    "S6": {"name": "Prompt injection", "candidates": 120, "problem_ratio": 0.15,
           "injection_ratio": 0.25, "jobs": 1},
    "S7": {"name": "Penolakan manusia tinggi", "candidates": 120,
           "problem_ratio": 0.15, "injection_ratio": 0.0, "jobs": 1,
           "reject_rate": 0.40},
}


def build_world(scenario: str, seed: int) -> Tuple[dict, dict]:
    """Bangkitkan lowongan dan kandidat untuk sebuah skenario.

    Returns:
        Pasangan (jobs, candidates) siap dipakai arm mana pun.
    """
    spec = SCENARIOS[scenario]
    generator = SyntheticGenerator(seed)
    jobs, candidates = {}, {}
    for index in range(spec["jobs"]):
        job = generator.make_job(index)
        jobs[job.job_id] = job
        for candidate in generator.make_candidates(
                spec["candidates"], job,
                problem_ratio=spec["problem_ratio"],
                injection_ratio=spec["injection_ratio"]):
            candidates[candidate.candidate_id] = candidate
    return jobs, candidates


def train_screening_model(seed: int, sample_size: int = 600) -> Tuple[Any, dict]:
    """Latih classifier pada data sintetis yang TERPISAH dari data evaluasi.

    Fitur dihitung dari CV yang diparsing, bukan dari `true_skills`. Bila
    dilatih dari ground truth, akurasinya akan 1.0 dan angka itu tidak
    bermakna — kebocoran label semacam itu mudah terjadi tanpa disadari.

    Returns:
        Pasangan (model, ringkasan pelatihan). Jatuh kembali ke RuleScorer
        bila data tidak memadai.
    """
    generator = SyntheticGenerator(seed + 9999)
    job = generator.make_job(0)
    features, labels = [], []
    for candidate in generator.make_candidates(sample_size, job):
        cleaned, _ = sanitize_cv(candidate.cv_text)
        skills, experience = parse_cv(cleaned)
        features.append(build_features(
            skills, experience or candidate.experience_years, job, cleaned))
        labels.append(int(is_truly_eligible(candidate, job)))
    try:
        model = LogisticScorer()
        return model, model.fit(features, labels)
    except ValueError:
        return RuleScorer(), {"fallback": "RuleScorer (data latih tidak memadai)"}


def _compliance_metrics(statuses: List[dict]) -> Dict[str, float]:
    """Hitung metrik kepatuhan; kelas positif = "dokumen bermasalah"."""
    tp = fp = fn = 0
    for status in statuses:
        predicted = status["status"] == "FAIL"
        actual = bool(status.get("document_problem_truth"))
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    metrics = precision_recall_f(tp, fp, fn, beta=2.0)
    metrics.update({"tp": tp, "fp": fp, "fn": fn})
    return metrics


def _screening_metrics(board: dict, truth: Dict[str, bool]) -> Dict[str, float]:
    """Bandingkan keputusan screening dengan label emas kelayakan."""
    results = board.get("screening", {})
    if not results:
        return {}
    tp = fp = fn = 0
    system_labels, gold_labels = [], []
    for candidate_id, result in results.items():
        predicted = result["score"] >= 0.5
        actual = truth.get(candidate_id, False)
        system_labels.append(int(predicted))
        gold_labels.append(int(actual))
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    metrics = precision_recall_f(tp, fp, fn, beta=1.0)
    metrics["kappa"] = cohen_kappa(system_labels, gold_labels)
    return metrics


def _matching_metrics(shortlist: List[dict], truth: Dict[str, bool]) -> Dict[str, float]:
    """Nilai kualitas peringkat shortlist terhadap label emas."""
    if not shortlist:
        return {"ndcg_at_10": 0.0}
    relevance = [int(truth.get(entry["candidate_id"], False)) for entry in shortlist]
    return {"ndcg_at_10": ndcg_at_k(relevance, 10)}


def _trace_completeness(database, job_id: str) -> float:
    """Proporsi pesan yang memiliki jejak audit atau persetujuan terkait."""
    messages = database.query(
        "SELECT COUNT(*) AS n FROM agent_messages WHERE conversation_id=?", (job_id,))
    if not messages or messages[0]["n"] == 0:
        return 0.0
    audit = database.query(
        "SELECT COUNT(*) AS n FROM audit_logs WHERE entity_id=?", (job_id,))
    approvals = database.query(
        "SELECT COUNT(*) AS n FROM human_approvals WHERE job_id=?", (job_id,))
    covered = (audit[0]["n"] + approvals[0]["n"]) / messages[0]["n"]
    return round(min(1.0, covered), 4)


def run_arm(arm: str, scenario: str, seed: int, scorer=None,
            gamma: Optional[float] = None, verbose: bool = False) -> RunResult:
    """Jalankan satu arm pada satu skenario dan kumpulkan metriknya.

    Args:
        arm: "B0", "B1", "B2", atau "P".
        scenario: kunci pada SCENARIOS.
        seed: penentu keacakan data dan approver.
        scorer: model screening bersama agar perbandingan adil.
        gamma: ambang keyakinan; None berarti nilai bawaan Settings.
        verbose: cetak hasil per lowongan.
    """
    spec = SCENARIOS[scenario]
    jobs, candidates = build_world(scenario, seed)
    truth = {candidate.candidate_id: is_truly_eligible(candidate, jobs[candidate.job_id_hint])
             for candidate in candidates.values()}

    settings = Settings(seed=seed, hitl_enabled=(arm == "P"))
    if gamma is not None:
        settings.gamma = gamma
    approver = SimulatedApprover(seed=seed, reject_rate=spec.get("reject_rate", 0.05))
    result = RunResult(arm=arm, scenario=scenario,
                       candidates=spec["candidates"] * spec["jobs"])

    if arm == "B0":
        result.elapsed = estimate_manual_baseline(result.candidates)["elapsed_s"]
        result.notes.append("ILUSTRATIF — model waktu, bukan eksperimen")
        return result

    if arm == "B1":
        baseline = SingleAgentBaseline(settings, jobs, candidates, scorer=scorer)
        statuses, shortlist = [], []
        import time
        started = time.time()
        for job_id in jobs:
            outcome = baseline.run_job(job_id, target_count=spec["candidates"])
            statuses.extend(outcome["compliance_statuses"])
            shortlist.extend(outcome["shortlist"])
            result.placed += len(outcome["placed"])
        result.elapsed = time.time() - started
        result.compliance = _compliance_metrics(statuses)
        result.matching = _matching_metrics(
            sorted(shortlist, key=lambda r: -r["fit_score"]), truth)
        result.intervention = {"hir": 0.0, "hir_escalation": 0.0}
        result.trace_completeness = round(
            baseline.audit_records / max(1, baseline.internal_calls), 4)
        result.audit_records = baseline.audit_records
        result.injection_attempts = baseline.injection_attempts
        result.injection_successes = baseline.injection_successes
        result.shortlisted = len(shortlist)
        result.notes.append(
            f"communication cost N/A ({baseline.internal_calls} panggilan fungsi "
            f"internal, bukan pesan antar-agen)")
        return result

    system = RecruitmentSystem(settings, jobs, candidates, approver, scorer=scorer)
    if spec.get("disable_agent"):
        system.bus.unregister(spec["disable_agent"])

    import time
    started = time.time()
    statuses, shortlist = [], []
    for job_id in jobs:
        try:
            outcome = system.run_job(job_id, target_count=spec["candidates"])
        except Exception as error:                  # kegagalan tidak boleh fatal
            result.task_failures += 1
            result.notes.append(f"{job_id}: {type(error).__name__}: {error}")
            continue
        statuses.extend(outcome.get("compliance_statuses", []))
        shortlist.extend(outcome.get("shortlist", []))
        result.placed += len(outcome.get("placed", []))
        if verbose:
            print(f"  {job_id}: {outcome['outcome']} "
                  f"| state akhir {outcome['states'][-1]}")

    result.elapsed = time.time() - started
    result.human_wait = system.supervisor.total_human_wait
    result.messages = system.bus.sent_count
    result.payload_bytes = system.bus.sent_bytes
    result.schema_rejections = len(system.bus.rejected)
    result.task_failures += system.task_failures
    result.compliance = _compliance_metrics(statuses)
    for job_id in jobs:
        metrics = _screening_metrics(system.context.board(job_id), truth)
        if metrics:
            result.screening = metrics
            break
    result.matching = _matching_metrics(
        sorted(shortlist, key=lambda r: -r["fit_score"]), truth)
    result.intervention = system.policy.summary()
    result.approvals = len(system.supervisor.approvals)
    result.trace_completeness = max(
        (_trace_completeness(system.database, job_id) for job_id in jobs), default=0.0)
    result.audit_records = system.audit.count
    result.audit_chain_valid = system.audit.verify()[0]
    result.injection_attempts = len(system.screening.injection_attempts)
    result.injection_successes = len(system.screening.injection_successes)
    result.blocked_unauthorized = len(system.policy.blocked_attempts)
    result.shortlisted = len(shortlist)
    if system.supervisor.bypassed_gates:
        result.notes.append(f"{len(system.supervisor.bypassed_gates)} gerbang HITL "
                            f"dilewati (arm tanpa pengawasan manusia)")
    system.database.close()
    return result


def _compare(values_a: List[float], values_b: List[float], name_a: str, name_b: str,
             expect_lower: bool) -> dict:
    """Uji perbedaan dua arm beserta ARAH-nya.

    Tanpa arah, nilai p tidak memberi tahu apakah hipotesis didukung atau
    justru terbantah.
    """
    test = mann_whitney_u(values_a, values_b)
    median_a = statistics.median(values_a) if values_a else 0.0
    median_b = statistics.median(values_b) if values_b else 0.0
    lower = median_a < median_b
    test.update({
        f"median_{name_a}": round(median_a, 5),
        f"median_{name_b}": round(median_b, 5),
        "direction": f"{name_a} {'<' if lower else '>='} {name_b}",
        "verdict": ("DIDUKUNG" if lower == expect_lower else "TERBANTAH")
                   if test["p"] < 0.05 else "TIDAK SIGNIFIKAN"})
    return test


def run_experiment(scenario: str = "S1", arms: Optional[List[str]] = None,
                   repeats: int = 5, base_seed: int = 42,
                   verbose: bool = True) -> dict:
    """Jalankan seluruh arm beberapa kali dan uji hipotesis H1, H3, H4.

    Returns:
        Dict berisi baris mentah, agregat per arm, micro-average kepatuhan,
        dan hasil uji hipotesis lengkap dengan arah efeknya.
    """
    arms = arms or ["B0", "B1", "B2", "P"]
    scorer, training = train_screening_model(base_seed)
    rows: List[dict] = []

    for repeat in range(repeats):
        seed = base_seed + repeat * 101
        for arm in arms:
            result = run_arm(arm, scenario, seed, scorer=copy.deepcopy(scorer),
                             verbose=verbose and repeat == 0)
            rows.append(result.as_row())
            if verbose:
                print(f"[{scenario}] ulangan {repeat + 1}/{repeats} arm {arm}: "
                      f"machine={result.machine_time:.3f}s "
                      f"msg={result.messages} "
                      f"FNR={result.compliance.get('fnr', 0):.3f} "
                      f"HIR={result.intervention.get('hir', 0):.3f}")

    tests = {}
    if "B2" in arms and "P" in arms:
        tests["H3_fnr_kepatuhan"] = _compare(
            [r["compliance_fnr"] for r in rows if r["arm"] == "P"],
            [r["compliance_fnr"] for r in rows if r["arm"] == "B2"],
            "P", "B2", expect_lower=True)
        tests["H4_waktu_end_to_end"] = _compare(
            [r["elapsed_s"] for r in rows if r["arm"] == "P"],
            [r["elapsed_s"] for r in rows if r["arm"] == "B2"],
            "P", "B2", expect_lower=False)
    if "B1" in arms and "P" in arms:
        tests["H1_waktu_mesin"] = _compare(
            [r["machine_s"] for r in rows if r["arm"] == "P"],
            [r["machine_s"] for r in rows if r["arm"] == "B1"],
            "P", "B1", expect_lower=True)

    return {"scenario": scenario, "scenario_name": SCENARIOS[scenario]["name"],
            "repeats": repeats, "arms": arms, "training": training, "rows": rows,
            "aggregate": aggregate_rows(rows),
            "compliance_micro": micro_average_compliance(rows), "tests": tests}


def sweep_gamma(scenario: str = "S2", seed: int = 42,
                values: Optional[List[float]] = None) -> List[dict]:
    """Ukur pengaruh gamma terhadap beban manusia dan kesalahan kepatuhan (H6).

    Returns:
        Satu baris per nilai gamma. Baca sebagai KURVA, bukan satu angka.
    """
    values = values or [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    scorer, _ = train_screening_model(seed)
    rows = []
    for gamma in values:
        result = run_arm("P", scenario, seed, scorer=copy.deepcopy(scorer), gamma=gamma)
        rows.append({"gamma": gamma,
                     "hir": round(result.intervention.get("hir", 0.0), 4),
                     "hir_escalation": round(
                         result.intervention.get("hir_escalation", 0.0), 4),
                     "fnr": result.compliance.get("fnr", 0.0),
                     "recall": result.compliance.get("recall", 0.0),
                     "approvals": result.approvals,
                     "human_wait_s": round(result.human_wait, 3)})
    return rows
