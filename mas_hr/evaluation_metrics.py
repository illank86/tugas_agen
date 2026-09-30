"""Metrik evaluasi (laporan Bagian 9.2).

Tiga prinsip yang dijaga di sini:

1. T_mesin DIPISAHKAN dari T_e2e. Tanpa itu, arm dengan HITL tampak lambat
   semata-mata karena manusia butuh waktu berpikir — yang merupakan fitur.
2. Kepatuhan dinilai dengan F-beta (beta=2) yang menekankan recall. False
   negative (dokumen bermasalah lolos) menimbulkan risiko hukum; false
   positive hanya menambah satu tinjauan. Keduanya tidak setara, sehingga F1
   biasa akan menyesatkan.
3. Rasio kepatuhan di-MICRO-AVERAGE lintas ulangan. Merata-ratakan rasio dari
   run yang tidak punya kasus positif menghasilkan angka menyesatkan.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import math


def precision_recall_f(tp: int, fp: int, fn: int, beta: float = 1.0) -> Dict[str, float]:
    """Hitung precision, recall, F-beta, dan false negative rate."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta_squared = beta * beta
    f_score = ((1 + beta_squared) * precision * recall
               / (beta_squared * precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f_beta": round(f_score, 4),
            "fnr": round(fn / (tp + fn), 4) if (tp + fn) else 0.0}


def ndcg_at_k(relevance: List[int], k: int = 10) -> float:
    """Hitung normalized discounted cumulative gain untuk peringkat shortlist."""
    def dcg(values: List[int]) -> float:
        """Discounted cumulative gain dari daftar relevansi terurut."""
        return sum((2 ** value - 1) / math.log2(index + 2)
                   for index, value in enumerate(values[:k]))
    ideal = dcg(sorted(relevance, reverse=True))
    return round(dcg(relevance) / ideal, 4) if ideal else 0.0


def precision_at_k(relevance: List[int], k: int = 10) -> float:
    """Proporsi kandidat relevan di antara k teratas."""
    top = relevance[:k]
    return round(sum(top) / len(top), 4) if top else 0.0


def cohen_kappa(a: List[int], b: List[int]) -> float:
    """Kesepakatan dua penilai yang dikoreksi terhadap peluang."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    rate_a, rate_b = sum(a) / n, sum(b) / n
    expected = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else 1.0


def mann_whitney_u(sample_a: List[float], sample_b: List[float]) -> Dict[str, float]:
    """Uji non-parametrik dua sampel, tanpa dependensi eksternal.

    Dipakai karena data waktu eksekusi umumnya tidak berdistribusi normal,
    sehingga uji-t tidak tepat. Mengembalikan pula effect size rank-biserial;
    nilai p saja tidak memberi tahu besar perbedaannya.
    """
    n1, n2 = len(sample_a), len(sample_b)
    if n1 == 0 or n2 == 0:
        return {"u": 0.0, "z": 0.0, "p": 1.0, "effect_size": 0.0}
    combined = sorted([(value, 0) for value in sample_a]
                      + [(value, 1) for value in sample_b])
    ranks = [0.0] * len(combined)
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1
    rank_sum_a = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u_a = rank_sum_a - n1 * (n1 + 1) / 2
    u = min(u_a, n1 * n2 - u_a)
    mean = n1 * n2 / 2
    deviation = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) or 1.0
    z = (u - mean) / deviation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"u": round(u, 2), "z": round(z, 3), "p": round(p, 4),
            "effect_size": round(1 - 2 * u / (n1 * n2), 4)}


@dataclass
class RunResult:
    """Metrik satu kali eksekusi satu arm pada satu skenario."""

    arm: str
    scenario: str = ""
    elapsed: float = 0.0
    human_wait: float = 0.0
    messages: int = 0
    payload_bytes: int = 0
    candidates: int = 0
    compliance: Dict[str, float] = field(default_factory=dict)
    screening: Dict[str, float] = field(default_factory=dict)
    matching: Dict[str, float] = field(default_factory=dict)
    intervention: Dict[str, float] = field(default_factory=dict)
    approvals: int = 0
    trace_completeness: float = 0.0
    audit_records: int = 0
    audit_chain_valid: bool = True
    task_failures: int = 0
    schema_rejections: int = 0
    injection_attempts: int = 0
    injection_successes: int = 0
    blocked_unauthorized: int = 0
    shortlisted: int = 0
    placed: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def machine_time(self) -> float:
        """Waktu eksekusi tanpa waktu tunggu manusia."""
        return max(0.0, self.elapsed - self.human_wait)

    @property
    def injection_resistance(self) -> float:
        """Proporsi upaya injeksi yang tidak mengubah keluaran sistem."""
        if not self.injection_attempts:
            return 1.0
        return round(1 - self.injection_successes / self.injection_attempts, 4)

    def as_row(self) -> dict:
        """Ratakan menjadi satu baris datar untuk tabel dan agregasi."""
        return {
            "arm": self.arm, "scenario": self.scenario,
            "elapsed_s": round(self.elapsed, 4),
            "machine_s": round(self.machine_time, 4),
            "human_wait_s": round(self.human_wait, 4),
            "messages": self.messages, "payload_bytes": self.payload_bytes,
            "candidates": self.candidates,
            "compliance_recall": self.compliance.get("recall", 0.0),
            "compliance_fnr": self.compliance.get("fnr", 0.0),
            "compliance_tp": self.compliance.get("tp", 0),
            "compliance_fp": self.compliance.get("fp", 0),
            "compliance_fn": self.compliance.get("fn", 0),
            "screening_f1": self.screening.get("f_beta", 0.0),
            "screening_kappa": self.screening.get("kappa", 0.0),
            "ndcg_at_10": self.matching.get("ndcg_at_10", 0.0),
            "hir": round(self.intervention.get("hir", 0.0), 4),
            "hir_escalation": round(self.intervention.get("hir_escalation", 0.0), 4),
            "approvals": self.approvals,
            "trace_completeness": self.trace_completeness,
            "audit_chain_valid": self.audit_chain_valid,
            "injection_resistance": self.injection_resistance,
            "blocked_unauthorized": self.blocked_unauthorized,
            "shortlisted": self.shortlisted, "placed": self.placed,
        }


def aggregate_rows(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Hitung rata-rata dan simpangan baku tiap metrik, dikelompokkan per arm."""
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    summary: Dict[str, Dict[str, float]] = {}
    for arm, arm_rows in grouped.items():
        stats: Dict[str, float] = {"runs": len(arm_rows)}
        numeric_keys = [key for key, value in arm_rows[0].items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)]
        for key in numeric_keys:
            values = [row[key] for row in arm_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            stats[f"{key}_mean"] = round(mean, 4)
            stats[f"{key}_sd"] = round(math.sqrt(variance), 4)
        summary[arm] = stats
    return summary


def micro_average_compliance(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Jumlahkan tp/fp/fn lintas ulangan lebih dulu, baru hitung rasionya.

    Menandai `enough_samples` bila kasus positif kurang dari 10, karena rasio
    dari sampel sekecil itu tidak layak ditarik kesimpulan.
    """
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    summary: Dict[str, Dict[str, float]] = {}
    for arm, arm_rows in grouped.items():
        tp = sum(row.get("compliance_tp", 0) for row in arm_rows)
        fp = sum(row.get("compliance_fp", 0) for row in arm_rows)
        fn = sum(row.get("compliance_fn", 0) for row in arm_rows)
        metrics = precision_recall_f(tp, fp, fn, beta=2.0)
        metrics.update({"tp": tp, "fp": fp, "fn": fn, "positives": tp + fn,
                        "enough_samples": (tp + fn) >= 10})
        summary[arm] = metrics
    return summary
