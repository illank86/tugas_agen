"""Arm pembanding B0 dan B1 (laporan Bagian 9.1).

B1 dibuat sejujur mungkin: satu fungsi monolitik, satu konteks, tanpa batas
agen, tanpa validasi skema, tanpa gerbang manusia, tanpa audit per-langkah.
Itu bukan strawman — itu memang bentuk sistem yang dibangun orang ketika
tidak memakai arsitektur agen.

Perbedaan terpenting bukan kecepatannya, melainkan: tidak ada pembatas
kewenangan, satu identitas untuk semua akses, jejak audit tidak lengkap, dan
ketidakpastian pembacaan dokumen diperlakukan sebagai lolos.
"""
from typing import Any, Dict, List
import time

from .agents.screening_agent import contains_injection, parse_cv
from .screening_model import RuleScorer, build_features
from .settings import Settings
from .skill_taxonomy import searchable_text
from .text_similarity import cosine, get_embedder

# Ambang yang dipakai monolit untuk "ragu-ragu tapi diteruskan".
MONOLITH_UNCERTAIN_THRESHOLD = 0.85


class SingleAgentBaseline:
    """Arm B1: tujuh fase dieksekusi berurutan dalam satu konteks tunggal."""

    arm = "B1"

    def __init__(self, settings: Settings, jobs: Dict[str, Any],
                 candidates: Dict[str, Any], scorer=None):
        """Siapkan baseline dengan sumber data yang sama seperti arm MAS."""
        self.settings = settings
        self.jobs = jobs
        self.candidates = candidates
        self.scorer = scorer or RuleScorer()
        self.embedder = get_embedder(False)
        self._vectors: Dict[str, dict] = {}
        self.internal_calls = 0
        self.audit_records = 0
        self.injection_attempts = 0
        self.injection_successes = 0

    def _vector(self, skill_id: str) -> dict:
        """Ambil (dan cache) vektor embedding sebuah skill."""
        if skill_id not in self._vectors:
            self._vectors[skill_id] = self.embedder.embed(searchable_text(skill_id))
        return self._vectors[skill_id]

    def _check_documents(self, candidate, job) -> str:
        """Verifikasi dokumen ala monolit.

        Dokumen yang tidak terbaca DILEWATI begitu saja — inilah sumber false
        negative kepatuhan yang membedakan arm ini dari arm dengan HITL.
        """
        for document in candidate.documents:
            if document.kind not in job.required_documents:
                continue
            if not document.present:
                return "FAIL"
            if document.ocr_confidence < MONOLITH_UNCERTAIN_THRESHOLD:
                continue                      # <-- ketidakpastian dianggap lolos
            if document.expires_at and document.expires_at < self.settings.today:
                return "FAIL"
            if not document.name_consistent:
                return "FAIL"
        return "PASS"

    def _naive_fit(self, skills: Dict[str, str], experience: float, job) -> float:
        """Hitung skor dengan cosine agregat, cara "naif" tanpa coverage-aware."""
        total_importance = sum(r["importance"] for r in job.required_skills) or 1.0
        weighted = 0.0
        for requirement in job.required_skills:
            requirement_vector = self._vector(requirement["skill_id"])
            best = max((cosine(requirement_vector, self._vector(skill_id))
                        for skill_id in skills), default=0.0)
            weighted += requirement["importance"] * best
        skill_value = weighted / total_importance
        experience_value = min(1.0, experience / max(0.5, job.min_experience_years))
        weights = self.settings.weights()
        return (weights["skill"] * skill_value + weights["exp"] * experience_value
                + weights["comp"] * 1.0 + weights["pref"] * 0.9)

    def run_job(self, job_id: str, target_count: int = 200) -> dict:
        """Jalankan seluruh fase secara berurutan untuk satu lowongan."""
        job = self.jobs[job_id]
        started = time.time()
        pool = list(self.candidates.values())[:target_count]
        shortlist: List[dict] = []
        compliance_statuses: List[dict] = []

        for candidate in pool:
            self.internal_calls += 1
            # Tidak ada sanitisasi: teks CV masuk apa adanya.
            skills, parsed_experience = parse_cv(candidate.cv_text)
            experience = parsed_experience or candidate.experience_years
            features = build_features(skills, experience, job, candidate.cv_text)
            score, _ = self.scorer.predict(features)
            if contains_injection(candidate.cv_text):
                self.injection_attempts += 1
                self.injection_successes += 1
                # Monolit mengambil "petunjuk" dari teks dokumen — pola yang
                # persis terjadi ketika prompt dan data tidak dipisahkan.
                score = min(1.0, score + 0.45)
            if score < self.settings.screening_pass_threshold:
                continue

            status = self._check_documents(candidate, job)
            compliance_statuses.append({
                "candidate_id": candidate.candidate_id, "status": status,
                "document_problem_truth": candidate.document_problem_truth})
            if status == "FAIL":
                continue
            fit = self._naive_fit(skills, experience, job)
            if fit >= self.settings.theta:
                shortlist.append({"candidate_id": candidate.candidate_id,
                                  "fit_score": round(fit, 4)})

        shortlist.sort(key=lambda item: -item["fit_score"])
        placed = [{"candidate_id": entry["candidate_id"], "job_id": job_id}
                  for entry in shortlist[:job.headcount]]
        self.audit_records = 1          # satu catatan untuk satu lowongan
        return {"job_id": job_id, "shortlist": shortlist, "placed": placed,
                "compliance_statuses": compliance_statuses,
                "elapsed": time.time() - started,
                "outcome": f"{len(placed)} ditempatkan tanpa persetujuan manusia"}


def estimate_manual_baseline(candidate_count: int,
                             seconds_per_screening: float = 180.0,
                             seconds_per_document_check: float = 120.0,
                             seconds_per_shortlist_decision: float = 300.0,
                             documents_per_candidate: int = 4,
                             shortlist_size: int = 10) -> dict:
    """Arm B0: model waktu proses manual.

    PENTING: ini BUKAN hasil eksperimen. Nilai default hanya placeholder agar
    kode dapat dijalankan. Ganti dengan studi waktu yang kelompok lakukan
    sendiri, dan beri label "ilustratif" pada setiap keluarannya.
    """
    total = (candidate_count * seconds_per_screening
             + candidate_count * 0.4 * documents_per_candidate
             * seconds_per_document_check
             + shortlist_size * seconds_per_shortlist_decision)
    return {"arm": "B0", "elapsed_s": round(total, 1),
            "note": "ILUSTRATIF — ganti dengan studi waktu kelompok"}
