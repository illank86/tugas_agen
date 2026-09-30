"""Matching Agent — AI, static, deliberatif. Fase 5: Matching. Titik BARRIER.

Skor skill dihitung PER SYARAT, bukan sebagai cosine antar-vektor agregat
(laporan 6.5.1). Cosine agregat mengukur kesamaan ARAH, bukan KECUKUPAN:
kandidat dengan separuh kemampuan di semua dimensi memperoleh cosine identik
dengan kandidat penuh, dan syarat yang tidak terpenuhi ikut tersamarkan.

    match_j = max_i cos(e(r_j), e(c_i)) * min(1, level_c / level_min_r)
    x_skill = sum_j beta_j * match_j / sum_j beta_j
"""
from typing import Dict, List, Optional, Tuple

from ..agent_messaging import Message, Performative, Risk
from ..settings import LEVELS
from ..skill_taxonomy import searchable_text
from ..text_similarity import cosine
from .agent_base import Agent


class MatchingAgent(Agent):
    """Menyusun peringkat kandidat dan mengusulkan shortlist."""

    name = "MatchingAgent"

    def __init__(self, context):
        """Siapkan agen beserta cache vektor skill."""
        super().__init__(context)
        self._skill_vectors: Dict[str, dict] = {}

    def _skill_vector(self, skill_id: str) -> dict:
        """Ambil (dan cache) vektor embedding sebuah skill kanonik."""
        if skill_id not in self._skill_vectors:
            self._skill_vectors[skill_id] = self.context.embedder.embed(
                searchable_text(skill_id))
        return self._skill_vectors[skill_id]

    def skill_score(self, candidate_skills: Dict[str, str],
                    job) -> Tuple[float, Dict[str, float]]:
        """Hitung skor kecukupan skill secara per-syarat.

        Args:
            candidate_skills: skill_id -> level milik kandidat.
            job: lowongan yang dinilai.

        Returns:
            Pasangan (skor agregat [0,1], skor per syarat). Skor per syarat
            dikembalikan agar recruiter dapat melihat syarat mana yang belum
            terpenuhi, bukan hanya angka akhirnya.
        """
        per_requirement: Dict[str, float] = {}
        numerator = denominator = 0.0
        for requirement in job.required_skills:
            requirement_vector = self._skill_vector(requirement["skill_id"])
            minimum_level = LEVELS[requirement["min_level"]]
            best = 0.0
            for skill_id, level in candidate_skills.items():
                similarity = cosine(requirement_vector, self._skill_vector(skill_id))
                level_ratio = min(1.0, LEVELS[level] / minimum_level)
                best = max(best, similarity * level_ratio)
            per_requirement[requirement["skill_id"]] = round(best, 4)
            numerator += requirement["importance"] * best
            denominator += requirement["importance"]
        return (numerator / denominator if denominator else 0.0), per_requirement

    def fit_score(self, screening: dict, compliance: dict, job,
                  client_preference: float = 0.9) -> dict:
        """Gabungkan seluruh komponen menjadi satu skor akhir dan penjelasannya.

        Kepatuhan berperan sebagai GERBANG PERKALIAN, bukan salah satu suku
        penjumlahan: kandidat yang gagal kepatuhan memperoleh skor 0 berapa
        pun kompetensinya (konflik K1).
        """
        weights = self.context.settings.weights()
        skill_value, per_requirement = self.skill_score(
            screening["extracted_skills"], job)
        experience_value = min(1.0, screening["experience_years"]
                               / max(0.5, job.min_experience_years))
        compliance_value = 1.0 if compliance["status"] == "PASS" else 0.0

        components = {"skill": weights["skill"] * skill_value,
                      "exp": weights["exp"] * experience_value,
                      "comp": weights["comp"] * compliance_value,
                      "pref": weights["pref"] * client_preference}
        raw_total = sum(components.values())
        gate = 1.0 if compliance["status"] == "PASS" else 0.0
        final_score = gate * raw_total
        divisor = raw_total or 1.0
        return {"candidate_id": screening["candidate_id"],
                "fit_score": round(final_score, 4),
                "compliance_gate": gate,
                "components": {k: round(v, 4) for k, v in components.items()},
                "contributions": {k: round(100 * v / divisor, 1)
                                  for k, v in components.items()},
                "per_requirement": per_requirement,
                "unmet_requirements": [rid for rid, value in per_requirement.items()
                                       if value < 0.6],
                "theta": self.context.settings.theta}

    def handle(self, message: Message) -> Optional[Message]:
        """Beri skor seluruh kandidat lalu usulkan shortlist.

        Ini titik BARRIER: seluruh kandidat harus sudah dinilai sebelum
        peringkat dapat disusun. Hasilnya berupa USULAN (level A1) yang masih
        menunggu persetujuan manusia.
        """
        if message.schema != "ComplianceStatus@1.0":
            raise NotImplementedError(message.schema)

        job = self.context.jobs[message.payload["job_id"]]
        screening_results = self.context.board(job.job_id).get("screening", {})
        ranking: List[dict] = []
        for status in message.payload["statuses"]:
            screening = screening_results.get(status["candidate_id"])
            if screening is None:
                continue
            ranking.append(self.fit_score(screening, status, job))
        ranking.sort(key=lambda item: item["fit_score"], reverse=True)

        # Peringkat LENGKAP ditaruh di blackboard, bukan di payload pesan:
        # hilir hanya butuh shortlist, tetapi antarmuka dan laporan butuh
        # seluruh kandidat yang dinilai beserta rincian skornya.
        self.context.board(job.job_id)["ranking"] = ranking

        settings = self.context.settings
        eligible = [r for r in ranking if r["compliance_gate"] == 1.0]
        shortlist = (eligible[:settings.top_k] if settings.top_k > 0
                     else [r for r in eligible if r["fit_score"] >= settings.theta])
        for rank, entry in enumerate(shortlist, start=1):
            entry["rank"] = rank
            self.context.database.execute(
                """INSERT INTO matching_scores
                   (candidate_id, job_id, fit_score, components_json, theta_used,
                    rank, created_at) VALUES (?,?,?,?,?,?,datetime('now'))""",
                (entry["candidate_id"], job.job_id, entry["fit_score"],
                 str(entry["components"]), settings.theta, rank))

        decision = self.request_permission("finalize_shortlist", confidence=0.95)
        self.record("job", job.job_id, "SHORTLIST_PROPOSED",
                    {"scored": len(ranking), "shortlisted": len(shortlist),
                     "theta": settings.theta, "autonomous": decision.autonomous})
        payload = {"job_id": job.job_id, "ranking": shortlist,
                   "n_scored": len(ranking), "theta": settings.theta,
                   "requires_approval": not decision.autonomous,
                   "checkpoint": decision.checkpoint}
        return self.reply(message, Performative.PROPOSE, "ShortlistProposal@1.0",
                          payload, Risk.HIGH)
