"""Intake Agent — AI, static, deliberatif. Fase 1: Request & Intake.

Menerima permintaan klien yang ambigu, mengklarifikasi, lalu menormalkan
skill ke taksonomi kanonik. Klarifikasi disimulasikan sebagai pengisian
field yang belum lengkap. Bila lowongan diekstrak DeepSeek dari teks bebas
(llm_parsing.py), skill yang tidak dapat dipetakan ke taksonomi ikut dicatat
sebagai ambiguitas agar dilihat manusia di gerbang konfirmasi lowongan.
"""
from typing import Optional

from ..agent_messaging import Message, Performative, Risk
from ..skill_taxonomy import canonical_name, normalize
from .agent_base import Agent


class IntakeAgent(Agent):
    """Mengubah permintaan klien menjadi StructuredJobRequirement."""

    name = "IntakeAgent"

    def handle(self, message: Message) -> Optional[Message]:
        """Susun spesifikasi lowongan terstruktur dari permintaan mentah.

        Args:
            message: pesan berskema JobRequestRaw@1.0.

        Returns:
            Balasan berskema StructuredJobRequirement@1.0, lengkap dengan
            jumlah putaran klarifikasi dan daftar ambiguitas yang tersisa.
        """
        if message.schema != "JobRequestRaw@1.0":
            raise NotImplementedError(message.schema)

        job = self.context.jobs[message.payload["job_id"]]
        self.request_permission("normalize_skills", confidence=1.0)

        normalized, unresolved, clarification_turns = [], [], 0
        for requirement in job.required_skills:
            skill_id = normalize(canonical_name(requirement["skill_id"]))
            if skill_id is None:
                unresolved.append(requirement["skill_id"])
                clarification_turns += 1
                continue
            normalized.append({"skill_id": skill_id,
                               "skill": canonical_name(skill_id),
                               "importance": requirement["importance"],
                               "min_level": requirement["min_level"]})

        for raw_skill in job.unresolved_skills:
            unresolved.append(raw_skill)
            clarification_turns += 1

        # Field kosong memicu satu putaran tanya-jawab tambahan ke klien.
        if not job.min_experience_years:
            clarification_turns += 1
        if not job.location:
            clarification_turns += 1

        payload = {
            "job_id": job.job_id, "client_id": job.client_id, "title": job.title,
            "headcount": job.headcount, "required_skills": normalized,
            "min_experience_years": job.min_experience_years,
            "location": job.location, "sla_days": job.sla_days,
            "required_documents": job.required_documents,
            "clarification_turns": clarification_turns,
            "unresolved_ambiguities": unresolved,
            "parsed_by": job.parsed_by,
        }
        self.record("job", job.job_id, "INTAKE_STRUCTURED",
                    {"skills": len(normalized), "turns": clarification_turns,
                     "unresolved": unresolved})
        return self.reply(message, Performative.INFORM,
                          "StructuredJobRequirement@1.0", payload, Risk.LOW)
