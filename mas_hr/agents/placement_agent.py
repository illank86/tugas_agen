"""Placement Agent — hibrida, static, paling terkurung. Fase 7: Placement.

Agen ini menyentuh dokumen hukum, maka ruang keluarannya dipersempit sampai
kesalahan berbahaya menjadi MUSTAHIL SECARA STRUKTURAL, bukan sekadar tidak
dianjurkan: ia hanya boleh mengisi variabel pada template yang sudah
disetujui, dan tidak dapat mengarang klausul baru.
"""
from datetime import timedelta
from typing import Dict, List, Optional

from ..agent_messaging import Message, Performative, Risk
from .agent_base import Agent

# Template kontrak yang disetujui. Hanya {placeholder} yang boleh diisi.
CONTRACT_TEMPLATE = [
    "Pasal 1 - Identitas: Pekerja {name} ditempatkan pada {client}.",
    "Pasal 2 - Jabatan: {title}, lokasi {location}.",
    "Pasal 3 - Jangka waktu: mulai {start_date} selama {duration_months} bulan.",
    "Pasal 4 - Kepatuhan: dokumen {documents} telah diverifikasi ({compliance_ref}).",
    "Pasal 5 - Persetujuan: disetujui oleh {approver_id} pada {approved_at}.",
]

ALLOWED_PLACEHOLDERS = {"name", "client", "title", "location", "start_date",
                        "duration_months", "documents", "compliance_ref",
                        "approver_id", "approved_at"}


class PlacementAgent(Agent):
    """Menyusun draf kontrak dan mengeksekusi penempatan setelah disetujui."""

    name = "PlacementAgent"

    def fill_template(self, values: Dict[str, str]) -> List[str]:
        """Isi template kontrak dengan nilai yang diizinkan.

        Raises:
            ValueError: bila ada placeholder di luar daftar izin. Ini
                pertahanan terhadap penyisipan klausul baru.
        """
        unknown = set(values) - ALLOWED_PLACEHOLDERS
        if unknown:
            raise ValueError(f"placeholder tidak diizinkan pada kontrak: {unknown}")
        return [line.format(**values) for line in CONTRACT_TEMPLATE]

    def handle(self, message: Message) -> Optional[Message]:
        """Susun draf kontrak untuk satu kandidat yang sudah disetujui di-hire.

        Menyusun draf diizinkan; MENGEKSEKUSI tidak. Supervisor yang membuka
        gerbang HITL-5 dan HITL-6.
        """
        if message.schema != "PlacementAuthorization@1.0":
            raise NotImplementedError(message.schema)

        job = self.context.jobs[message.payload["job_id"]]
        candidate_id = message.payload["candidate_id"]
        candidate = self.context.candidates[candidate_id]
        start_date = self.context.settings.today + timedelta(days=7)

        clauses = self.fill_template({
            "name": candidate.name, "client": job.client_id, "title": job.title,
            "location": job.location, "start_date": start_date.isoformat(),
            "duration_months": "12",
            "documents": ", ".join(job.required_documents),
            "compliance_ref": f"compliance_checks/{candidate_id}",
            "approver_id": message.payload["approver_id"],
            "approved_at": self.context.settings.today.isoformat()})

        decision = self.request_permission("approve_contract", confidence=0.99)
        self.record("candidate", candidate_id, "CONTRACT_DRAFTED",
                    {"job_id": job.job_id, "clauses": len(clauses),
                     "autonomous": decision.autonomous})
        payload = {"job_id": job.job_id, "candidate_id": candidate_id,
                   "clauses": clauses, "start_date": start_date.isoformat(),
                   "requires_approval": not decision.autonomous,
                   "checkpoint": decision.checkpoint}
        return self.reply(message, Performative.PROPOSE, "ContractDraft@1.0",
                          payload, Risk.HIGH)

    def execute_placement(self, job, candidate_id: str, approver_id: str) -> dict:
        """Eksekusi penempatan. Hanya dipanggil setelah HITL-6 disetujui.

        Returns:
            Ringkasan penempatan berisi id, tanggal mulai, dan status onboarding.
        """
        decision = self.request_permission("execute_placement", confidence=0.99)
        start_date = (self.context.settings.today + timedelta(days=7)).isoformat()
        cursor = self.context.database.execute(
            """INSERT INTO placements
               (candidate_id, job_id, contract_ref, start_date, location, status,
                signed_at) VALUES (?,?,?,?,?,?,datetime('now'))""",
            (candidate_id, job.job_id, f"CTR-{job.job_id}-{candidate_id}",
             start_date, job.location, "PLACED"))
        self.record("candidate", candidate_id, "PLACEMENT_EXECUTED",
                    {"job_id": job.job_id, "approver": approver_id,
                     "autonomous": decision.autonomous})
        return {"placement_id": cursor.lastrowid, "job_id": job.job_id,
                "candidate_id": candidate_id, "start_date": start_date,
                "onboarding_status": "IN_PROGRESS"}
