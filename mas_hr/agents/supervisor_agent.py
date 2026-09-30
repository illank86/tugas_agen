"""Supervisor / Orchestrator Agent — AI, static, deliberatif.

Tanggung jawab: manajemen tujuan, perencanaan dan routing tugas, manajemen
state, resolusi konflik, pemantauan KPI, dan pembukaan gerbang HITL.

Supervisor tidak menyimpan state global di memori proses: state ada pada
JobStateMachine dan database. Ini mitigasi terhadap risiko Supervisor menjadi
titik kompleksitas tunggal.
"""
from typing import Dict, List, Optional
import time

from ..agent_messaging import Message, Performative, Risk
from ..human_approval import ApprovalRequest
from ..job_state_machine import JobStateMachine
from .agent_base import Agent

# Pesan yang hanya diamati Supervisor untuk KPI; alurnya dikendalikan
# secara eksplisit oleh workflow, bukan oleh balasan atas pesan ini.
OBSERVED_SCHEMAS = {"SourcingBid@1.0", "ComplianceEscalation@1.0"}


class SupervisorAgent(Agent):
    """Mengoordinasikan tujuh agen spesialis dan menjaga kendali manusia."""

    name = "SupervisorAgent"

    def __init__(self, context):
        """Siapkan supervisor tanpa state global di memori."""
        super().__init__(context)
        self.state_machines: Dict[str, JobStateMachine] = {}
        self.approvals: List[dict] = []
        self.bypassed_gates: List[dict] = []
        self.escalations: List[dict] = []
        self.conflicts: List[dict] = []
        self.total_human_wait = 0.0

    def state_machine(self, job_id: str) -> JobStateMachine:
        """Ambil (atau buat) mesin status untuk sebuah lowongan."""
        if job_id not in self.state_machines:
            self.state_machines[job_id] = JobStateMachine(job_id, self.context.audit)
        return self.state_machines[job_id]

    def handle(self, message: Message) -> Optional[Message]:
        """Terima pesan pengamatan; alur utama dikendalikan oleh workflow."""
        if message.schema == "ComplianceEscalation@1.0":
            self.escalations.append(dict(message.payload))
            return None
        if message.schema in OBSERVED_SCHEMAS:
            return None
        raise NotImplementedError(f"Supervisor tidak menangani {message.schema}")

    def request_approval(self, checkpoint: str, decision_type: str, evidence: dict,
                         job_id: str, candidate_id: Optional[str] = None) -> dict:
        """Buka gerbang HITL dan tunggu keputusan manusia.

        Ini titik BARRIER: proses benar-benar berhenti sampai ada jawaban.
        Bila `hitl_enabled` False (arm B2), gerbang dilewati dan dicatat
        terpisah dari approval — menggabungkannya akan membuat arm tanpa
        pengawasan tampak seolah punya pengawasan.

        Returns:
            Dict keputusan berisi decision, approver_id, approver_role, reason.
        """
        if not self.context.settings.hitl_enabled:
            self.bypassed_gates.append({"checkpoint": checkpoint, "job_id": job_id,
                                        "decision_type": decision_type})
            self.record("job", job_id, f"{checkpoint}_BYPASSED_NO_HITL",
                        {"decision_type": decision_type})
            return {"decision": "APPROVED", "approver_id": "SYSTEM",
                    "approver_role": "SYSTEM",
                    "reason": "HITL dimatikan (arm B2)", "latency_seconds": 0.0}

        # Label emas tidak pernah ikut dalam pesan yang melintasi bus.
        self.send("Human", Performative.REQUEST, "ApprovalRequest@1.0",
                  {"checkpoint_id": checkpoint, "decision_type": decision_type,
                   "evidence": {k: v for k, v in evidence.items()
                                if k != "document_problem_truth"}},
                  job_id, job_id, Risk.HIGH)

        started = time.time()
        result = self.context.approver.review(ApprovalRequest(
            checkpoint_id=checkpoint, decision_type=decision_type, evidence=evidence,
            job_id=job_id, candidate_id=candidate_id))
        waited = time.time() - started
        self.total_human_wait += waited

        self.context.database.execute(
            """INSERT INTO human_approvals
               (job_id, candidate_id, checkpoint_id, decision, approver_id,
                approver_role, reason, decided_at, latency_seconds)
               VALUES (?,?,?,?,?,?,?,datetime('now'),?)""",
            (job_id, candidate_id, checkpoint, result.decision, result.approver_id,
             result.approver_role, result.reason, result.latency_seconds))
        self.approvals.append({"checkpoint": checkpoint, "decision": result.decision,
                               "approver_id": result.approver_id, "latency": waited,
                               "job_id": job_id, "candidate_id": candidate_id})
        self.record("job", job_id, f"{checkpoint}_{result.decision}",
                    {"approver": result.approver_id, "role": result.approver_role,
                     "reason": result.reason, "candidate_id": candidate_id})
        return {"decision": result.decision, "approver_id": result.approver_id,
                "approver_role": result.approver_role, "reason": result.reason,
                "latency_seconds": result.latency_seconds}

    def resolve_conflict(self, candidate_id: str, screening: dict,
                         compliance: dict) -> str:
        """Selesaikan konflik antara skor screening dan status kepatuhan.

        Aturan K1: compliance FAIL memiliki prioritas leksikografis atas skor
        kompetensi setinggi apa pun. Risiko hukum tidak dapat dikompensasi.

        Returns:
            "REJECT" bila kandidat harus digugurkan, "CONTINUE" bila tidak.
        """
        if compliance["status"] == "FAIL" and screening["score"] >= 0.7:
            self.conflicts.append({"code": "K1", "candidate_id": candidate_id,
                                   "screening_score": screening["score"],
                                   "resolution": "VETO_COMPLIANCE"})
            self.record("candidate", candidate_id, "CONFLICT_K1_RESOLVED",
                        {"screening_score": screening["score"],
                         "resolution": "veto compliance (prioritas leksikografis)"})
            return "REJECT"
        return "CONTINUE"
