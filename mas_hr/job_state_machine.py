"""Mesin status permintaan lowongan (laporan 5.6.4).

Transisi yang tidak terdaftar ditolak. Inilah yang menjamin invarian: tidak
ada jalan menuju PLACED tanpa melewati persetujuan kontrak dan otorisasi
penempatan.
"""
from typing import Dict, List, Optional, Set

TRANSITIONS: Dict[str, Set[str]] = {
    "DRAFT": {"INTAKE_CLARIFYING"},
    "INTAKE_CLARIFYING": {"REQ_PENDING_APPROVAL"},
    "REQ_PENDING_APPROVAL": {"SOURCING", "INTAKE_CLARIFYING", "CLOSED"},
    "SOURCING": {"SCREENING", "ON_HOLD"},
    "SCREENING": {"COMPLIANCE_CHECK", "ON_HOLD"},
    "COMPLIANCE_CHECK": {"COMPLIANCE_REVIEW", "MATCHING", "CLOSED"},
    "COMPLIANCE_REVIEW": {"COMPLIANCE_CHECK", "MATCHING"},
    "MATCHING": {"SHORTLIST_PENDING_APPROVAL"},
    "SHORTLIST_PENDING_APPROVAL": {"INTERVIEW_SCHEDULING", "MATCHING", "SOURCING", "CLOSED"},
    "INTERVIEW_SCHEDULING": {"INTERVIEW_DONE", "ON_HOLD"},
    "INTERVIEW_DONE": {"HIRE_PENDING_APPROVAL"},
    "HIRE_PENDING_APPROVAL": {"CONTRACT_DRAFTING", "INTERVIEW_SCHEDULING", "CLOSED"},
    "CONTRACT_DRAFTING": {"CONTRACT_PENDING_APPROVAL", "CLOSED"},
    "CONTRACT_PENDING_APPROVAL": {"PLACEMENT_AUTHORIZED", "CONTRACT_DRAFTING", "CLOSED"},
    "PLACEMENT_AUTHORIZED": {"PLACED", "CLOSED"},
    "PLACED": {"MONITORING"},
    "MONITORING": {"CLOSED"},
    "ON_HOLD": {"SOURCING", "SCREENING", "CLOSED"},
    "CLOSED": set(),
}

# State yang hanya boleh dimasuki dari state tertentu (invarian keras).
GATED_ENTRY: Dict[str, Set[str]] = {
    "PLACED": {"PLACEMENT_AUTHORIZED"},
    "PLACEMENT_AUTHORIZED": {"CONTRACT_PENDING_APPROVAL"},
}


class IllegalTransition(Exception):
    """Percobaan transisi yang melanggar mesin status atau invarian gerbang."""


class JobStateMachine:
    """Melacak posisi satu permintaan lowongan dalam alur tujuh fase."""

    def __init__(self, job_id: str, audit=None):
        """Mulai dari state DRAFT.

        Args:
            job_id: identitas lowongan yang dilacak.
            audit: AuditTrail opsional; setiap transisi ikut tercatat.
        """
        self.job_id = job_id
        self.state = "DRAFT"
        self.history: List[str] = ["DRAFT"]
        self.audit = audit

    def can(self, target: str) -> bool:
        """Periksa apakah transisi ke `target` diizinkan dari state saat ini."""
        if target not in TRANSITIONS.get(self.state, set()):
            return False
        if target in GATED_ENTRY and self.state not in GATED_ENTRY[target]:
            return False
        return True

    def to(self, target: str) -> str:
        """Pindah ke state `target`.

        Raises:
            IllegalTransition: bila transisi tidak diizinkan.
        """
        if not self.can(target):
            raise IllegalTransition(
                f"{self.job_id}: transisi ilegal {self.state} -> {target}")
        previous, self.state = self.state, target
        self.history.append(target)
        if self.audit:
            self.audit.record("job", self.job_id, f"STATE_{target}", "SYSTEM",
                              "JobStateMachine", before={"state": previous},
                              after={"state": target})
        return self.state

    def close_if_possible(self) -> None:
        """Tutup permintaan bila state saat ini mengizinkan transisi ke CLOSED.

        Dipakai saat alur berhenti di tengah (mis. HITL-6 menolak) supaya
        tidak ada permintaan yang menggantung.
        """
        if self.state == "MONITORING" or self.can("CLOSED"):
            self.to("CLOSED")
