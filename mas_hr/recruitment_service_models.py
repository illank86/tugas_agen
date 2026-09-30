"""Tipe data ringan yang dipakai lintas layanan dan server API."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunSummary:
    """Ringkasan satu run untuk daftar; sengaja tanpa data kandidat."""

    run_id: str
    job_id: str
    job_title: str
    status: str
    submitted_at: str
    finished_at: Optional[str] = None
    candidates_evaluated: int = 0
    shortlisted: int = 0
    placed: int = 0
