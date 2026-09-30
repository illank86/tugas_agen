"""Struktur data domain yang dipakai bersama seluruh modul.

Dipisahkan dari generator dan pembaca berkas supaya sumber data mana pun —
sintetis, berkas .txt, atau PDF — menghasilkan objek yang sama, dan agen
tidak perlu tahu dari mana data itu berasal.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
import hashlib


@dataclass
class Document:
    """Satu dokumen kepatuhan milik kandidat (KTP, ijazah, SKCK, dsb.).

    Attributes:
        ocr_confidence: keyakinan pembacaan dokumen. Pada prototipe nilainya
            berasal dari data sintetis atau metadata berkas, bukan dari OCR
            sungguhan.
        name_consistent: apakah nama pada dokumen konsisten dengan data
            kandidat; sumber temuan aturan R-003.
        present: False berarti dokumen tidak diserahkan sama sekali.
    """

    doc_id: str
    kind: str
    issue_date: date
    expires_at: Optional[date] = None
    ocr_confidence: float = 0.95
    name_consistent: bool = True
    present: bool = True

    @property
    def sha256(self) -> str:
        """Hash identitas dokumen; disimpan menggantikan isi dokumen itu sendiri."""
        return hashlib.sha256(self.doc_id.encode()).hexdigest()


@dataclass
class JobRequirement:
    """Spesifikasi lowongan yang sudah terstruktur dan ternormalisasi.

    Attributes:
        required_skills: daftar dict {skill_id, importance, min_level}.
            `importance` adalah beta_j pada rumus coverage-aware matching dan
            harus berjumlah 1.0.
    """

    job_id: str
    title: str
    client_id: str = "CLI-0031"
    headcount: int = 1
    required_skills: List[dict] = field(default_factory=list)
    min_experience_years: float = 0.0
    location: str = "Sleman, DIY"
    sla_days: int = 14
    required_documents: List[str] = field(default_factory=lambda: [
        "KTP", "Ijazah", "SKCK", "Surat Keterangan Sehat"])

    def validate(self) -> None:
        """Pastikan bobot skill berjumlah 1.0.

        Raises:
            ValueError: bila kosong atau jumlah importance bukan 1.0. Bobot
                yang tidak berjumlah 1 membuat skor skill keluar dari skala
                [0,1] sehingga tidak lagi sebanding antar-lowongan — padahal
                optimasi penugasan lintas lowongan mengandalkan itu.
        """
        if not self.required_skills:
            raise ValueError(f"{self.job_id}: required_skills kosong")
        total = sum(s["importance"] for s in self.required_skills)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{self.job_id}: jumlah importance = {total:.3f}, harus 1.0")


@dataclass
class Candidate:
    """Profil kandidat beserta dokumen dan label evaluasi.

    Attributes:
        true_skills: skill sebenarnya. Hanya terisi pada data sintetis, di
            mana kita yang membangkitkannya. Untuk CV dari berkas, dict ini
            kosong dan metrik akurasi tidak berlaku.
        document_problem_truth: label emas kepatuhan, hanya untuk pengukuran
            dan untuk approver tersimulasi. Nilai ini TIDAK PERNAH dikirim
            melalui message bus.
        interview_quality: nilai yang dipakai simulator wawancara. Untuk CV
            dari berkas nilainya placeholder dan hasil wawancara tidak
            bermakna — pakai mode interaktif agar manusia yang menilai.
    """

    candidate_id: str
    name: str
    location: str = ""
    experience_years: float = 0.0
    cv_text: str = ""
    documents: List[Document] = field(default_factory=list)
    true_skills: Dict[str, str] = field(default_factory=dict)
    source: str = "db-internal"
    persona: str = "unknown"
    job_id_hint: str = ""
    has_injection: bool = False
    document_problem_truth: bool = False
    interview_quality: float = 0.7
    dedup_hash: str = ""


def is_truly_eligible(candidate: Candidate, job: JobRequirement) -> bool:
    """Label emas kelayakan kandidat terhadap sebuah lowongan.

    Hanya bermakna untuk data sintetis, karena bergantung pada `true_skills`.
    Untuk CV dari berkas selalu mengembalikan False dan tidak boleh dipakai
    sebagai dasar metrik.
    """
    from .settings import LEVELS
    if not candidate.true_skills:
        return False
    total = sum(r["importance"] for r in job.required_skills) or 1.0
    covered = 0.0
    for requirement in job.required_skills:
        level = candidate.true_skills.get(requirement["skill_id"])
        if level and LEVELS[level] >= LEVELS[requirement["min_level"]]:
            covered += requirement["importance"]
    return (covered / total >= 0.72
            and candidate.experience_years >= job.min_experience_years)
