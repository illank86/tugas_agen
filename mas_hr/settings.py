"""Parameter sistem, matriks risiko aksi, dan allowlist aksi per agen.

Satu dataclass datar `Settings` menggantikan empat dataclass bersarang pada
versi sebelumnya. Seluruh parameter yang bisa disetel ada di satu tempat,
sehingga mudah dilihat dan diubah saat demo.

CATATAN KALIBRASI: `theta` dan bobot `w_*` masih ditetapkan secara arbitrer
(lihat laporan Bagian 6.5.1). Keduanya harus dikalibrasi bila data historis
tersedia, dan sampai itu terjadi jangan diklaim sebagai nilai optimal.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Set

# Level kompetensi ordinal; dipakai coverage-aware matching dan rule engine.
LEVELS: Dict[str, int] = {"basic": 1, "intermediate": 2, "advanced": 3}


@dataclass
class Settings:
    """Seluruh parameter sistem dalam satu objek.

    Attributes:
        w_skill, w_exp, w_comp, w_pref: bobot komponen skor akhir; harus
            berjumlah 1.0 agar skor berada pada skala [0,1] yang sebanding
            antar-lowongan.
        theta: ambang minimum masuk shortlist.
        top_k: bila > 0, ambil k teratas dan abaikan theta.
        screening_pass_threshold: ambang skor lolos ke fase compliance.
        batch_size: ukuran batch streaming Sourcing -> Screening. Bila
            disetel >= jumlah kandidat, pipeline berubah menjadi serial murni
            dan keunggulan paralelisme hilang.
        max_workers: jumlah thread untuk paralelisme antar-kandidat/dokumen.
        ocr_unreadable_threshold: di bawah ini dokumen dianggap TIDAK TERBACA.
        skck_max_age_days: batas usia SKCK [ASUMSI, perlu verifikasi kebijakan].
        gamma: ambang keyakinan minimum untuk eksekusi tanpa manusia.
        tau: ambang risiko R(a)=L(a)*I(a) ternormalisasi.
        hitl_enabled: False berarti seluruh gerbang manusia dimatikan (arm B2).
    """

    # -- matching ----------------------------------------------------------
    w_skill: float = 0.40
    w_exp: float = 0.30
    w_comp: float = 0.20
    w_pref: float = 0.10
    theta: float = 0.75
    top_k: int = 0

    # -- screening ---------------------------------------------------------
    screening_pass_threshold: float = 0.45
    batch_size: int = 20
    max_workers: int = 8

    # -- compliance --------------------------------------------------------
    ocr_unreadable_threshold: float = 0.70
    skck_max_age_days: int = 180

    # -- bounded autonomy --------------------------------------------------
    gamma: float = 0.85
    tau: float = 0.60
    hitl_enabled: bool = True

    # -- umum --------------------------------------------------------------
    seed: int = 42
    today: date = date(2026, 9, 15)
    hmac_key: bytes = b"prototype-only-not-a-real-secret"

    def weights(self) -> Dict[str, float]:
        """Kembalikan bobot komponen skor sebagai dict."""
        return {"skill": self.w_skill, "exp": self.w_exp,
                "comp": self.w_comp, "pref": self.w_pref}

    def validate(self) -> None:
        """Pastikan bobot berjumlah 1.0; lempar ValueError bila tidak."""
        total = sum(self.weights().values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"bobot harus berjumlah 1.0, sekarang {total}")


# ---------------------------------------------------------------------------
# Matriks risiko-otonomi (laporan Bagian 5.9.3)
#   impact     : I(a), skala 1-5
#   level      : A3 otonom | A2 otonom+audit | A1 usul+approval | A0 manusia
#   reversible : kriteria utama penentuan level
#   checkpoint : gerbang HITL yang dibuka bila aksi tidak otonom
# ---------------------------------------------------------------------------
ACTIONS: Dict[str, dict] = {
    "parse_cv":            dict(impact=1, level="A3", reversible=True,  checkpoint=None),
    "deduplicate":         dict(impact=1, level="A3", reversible=True,  checkpoint=None),
    "normalize_skills":    dict(impact=1, level="A3", reversible=True,  checkpoint=None),
    "search_candidates":   dict(impact=1, level="A3", reversible=True,  checkpoint=None),
    "schedule_interview":  dict(impact=2, level="A3", reversible=True,  checkpoint=None),
    "notify_candidate":    dict(impact=2, level="A2", reversible=False, checkpoint=None),
    "screen_candidate":    dict(impact=2, level="A2", reversible=True,  checkpoint=None),
    "confirm_requirement": dict(impact=2, level="A1", reversible=True,  checkpoint="HITL-1"),
    "ocr_extract":         dict(impact=3, level="A2", reversible=True,  checkpoint=None),
    "compliance_verdict":  dict(impact=4, level="A3", reversible=False, checkpoint="HITL-2"),
    "finalize_shortlist":  dict(impact=4, level="A1", reversible=True,  checkpoint="HITL-3"),
    "hire_decision":       dict(impact=5, level="A0", reversible=False, checkpoint="HITL-4"),
    "approve_contract":    dict(impact=5, level="A0", reversible=False, checkpoint="HITL-5"),
    "execute_placement":   dict(impact=5, level="A0", reversible=False, checkpoint="HITL-6"),
}

# Allowlist aksi per agen. Ditegakkan secara mekanis: agen yang mencoba aksi
# di luar daftarnya akan ditolak PolicyEngine, sehingga kendali tidak
# bergantung pada kepatuhan model terhadap prompt.
AGENT_ACTIONS: Dict[str, Set[str]] = {
    "SupervisorAgent": {"confirm_requirement", "finalize_shortlist", "hire_decision",
                        "approve_contract", "execute_placement"},
    "IntakeAgent":     {"normalize_skills", "confirm_requirement"},
    "SourcingAgent":   {"search_candidates", "deduplicate"},
    "ScreeningAgent":  {"parse_cv", "normalize_skills", "screen_candidate"},
    "ComplianceAgent": {"ocr_extract", "compliance_verdict"},
    "MatchingAgent":   {"finalize_shortlist"},
    "InterviewAgent":  {"schedule_interview", "notify_candidate"},
    "PlacementAgent":  {"approve_contract", "execute_placement", "notify_candidate"},
}

# Metadata agen untuk tabel `agents` dan untuk laporan Bagian 5.3.
AGENT_PROFILE: Dict[str, tuple] = {
    "SupervisorAgent": ("AI", "STATIC", "A2"),
    "IntakeAgent":     ("AI", "STATIC", "A3"),
    "SourcingAgent":   ("AI", "STATIC", "A3"),
    "ScreeningAgent":  ("AI", "STATIC", "A2"),
    "ComplianceAgent": ("HYBRID", "STATIC (mobile = future work)", "A3/A1"),
    "MatchingAgent":   ("AI", "STATIC", "A1"),
    "InterviewAgent":  ("AI", "STATIC", "A3"),
    "PlacementAgent":  ("HYBRID", "STATIC", "A0"),
}

DOCUMENT_TYPES = ["KTP", "Ijazah", "SKCK", "Surat Keterangan Sehat", "Sertifikat"]
