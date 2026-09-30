"""Taksonomi skill kanonik — ontologi bersama antar-agen (laporan 5.8.2).

Tanpa ini, LLM menulis "Ms. Excel", OCR membaca "MICROSOFT EXCEL", dan
classifier mengenal kode internal; tiga agen menganggapnya tiga skill berbeda.
Seluruh agen wajib menormalkan entitas ke `skill_id` sebelum mengirim pesan.
"""
from typing import Dict, List, Optional, Tuple
import re
import unicodedata

# skill_id -> (nama kanonik, daftar sinonim, skill induk)
SKILLS: Dict[str, Tuple[str, List[str], Optional[str]]] = {
    "SKL-000": ("Aplikasi Perkantoran", ["microsoft office", "office suite"], None),
    "SKL-001": ("Microsoft Excel", ["excel", "ms excel", "msexcel", "spreadsheet",
                                    "microsoft office excel", "pengolah angka"], "SKL-000"),
    "SKL-002": ("SOP Pergudangan", ["sop gudang", "standar operasional gudang",
                                    "warehouse sop", "prosedur pergudangan"], "SKL-010"),
    "SKL-003": ("Komunikasi", ["communication", "komunikasi efektif",
                               "interpersonal", "public speaking"], None),
    "SKL-004": ("Administrasi Perkantoran", ["admin", "administrasi", "clerical",
                                             "office administration"], None),
    "SKL-005": ("Manajemen Inventori", ["inventory management", "stock opname",
                                        "manajemen stok", "stock control"], "SKL-010"),
    "SKL-006": ("Microsoft Word", ["word", "ms word", "pengolah kata"], "SKL-000"),
    "SKL-007": ("SAP MM", ["sap", "sap material management", "erp sap"], "SKL-011"),
    "SKL-008": ("Bahasa Inggris", ["english", "toefl", "bahasa inggris aktif"], None),
    "SKL-009": ("Forklift", ["operator forklift", "sim forklift", "lift truck"], "SKL-010"),
    "SKL-010": ("Operasional Logistik", ["logistik", "logistics", "supply chain"], None),
    "SKL-011": ("Sistem ERP", ["erp", "enterprise resource planning"], None),
    "SKL-012": ("Pembukuan Dasar", ["bookkeeping", "akuntansi dasar",
                                    "jurnal umum", "kas kecil"], None),
    "SKL-013": ("Ketelitian Data", ["data entry", "input data", "teliti",
                                    "attention to detail"], None),
}

# Indeks pencarian: setiap nama dan sinonim menunjuk ke satu skill_id.
_ALIAS_INDEX: Dict[str, str] = {}
for _sid, (_name, _syn, _parent) in SKILLS.items():
    _ALIAS_INDEX[_name.lower()] = _sid
    for _s in _syn:
        _ALIAS_INDEX[_s.lower()] = _sid


def _slugify(text: str) -> str:
    """Turunkan teks bebas menjadi bentuk banding: huruf kecil, tanpa simbol."""
    text = unicodedata.normalize("NFKD", text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(raw: str) -> Optional[str]:
    """Ubah nama skill bebas menjadi skill_id kanonik.

    Args:
        raw: teks apa adanya dari CV atau berkas job requirement.

    Returns:
        skill_id bila dikenali, None bila entitas berada di luar taksonomi.
        Pemanggil WAJIB menangani None secara eksplisit — mendiamkan entitas
        tak dikenal berarti membiarkan agen berbicara bahasa berbeda.
    """
    slug = _slugify(raw)
    if not slug:
        return None
    if slug in _ALIAS_INDEX:
        return _ALIAS_INDEX[slug]
    # Pencocokan longgar: alias terkandung di teks atau sebaliknya.
    # Alias terpanjang menang agar "microsoft excel" tidak kalah oleh "excel".
    best, best_len = None, 0
    for alias, sid in _ALIAS_INDEX.items():
        if (alias in slug or slug in alias) and len(alias) > best_len:
            best, best_len = sid, len(alias)
    return best


def canonical_name(skill_id: str) -> str:
    """Kembalikan nama tampilan sebuah skill_id; fallback ke id itu sendiri."""
    return SKILLS[skill_id][0] if skill_id in SKILLS else skill_id


def searchable_text(skill_id: str) -> str:
    """Gabungan nama + sinonim, dipakai sebagai basis vektor embedding skill.

    Sinonim ikut masuk agar kemiripan semantik tidak bergantung pada satu
    penulisan saja.
    """
    if skill_id not in SKILLS:
        return skill_id
    name, synonyms, _ = SKILLS[skill_id]
    return " ".join([name] + synonyms)
