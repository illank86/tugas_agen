"""Pembaca job requirement dari berkas .txt/.md/.pdf (satu berkas atau folder).

Format sengaja dibuat `kunci: nilai` agar dapat ditulis siapa pun tanpa tahu
JSON. Contoh berkas lengkap ada di data/job_requirements/.

    judul: Staff Admin Gudang
    klien: CLI-0031
    jumlah: 3
    pengalaman_minimal: 1
    lokasi: Sleman, DIY
    sla_hari: 14
    dokumen: KTP, Ijazah, SKCK, Surat Keterangan Sehat
    skill: Microsoft Excel | 0.40 | intermediate
    skill: SOP Pergudangan | 0.30 | basic
    skill: Komunikasi      | 0.30 | intermediate

Baris diawali '#' dianggap komentar. Nama skill boleh berupa skill_id
(SKL-001) atau nama/sinonim yang dapat dinormalkan ("excel"). Skill di luar
taksonomi DITOLAK, bukan didiamkan — itu penegakan ontologi bersama MIMA.

Berkas yang BUKAN format `kunci: nilai` — job description teks bebas seperti
iklan lowongan, termasuk PDF — diekstrak dengan LLM DeepSeek bila
DEEPSEEK_API_KEY disetel (lihat llm_parsing.py). Hasilnya tetap melewati
taksonomi dan JobRequirement.validate().
"""
from pathlib import Path
from typing import Dict, List, Tuple

from .deepseek_client import is_available as llm_available
from .domain_models import JobRequirement
from .settings import LEVELS
from .skill_taxonomy import SKILLS, normalize

# Alias kunci: pengguna boleh menulis dalam Indonesia atau Inggris.
KEY_ALIASES: Dict[str, str] = {
    "judul": "title", "title": "title", "jabatan": "title", "posisi": "title",
    "job_id": "job_id", "id": "job_id", "kode": "job_id",
    "klien": "client_id", "client": "client_id", "client_id": "client_id",
    "jumlah": "headcount", "headcount": "headcount", "kuota": "headcount",
    "pengalaman_minimal": "min_experience_years",
    "pengalaman": "min_experience_years",
    "min_experience_years": "min_experience_years",
    "lokasi": "location", "location": "location",
    "sla_hari": "sla_days", "sla": "sla_days", "sla_days": "sla_days",
    "dokumen": "documents", "documents": "documents", "berkas": "documents",
}
SKILL_KEYS = ("skill", "skills", "keahlian")
JOB_SUFFIXES = (".txt", ".md", ".pdf")


def _parse_skill_line(value: str, line_number: int) -> dict:
    """Uraikan satu baris `skill:` menjadi dict syarat skill.

    Args:
        value: isi setelah tanda titik dua, mis. "Excel | 0.40 | intermediate".
        line_number: nomor baris, dipakai pada pesan galat.

    Returns:
        Dict {skill_id, importance, min_level}.

    Raises:
        ValueError: bila format salah, skill di luar taksonomi, atau level
            tidak dikenal.
    """
    parts = [p.strip() for p in value.split("|")]
    if len(parts) < 2:
        raise ValueError(
            f"baris {line_number}: format skill harus "
            f"'nama | importance | min_level', dapat '{value}'")
    name, importance_text = parts[0], parts[1]
    level = (parts[2].lower() if len(parts) > 2 else "basic")

    skill_id = name if name in SKILLS else normalize(name)
    if skill_id is None:
        raise ValueError(
            f"baris {line_number}: skill '{name}' tidak ada di taksonomi kanonik. "
            f"Tambahkan ke mas_hr/skill_taxonomy.py beserta sinonimnya, atau "
            f"lihat daftar dengan: python -m mas_hr.cli skills")
    if level not in LEVELS:
        raise ValueError(
            f"baris {line_number}: level '{level}' tidak dikenal; "
            f"pilih {sorted(LEVELS)}")
    try:
        importance = float(importance_text)
    except ValueError:
        raise ValueError(
            f"baris {line_number}: importance '{importance_text}' bukan angka")
    return {"skill_id": skill_id, "importance": importance, "min_level": level}


def parse_job_text(text: str, fallback_job_id: str) -> JobRequirement:
    """Ubah isi berkas .txt menjadi objek JobRequirement tervalidasi.

    Args:
        text: seluruh isi berkas.
        fallback_job_id: dipakai bila berkas tidak mencantumkan `job_id`.

    Returns:
        JobRequirement yang sudah lolos validasi bobot.

    Raises:
        ValueError: bila ada baris salah format, skill tak dikenal, atau
            jumlah importance bukan 1.0.
    """
    fields: Dict[str, str] = {}
    skills: List[dict] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"baris {line_number}: tidak ada ':' pada '{line}'")
        key, value = line.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key in SKILL_KEYS:
            skills.append(_parse_skill_line(value, line_number))
        elif key in KEY_ALIASES:
            fields[KEY_ALIASES[key]] = value
        else:
            raise ValueError(
                f"baris {line_number}: kunci '{key}' tidak dikenal. "
                f"Kunci yang tersedia: {sorted(set(KEY_ALIASES))} atau 'skill'")

    documents = ([d.strip() for d in fields["documents"].split(",") if d.strip()]
                 if "documents" in fields else
                 ["KTP", "Ijazah", "SKCK", "Surat Keterangan Sehat"])

    job = JobRequirement(
        job_id=fields.get("job_id", fallback_job_id),
        title=fields.get("title", fallback_job_id),
        client_id=fields.get("client_id", "CLI-0031"),
        headcount=int(float(fields.get("headcount", 1))),
        required_skills=skills,
        min_experience_years=float(fields.get("min_experience_years", 0)),
        location=fields.get("location", "Sleman, DIY"),
        sla_days=int(float(fields.get("sla_days", 14))),
        required_documents=documents)
    job.validate()
    return job


def parse_job_document(text: str, fallback_job_id: str) -> JobRequirement:
    """Pilih jalur parsing: format `kunci: nilai` atau teks bebas lewat LLM.

    Berkas berformat kunci-nilai selalu diurai secara ketat supaya galat
    penulisan tetap terlihat, bukan ditebak oleh LLM.

    Raises:
        ValueError: galat format, atau teks bebas tanpa DEEPSEEK_API_KEY.
    """
    from .llm_parsing import looks_like_key_value, parse_job_description

    if looks_like_key_value(text, set(KEY_ALIASES) | set(SKILL_KEYS)):
        return parse_job_text(text, fallback_job_id)
    if not llm_available():
        raise ValueError(
            "berkas bukan format 'kunci: nilai'. Untuk membaca job description "
            "teks bebas dengan LLM, setel variabel lingkungan DEEPSEEK_API_KEY")
    return parse_job_description(text, fallback_job_id)


def _read_text(file_path: Path) -> str:
    """Ambil teks berkas job requirement; PDF memakai ekstraktor CV."""
    if file_path.suffix.lower() == ".pdf":
        from .cv_reader import extract_pdf_text
        text, _ = extract_pdf_text(file_path)
        if not text.strip():
            raise ValueError("tidak ada teks terbaca dari PDF (hasil pindaian?)")
        return text
    return file_path.read_text(encoding="utf-8")


def read_job_file(path: str) -> JobRequirement:
    """Baca satu berkas job requirement (.txt, .md, atau .pdf).

    Raises:
        FileNotFoundError: bila berkas tidak ada.
        ValueError: diteruskan dari parse_job_text, dengan nama berkas
            ditambahkan agar mudah dilacak.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"berkas job requirement tidak ditemukan: {path}")
    try:
        return parse_job_document(_read_text(file_path), file_path.stem)
    except ValueError as error:
        raise ValueError(f"{file_path.name}: {error}") from error


def read_job_folder(path: str) -> List[JobRequirement]:
    """Baca seluruh berkas job requirement dalam folder sebagai daftar lowongan.

    Berguna untuk skenario multi-lowongan dan untuk optimasi penugasan lintas
    lowongan, karena beberapa klien dilayani bersamaan.

    Raises:
        FileNotFoundError: bila folder tidak ada atau tidak berisi berkas
            .txt/.md/.pdf.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder job requirement tidak ditemukan: {path}")
    files = sorted(f for f in folder.iterdir()
                   if f.suffix.lower() in JOB_SUFFIXES)
    if not files:
        raise FileNotFoundError(
            f"tidak ada berkas {'/'.join(JOB_SUFFIXES)} di {path}")
    return [read_job_file(str(f)) for f in files]


def load_jobs(path: str) -> List[JobRequirement]:
    """Muat lowongan dari sebuah path, baik berkas maupun folder.

    Pemanggil tidak perlu tahu mana yang diberikan pengguna.
    """
    target = Path(path)
    if target.is_dir():
        return read_job_folder(path)
    return [read_job_file(path)]
