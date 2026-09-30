"""Pembaca CV dari folder: .pdf, .txt, dan .md.

Ekstraksi teks PDF dicoba berurutan dari yang paling andal:
  1. pypdf        — bila paket terpasang (paling akurat)
  2. pdftotext    — bila utilitas Poppler tersedia di PATH
  3. bawaan       — parser minimal berbasis zlib + operator teks PDF

Jalur ketiga ada supaya repositori tetap berjalan tanpa dependensi apa pun.

Bila DEEPSEEK_API_KEY disetel, teks CV juga diekstrak LLM DeepSeek (nama,
lokasi, pengalaman, skill kanonik; lihat llm_parsing.py). Metadata .meta.txt
yang ditulis manusia selalu menang atas hasil LLM, dan dokumen kepatuhan
tidak pernah diambil dari LLM.
Keterbatasannya nyata: PDF hasil pindaian (gambar) tidak menghasilkan teks
sama sekali, dan font Type0/CID dengan encoding khusus bisa keluar sebagai
karakter salah. Bila hasilnya kacau, pasang pypdf.

PERINGATAN DATA PRIBADI: jalur utama prototipe adalah data sintetis, justru
karena domain ini memuat KTP, ijazah, dan SKCK. Bila memasukkan CV orang
sungguhan, pastikan ada persetujuan dan jangan commit foldernya ke repositori
publik. Untuk presentasi kelas, CV buatan sendiri sudah cukup.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import shutil
import subprocess
import zlib

from .deepseek_client import DeepSeekError, is_available as llm_available
from .domain_models import Candidate, Document, JobRequirement

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Escape sequence pada literal string PDF.
_PDF_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
                b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}


def _unescape_pdf_string(raw: bytes) -> str:
    """Terjemahkan literal string PDF (tanpa kurung) menjadi teks biasa."""
    output = bytearray()
    index = 0
    while index < len(raw):
        char = raw[index:index + 1]
        if char == b"\\" and index + 1 < len(raw):
            nxt = raw[index + 1:index + 2]
            if nxt in _PDF_ESCAPES:
                output += _PDF_ESCAPES[nxt]
                index += 2
                continue
            if nxt.isdigit():                       # escape oktal \ooo
                digits = raw[index + 1:index + 4]
                octal = bytes([int(digits, 8) & 0xFF])
                output += octal
                index += 1 + len(digits)
                continue
            index += 2
            continue
        output += char
        index += 1
    return output.decode("utf-8", errors="replace")


def _parse_tounicode_cmaps(data: bytes) -> Dict[str, str]:
    """Kumpulkan peta kode glyph -> karakter dari seluruh CMap ToUnicode.

    PDF yang dihasilkan pengolah kata modern (LibreOffice, Word) memakai font
    subset: teksnya tersimpan sebagai `<0102...>Tj`, yaitu nomor glyph, bukan
    huruf. Tanpa membaca CMap ToUnicode, keluarannya hanya angka.

    KETERBATASAN: peta dari semua font digabung. Pada dokumen dengan banyak
    font subset, kode yang sama bisa berarti huruf berbeda dan sebagian teks
    akan salah. Untuk jalur cadangan tanpa dependensi ini dapat diterima;
    pasang pypdf bila hasilnya meragukan.
    """
    mapping: Dict[str, str] = {}

    def _to_char(hex_value: str) -> str:
        """Ubah rangkaian heksadesimal UTF-16BE menjadi karakter.

        Satu entri CMap dapat memetakan ke lebih dari satu karakter (ligatur),
        sehingga hex dibaca per empat digit, bukan sekali jadi.
        """
        try:
            return "".join(chr(int(hex_value[i:i + 4], 16))
                           for i in range(0, len(hex_value), 4))
        except ValueError:
            return ""

    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = match.group(1)
        try:
            content = zlib.decompress(raw)
        except zlib.error:
            content = raw
        if b"beginbfchar" not in content and b"beginbfrange" not in content:
            continue
        text = content.decode("latin-1", errors="replace")

        for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
            for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                mapping.setdefault(src.upper(), _to_char(dst))

        for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
            for low, high, dst in re.findall(
                    r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                start, end, base = int(low, 16), int(high, 16), int(dst, 16)
                width = len(low)
                for offset in range(min(end - start + 1, 512)):
                    code = f"{start + offset:0{width}X}"
                    mapping.setdefault(code, chr(base + offset))
    return mapping


def _decode_hex_string(hex_text: str, cmap: Dict[str, str]) -> str:
    """Terjemahkan literal heksadesimal PDF menjadi teks memakai CMap."""
    hex_text = re.sub(r"\s+", "", hex_text).upper()
    if not cmap:
        return ""
    width = len(next(iter(cmap)))
    if width not in (2, 4):
        width = 4
    if len(hex_text) % width:
        hex_text = hex_text[:len(hex_text) - len(hex_text) % width]
    return "".join(cmap.get(hex_text[i:i + width], "")
                   for i in range(0, len(hex_text), width))


def _text_from_content_stream(content: bytes,
                              cmap: Optional[Dict[str, str]] = None) -> str:
    """Ambil teks dari satu content stream PDF yang sudah ter-dekompresi.

    Menangani literal teks `(...)` maupun heksadesimal `<...>`, dan menyisipkan
    baris baru pada operator pemindah posisi (`Td`, `TD`, `T*`) serta akhir
    blok teks.

    Args:
        content: isi content stream yang sudah ter-dekompresi.
        cmap: peta glyph -> karakter dari ToUnicode, diperlukan untuk PDF
            berfont subset.
    """
    pieces: List[str] = []
    pattern = re.compile(
        rb"\((?:\\.|[^()\\])*\)|<[0-9A-Fa-f\s]+>|\bT[dDjJ*]\b|\bET\b", re.S)
    for match in pattern.finditer(content):
        token = match.group(0)
        if token.startswith(b"("):
            pieces.append(_unescape_pdf_string(token[1:-1]))
        elif token.startswith(b"<"):
            pieces.append(_decode_hex_string(
                token[1:-1].decode("latin-1"), cmap or {}))
        elif token in (b"Td", b"TD", b"T*", b"ET"):
            pieces.append("\n")
    text = "".join(pieces)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_content_stream(content: bytes) -> bool:
    """Tebak apakah sebuah stream berisi instruksi teks halaman.

    Berkas PDF juga menyimpan program font dan citra sebagai stream. Tanpa
    penyaringan ini, isi font ikut terbaca dan keluarannya jadi sampah biner.
    Penanda yang dipakai: ada blok teks `BT ... ET` dan mayoritas byte-nya
    dapat dicetak.
    """
    if b"BT" not in content or b"ET" not in content:
        return False
    sample = content[:4000]
    if not sample:
        return False
    printable = sum(1 for byte in sample if 9 <= byte <= 126)
    return printable / len(sample) > 0.85


def _extract_pdf_builtin(data: bytes) -> str:
    """Ekstraksi PDF tanpa dependensi: dekompresi stream lalu baca operator teks.

    Hanya stream yang lolos `_is_content_stream` yang diproses. PDF hasil
    pindaian tidak memiliki stream semacam itu sehingga mengembalikan string
    kosong — itu jawaban yang benar, bukan kegagalan diam-diam.
    """
    cmap = _parse_tounicode_cmaps(data)
    chunks: List[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = match.group(1)
        try:
            content = zlib.decompress(raw)
        except zlib.error:
            content = raw                      # stream tidak terkompresi
        if not _is_content_stream(content):
            continue
        piece = _text_from_content_stream(content, cmap)
        if piece:
            chunks.append(piece)
    return "\n".join(chunks).strip()


def extract_pdf_text(path: Path) -> Tuple[str, str]:
    """Ambil teks dari sebuah berkas PDF.

    Returns:
        Pasangan (teks, nama_metode) sehingga pemanggil dapat melaporkan
        metode mana yang dipakai — penting agar kualitas ekstraksi yang buruk
        tidak disalahartikan sebagai kesalahan sistem.
    """
    data = path.read_bytes()

    try:                                        # 1. pypdf
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text.strip(), "pypdf"
    except Exception:
        pass

    if shutil.which("pdftotext"):               # 2. Poppler
        try:
            result = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                    capture_output=True, timeout=30)
            text = result.stdout.decode("utf-8", errors="replace")
            if text.strip():
                return text.strip(), "pdftotext"
        except Exception:
            pass

    text = _extract_pdf_builtin(data)           # 3. bawaan
    return text, "builtin"


def read_cv_text(path: Path) -> Tuple[str, str]:
    """Baca isi satu berkas CV, apa pun formatnya yang didukung."""
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace"), "plaintext"


# ---------------------------------------------------------------------------
# Metadata pendamping (.meta.txt)
# ---------------------------------------------------------------------------
def _parse_metadata(path: Path) -> dict:
    """Baca berkas `<nama>.meta.txt` berisi identitas dan dokumen kandidat.

    Formatnya sama dengan job requirement: `kunci: nilai`, satu baris
    `dokumen:` per dokumen:

        nama: Andi Santoso
        lokasi: Sleman, DIY
        pengalaman: 3
        dokumen: KTP  | ada | 0.97
        dokumen: SKCK | ada | 0.62 | 2026-06-01 | 2026-11-28

    Kolom dokumen: jenis | ada/tidak | ocr_confidence | terbit | kedaluwarsa.
    """
    if not path.is_file():
        return {}
    meta: dict = {"documents": []}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key in ("dokumen", "document"):
            parts = [p.strip() for p in value.split("|")]
            meta["documents"].append({
                "kind": parts[0],
                "present": len(parts) < 2 or parts[1].lower() in ("ada", "yes", "true"),
                "ocr_confidence": float(parts[2]) if len(parts) > 2 else 0.95,
                "issue_date": parts[3] if len(parts) > 3 else "",
                "expires_at": parts[4] if len(parts) > 4 else "",
            })
        elif key in ("nama", "name"):
            meta["name"] = value
        elif key in ("lokasi", "location"):
            meta["location"] = value
        elif key in ("pengalaman", "experience", "pengalaman_tahun"):
            meta["experience_years"] = float(value)
        elif key in ("sumber", "source"):
            meta["source"] = value
    return meta


def _build_documents(candidate_id: str, meta: dict, required: List[str],
                     today: date, assume_valid: bool) -> List[Document]:
    """Bangun daftar Document dari metadata, atau tandai hilang bila tak ada.

    Tanpa metadata, dokumen dinyatakan TIDAK ADA — bukan dianggap sah. Sistem
    kepatuhan tidak boleh mengarang bukti. `assume_valid` hanya aktif bila
    pengguna memintanya secara eksplisit lewat flag CLI.
    """
    by_kind = {d["kind"]: d for d in meta.get("documents", [])}
    documents: List[Document] = []
    for index, kind in enumerate(required):
        entry = by_kind.get(kind)
        doc_id = f"DOC-{candidate_id}-{index}"
        if entry is None:
            if assume_valid:
                documents.append(Document(
                    doc_id=doc_id, kind=kind, issue_date=today - timedelta(days=30),
                    expires_at=(today + timedelta(days=150)
                                if kind in ("SKCK", "Surat Keterangan Sehat") else None),
                    ocr_confidence=0.95, name_consistent=True, present=True))
            else:
                documents.append(Document(
                    doc_id=doc_id, kind=kind, issue_date=today, expires_at=None,
                    ocr_confidence=0.0, name_consistent=True, present=False))
            continue
        issue = (date.fromisoformat(entry["issue_date"])
                 if entry.get("issue_date") else today)
        expires = (date.fromisoformat(entry["expires_at"])
                   if entry.get("expires_at") else None)
        documents.append(Document(
            doc_id=doc_id, kind=kind, issue_date=issue, expires_at=expires,
            ocr_confidence=entry.get("ocr_confidence", 0.95),
            name_consistent=True, present=entry.get("present", True)))
    return documents


def _parse_with_llm(texts: Dict[str, str]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Ekstrak CV dengan DeepSeek secara paralel.

    Returns:
        Pasangan (hasil per nama berkas, galat per nama berkas). CV yang gagal
        tetap diproses dengan parser aturan; kegagalan LLM tidak menggagalkan
        pembacaan folder.
    """
    from .llm_parsing import parse_cv_text

    def work(item: Tuple[str, str]) -> Tuple[str, Optional[dict], str]:
        name, text = item
        try:
            return name, parse_cv_text(text), ""
        except DeepSeekError as error:
            return name, None, str(error)

    parsed: Dict[str, dict] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, result, error in executor.map(work, texts.items()):
            if result is None:
                errors[name] = error
            else:
                parsed[name] = result
    return parsed, errors


def read_cv_folder(folder_path: str, job: JobRequirement, today: date,
                   assume_documents_valid: bool = False,
                   use_llm: Optional[bool] = None) -> Tuple[Dict[str, Candidate], dict]:
    """Muat seluruh CV dalam sebuah folder menjadi kandidat.

    Struktur folder yang diharapkan:

        data/cv/
          andi.pdf
          andi.meta.txt        (opsional)
          budi.pdf

    Args:
        folder_path: folder berisi berkas CV.
        job: lowongan yang dilamar; menentukan dokumen apa yang diperiksa.
        today: tanggal acuan untuk masa berlaku dokumen.
        assume_documents_valid: bila True, dokumen tanpa metadata dianggap
            sah. Ini MENGARANG bukti kepatuhan dan hanya boleh dipakai untuk
            demo cepat.
        use_llm: ekstrak CV dengan DeepSeek. None berarti otomatis: aktif
            bila DEEPSEEK_API_KEY disetel.

    Returns:
        Pasangan (kandidat per id, ringkasan pembacaan). Ringkasan memuat
        metode ekstraksi tiap berkas, berkas yang tidak menghasilkan teks
        (supaya masalah PDF pindaian terlihat), dan berkas yang dilewati
        karena orangnya sudah terbaca dari format lain.

    Raises:
        FileNotFoundError: bila folder tidak ada atau tidak berisi berkas
            yang didukung.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder CV tidak ditemukan: {folder_path}")
    all_files = sorted(f for f in folder.iterdir()
                       if f.suffix.lower() in SUPPORTED_SUFFIXES
                       and not f.name.endswith(".meta.txt"))
    # Satu orang boleh punya beberapa format (andi.pdf dan andi.txt). Ambil
    # satu saja per nama, sesuai urutan preferensi, agar tidak muncul sebagai
    # dua kandidat berbeda.
    preference = {".pdf": 0, ".txt": 1, ".md": 2}
    chosen: Dict[str, Path] = {}
    skipped: List[str] = []
    for candidate_file in all_files:
        current = chosen.get(candidate_file.stem)
        if current is None:
            chosen[candidate_file.stem] = candidate_file
        elif preference[candidate_file.suffix.lower()] < preference[current.suffix.lower()]:
            skipped.append(current.name)
            chosen[candidate_file.stem] = candidate_file
        else:
            skipped.append(candidate_file.name)
    files = [chosen[stem] for stem in sorted(chosen)]
    if not files:
        raise FileNotFoundError(
            f"tidak ada berkas {'/'.join(sorted(SUPPORTED_SUFFIXES))} di {folder_path}")

    candidates: Dict[str, Candidate] = {}
    methods: Dict[str, str] = {}
    empty: List[str] = []

    texts: Dict[str, str] = {}
    for file_path in files:
        text, method = read_cv_text(file_path)
        methods[file_path.name] = method
        texts[file_path.name] = text
        if not text.strip():
            empty.append(file_path.name)

    if use_llm is None:
        use_llm = llm_available()
    llm_parsed: Dict[str, dict] = {}
    llm_errors: Dict[str, str] = {}
    if use_llm:
        llm_parsed, llm_errors = _parse_with_llm(
            {name: text for name, text in texts.items() if text.strip()})

    for index, file_path in enumerate(files):
        text = texts[file_path.name]
        meta = _parse_metadata(folder / f"{file_path.stem}.meta.txt")
        llm = llm_parsed.get(file_path.name, {})
        candidate_id = f"CND-FILE-{index:03d}"
        # Urutan prioritas: metadata manusia > hasil LLM > nilai bawaan.
        experience = meta.get("experience_years")
        if experience is None:
            experience = llm.get("experience_years")
        candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            name=meta.get("name") or llm.get("name") or file_path.stem,
            location=meta.get("location") or llm.get("location") or job.location,
            experience_years=experience if experience is not None else 0.0,
            cv_text=text,
            documents=_build_documents(candidate_id, meta, job.required_documents,
                                       today, assume_documents_valid),
            true_skills={},                 # tidak ada ground truth untuk CV nyata
            source=meta.get("source", "db-internal"),
            persona="from_file",
            job_id_hint=job.job_id,
            document_problem_truth=False,   # tidak diketahui; jangan dipakai metrik
            interview_quality=0.70,         # placeholder; pakai mode interaktif
            llm_skills=llm.get("skills", {}))
    return candidates, {"extraction_methods": methods, "empty_files": empty,
                        "total": len(files), "skipped_duplicates": skipped,
                        "llm_parsed": sorted(llm_parsed), "llm_errors": llm_errors}
