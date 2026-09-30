"""Pemuat berkas `.env` minimal, hanya memakai standard library.

Supaya API key tidak perlu disetel ulang di setiap terminal, tulis sekali di
berkas `.env` pada folder proyek (salin dari `.env.example`):

    DEEPSEEK_API_KEY=sk-...

Berkas `.env` masuk .gitignore sehingga kunci tidak ikut ter-commit.

Aturan:
  - Variabel lingkungan yang SUDAH disetel selalu menang atas isi `.env`,
    sehingga `$env:DEEPSEEK_API_KEY=...` tetap dapat menimpa sementara.
  - Dicari berurutan: `.env` di folder kerja, lalu di akar proyek (folder
    induk paket mas_hr). Keduanya dibaca; yang ditemukan lebih dulu menang.
  - Format: `KUNCI=nilai` per baris; baris kosong dan `#` diabaikan; awalan
    `export ` dan tanda kutip pembungkus nilai diperbolehkan.
"""
from pathlib import Path
from typing import Dict, List, Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_env_text(text: str) -> Dict[str, str]:
    """Uraikan isi berkas `.env` menjadi dict kunci -> nilai."""
    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:                 # komentar di ujung baris
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_env(paths: Optional[List[Path]] = None) -> List[Path]:
    """Muat `.env` ke os.environ tanpa menimpa variabel yang sudah ada.

    Returns:
        Daftar berkas `.env` yang berhasil dibaca, untuk keperluan diagnosis.
    """
    if paths is None:
        paths = [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    loaded: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            values = parse_env_text(resolved.read_text(encoding="utf-8-sig"))
        except OSError:
            continue
        for key, value in values.items():
            if key not in os.environ:
                os.environ[key] = value
        loaded.append(resolved)
    return loaded
