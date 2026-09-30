"""Menjalankan server API dan antarmuka Streamlit sekaligus dalam satu perintah.

Keduanya adalah proses terpisah dengan siklus hidupnya sendiri: uvicorn
menyajikan API, Streamlit menyajikan antarmuka. Modul ini menyalakan keduanya,
menyatukan keluaran lognya, dan memastikan keduanya ikut mati bila salah satu
berhenti atau bila pengguna menekan Ctrl+C — supaya tidak ada proses yatim
yang menahan port.
"""
from pathlib import Path
from typing import List, Optional
import signal
import subprocess
import sys
import threading
import time

# Diberi warna agar keluaran dua proses mudah dibedakan di satu terminal.
COLORS = {"API": "\033[36m", "UI": "\033[35m", "RESET": "\033[0m"}


def _stream_output(process: subprocess.Popen, label: str) -> None:
    """Teruskan keluaran satu proses ke terminal dengan awalan penanda."""
    color, reset = COLORS.get(label, ""), COLORS["RESET"]
    for line in iter(process.stdout.readline, ""):
        if line:
            sys.stdout.write(f"{color}[{label}]{reset} {line}")
            sys.stdout.flush()


def _spawn(command: List[str], label: str) -> subprocess.Popen:
    """Jalankan satu proses anak dan sambungkan keluarannya ke terminal."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    thread = threading.Thread(target=_stream_output, args=(process, label),
                              daemon=True)
    thread.start()
    return process


def _check_dependencies() -> Optional[str]:
    """Pastikan paket untuk kedua layanan tersedia.

    Returns:
        Pesan galat bila ada yang kurang, atau None bila lengkap.
    """
    missing: List[str] = []
    for module, package in (("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"),
                            ("streamlit", "streamlit"), ("plotly", "plotly"),
                            ("pandas", "pandas")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        return ("Paket berikut belum terpasang: " + ", ".join(missing)
                + "\n    pip install " + " ".join(f'"{p}"' for p in missing))
    return None


def start_all(api_host: str = "127.0.0.1", api_port: int = 8000,
              ui_port: int = 8501, workers: int = 4,
              database: str = ":memory:") -> int:
    """Nyalakan server API dan antarmuka Streamlit, lalu tunggu keduanya.

    Args:
        api_host: alamat bind server API.
        api_port: port server API.
        ui_port: port antarmuka Streamlit.
        workers: jumlah alur rekrutmen yang berjalan bersamaan di sisi API.
        database: berkas SQLite bersama, atau ":memory:".

    Returns:
        Kode keluar: 0 bila dihentikan pengguna, 1 bila salah satu proses
        berhenti sendiri karena galat.
    """
    problem = _check_dependencies()
    if problem:
        print(problem)
        return 1

    package_root = Path(__file__).resolve().parent
    api_command = [sys.executable, "-m", "mas_hr.cli", "serve",
                   "--host", api_host, "--port", str(api_port),
                   "--workers", str(workers), "--database", database]
    ui_command = [sys.executable, "-m", "streamlit", "run",
                  str(package_root / "ui_app.py"),
                  "--server.port", str(ui_port), "--server.headless", "true"]

    print("Menyalakan dua layanan sekaligus:")
    print(f"  API  : http://{api_host}:{api_port}      (dokumentasi: /docs)")
    print(f"  UI   : http://localhost:{ui_port}")
    print(f"  Pekerja paralel: {workers} | basis data: {database}")
    print("Tekan Ctrl+C untuk menghentikan keduanya.\n")

    processes = {"API": _spawn(api_command, "API"), "UI": _spawn(ui_command, "UI")}

    def _terminate(*_args) -> None:
        """Hentikan seluruh proses anak dengan rapi, lalu paksa bila perlu."""
        print("\nMenghentikan kedua layanan...")
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 8
        for process in processes.values():
            remaining = max(0.1, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()

    signal.signal(signal.SIGINT, lambda *_: (_terminate(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_terminate(), sys.exit(0)))

    try:
        while True:
            for label, process in processes.items():
                code = process.poll()
                if code is not None:
                    # Satu proses mati berarti sistemnya tidak lagi utuh;
                    # matikan pasangannya supaya tidak ada layanan setengah jalan.
                    print(f"\n[{label}] berhenti dengan kode {code}.")
                    _terminate()
                    return 1 if code else 0
            time.sleep(0.4)
    except KeyboardInterrupt:
        _terminate()
        return 0
