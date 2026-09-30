"""Antarmuka Streamlit untuk menjalankan dan menelaah hasil rekrutmen.

Jalankan lewat:  python -m mas_hr.cli ui

Tiga layar:
  1. Pengaturan  — pilih folder job requirement dan folder CV, atur parameter
  2. Daftar hasil — satu baris per kandidat, terurut menurut skor kecocokan
  3. Detail      — rincian satu kandidat, termasuk radar kekuatan skornya

Prinsip yang dipertahankan dari sistem inti: antarmuka menampilkan kandidat
yang GUGUR beserta alasannya, bukan hanya yang lolos. Daftar yang hanya
menampilkan pemenang tidak dapat diaudit.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import html
import sys

import streamlit as st
import time

# Streamlit mengeksekusi berkas ini sebagai SKRIP, bukan sebagai bagian dari
# paket, sehingga impor relatif tidak tersedia. Akar proyek ditambahkan ke
# sys.path agar impor absolut di bawah selalu berhasil, baik saat dijalankan
# lewat `python -m mas_hr.cli ui` maupun `streamlit run mas_hr/ui_app.py`.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mas_hr.cv_reader import read_cv_folder
from mas_hr.human_approval import CHECKPOINT_ROLES, SimulatedApprover
from mas_hr.recruitment_service import RecruitmentService
from mas_hr.job_requirement_reader import JOB_SUFFIXES, load_jobs, read_job_file
from mas_hr.llm_narrator import is_available as llm_available
from mas_hr.llm_narrator import narrate_candidate, narrate_run
from mas_hr.recruitment_workflow import RecruitmentSystem
from mas_hr.result_reporting import (build_candidate_report, radar_axes,
                                     rejection_reason)
from mas_hr.settings import Settings
from mas_hr.skill_taxonomy import canonical_name
from mas_hr.synthetic_data import SyntheticGenerator

DEFAULT_JOB_FOLDER = "data/job_requirements"
DEFAULT_CV_FOLDER = "data/cv"

# Apa yang sebenarnya diputuskan manusia pada tiap gerbang. Ditampilkan agar
# penyetuju tahu konsekuensi keputusannya, bukan sekadar menekan tombol.
GATE_EXPLANATION = {
    "HITL-1": "Menyetujui spesifikasi lowongan sebelum pencarian kandidat dimulai. "
              "Salah di sini berarti seluruh proses mencari orang yang keliru.",
    "HITL-2": "Memutuskan dokumen yang tidak dapat dibaca sistem dengan cukup "
              "yakin. Sistem sengaja TIDAK menebak pada titik ini.",
    "HITL-3": "Memfinalkan shortlist. Menentukan siapa yang maju ke wawancara.",
    "HITL-4": "Keputusan hire. Tidak dapat dibatalkan dan mengikat klien.",
    "HITL-5": "Menyetujui isi kontrak kerja. Dokumen hukum yang mengikat.",
    "HITL-6": "Mengotorisasi penempatan di lokasi klien.",
}

# Urutan gerbang sepanjang tujuh fase. Dipakai untuk menggambar linimasa
# sehingga posisi proses terlihat, bukan hanya gerbang yang sedang aktif.
GATE_SEQUENCE = ["HITL-1", "HITL-2", "HITL-3", "HITL-4", "HITL-5", "HITL-6"]

GATE_SHORT = {
    "HITL-1": "Konfirmasi lowongan",
    "HITL-2": "Tinjauan dokumen",
    "HITL-3": "Finalisasi shortlist",
    "HITL-4": "Keputusan hire",
    "HITL-5": "Persetujuan kontrak",
    "HITL-6": "Otorisasi penempatan",
}

# Pilihan perilaku simulasi per gerbang. "Tanya saya" adalah perilaku
# sesungguhnya; dua lainnya membiarkan sistem memutuskan sendiri agar demo
# tidak tersendat di gerbang yang tidak sedang dibahas.
GATE_BEHAVIOURS = ["Tanya saya", "Setujui otomatis", "Tolak otomatis"]
AUTO_DECISION = {"Setujui otomatis": "APPROVED", "Tolak otomatis": "REJECTED"}

FINISHED = ("DONE", "FAILED", "CANCELLED")
QUEUE_GRACE_SECONDS = 10
RUN_STATUS = {"QUEUED": "Antre", "RUNNING": "Berjalan",
              "WAITING_HUMAN": "Menunggu keputusan"}

# Lama halaman menunggu agen mencapai titik persetujuan berikutnya sebelum
# menyerah dan menampilkan tombol periksa manual.
WAIT_FOR_AGENT_SECONDS = 60

# Gerbang yang keputusannya mengikat pihak luar dan tidak dapat dibatalkan.
# Diberi peringatan menonjol agar tidak disetujui secara refleks.
IRREVERSIBLE_GATES = {"HITL-4", "HITL-5", "HITL-6"}

DECISION_LABELS = {"APPROVED": "Disetujui", "REJECTED": "Ditolak",
                   "REQUEST_REVISION": "Minta revisi"}

# Apa yang TERJADI bila revisi diminta pada tiap titik persetujuan. Label
# tombol dibuat spesifik supaya penyetuju tahu akibatnya sebelum menekan.
REVISION_ACTION = {
    "HITL-1": ("Minta revisi", "Intake Agent menyusun ulang spesifikasi "
                               "lowongan, lalu diajukan kembali kepada Anda."),
    "HITL-2": ("Minta dokumen ulang", "Dokumen dianggap belum sah, sehingga "
                                      "kandidat tidak diteruskan pada proses ini."),
    "HITL-3": ("Minta revisi", "Matching Agent menghitung ulang shortlist, "
                               "lalu diajukan kembali kepada Anda."),
    "HITL-4": ("Wawancara ulang", "Kandidat diwawancara sekali lagi, lalu "
                                  "keputusan hire diajukan kembali."),
    "HITL-5": ("Minta revisi", "Placement Agent menyusun ulang draf kontrak, "
                               "lalu diajukan kembali kepada Anda."),
    "HITL-6": ("Minta revisi", "Otorisasi diajukan kembali bersama catatan Anda."),
}

# Nama pustaka pembaca diganti label yang dipahami pengguna non-teknis.
METHOD_LABELS = {"pypdf": "PDF (pypdf)", "pdftotext": "PDF (pdftotext)",
                 "builtin": "PDF (pembaca bawaan)", "plaintext": "Teks biasa"}

COMPONENT_LABELS = {"skill": "Skill", "exp": "Pengalaman",
                    "comp": "Kepatuhan", "pref": "Preferensi klien"}

STAGE_LABELS = {
    "SOURCED": "Terkumpul", "SCREENED": "Lolos screening",
    "SCREENING_FAILED": "Gugur di screening",
    "COMPLIANCE_CHECKED": "Dokumen diperiksa",
    "COMPLIANCE_FAILED": "Gugur kepatuhan", "SCORED": "Dinilai",
    "SHORTLISTED": "Masuk shortlist", "INTERVIEWED": "Diwawancara",
    "PLACED": "Ditempatkan",
}


# ---------------------------------------------------------------------------
# Pemilihan folder
# ---------------------------------------------------------------------------
def describe_folder(path: str, suffixes: List[str]) -> Dict[str, Any]:
    """Periksa sebuah folder dan ringkas isinya.

    Streamlit berjalan di peramban sehingga tidak dapat membuka dialog folder
    milik sistem operasi. Yang bisa dilakukan adalah memvalidasi jalur yang
    diketik dan menunjukkan isinya, agar pengguna tahu folder yang dimaksud
    benar sebelum menjalankan proses.
    """
    folder = Path(path).expanduser()
    if not path.strip():
        return {"ok": False, "message": "Folder belum diisi.", "files": []}
    if not folder.exists():
        return {"ok": False, "message": f"Folder tidak ditemukan: {folder}",
                "files": []}
    if not folder.is_dir():
        return {"ok": False, "message": f"Bukan folder: {folder}", "files": []}
    files = sorted(f.name for f in folder.iterdir()
                   if f.suffix.lower() in suffixes
                   and not f.name.endswith(".meta.txt"))
    if not files:
        return {"ok": False,
                "message": f"Tidak ada berkas {'/'.join(suffixes)} di {folder}",
                "files": []}
    return {"ok": True, "message": f"{len(files)} berkas ditemukan", "files": files}


def save_uploads(uploaded_files, destination: str) -> List[str]:
    """Simpan berkas unggahan ke folder tujuan dan kembalikan namanya.

    Berguna ketika aplikasi dijalankan di komputer lain: pengguna tidak perlu
    menyalin berkas secara manual ke server.
    """
    folder = Path(destination).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files or []:
        target = folder / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved.append(uploaded.name)
    return saved


# ---------------------------------------------------------------------------
# Menjalankan pipeline
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_service() -> RecruitmentService:
    """Layanan latar belakang untuk mode persetujuan manual.

    Alur dijalankan di thread tersendiri supaya dapat BERHENTI di gerbang dan
    menunggu keputusan, sesuatu yang tidak mungkin dilakukan bila alur
    dijalankan langsung di dalam skrip Streamlit yang berjalan sekali jalan.
    """
    return RecruitmentService(max_workers=2)


@st.cache_resource(show_spinner=False)
def load_screening_model(seed: int):
    """Latih model screening sekali lalu gunakan ulang antar-interaksi."""
    from mas_hr.experiment_runner import train_screening_model
    return train_screening_model(seed)


def start_manual_run(job_folder: str, cv_folder: str, settings: Settings,
                     use_synthetic_candidates: bool, synthetic_count: int) -> str:
    """Mulai alur yang akan BERHENTI di setiap gerbang menunggu keputusan Anda.

    Returns:
        run_id yang dipakai untuk memantau gerbang dan mengirim keputusan.
    """
    record = get_service().submit(
        job_path=job_folder,
        cv_folder=None if use_synthetic_candidates else cv_folder,
        settings=settings, synthetic_count=synthetic_count,
        approval_mode="manual", approval_timeout=3600.0)
    return record.run_id


def result_from_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Ubah run yang sudah selesai menjadi bentuk hasil yang dipakai tampilan."""
    record = get_service().get(run_id)
    if record is None or record.status != "DONE":
        return None
    return {"system": record.system, "job": record.job, "jobs": {},
            "outcome": record.outcome, "rows": record.rows,
            # getattr: layanan yang di-cache sebelum kode diperbarui masih
            # membuat RunRecord versi lama tanpa atribut ini.
            "notes": record.notes,
            "source": getattr(record, "source_report", None),
            "training": {}}


def run_pipeline(job_folder: str, cv_folder: str, settings: Settings,
                 use_synthetic_candidates: bool,
                 synthetic_count: int) -> Dict[str, Any]:
    """Jalankan alur rekrutmen untuk lowongan pertama pada folder yang dipilih.

    Args:
        job_folder: berkas atau folder berisi job requirement .txt.
        cv_folder: folder berisi CV; diabaikan bila memakai kandidat sintetis.
        settings: parameter sistem dari panel pengaturan.
        use_synthetic_candidates: True untuk membangkitkan kandidat sintetis.
        synthetic_count: jumlah kandidat sintetis yang dibangkitkan.

    Returns:
        Dict berisi sistem, lowongan, hasil alur, laporan per kandidat, dan
        catatan yang perlu ditampilkan ke pengguna.
    """
    jobs_list = load_jobs(job_folder)
    jobs = {job.job_id: job for job in jobs_list}
    primary_job = jobs_list[0]

    if use_synthetic_candidates:
        generator = SyntheticGenerator(settings.seed, today=settings.today)
        candidates = {}
        for candidate in generator.make_candidates(synthetic_count, primary_job):
            candidates[candidate.candidate_id] = candidate
        source = {"kind": "synthetic", "candidates": len(candidates)}
    else:
        candidates, report = read_cv_folder(cv_folder, primary_job, settings.today)
        source = {"kind": "files", "folder": cv_folder,
                  "candidates": len(candidates), **report}

    scorer, training = load_screening_model(settings.seed)
    system = RecruitmentSystem(
        settings, jobs, candidates, SimulatedApprover(seed=settings.seed),
        scorer=scorer, source_all_channels=not use_synthetic_candidates)
    outcome = system.run_job(primary_job.job_id, target_count=max(len(candidates), 1))
    report_rows = build_candidate_report(system, primary_job, outcome)
    return {"system": system, "job": primary_job, "jobs": jobs, "outcome": outcome,
            "rows": report_rows, "notes": [], "source": source,
            "training": training}


# ---------------------------------------------------------------------------
# Komponen tampilan
# ---------------------------------------------------------------------------
def radar_figure(row: dict, job):
    """Bangun radar chart kekuatan kandidat.

    Sumbu berisi pemenuhan tiap syarat skill ditambah pengalaman, kepatuhan,
    dan preferensi klien, sehingga terlihat komponen mana yang mengangkat atau
    menjatuhkan skor akhir.
    """
    import plotly.graph_objects as go

    axes = radar_axes(row, job, canonical_name)
    labels = list(axes) + [list(axes)[0]]
    values = list(axes.values()) + [list(axes.values())[0]]

    figure = go.Figure()
    figure.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill="toself", name=row["nama"],
        line={"width": 2}))
    figure.add_trace(go.Scatterpolar(
        r=[1.0] * len(labels), theta=labels, name="Target penuh",
        line={"dash": "dot", "width": 1}))
    figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True, height=420, margin={"l": 60, "r": 60, "t": 40, "b": 40})
    return figure


def contribution_figure(row: dict):
    """Bangun diagram batang kontribusi tiap komponen terhadap skor akhir."""
    import plotly.graph_objects as go

    labels = {"skill": "Skill", "exp": "Pengalaman",
              "comp": "Kepatuhan", "pref": "Preferensi klien"}
    keys = [k for k in labels if k in row["komponen"]]
    figure = go.Figure(go.Bar(
        x=[row["komponen"][k] for k in keys],
        y=[labels[k] for k in keys], orientation="h",
        text=[f"{row['kontribusi'].get(k, 0)}%" for k in keys],
        textposition="outside"))
    figure.update_layout(height=260, xaxis_title="Sumbangan ke skor akhir",
                         margin={"l": 10, "r": 40, "t": 20, "b": 30})
    return figure


def select_job(path: str) -> Optional[str]:
    """Tampilkan pilihan lowongan dan kembalikan jalur berkas yang dipilih.

    Sebelumnya folder berisi beberapa lowongan diam-diam hanya memakai berkas
    pertama menurut abjad. Kini pengguna memilih sendiri, melihat ringkasan
    lowongannya, dan diberi tahu bila ada berkas yang tidak valid.
    """
    target = Path(path).expanduser()
    files = (sorted(f for f in target.iterdir() if f.suffix.lower() in JOB_SUFFIXES)
             if target.is_dir() else [target])
    jobs: Dict[str, Any] = {}
    invalid: List[str] = []
    for file_path in files:
        try:
            jobs[str(file_path)] = read_job_file(str(file_path))
        except (FileNotFoundError, ValueError) as error:
            invalid.append(f"{file_path.name}: {error}")

    if invalid:
        st.sidebar.warning("Berkas tidak valid dan dilewati:\n\n"
                           + "\n\n".join(f"- {item}" for item in invalid),
                           icon=":material/warning:")
    if not jobs:
        st.sidebar.error("Tidak ada job requirement yang dapat dibaca.")
        return None

    chosen = st.sidebar.selectbox(
        f"Lowongan yang diproses ({len(jobs)} tersedia)", list(jobs),
        format_func=lambda key: f"{jobs[key].title} — {Path(key).name}",
        help="Satu proses menangani satu lowongan. Kandidat dinilai terhadap "
             "syarat lowongan yang dipilih di sini.")
    job = jobs[chosen]
    skills = ", ".join(canonical_name(s["skill_id"]) for s in job.required_skills)
    st.sidebar.caption(
        f"{job.job_id} · klien {job.client_id} · {job.headcount} posisi · "
        f"{job.location} · min. {job.min_experience_years} th pengalaman  \n"
        f"Skill: {skills or '-'}")
    return chosen


def render_sidebar() -> Dict[str, Any]:
    """Gambar panel pengaturan dan kembalikan pilihan pengguna."""
    st.sidebar.header("Sumber data")
    job_folder = st.sidebar.text_input(
        "Folder / berkas job requirement", value=DEFAULT_JOB_FOLDER,
        help="Folder atau satu berkas job requirement. Format 'kunci: nilai' "
             "dibaca langsung; job description teks bebas (.txt/.md/.pdf) "
             "dibaca LLM DeepSeek bila DEEPSEEK_API_KEY disetel.")
    job_status = describe_folder(job_folder, list(JOB_SUFFIXES)) if Path(job_folder).is_dir() \
        else {"ok": Path(job_folder).is_file(),
              "message": ("Berkas ditemukan" if Path(job_folder).is_file()
                          else f"Tidak ditemukan: {job_folder}"),
              "files": [Path(job_folder).name] if Path(job_folder).is_file() else []}
    selected_job = None
    if not job_status["ok"]:
        st.sidebar.error(job_status["message"])
    else:
        selected_job = select_job(job_folder)

    source = st.sidebar.radio(
        "Sumber kandidat", ["Folder CV", "Data sintetis"],
        help="Data sintetis punya ground truth sehingga metrik akurasi berlaku. "
             "CV dari berkas tidak.")
    cv_folder = DEFAULT_CV_FOLDER
    synthetic_count = 150
    if source == "Folder CV":
        cv_folder = st.sidebar.text_input("Folder CV", value=DEFAULT_CV_FOLDER,
                                          help="Berisi .pdf, .txt, atau .md.")
        cv_status = describe_folder(cv_folder, [".pdf", ".txt", ".md"])
        (st.sidebar.success if cv_status["ok"] else st.sidebar.error)(
            cv_status["message"])
        if cv_status["files"]:
            st.sidebar.caption(", ".join(cv_status["files"][:8]))
        uploads = st.sidebar.file_uploader(
            "Unggah CV ke folder di atas", accept_multiple_files=True,
            type=["pdf", "txt", "md"])
        if uploads and st.sidebar.button("Simpan unggahan"):
            saved = save_uploads(uploads, cv_folder)
            st.sidebar.success(f"Tersimpan: {', '.join(saved)}")
            st.rerun()
    else:
        synthetic_count = st.sidebar.slider("Jumlah kandidat sintetis",
                                            20, 400, 150, step=10)

    st.sidebar.header("Parameter")
    theta = st.sidebar.slider(
        "Ambang shortlist (theta)", 0.0, 1.0, 0.75, 0.01,
        help="Belum dikalibrasi secara empiris; ubah dan amati dampaknya.")
    gamma = st.sidebar.slider(
        "Ambang keyakinan otonomi (gamma)", 0.5, 0.99, 0.85, 0.01,
        help="Makin tinggi, makin sering sistem bertanya kepada manusia. "
             "Dokumen yang keyakinan pembacaannya di bawah nilai ini dikirim "
             "ke Tinjauan dokumen (persetujuan ke-2).")
    top_k = st.sidebar.number_input(
        "Ambil k teratas (0 = pakai theta)", min_value=0, max_value=50, value=0)
    st.sidebar.header("Peran manusia")
    approval_mode = st.sidebar.radio(
        "Mode persetujuan",
        ["Manual — Anda yang memutuskan", "Otomatis (simulasi)",
         "Tanpa persetujuan manusia (arm B2)"],
        help="Manual: alur BERHENTI di enam titik persetujuan dan menunggu "
             "keputusan Anda. Otomatis: penyetuju tersimulasi, dipakai untuk "
             "eksperimen. Tanpa persetujuan manusia: seluruh pengawasan "
             "manusia dimatikan.")
    hitl = not approval_mode.startswith("Tanpa")

    if approval_mode.startswith("Manual"):
        with st.sidebar.expander("Atur tiap titik persetujuan", expanded=False):
            st.caption("Pilih persetujuan yang ingin Anda putuskan sendiri. "
                       "Sisanya diputuskan otomatis dan ditandai SIM-AUTO pada "
                       "jejak audit.")
            behaviour: Dict[str, str] = {}
            for number, checkpoint in enumerate(GATE_SEQUENCE, start=1):
                behaviour[checkpoint] = st.selectbox(
                    f"{number}. {GATE_SHORT[checkpoint]}", GATE_BEHAVIOURS,
                    key=f"behaviour-{checkpoint}")
            st.session_state["gate_behaviour"] = behaviour


    seed = st.sidebar.number_input(
        "Nomor variasi data", min_value=0, value=42,
        help="Menentukan hasil acak: kandidat sintetis, hasil wawancara, dan "
             "keputusan penyetuju tersimulasi. Nomor yang sama selalu "
             "memberi hasil yang sama persis, sehingga demo dapat diulang. "
             "Ganti nomornya untuk mendapat variasi baru. (Istilah teknis: seed.)")

    return {"approval_mode": approval_mode,
            # Satu berkas lowongan yang dipilih, bukan foldernya: alur hanya
            # memproses satu lowongan per jalan.
            "job_folder": selected_job, "cv_folder": cv_folder,
            "use_synthetic": source == "Data sintetis",
            "synthetic_count": synthetic_count, "theta": theta, "gamma": gamma,
            "top_k": int(top_k), "hitl": hitl, "seed": int(seed),
            "ready": selected_job is not None}


def render_source_panel(source: Dict[str, Any]) -> None:
    """Ringkas asal kandidat: berapa CV dibaca, dilewati, atau gagal dibaca.

    Menggantikan deretan kotak info berisi kalimat panjang. Angka penting
    tampil sekilas; rincian per berkas tersedia bila diperlukan.
    """
    with st.container(border=True):
        if source.get("kind") == "synthetic":
            st.markdown(
                '<div class="src-head"><div>'
                '<div class="hitl-section">Sumber kandidat</div>'
                '<div class="src-title">Data sintetis</div></div>'
                '<span class="src-pill is-good">Ground truth tersedia</span></div>'
                + kv_grid([("Kandidat dibangkitkan", source.get("candidates", 0))])
                + '<div class="src-foot">Label kebenaran diketahui, sehingga '
                  'metrik akurasi dapat dihitung.</div>',
                unsafe_allow_html=True)
            return

        methods: Dict[str, str] = source.get("extraction_methods", {})
        skipped: List[str] = source.get("skipped_duplicates", [])
        empty: List[str] = source.get("empty_files", [])
        used_by_stem = {Path(name).stem: name for name in methods}
        method_names = sorted({METHOD_LABELS.get(m, m) for m in methods.values()})

        duplicates = ""
        if skipped:
            chips = "".join(
                f'<span class="src-dup"><s>{_esc(name)}</s> &rarr; '
                f'{_esc(used_by_stem.get(Path(name).stem, "?"))}</span>'
                for name in skipped)
            duplicates = (f'<div class="hitl-kv-label" style="margin-top:.75rem">'
                          f'Format ganda — satu orang cukup dibaca sekali</div>'
                          f'<div class="hitl-chips">{chips}</div>')
        unreadable = ""
        if empty:
            unreadable = (f'<div class="hitl-note is-bad" style="margin-top:.75rem">'
                          f'<b>Tidak ada teks terbaca:</b> {_esc(", ".join(empty))}. '
                          f'Kemungkinan PDF hasil pindaian — pasang pypdf atau '
                          f'sediakan versi .txt.</div>')

        st.markdown(
            f'<div class="src-head"><div>'
            f'<div class="hitl-section">Sumber kandidat</div>'
            f'<div class="src-title">Folder CV'
            f'<span class="mas-muted"> · {_esc(source.get("folder", ""))}</span>'
            f'</div></div>'
            f'<span class="src-pill is-warn">Tanpa ground truth</span></div>'
            + kv_grid([
                ("CV dibaca", source.get("total", len(methods))),
                ("Duplikat dilewati", len(skipped)),
                ("Gagal dibaca", len(empty)),
                ("Metode baca", ", ".join(method_names) or "-"),
            ])
            + duplicates + unreadable
            + '<div class="src-foot">CV dari berkas tidak punya label kebenaran, '
              'jadi metrik akurasi tidak dihitung dan nilai wawancara memakai '
              'angka placeholder.</div>',
            unsafe_allow_html=True)

        files = ([{"Berkas": name,
                   "Status": "Tidak ada teks" if name in empty else "Dibaca",
                   "Metode": METHOD_LABELS.get(method, method)}
                  for name, method in methods.items()]
                 + [{"Berkas": name,
                     "Status": f"Dilewati — memakai "
                               f"{used_by_stem.get(Path(name).stem, '?')}",
                     "Metode": "-"} for name in skipped])
        with st.expander(f"Rincian per berkas ({len(files)})",
                         icon=":material/description:"):
            st.dataframe(sorted(files, key=lambda f: f["Berkas"]),
                         hide_index=True, width="stretch")


def humanize_outcome(text: str) -> str:
    """Ganti kode internal pada ringkasan hasil dengan istilah tampilan.

    Contoh: "berhenti pada HITL-1 (REJECTED)" menjadi
    "berhenti pada Konfirmasi lowongan (ditolak)".
    """
    for checkpoint, label in GATE_SHORT.items():
        text = text.replace(checkpoint, label)
    for code, label in DECISION_LABELS.items():
        text = text.replace(code, label.lower())
    return text


def render_narrative(cache_key: str, button_label: str, produce) -> None:
    """Tampilkan narasi DeepSeek yang dibuat hanya saat tombol ditekan.

    Narasi disimpan di session_state supaya rerun Streamlit tidak memanggil
    API berulang kali. `produce` mengembalikan pasangan (narasi, sumber).
    """
    store = st.session_state.setdefault("narratives", {})
    if cache_key not in store:
        if not llm_available():
            st.caption("Narasi LLM nonaktif: setel DEEPSEEK_API_KEY untuk "
                       "memakai DeepSeek; tombol di bawah memakai narasi template.")
        if st.button(button_label, key=f"narrate-{cache_key}",
                     icon=":material/auto_awesome:"):
            with st.spinner("Menyusun narasi..."):
                store[cache_key] = produce()
    if cache_key in store:
        text, source = store[cache_key]
        with st.container(border=True):
            st.markdown(text)
            st.caption("Ditulis oleh LLM DeepSeek dari fakta terstruktur; tidak "
                       "memengaruhi skor maupun keputusan."
                       if source == "deepseek" else f"Sumber: {source}")


def render_summary(result: Dict[str, Any]) -> None:
    """Tampilkan ringkasan lowongan dan hasil alur."""
    job, outcome, system = result["job"], result["outcome"], result["system"]
    st.subheader(f"{job.title} — {job.job_id}")
    caption = (f"Klien {job.client_id} · {job.headcount} posisi · {job.location} · "
               f"SLA {job.sla_days} hari · pengalaman minimal "
               f"{job.min_experience_years} tahun")
    st.caption(caption)

    metric_cards([
        ("Kandidat dinilai", len(result["rows"]), "Seluruh CV yang masuk screening"),
        ("Masuk shortlist",
         sum(1 for row in result["rows"] if row["masuk_shortlist"]),
         "Lolos ambang theta dan kepatuhan"),
        ("Ditempatkan", len(outcome.get("placed", [])),
         "Lolos seluruh titik persetujuan"),
        ("Keputusan manusia", len(system.supervisor.approvals),
         "Jumlah keputusan yang diambil di titik persetujuan"),
        ("Pesan antar-agen", system.bus.sent_count,
         "Communication cost"),
    ])

    if result.get("source"):
        render_source_panel(result["source"])
        llm_parsed = result["source"].get("llm_parsed") or []
        llm_errors = result["source"].get("llm_errors") or {}
        if llm_parsed:
            st.caption(f"{len(llm_parsed)} CV juga diekstrak LLM DeepSeek "
                       f"(skill, pengalaman, identitas). Metadata .meta.txt "
                       f"tetap diutamakan.")
        if llm_errors:
            st.warning("Ekstraksi DeepSeek gagal, memakai parser aturan saja: "
                       + "; ".join(f"{name} ({error})"
                                   for name, error in llm_errors.items()),
                       icon=":material/warning:")
    else:
        for note in result["notes"]:
            st.caption(note)
    approvals = system.supervisor.approvals
    human_ids = sorted({a["approver_id"] for a in approvals})
    if approvals:
        st.caption(f"Keputusan persetujuan diambil oleh: {', '.join(human_ids)}")
    if system.supervisor.bypassed_gates:
        st.error(f"{len(system.supervisor.bypassed_gates)} titik persetujuan "
                 f"dilewati tanpa pengawasan manusia (mode arm B2).",
                 icon=":material/warning:")
    st.caption(f"Hasil: {humanize_outcome(outcome['outcome'])} · "
               f"waktu {outcome['elapsed']:.3f} detik "
               f"(tunggu manusia {system.supervisor.total_human_wait:.3f} detik)")
    render_narrative(f"run-{job.job_id}-{outcome['elapsed']}", "Buat ringkasan naratif",
                     lambda: narrate_run(job, result["rows"], outcome))


def render_results_table(rows: List[dict]) -> Optional[int]:
    """Tampilkan tabel hasil dan kembalikan indeks baris yang dipilih.

    Tabel memuat kandidat yang gugur beserta alasannya, bukan hanya yang
    lolos.
    """
    import pandas as pd

    table = pd.DataFrame([{
        # Selalu string: kolom campuran angka dan "-" gagal
        # dikonversi ke Arrow saat tabel dirender.
        "Peringkat": f"#{row['peringkat']}" if row["peringkat"] else "-",
        "Nama": row["nama"],
        "Fit score": row["fit_score"],
        "Skor screening": row["skor_screening"],
        "Pengalaman (th)": row["pengalaman_tahun"],
        "Kepatuhan": row["status_kepatuhan"],
        "Tahap": STAGE_LABELS.get(row["tahap"], row["tahap"]),
        "Alasan gugur": rejection_reason(row),
    } for row in rows])

    selection = st.dataframe(
        table, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Fit score": st.column_config.ProgressColumn(
                "Fit score", min_value=0.0, max_value=1.0, format="%.3f"),
            "Skor screening": st.column_config.NumberColumn(format="%.3f"),
        })
    chosen = selection.selection.rows if hasattr(selection, "selection") else []
    return chosen[0] if chosen else None


def render_detail(row: dict, job, result: Optional[Dict[str, Any]] = None,
                  key_prefix: str = "detail") -> None:
    """Tampilkan halaman detail satu kandidat."""
    st.markdown(f"### {row['nama']}")
    st.caption(f"{row['candidate_id']} · {row['lokasi'] or 'lokasi tidak dicatat'} · "
               f"sumber {row['sumber']}")

    metric_cards([
        ("Fit score", f"{row['fit_score']:.3f}", "Skor akhir setelah pemeriksaan kepatuhan"),
        ("Skor screening", f"{row['skor_screening']:.3f}",
         f"keyakinan model {row['keyakinan_screening']:.2f}"),
        ("Pengalaman", f"{row['pengalaman_tahun']:.1f} th"),
        ("Kepatuhan", row["status_kepatuhan"]),
        ("Peringkat", f"#{row['peringkat']}" if row["peringkat"] else "-"),
    ])

    chips = []
    if row["ditempatkan"]:
        chips.append(badge("Ditempatkan", "done"))
    elif row["masuk_shortlist"]:
        chips.append(badge("Masuk shortlist", "done"))
    else:
        chips.append(badge("Gugur", "stop"))
    chips.append(badge(STAGE_LABELS.get(row["tahap"], row["tahap"]), "idle"))
    if row["ditinjau_manusia"]:
        chips.append(badge("Ditinjau manusia", "wait"))
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    if not row["masuk_shortlist"]:
        st.warning(f"Tidak masuk shortlist: {rejection_reason(row)}")
    if row["ada_indikasi_injeksi"]:
        st.error("CV memuat pola instruksi yang menyerupai prompt injection. "
                 "Baris tersebut dinetralkan dan skor tetap dihitung dari fitur "
                 "terstruktur.", icon=":material/warning:")
    if row["ditinjau_manusia"]:
        st.info("Status kepatuhan kandidat ini ditentukan oleh peninjau manusia "
                "pada tahap Tinjauan dokumen.", icon=":material/person:")

    render_narrative(f"{key_prefix}-{row['candidate_id']}-{row['tahap']}",
                     "Buat narasi penjelasan", lambda: narrate_candidate(row, job))

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Kekuatan pembentuk skor**")
            st.plotly_chart(radar_figure(row, job), width="stretch")
            st.markdown('<div class="mas-muted">Nilai 1.0 berarti syarat '
                        'terpenuhi penuh. Garis putus-putus adalah target '
                        'penuh sebagai pembanding.</div>',
                        unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("**Sumbangan tiap komponen**")
            st.plotly_chart(contribution_figure(row), width="stretch")
        if row["syarat_belum_terpenuhi"]:
            st.warning("Syarat belum terpenuhi: "
                       + ", ".join(canonical_name(skill_id)
                                   for skill_id in row["syarat_belum_terpenuhi"]))

    tabs = st.tabs(["Skill terbaca", "Dokumen", "Temuan kepatuhan", "Teks CV"])
    with tabs[0]:
        if row["skill_terbaca"]:
            st.table([{
                "Skill": canonical_name(skill_id), "Level": level,
                # Diformat sebagai teks: kolom yang mencampur angka dan "-"
                # gagal dikonversi ke Arrow saat tabel dirender.
                "Pemenuhan syarat": (f"{row['per_syarat'][skill_id]:.2f}"
                                     if skill_id in row["per_syarat"]
                                     else "bukan syarat lowongan ini")}
                for skill_id, level in row["skill_terbaca"].items()])
        else:
            st.write("Tidak ada skill kanonik yang berhasil diekstraksi dari CV.")
    with tabs[1]:
        if row["dokumen"]:
            st.table([{"Jenis": d["jenis"],
                       "Ada": "ya" if d["ada"] else "tidak",
                       "Keyakinan OCR": f"{d['ocr_confidence']:.2f}",
                       "Terbit": d["terbit"], "Kedaluwarsa": d["kedaluwarsa"]}
                      for d in row["dokumen"]])
        else:
            st.write("Tidak ada data dokumen untuk kandidat ini.")
    with tabs[2]:
        if row["temuan_kepatuhan"]:
            st.table([{"Aturan": finding["rule_id"], "Dokumen": finding["document"],
                       "Hasil": finding["result"], "Temuan": finding["finding"],
                       "Keyakinan OCR": f"{finding.get('ocr_confidence', 0):.2f}"}
                      for finding in row["temuan_kepatuhan"]])
        else:
            st.write("Kandidat ini belum sampai ke tahap pemeriksaan dokumen.")
    with tabs[3]:
        st.text(row["cv_text"] or "(teks CV tidak tersedia)")


def live_candidate_rows(run_id: str) -> List[dict]:
    """Susun laporan kandidat dari alur yang MASIH berjalan.

    build_candidate_report biasanya dipanggil setelah alur selesai. Di sini ia
    dipanggil di tengah jalan dengan outcome kosong, supaya penyetuju dapat
    menelaah kandidat SEBELUM memutuskan — bukan setelah semuanya terlanjur
    diputuskan.
    """
    record = get_service().get(run_id)
    if record is None or record.system is None or record.job is None:
        return []
    try:
        return build_candidate_report(record.system, record.job, {})
    except Exception:
        return []


def render_candidate_browser(rows: List[dict], job, key_prefix: str) -> None:
    """Tampilkan tabel kandidat dan detail pilihan, dapat dipakai ulang.

    Dipakai di dua tempat: saat gerbang menunggu keputusan, dan setelah alur
    selesai. Keduanya memerlukan tampilan yang sama, sehingga logikanya tidak
    digandakan.
    """
    selection_key = f"{key_prefix}-selected"
    chosen = render_results_table(rows)
    if chosen is not None:
        st.session_state[selection_key] = chosen
    index = st.session_state.get(selection_key)
    if index is not None and index < len(rows):
        with st.expander(f"Detail: {rows[index]['nama']}", expanded=True):
            render_detail(rows[index], job, {}, key_prefix)


def render_gate_timeline(run_id: str, active_checkpoint: Optional[str] = None) -> None:
    """Gambar posisi proses pada enam gerbang human-in-the-loop.

    Tanpa ini, mode manual terasa seperti serangkaian dialog tanpa konteks:
    pengguna tidak tahu sudah sampai mana dan berapa gerbang lagi tersisa.
    """
    service = get_service()
    record = service.get(run_id)
    if record is None or record.system is None:
        return

    decided: Dict[str, List[dict]] = {}
    for approval in record.system.supervisor.approvals:
        decided.setdefault(approval["checkpoint"], []).append(approval)

    # Gerbang sebelum gerbang aktif yang tidak punya keputusan memang tidak
    # dipicu (mis. HITL-2 bila semua dokumen terbaca yakin), bukan terlewat.
    steps = []
    reached = False
    for number, checkpoint in enumerate(GATE_SEQUENCE, start=1):
        entries = decided.get(checkpoint, [])
        rejected = sum(1 for e in entries if e["decision"] == "REJECTED")
        revised = sum(1 for e in entries if e["decision"] == "REQUEST_REVISION")
        extra = ((f" · {rejected} ditolak" if rejected else "")
                 + (f" · {revised} revisi" if revised else ""))
        if checkpoint == active_checkpoint:
            state, mark, reached = "active", str(number), True
            meta = "menunggu Anda" + (f" · {len(entries)} selesai" if entries else "")
        elif entries:
            # Revisi yang tidak pernah berujung persetujuan (batas revisi
            # tercapai) juga menghentikan proses, jadi tidak boleh bertanda ✓.
            approved = any(e["decision"] == "APPROVED" for e in entries)
            stopped = rejected or not approved
            state = "stop" if stopped else "done"
            mark = "!" if stopped else "&#10003;"
            meta = f"{len(entries)} keputusan{extra}"
        elif reached:
            state, mark, meta = "idle", str(number), "berikutnya"
        else:
            state, mark, meta = "skip", "&ndash;", "tidak dipicu"
        steps.append(
            f'<div class="hitl-step is-{state}">'
            f'<div class="hitl-dot">{mark}</div>'
            f'<div class="hitl-step-label">{GATE_SHORT[checkpoint]}</div>'
            f'<div class="hitl-step-meta">{meta}</div></div>')
    st.markdown(f'<div class="hitl-stepper">{"".join(steps)}</div>',
                unsafe_allow_html=True)

    if decided:
        with st.expander(f"Riwayat keputusan "
                         f"({len(record.system.supervisor.approvals)})",
                         expanded=False, icon=":material/history:"):
            st.table([{
                "Titik persetujuan": GATE_SHORT.get(approval["checkpoint"],
                                                    approval["checkpoint"]),
                "Keputusan": DECISION_LABELS.get(approval["decision"],
                                                 approval["decision"]),
                "Penyetuju": approval["approver_id"],
                "Kandidat": approval.get("candidate_id") or "-",
                "Waktu berpikir (detik)": round(approval["latency"], 2),
            } for approval in record.system.supervisor.approvals])


def _esc(value: Any) -> str:
    """Amankan nilai sebelum disisipkan ke HTML; bukti berasal dari CV."""
    return html.escape(str(value))


def kv_grid(items: List[tuple]) -> str:
    """Bentuk kisi label–nilai ringkas untuk bukti gerbang."""
    cells = "".join(f'<div class="hitl-kv-item"><div class="hitl-kv-label">'
                    f'{_esc(label)}</div><div class="hitl-kv-value">{_esc(value)}'
                    f'</div></div>' for label, value in items)
    return f'<div class="hitl-kv">{cells}</div>'


def meter(label: str, value: float, note: str = "") -> str:
    """Bentuk batang nilai 0–1 berwarna menurut tingginya nilai."""
    value = max(0.0, min(1.0, float(value)))
    tone = "good" if value >= 0.75 else "mid" if value >= 0.5 else "low"
    return (f'<div class="hitl-meter"><div class="hitl-meter-head">'
            f'<span>{_esc(label)}</span><b>{value:.2f}</b></div>'
            f'<div class="hitl-meter-track"><div class="hitl-meter-fill '
            f'is-{tone}" style="width:{value * 100:.0f}%"></div></div>'
            + (f'<div class="hitl-meter-note">{_esc(note)}</div>' if note else "")
            + '</div>')


def render_gate_evidence(checkpoint: str, evidence: dict,
                         names: Dict[str, str]) -> None:
    """Tampilkan bukti gerbang dalam bentuk yang sesuai jenis keputusannya.

    Bukti mentah berupa dict; menampilkannya apa adanya memaksa penyetuju
    membaca struktur data, bukan menimbang keputusan.
    """
    def who(candidate_id: Optional[str]) -> str:
        if not candidate_id:
            return "-"
        name = names.get(candidate_id)
        return f"{name} ({candidate_id})" if name else candidate_id

    st.markdown('<div class="hitl-section">Bukti untuk ditimbang</div>',
                unsafe_allow_html=True)

    if checkpoint == "HITL-1":
        st.markdown(kv_grid([
            ("Posisi", evidence.get("title", "-")),
            ("Jumlah posisi", evidence.get("headcount", "-")),
            ("Putaran klarifikasi", evidence.get("clarification_turns", "-")),
        ]), unsafe_allow_html=True)
        skills = evidence.get("skills") or []
        chips = "".join(f'<span class="hitl-chip">{_esc(canonical_name(s))}</span>'
                        for s in skills) or '<span class="mas-muted">tidak ada</span>'
        st.markdown(f'<div class="hitl-kv-label" style="margin-top:.8rem">'
                    f'Skill wajib ({len(skills)})</div>'
                    f'<div class="hitl-chips">{chips}</div>',
                    unsafe_allow_html=True)
        if evidence.get("parsed_by") == "deepseek":
            st.caption("Spesifikasi ini diekstrak LLM DeepSeek dari job description "
                       "teks bebas. Periksa kembali sebelum menyetujui.")
        unresolved = evidence.get("unresolved") or []
        if unresolved:
            st.warning("Tidak dapat dipetakan ke taksonomi skill: "
                       + ", ".join(unresolved), icon=":material/help:")

    elif checkpoint == "HITL-2":
        st.markdown(kv_grid([("Kandidat", who(evidence.get("candidate_id")))]),
                    unsafe_allow_html=True)
        st.markdown(meter("Keyakinan pembacaan dokumen",
                          evidence.get("confidence", 0),
                          "Di bawah ambang gamma — sistem sengaja tidak menebak."),
                    unsafe_allow_html=True)
        findings = evidence.get("findings") or []
        items = "".join(f"<li>{_esc(f)}</li>" for f in findings) \
            or "<li>Tidak ada temuan tercatat.</li>"
        st.markdown(f'<div class="hitl-kv-label" style="margin-top:.8rem">'
                    f'Temuan pemeriksaan</div><ul class="hitl-list">{items}</ul>',
                    unsafe_allow_html=True)
        st.caption("Setujui bila dokumen sah; tolak bila dokumen memang "
                   "bermasalah; minta dokumen ulang bila perlu versi yang "
                   "lebih jelas.")

    elif checkpoint == "HITL-3" and isinstance(evidence.get("top"), list):
        st.markdown(kv_grid([
            ("Ukuran shortlist", f"{evidence.get('n_shortlist')} kandidat"),
            ("Ambang theta", evidence.get("theta")),
        ]), unsafe_allow_html=True)
        import pandas as pd
        table = pd.DataFrame([{
            "Kandidat": who(item["candidate_id"]),
            "Fit": float(item["fit"]),
            "Kontribusi": " · ".join(
                f"{COMPONENT_LABELS.get(k, k)} {v:.2f}" if isinstance(v, (int, float))
                else f"{COMPONENT_LABELS.get(k, k)} {v}"
                for k, v in (item.get("contributions") or {}).items()),
            "Belum terpenuhi": ", ".join(canonical_name(s)
                                         for s in item.get("unmet") or []) or "-",
        } for item in evidence["top"]])
        st.caption(f"{len(table)} teratas dari shortlist:")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Fit": st.column_config.ProgressColumn(
                "Fit", min_value=0.0, max_value=1.0, format="%.3f")})

    elif checkpoint == "HITL-4" and "competency" in evidence:
        competency = evidence["competency"]
        st.markdown(kv_grid([
            ("Kandidat", who(evidence.get("candidate_id"))),
            ("Posisi terisi", evidence.get("posisi_terisi", "-")),
        ]), unsafe_allow_html=True)
        st.markdown("".join(meter(label, competency.get(key, 0))
                            for key, label in (("teknis", "Teknis"),
                                               ("komunikasi", "Komunikasi"),
                                               ("sikap", "Sikap"))),
                    unsafe_allow_html=True)
        recommended = evidence.get("rekomendasi_pewawancara") == "RECOMMEND"
        # Rekomendasi pewawancara adalah bukti, bukan penyaring: keputusan
        # hire tetap milik manusia, termasuk melawan rekomendasi itu.
        st.markdown(
            f'<div class="hitl-note is-{"good" if recommended else "bad"}">'
            f'<b>{"Pewawancara merekomendasikan" if recommended else "Pewawancara TIDAK merekomendasikan"}</b>'
            f'<br>{_esc(evidence.get("catatan", ""))}</div>',
            unsafe_allow_html=True)

    elif checkpoint == "HITL-5" and isinstance(evidence.get("clauses"), list):
        st.markdown(kv_grid([("Kandidat", who(evidence.get("candidate_id")))]),
                    unsafe_allow_html=True)
        clauses = "".join(f"<li>{_esc(c)}</li>" for c in evidence["clauses"])
        st.markdown(f'<div class="hitl-kv-label" style="margin-top:.8rem">'
                    f'Draf kontrak ({len(evidence["clauses"])} klausul)</div>'
                    f'<ol class="hitl-doc">{clauses}</ol>',
                    unsafe_allow_html=True)

    elif checkpoint == "HITL-6":
        st.markdown(kv_grid([
            ("Kandidat", who(evidence.get("candidate_id"))),
            ("Tanggal mulai", evidence.get("start_date", "-")),
            ("Lokasi penempatan", evidence.get("location", "-")),
        ]), unsafe_allow_html=True)

    else:
        st.markdown(kv_grid([(key.replace("_", " ").capitalize(), value)
                             for key, value in evidence.items()
                             if key != "revisi"]),
                    unsafe_allow_html=True)


def render_queue_notice(run_id: str) -> None:
    """Jelaskan bahwa run menunggu giliran, dan beri jalan keluarnya.

    Layanan hanya menjalankan beberapa run sekaligus. Run manual yang
    ditinggalkan (mis. halaman di-refresh) tetap memegang slotnya sambil
    menunggu keputusan, sehingga run baru tertahan tanpa sebab yang terlihat.
    """
    service = get_service()
    others = service.active_runs(exclude=run_id)
    st.warning(
        f"Proses Anda menunggu giliran. {len(others)} proses lain masih "
        f"berjalan atau menunggu keputusan — biasanya sisa percobaan "
        f"sebelumnya atau dari tab lain yang ditinggalkan.",
        icon=":material/queue:")
    if others:
        st.table([{"Lowongan": r.job_title, "Status": RUN_STATUS.get(r.status, r.status),
                   "Dimulai": (r.started_at or r.submitted_at)[11:19]}
                  for r in others])
    left, right = st.columns(2)
    if left.button("Batalkan proses lain & lanjutkan", type="primary",
                   icon=":material/cancel:", disabled=not others,
                   width="stretch"):
        for other in others:
            service.cancel(other.run_id)
        st.session_state["flash"] = f"{len(others)} proses lain dibatalkan"
        time.sleep(0.5)                  # beri waktu slot pekerja terbebas
        st.rerun()
    if right.button("Periksa lagi", icon=":material/refresh:", width="stretch"):
        st.rerun()


def render_approval_panel(run_id: str) -> bool:
    """Tampilkan gerbang yang sedang menunggu dan terima keputusan pengguna.

    Returns:
        True bila alur masih berjalan (sehingga halaman perlu menunggu lagi),
        False bila alur sudah selesai atau gagal.
    """
    service = get_service()
    record = service.get(run_id)
    if record is None:
        return False
    if record.status in FINISHED:
        return False
    if record.status == "QUEUED":
        # Slot pekerja dari run yang baru dibatalkan butuh sesaat untuk bebas;
        # tunggu dulu sebelum menyimpulkan bahwa run ini benar-benar antre.
        with st.spinner("Menyiapkan proses…"):
            deadline = time.time() + QUEUE_GRACE_SECONDS
            while record.status == "QUEUED" and time.time() < deadline:
                time.sleep(0.2)
        if record.status == "QUEUED":
            render_queue_notice(run_id)
            return True

    # Tunggu alur mencapai titik persetujuan berikutnya. Setelah revisi, agen
    # mengerjakan ulang tahapnya dan itu bisa makan waktu; selama itu halaman
    # menampilkan indikator dan memeriksa sendiri, bukan diam menunggu klik.
    pending = service.pending_approvals(run_id)
    if not pending and record.status not in FINISHED:
        last = st.session_state.get("last_decision")
        message = ("Revisi dikirim — agen sedang mengerjakan ulang…"
                   if last == "REQUEST_REVISION"
                   else "Agen sedang bekerja menuju titik persetujuan berikutnya…")
        with st.spinner(message, show_time=True):
            deadline = time.time() + WAIT_FOR_AGENT_SECONDS
            while time.time() < deadline:
                pending = service.pending_approvals(run_id)
                if pending or record.status in FINISHED:
                    break
                time.sleep(0.2)

    if not pending:
        if record.status in FINISHED:
            return False
        st.info(f"Agen masih bekerja setelah {WAIT_FOR_AGENT_SECONDS} detik. "
                f"Proses tidak hilang; halaman akan menampilkan permintaan "
                f"berikutnya begitu siap.", icon=":material/hourglass:")
        if st.button("Periksa sekarang", type="primary",
                     icon=":material/refresh:"):
            st.rerun()
        return True
    st.session_state.pop("last_decision", None)

    gate = pending[0]
    checkpoint = gate["checkpoint_id"]
    role = CHECKPOINT_ROLES.get(checkpoint, "Recruiter")

    render_gate_timeline(run_id, active_checkpoint=checkpoint)

    # Gerbang yang disetel otomatis diputuskan tanpa menunggu klik, tetapi
    # penyetujunya dicatat sebagai SIM-AUTO sehingga jejak audit tetap jujur
    # membedakan keputusan manusia dari keputusan simulasi.
    behaviour = st.session_state.get("gate_behaviour", {}).get(checkpoint, "Tanya saya")
    if behaviour in AUTO_DECISION:
        service.decide(run_id, gate["gate_id"], AUTO_DECISION[behaviour],
                       "SIM-AUTO", f"disimulasikan otomatis ({behaviour.lower()})")
        st.session_state.setdefault("auto_log", []).append(
            f"{GATE_SHORT.get(checkpoint, checkpoint)} → "
            f"{DECISION_LABELS[AUTO_DECISION[behaviour]].lower()}")
        st.rerun()

    gate_id = gate["gate_id"]
    rows = live_candidate_rows(run_id)
    names = {row["candidate_id"]: row["nama"] for row in rows}
    highlight = gate.get("candidate_id")
    number = GATE_SEQUENCE.index(checkpoint) + 1 if checkpoint in GATE_SEQUENCE else "?"
    try:
        requested = datetime.fromisoformat(gate["requested_at"]).astimezone()
        requested_text = f"diminta {requested:%H:%M:%S}"
    except (KeyError, TypeError, ValueError):
        requested_text = ""

    with st.container(border=True):
        subject = ""
        if highlight:
            subject = (f'<div class="hitl-subject">Menyangkut kandidat '
                       f'<b>{_esc(names.get(highlight, highlight))}</b>'
                       f'<span class="mas-muted"> · {_esc(highlight)}</span></div>')
        risk = ""
        if checkpoint in IRREVERSIBLE_GATES:
            risk = ('<div class="hitl-note is-bad">Keputusan ini mengikat klien '
                    'dan tidak dapat dibatalkan setelah dikirim.</div>')
        queue = (f' · {len(pending) - 1} keputusan lain mengantre'
                 if len(pending) > 1 else "")
        revision_note = ""
        revision = gate["evidence"].get("revisi")
        if revision:
            # Tanpa penanda ini, permintaan yang muncul lagi setelah revisi
            # tampak seperti bug: kartu yang sama seolah tidak berubah.
            if not revision.get("dikerjakan_ulang"):
                outcome = "Diajukan ulang bersama catatan Anda."
            elif revision.get("berubah"):
                outcome = "Agen sudah mengerjakan ulang; bukti di bawah adalah hasil terbaru."
            else:
                outcome = ("Agen sudah mengerjakan ulang, tetapi hasilnya sama "
                           "dengan sebelumnya karena data masukannya tidak berubah.")
            revision_note = (
                f'<div class="hitl-note is-info"><b>Revisi ke-{revision["ke"]} '
                f'dari maksimal {revision["maks"]}</b> · {outcome}<br>'
                f'<span class="mas-muted">Catatan revisi: '
                f'“{_esc(revision.get("catatan", ""))}”</span></div>')
        st.markdown(
            f'<div class="hitl-head"><div>'
            f'<div class="hitl-eyebrow">Persetujuan {number} dari '
            f'{len(GATE_SEQUENCE)}</div>'
            f'<div class="hitl-title">{_esc(gate["decision_type"])}</div>'
            f'<div class="mas-muted">{GATE_EXPLANATION.get(checkpoint, "")}</div>'
            f'</div><div class="hitl-head-side">'
            f'<span class="hitl-pill"><span class="hitl-pulse"></span>'
            f'Menunggu keputusan</span>'
            f'<div class="mas-muted">Wewenang: <b>{_esc(role)}</b></div>'
            f'<div class="mas-muted">{requested_text}{queue}</div>'
            f'</div></div>{subject}{revision_note}{risk}',
            unsafe_allow_html=True)

        evidence_col, decision_col = st.columns([3, 2], gap="large")
        with evidence_col:
            render_gate_evidence(checkpoint, gate["evidence"], names)

        with decision_col:
            st.markdown('<div class="hitl-section">Keputusan Anda</div>',
                        unsafe_allow_html=True)
            # Form: mengetik alasan tidak memicu rerun halaman per ketukan.
            with st.form(key=f"decide-{gate_id}", border=False):
                approver_id = st.text_input(
                    "Identitas penyetuju", value="USR-HRM-01",
                    key=f"approver-{gate_id}")
                reason = st.text_area(
                    "Alasan keputusan", key=f"reason-{gate_id}", height=110,
                    placeholder="Tuliskan pertimbangan Anda…",
                    help="Wajib diisi. Tercatat di jejak audit bersama "
                         "identitas penyetuju.")
                approve = st.form_submit_button(
                    "Setujui", type="primary", icon=":material/check:",
                    width="stretch")
                revise_label, revise_help = REVISION_ACTION.get(
                    checkpoint, ("Minta revisi", None))
                left, right = st.columns(2)
                revise = left.form_submit_button(
                    revise_label, icon=":material/edit_note:", width="stretch",
                    help=revise_help)
                reject = right.form_submit_button(
                    "Tolak", icon=":material/close:", width="stretch")

            decision = ("APPROVED" if approve else "REQUEST_REVISION" if revise
                        else "REJECTED" if reject else None)
            if decision:
                if not reason.strip():
                    st.error("Alasan wajib diisi: keputusan tanpa alasan tidak "
                             "dapat diaudit.", icon=":material/error:")
                elif not service.decide(run_id, gate_id, decision,
                                        approver_id.strip() or "USR-UNKNOWN",
                                        reason.strip()):
                    # Permintaan sudah diputuskan (mis. klik ganda) atau
                    # kedaluwarsa; muat ulang agar tampil permintaan terkini.
                    st.session_state["flash"] = ("permintaan ini sudah diputuskan "
                                                 "sebelumnya; menampilkan yang terbaru")
                    st.rerun()
                else:
                    st.session_state["last_decision"] = decision
                    label =(revise_label if decision == "REQUEST_REVISION"
                             else DECISION_LABELS[decision])
                    st.session_state["flash"] = (
                        f"{GATE_SHORT.get(checkpoint, checkpoint)}: "
                        f"{label.lower()} oleh "
                        f"{approver_id.strip() or 'USR-UNKNOWN'}")
                    st.rerun()
            st.caption("Tanpa keputusan dalam 60 menit, permintaan ini otomatis "
                       "ditolak — diam tidak dianggap persetujuan.")

    # Telaah kandidat langsung di gerbang. Tanpa ini, penyetuju hanya melihat
    # ringkasan bukti dan baru bisa membuka detail setelah semuanya selesai —
    # yaitu ketika keputusannya sudah tidak ada gunanya lagi.
    if rows and checkpoint in ("HITL-3", "HITL-4", "HITL-5", "HITL-6"):
        with st.expander(f"Telaah kandidat sebelum memutuskan ({len(rows)})",
                         expanded=bool(highlight), icon=":material/person_search:"):
            if highlight:
                match = next((i for i, row in enumerate(rows)
                              if row["candidate_id"] == highlight), None)
                if match is not None:
                    st.caption(f"Baris **{rows[match]['nama']}** dipilih; baris "
                               f"lain ditampilkan sebagai pembanding.")
                    st.session_state.setdefault(f"gate-{checkpoint}-selected", match)
            render_candidate_browser(rows, record.job, f"gate-{checkpoint}")
    return True


CUSTOM_STYLE = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1400px; }
  div[data-testid="stMetricValue"] { font-size: 1.45rem; }
  div[data-testid="stMetricLabel"] { opacity: .75; font-size: .78rem;
                                     text-transform: uppercase;
                                     letter-spacing: .04em; }
  .mas-badge { display:inline-block; padding:.18rem .6rem; border-radius:999px;
               font-size:.72rem; font-weight:600; letter-spacing:.03em; }
  .mas-badge-wait   { background:#fef3c7; color:#92400e; }
  .mas-badge-done   { background:#dcfce7; color:#166534; }
  .mas-badge-stop   { background:#fee2e2; color:#991b1b; }
  .mas-badge-idle   { background:#f1f5f9; color:#475569; }
  .mas-gate-title   { font-size:1.05rem; font-weight:700; margin:.1rem 0 .15rem; }
  .mas-muted        { color:#64748b; font-size:.82rem; line-height:1.45; }

  /* Linimasa titik persetujuan */
  .hitl-stepper { display:flex; overflow-x:auto; padding:.4rem 0 .6rem;
                  margin-bottom:.6rem; }
  .hitl-step { flex:1 1 0; min-width:118px; position:relative;
               text-align:center; padding:0 .3rem; }
  .hitl-step::before { content:""; position:absolute; top:16px; left:-50%;
                       width:100%; height:2px; background:rgba(148,163,184,.35); }
  .hitl-step:first-child::before { display:none; }
  .hitl-step.is-done::before, .hitl-step.is-stop::before,
  .hitl-step.is-active::before { background:#22c55e; }
  .hitl-dot { position:relative; z-index:1; width:34px; height:34px;
              margin:0 auto .45rem; border-radius:50%; display:flex;
              align-items:center; justify-content:center; font-weight:700;
              font-size:.85rem; background:#e2e8f0; color:#475569; }
  .is-done   .hitl-dot { background:#16a34a; color:#fff; }
  .is-stop   .hitl-dot { background:#dc2626; color:#fff; }
  .is-active .hitl-dot { background:#f59e0b; color:#fff;
                         animation:hitl-ring 1.8s ease-in-out infinite; }
  .is-skip   .hitl-dot { background:transparent; color:#94a3b8;
                         border:2px dashed rgba(148,163,184,.7); }
  .hitl-step-code  { font-size:.68rem; font-weight:700; letter-spacing:.06em;
                     opacity:.55; }
  .hitl-step-label { font-size:.82rem; font-weight:600; line-height:1.25; }
  .hitl-step-meta  { font-size:.72rem; color:#64748b; margin-top:.15rem; }
  .is-skip .hitl-step-label { opacity:.55; }
  .is-active .hitl-step-meta { color:#d97706; font-weight:600; }
  .is-stop   .hitl-step-meta { color:#dc2626; }
  @keyframes hitl-ring {
    0%,100% { box-shadow:0 0 0 4px rgba(245,158,11,.28); }
    50%     { box-shadow:0 0 0 9px rgba(245,158,11,.06); } }

  /* Kartu keputusan persetujuan */
  .hitl-head { display:flex; justify-content:space-between; gap:1rem;
               flex-wrap:wrap; padding-bottom:.9rem; margin-bottom:.9rem;
               border-bottom:1px solid rgba(148,163,184,.25); }
  .hitl-head > div:first-child { flex:1 1 320px; }
  .hitl-head-side { display:flex; flex-direction:column; align-items:flex-end;
                    gap:.3rem; text-align:right; }
  .hitl-eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.08em;
                  text-transform:uppercase; color:#d97706; }
  .hitl-title { font-size:1.35rem; font-weight:700; margin:.15rem 0 .3rem;
                line-height:1.25; }
  .hitl-pill { display:inline-flex; align-items:center; gap:.45rem;
               padding:.25rem .7rem; border-radius:999px; font-size:.75rem;
               font-weight:600; background:rgba(245,158,11,.15); color:#d97706; }
  .hitl-pulse { width:8px; height:8px; border-radius:50%; background:#f59e0b;
                animation:hitl-ring 1.8s ease-in-out infinite; }
  .hitl-subject { font-size:.92rem; margin:-.2rem 0 .8rem; }
  .hitl-section { font-size:.72rem; font-weight:700; letter-spacing:.08em;
                  text-transform:uppercase; opacity:.6; margin-bottom:.55rem; }
  .hitl-note { border-left:3px solid; border-radius:6px; padding:.55rem .8rem;
               font-size:.85rem; line-height:1.45; margin:.2rem 0 .9rem; }
  .hitl-note.is-bad  { border-color:#dc2626; background:rgba(220,38,38,.08); }
  .hitl-note.is-good { border-color:#16a34a; background:rgba(22,163,74,.08); }
  .hitl-note.is-info { border-color:#3b82f6; background:rgba(59,130,246,.08); }
  .hitl-kv { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
             gap:.5rem; margin-bottom:.4rem; }
  .hitl-kv-item { padding:.5rem .7rem; border-radius:8px;
                  background:rgba(148,163,184,.1); }
  .hitl-kv-label { font-size:.72rem; color:#64748b; margin-bottom:.15rem; }
  .hitl-kv-value { font-size:.95rem; font-weight:600; overflow-wrap:anywhere; }
  .hitl-chips { display:flex; flex-wrap:wrap; gap:.35rem; }
  .hitl-chip { padding:.2rem .6rem; border-radius:999px; font-size:.78rem;
               background:rgba(59,130,246,.12); color:#3b82f6; font-weight:500; }
  .hitl-list { margin:.2rem 0 .4rem 1.1rem; padding:0; font-size:.88rem; }
  .hitl-doc { margin:.3rem 0 .4rem; padding:.8rem .9rem .8rem 2.2rem;
              max-height:280px; overflow-y:auto; border-radius:8px;
              border:1px solid rgba(148,163,184,.3); font-size:.86rem;
              line-height:1.55; font-family:Georgia,"Times New Roman",serif; }
  .hitl-doc li { margin-bottom:.3rem; }
  .hitl-meter { margin:.6rem 0; }
  .hitl-meter-head { display:flex; justify-content:space-between;
                     font-size:.82rem; margin-bottom:.25rem; }
  .hitl-meter-track { height:8px; border-radius:999px;
                      background:rgba(148,163,184,.2); overflow:hidden; }
  .hitl-meter-fill { height:100%; border-radius:999px; }
  .hitl-meter-fill.is-good { background:#16a34a; }
  .hitl-meter-fill.is-mid  { background:#f59e0b; }
  .hitl-meter-fill.is-low  { background:#dc2626; }
  .hitl-meter-note { font-size:.72rem; color:#64748b; margin-top:.2rem; }

  /* Panel sumber kandidat */
  .src-head { display:flex; justify-content:space-between; align-items:flex-start;
              gap:.75rem; flex-wrap:wrap; margin-bottom:.75rem; }
  .src-title { font-size:1.05rem; font-weight:700; }
  .src-pill { padding:.22rem .7rem; border-radius:999px; font-size:.74rem;
              font-weight:600; white-space:nowrap; }
  .src-pill.is-warn { background:rgba(245,158,11,.15); color:#d97706; }
  .src-pill.is-good { background:rgba(22,163,74,.13); color:#16a34a; }
  .src-dup { padding:.2rem .6rem; border-radius:6px; font-size:.78rem;
             background:rgba(148,163,184,.12); }
  .src-dup s { opacity:.6; }
  .src-foot { font-size:.76rem; color:#64748b; margin-top:.8rem; padding-top:.6rem;
              border-top:1px dashed rgba(148,163,184,.3); }
</style>
"""


def badge(text: str, kind: str = "idle") -> str:
    """Bentuk label kecil berwarna untuk status gerbang dan kandidat."""
    return f'<span class="mas-badge mas-badge-{kind}">{text}</span>'


def metric_cards(items: List[tuple]) -> None:
    """Tampilkan sederet nilai sebagai kartu, satu kolom per nilai.

    Args:
        items: daftar (label, nilai, keterangan opsional).
    """
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        label, value = item[0], item[1]
        helper = item[2] if len(item) > 2 else None
        with column:
            with st.container(border=True):
                st.metric(label, value, help=helper)


def main() -> None:
    """Titik masuk aplikasi Streamlit."""
    st.set_page_config(page_title="Agentic HR Outsourcing",
                       page_icon=":material/groups:", layout="wide")
    st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)
    st.title("Agentic AI untuk Outsourcing HRD")
    st.caption("Multi-agent system dengan bounded autonomy dan human-in-the-loop. "
               "Seluruh data pada demo ini sintetis atau berkas yang Anda sediakan.")

    options = render_sidebar()
    manual = options["approval_mode"].startswith("Manual")

    if st.sidebar.button("Jalankan proses", type="primary",
                         disabled=not options["ready"]):
        settings = Settings(seed=options["seed"], theta=options["theta"],
                            gamma=options["gamma"], top_k=options["top_k"],
                            hitl_enabled=options["hitl"])
        st.session_state.pop("selected", None)
        st.session_state.pop("result", None)
        # Run lama yang belum selesai dibatalkan, bukan sekadar dilupakan:
        # kalau tidak, ia tetap memegang slot pekerja sambil menunggu
        # keputusan dan run baru bisa tertahan di antrean.
        previous = st.session_state.pop("run_id", None)
        if previous:
            get_service().cancel(previous)
        st.session_state.pop("auto_log", None)
        st.session_state.pop("last_decision", None)
        try:
            if manual:
                st.session_state["run_id"] = start_manual_run(
                    options["job_folder"], options["cv_folder"], settings,
                    options["use_synthetic"], options["synthetic_count"])
            else:
                with st.spinner("Menjalankan delapan agen..."):
                    st.session_state["result"] = run_pipeline(
                        options["job_folder"], options["cv_folder"], settings,
                        options["use_synthetic"], options["synthetic_count"])
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))

    # Mode manual: tampilkan gerbang sampai alur selesai.
    run_id = st.session_state.get("run_id")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(f"Keputusan tercatat — {flash}", icon=":material/task_alt:")
    if run_id and "result" not in st.session_state:
        still_running = render_approval_panel(run_id)
        if still_running:
            return
        finished = result_from_run(run_id)
        if finished is None:
            record = get_service().get(run_id)
            if record is not None and record.status == "CANCELLED":
                st.info("Proses ini sudah dibatalkan. Tekan **Jalankan proses** "
                        "untuk memulai lagi.", icon=":material/cancel:")
            else:
                st.error(f"Alur gagal: "
                         f"{record.error if record else 'tidak diketahui'}")
            return
        st.session_state["result"] = finished

    if not manual and "result" not in st.session_state:
        st.caption("Mode persetujuan otomatis dipilih: setiap keputusan dibuat "
                   "oleh penyetuju TERSIMULASI, bukan oleh manusia. Pakai mode "
                   "manual bila ingin memutuskan sendiri.")

    result = st.session_state.get("result")
    if not result:
        st.info("Pilih folder job requirement dan sumber kandidat di panel kiri, "
                "lalu tekan **Jalankan proses**.")
        return

    render_summary(result)
    if st.session_state.get("run_id"):
        render_gate_timeline(st.session_state["run_id"])
        auto_log = st.session_state.get("auto_log") or []
        if auto_log:
            st.caption("Diputuskan otomatis oleh simulasi: " + " · ".join(auto_log))
    st.markdown("### Hasil per kandidat")
    st.caption("Terurut menurut fit score. Klik satu baris untuk melihat detail.")
    render_candidate_browser(result["rows"], result["job"], "final")


if __name__ == "__main__":
    main()
