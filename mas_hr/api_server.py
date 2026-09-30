"""Server FastAPI: menjalankan sistem sebagai layanan yang hidup terus-menerus.

Perbedaan utama dengan mode CLI: di sini agen tidak dijalankan satu per satu
untuk satu lowongan lalu selesai. Server menerima banyak permintaan lowongan,
memasukkannya ke antrean, dan memprosesnya BERSAMAAN di beberapa thread
pekerja. Satu alur yang sedang menunggu keputusan manusia tidak menghentikan
alur lain.

Jalankan dengan:  python -m mas_hr.cli serve
Dokumentasi interaktif tersedia di /docs setelah server hidup.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .job_requirement_reader import JOB_SUFFIXES
from .recruitment_service import RecruitmentService
from .settings import Settings

# Folder kerja tempat berkas unggahan disimpan sebelum diproses.
UPLOAD_ROOT = Path("data/uploads")

service: Optional[RecruitmentService] = None


# ---------------------------------------------------------------------------
# Skema permintaan dan tanggapan
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    """Permintaan menjalankan satu alur rekrutmen."""

    job_path: str = Field(..., description="Berkas atau folder job requirement (.txt)")
    cv_folder: Optional[str] = Field(
        None, description="Folder CV (.pdf/.txt/.md). Kosong = kandidat sintetis.")
    synthetic_count: int = Field(150, ge=1, le=2000,
                                 description="Jumlah kandidat sintetis")
    approval_mode: str = Field(
        "simulated",
        description="'simulated' = approver otomatis; 'manual' = alur menunggu "
                    "keputusan lewat endpoint /api/approvals")
    approval_timeout: float = Field(1800.0, ge=1.0,
                                    description="Batas tunggu keputusan (detik)")
    theta: float = Field(0.75, ge=0.0, le=1.0, description="Ambang shortlist")
    gamma: float = Field(0.85, ge=0.5, le=0.99,
                         description="Ambang keyakinan otonomi")
    top_k: int = Field(0, ge=0, le=100, description="0 = pakai theta")
    hitl_enabled: bool = Field(True, description="False = arm B2 tanpa gerbang")
    seed: int = Field(42, ge=0)

    def to_settings(self) -> Settings:
        """Ubah parameter permintaan menjadi objek Settings."""
        return Settings(seed=self.seed, theta=self.theta, gamma=self.gamma,
                        top_k=self.top_k, hitl_enabled=self.hitl_enabled)


class DecisionRequest(BaseModel):
    """Keputusan manusia atas satu gerbang yang sedang menunggu."""

    decision: str = Field(..., description="APPROVED | REJECTED | REQUEST_REVISION")
    approver_id: str = Field(..., description="Identitas pemberi keputusan")
    reason: str = Field("", description="Alasan keputusan, untuk jejak audit")


# ---------------------------------------------------------------------------
# Siklus hidup aplikasi
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Siapkan layanan saat server hidup dan matikan pool saat server berhenti."""
    global service
    service = RecruitmentService(
        max_workers=getattr(app.state, "max_workers", 4),
        database_path=getattr(app.state, "database_path", ":memory:"))
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    service.shutdown(wait=False)


app = FastAPI(
    title="Agentic HR Outsourcing — Multi-Agent Service",
    description=__doc__,
    version="2.1.0",
    lifespan=lifespan)


def require_service() -> RecruitmentService:
    """Pastikan layanan sudah siap; dipakai seluruh endpoint."""
    if service is None:
        raise HTTPException(503, "layanan belum siap")
    return service


def require_run(run_id: str):
    """Ambil run atau kembalikan 404 dengan pesan yang jelas."""
    record = require_service().get(run_id)
    if record is None:
        raise HTTPException(404, f"run tidak ditemukan: {run_id}")
    return record


# ---------------------------------------------------------------------------
# Endpoint umum
# ---------------------------------------------------------------------------
@app.get("/health", tags=["umum"])
def health() -> dict:
    """Status layanan, jumlah pekerja, dan rekap run menurut statusnya."""
    return {"status": "ok", **require_service().stats()}


@app.get("/api/skills", tags=["umum"])
def list_skills() -> List[dict]:
    """Taksonomi skill kanonik; dipakai saat menyusun job requirement."""
    from .skill_taxonomy import SKILLS
    return [{"skill_id": skill_id, "name": name, "synonyms": synonyms,
             "parent": parent}
            for skill_id, (name, synonyms, parent) in sorted(SKILLS.items())]


@app.get("/api/agents", tags=["umum"])
def list_agents() -> List[dict]:
    """Delapan agen beserta tipe, mobilitas, dan allowlist aksinya."""
    from .settings import AGENT_ACTIONS, AGENT_PROFILE
    return [{"agent_id": agent,
             "type": AGENT_PROFILE.get(agent, ("AI", "STATIC", "A3"))[0],
             "mobility": AGENT_PROFILE.get(agent, ("AI", "STATIC", "A3"))[1],
             "autonomy_level": AGENT_PROFILE.get(agent, ("AI", "STATIC", "A3"))[2],
             "allowed_actions": sorted(actions)}
            for agent, actions in AGENT_ACTIONS.items()]


@app.get("/api/folders", tags=["umum"])
def inspect_folder(path: str = Query(..., description="Jalur folder yang diperiksa"),
                   kind: str = Query("cv", pattern="^(cv|job)$")) -> dict:
    """Periksa isi sebuah folder sebelum dipakai sebagai sumber data.

    Membantu klien memilih folder tanpa menebak: mengembalikan daftar berkas
    yang dikenali beserta pesan bila folder tidak valid.
    """
    suffixes = list(JOB_SUFFIXES) if kind == "job" else [".pdf", ".txt", ".md"]
    folder = Path(path).expanduser()
    if not folder.exists():
        raise HTTPException(404, f"folder tidak ditemukan: {folder}")
    if not folder.is_dir():
        raise HTTPException(400, f"bukan folder: {folder}")
    files = sorted(f.name for f in folder.iterdir()
                   if f.suffix.lower() in suffixes
                   and not f.name.endswith(".meta.txt"))
    return {"path": str(folder), "kind": kind, "files": files, "count": len(files),
            "subfolders": sorted(f.name for f in folder.iterdir() if f.is_dir())}


# ---------------------------------------------------------------------------
# Menjalankan alur rekrutmen
# ---------------------------------------------------------------------------
@app.post("/api/runs", status_code=202, tags=["run"])
def submit_run(request: RunRequest) -> dict:
    """Daftarkan satu alur rekrutmen untuk diproses di latar belakang.

    Mengembalikan 202 beserta run_id segera; pemrosesan berjalan di thread
    pekerja sehingga beberapa lowongan dapat diproses bersamaan.
    """
    if request.approval_mode not in ("simulated", "manual"):
        raise HTTPException(422, "approval_mode harus 'simulated' atau 'manual'")
    try:
        record = require_service().submit(
            job_path=request.job_path, cv_folder=request.cv_folder,
            settings=request.to_settings(), synthetic_count=request.synthetic_count,
            approval_mode=request.approval_mode,
            approval_timeout=request.approval_timeout)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))
    except ValueError as error:
        raise HTTPException(422, str(error))
    return {"run_id": record.run_id, "status": record.status,
            "job_id": record.job_id, "job_title": record.job_title,
            "poll": f"/api/runs/{record.run_id}"}


@app.post("/api/runs/upload", status_code=202, tags=["run"])
async def submit_run_with_upload(
        job_file: UploadFile = File(..., description="Job requirement .txt (kunci: nilai) "
                                        "atau job description teks bebas .txt/.md/.pdf"),
        cv_files: List[UploadFile] = File(default=[], description="Berkas CV"),
        meta_files: List[UploadFile] = File(default=[],
                                            description="Berkas *.meta.txt"),
        approval_mode: str = Form("simulated"),
        theta: float = Form(0.75),
        gamma: float = Form(0.85),
        seed: int = Form(42)) -> dict:
    """Unggah job requirement dan sekumpulan CV, lalu langsung proses.

    Berkas disimpan ke folder kerja tersendiri per pengiriman, sehingga
    unggahan dari beberapa klien tidak saling menimpa.
    """
    import uuid

    workspace = UPLOAD_ROOT / f"upload-{uuid.uuid4().hex[:10]}"
    (workspace / "cv").mkdir(parents=True, exist_ok=True)
    job_path = workspace / (job_file.filename or "job.txt")
    job_path.write_bytes(await job_file.read())
    for uploaded in list(cv_files) + list(meta_files):
        if not uploaded.filename:
            continue
        (workspace / "cv" / uploaded.filename).write_bytes(await uploaded.read())

    has_cv = any((workspace / "cv").iterdir())
    try:
        record = require_service().submit(
            job_path=str(job_path),
            cv_folder=str(workspace / "cv") if has_cv else None,
            settings=Settings(seed=seed, theta=theta, gamma=gamma),
            approval_mode=approval_mode)
    except (FileNotFoundError, ValueError) as error:
        shutil.rmtree(workspace, ignore_errors=True)
        raise HTTPException(422, str(error))
    return {"run_id": record.run_id, "status": record.status,
            "workspace": str(workspace),
            "cv_uploaded": len(list((workspace / "cv").iterdir())),
            "poll": f"/api/runs/{record.run_id}"}


@app.get("/api/runs", tags=["run"])
def list_runs() -> List[dict]:
    """Ringkasan seluruh run yang pernah dikirim, terbaru lebih dulu."""
    return require_service().list_runs()


@app.get("/api/runs/{run_id}", tags=["run"])
def get_run(run_id: str) -> dict:
    """Status dan ringkasan satu run, termasuk catatan pembacaan berkas."""
    record = require_run(run_id)
    return {**record.summary(), "notes": record.notes}


@app.get("/api/runs/{run_id}/candidates", tags=["hasil"])
def get_candidates(run_id: str,
                   limit: int = Query(100, ge=1, le=1000),
                   offset: int = Query(0, ge=0),
                   shortlisted_only: bool = Query(False)) -> dict:
    """Hasil per kandidat, terurut menurut fit score.

    Kandidat yang gugur tetap disertakan beserta alasannya: daftar yang hanya
    memuat pemenang tidak dapat diaudit.
    """
    from .result_reporting import rejection_reason

    record = require_run(run_id)
    rows = [row for row in record.rows
            if not shortlisted_only or row["masuk_shortlist"]]
    page = rows[offset:offset + limit]
    return {"run_id": run_id, "total": len(rows), "limit": limit, "offset": offset,
            "candidates": [{
                "candidate_id": row["candidate_id"], "nama": row["nama"],
                "peringkat": row["peringkat"], "fit_score": row["fit_score"],
                "skor_screening": row["skor_screening"],
                "pengalaman_tahun": row["pengalaman_tahun"],
                "status_kepatuhan": row["status_kepatuhan"],
                "tahap": row["tahap"], "masuk_shortlist": row["masuk_shortlist"],
                "ditempatkan": row["ditempatkan"],
                "kontribusi": row["kontribusi"],
                "alasan_gugur": rejection_reason(row)} for row in page]}


@app.get("/api/runs/{run_id}/candidates/{candidate_id}", tags=["hasil"])
def get_candidate_detail(run_id: str, candidate_id: str) -> dict:
    """Detail satu kandidat: rincian skor, dokumen, temuan, dan sumbu radar."""
    require_run(run_id)
    detail = require_service().candidate_detail(run_id, candidate_id)
    if detail is None:
        raise HTTPException(404, f"kandidat tidak ditemukan: {candidate_id}")
    return detail


@app.get("/api/runs/{run_id}/candidates/{candidate_id}/narrative", tags=["hasil"])
def get_candidate_narrative(run_id: str, candidate_id: str) -> dict:
    """Narasi DeepSeek yang menjelaskan hasil satu kandidat.

    `sumber` bernilai "deepseek" atau "template (...)" bila LLM tidak tersedia.
    """
    require_run(run_id)
    narrative = require_service().candidate_narrative(run_id, candidate_id)
    if narrative is None:
        raise HTTPException(404, f"kandidat tidak ditemukan: {candidate_id}")
    return narrative


@app.get("/api/runs/{run_id}/narrative", tags=["hasil"])
def get_run_narrative(run_id: str) -> dict:
    """Ringkasan naratif DeepSeek untuk seluruh hasil satu run."""
    require_run(run_id)
    narrative = require_service().run_narrative(run_id)
    if narrative is None:
        raise HTTPException(409, "run belum memiliki hasil")
    return narrative


@app.get("/api/runs/{run_id}/messages", tags=["hasil"])
def get_messages(run_id: str, limit: int = Query(200, ge=1, le=2000)) -> dict:
    """Jejak pesan antar-agen untuk satu run."""
    require_run(run_id)
    messages = require_service().messages(run_id, limit)
    return {"run_id": run_id, "count": len(messages), "messages": messages}


@app.get("/api/runs/{run_id}/governance", tags=["hasil"])
def get_governance(run_id: str) -> dict:
    """Ringkasan tata kelola: otonomi, gerbang manusia, audit, dan keamanan."""
    require_run(run_id)
    report = require_service().governance(run_id)
    if report is None:
        raise HTTPException(409, "run belum memiliki sistem yang berjalan")
    return report


# ---------------------------------------------------------------------------
# Gerbang human-in-the-loop lewat HTTP
# ---------------------------------------------------------------------------
@app.get("/api/approvals", tags=["hitl"])
def list_pending_approvals(run_id: Optional[str] = None) -> dict:
    """Gerbang yang sedang menunggu keputusan manusia.

    Hanya terisi pada run dengan approval_mode "manual": di mode itu alur
    benar-benar berhenti dan menunggu keputusan datang lewat HTTP.
    """
    pending = require_service().pending_approvals(run_id)
    return {"count": len(pending), "pending": pending}


@app.post("/api/approvals/{run_id}/{gate_id}", tags=["hitl"])
def submit_decision(run_id: str, gate_id: str, request: DecisionRequest) -> dict:
    """Kirim keputusan manusia untuk satu gerbang sehingga alur berlanjut."""
    require_run(run_id)
    try:
        accepted = require_service().decide(
            run_id, gate_id, request.decision, request.approver_id, request.reason)
    except ValueError as error:
        raise HTTPException(422, str(error))
    if not accepted:
        raise HTTPException(
            404, "gerbang tidak ditemukan atau sudah diputuskan; periksa "
                 "/api/approvals untuk daftar terkini")
    return {"run_id": run_id, "gate_id": gate_id, "decision": request.decision,
            "status": "diterima"}


@app.exception_handler(Exception)
async def unhandled_error(request, error: Exception) -> JSONResponse:
    """Kembalikan galat tak terduga sebagai JSON, tanpa menjatuhkan server."""
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(error).__name__}: {error}"})


def create_app(max_workers: int = 4, database_path: str = ":memory:") -> FastAPI:
    """Bentuk aplikasi dengan konfigurasi pool dan basis data tertentu."""
    app.state.max_workers = max_workers
    app.state.database_path = database_path
    return app
