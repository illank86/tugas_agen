"""Layanan yang menjalankan banyak alur rekrutmen secara bersamaan.

Modul ini yang membuat sistem berjalan terus-menerus, bukan sekali jalan lalu
selesai. Setiap permintaan lowongan menjadi satu "run": dijalankan di thread
tersendiri dari sebuah pool, sehingga beberapa lowongan diproses paralel dan
satu alur yang sedang menunggu keputusan manusia tidak menghentikan alur lain.

Catatan konkurensi:
  - Setiap run memiliki objek RecruitmentSystem sendiri, jadi tidak ada state
    agen yang dipakai bersama antar-run.
  - Bila memakai basis data berkas, SQLite dijalankan dalam mode WAL sehingga
    banyak koneksi dapat menulis bergantian tanpa saling memblokir lama.
  - conversation_id memakai run_id, bukan job_id, agar dua run untuk lowongan
    yang sama tidak mencampur jejak pesannya.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import threading
import traceback
import uuid

from .cv_reader import read_cv_folder
from .human_approval import QueuedApprover, SimulatedApprover
from .job_requirement_reader import load_jobs
from .llm_narrator import narrate_candidate, narrate_run
from .recruitment_service_models import RunSummary
from .result_reporting import build_candidate_report, radar_axes, rejection_reason
from .settings import Settings
from .skill_taxonomy import canonical_name
from .synthetic_data import SyntheticGenerator


def _now() -> str:
    """Waktu sekarang dalam ISO-8601 UTC."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    """Satu proses rekrutmen beserta seluruh keadaannya."""

    run_id: str
    job_id: str
    job_title: str
    status: str = "QUEUED"          # QUEUED | RUNNING | WAITING_HUMAN | DONE | FAILED | CANCELLED
    cancelled: bool = False
    submitted_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    outcome: Optional[dict] = None
    rows: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # Ringkasan pembacaan sumber kandidat dalam bentuk terstruktur, supaya
    # antarmuka tidak perlu mengurai kalimat pada `notes`.
    source_report: dict = field(default_factory=dict)
    approval_mode: str = "simulated"
    system: Any = None
    job: Any = None
    future: Optional[Future] = None

    def summary(self) -> dict:
        """Ringkasan ringan untuk daftar run, tanpa data kandidat."""
        placed = len(self.outcome.get("placed", [])) if self.outcome else 0
        shortlisted = sum(1 for row in self.rows if row["masuk_shortlist"])
        return {"run_id": self.run_id, "job_id": self.job_id,
                "job_title": self.job_title, "status": self.status,
                "submitted_at": self.submitted_at, "started_at": self.started_at,
                "finished_at": self.finished_at,
                "approval_mode": self.approval_mode,
                "candidates_evaluated": len(self.rows),
                "shortlisted": shortlisted, "placed": placed,
                "outcome": self.outcome.get("outcome") if self.outcome else None,
                "error": self.error}


class RecruitmentService:
    """Antrean dan registri run; dipakai oleh server API maupun skrip lain."""

    def __init__(self, max_workers: int = 4, database_path: str = ":memory:",
                 model_seed: int = 42):
        """Siapkan pool pekerja dan latih model screening satu kali.

        Args:
            max_workers: jumlah alur rekrutmen yang boleh berjalan bersamaan.
            database_path: berkas SQLite bersama, atau ":memory:" agar tiap
                run memakai basis data sementaranya sendiri.
            model_seed: seed pelatihan model screening.
        """
        from .experiment_runner import train_screening_model

        self.max_workers = max_workers
        self.database_path = database_path
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="recruit")
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.RLock()
        self.scorer, self.training_info = train_screening_model(model_seed)
        self.started_at = _now()

    # -- registri ----------------------------------------------------------
    def get(self, run_id: str) -> Optional[RunRecord]:
        """Ambil satu run berdasarkan id-nya."""
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> List[dict]:
        """Ringkasan seluruh run, terbaru lebih dulu."""
        with self._lock:
            records = list(self._runs.values())
        return [record.summary() for record in
                sorted(records, key=lambda r: r.submitted_at, reverse=True)]

    def stats(self) -> dict:
        """Statistik layanan untuk endpoint kesehatan dan pemantauan."""
        with self._lock:
            records = list(self._runs.values())
        by_status: Dict[str, int] = {}
        for record in records:
            by_status[record.status] = by_status.get(record.status, 0) + 1
        return {"started_at": self.started_at, "max_workers": self.max_workers,
                "database_path": self.database_path, "total_runs": len(records),
                "by_status": by_status,
                "screening_model": getattr(self.scorer, "version", "unknown")}

    # -- penyiapan data ----------------------------------------------------
    def _load_candidates(self, job, cv_folder: Optional[str],
                         synthetic_count: int, settings: Settings) -> tuple:
        """Muat kandidat dari folder CV, atau bangkitkan kandidat sintetis."""
        notes: List[str] = []
        if cv_folder:
            candidates, report = read_cv_folder(cv_folder, job, settings.today)
            methods = sorted(set(report["extraction_methods"].values()))
            notes.append(f"{report['total']} berkas CV terbaca "
                         f"(metode: {', '.join(methods)}).")
            if report.get("skipped_duplicates"):
                notes.append(f"Dilewati karena orangnya sudah terbaca dari format "
                             f"lain: {', '.join(report['skipped_duplicates'])}.")
            if report.get("llm_parsed"):
                notes.append(f"{len(report['llm_parsed'])} CV juga diekstrak "
                             f"LLM DeepSeek (skill, pengalaman, identitas).")
            if report.get("llm_errors"):
                notes.append(f"Ekstraksi DeepSeek gagal untuk "
                             f"{', '.join(report['llm_errors'])}; memakai "
                             f"parser aturan saja.")
            if report["empty_files"]:
                notes.append(f"Tidak ada teks terbaca dari "
                             f"{', '.join(report['empty_files'])} — kemungkinan "
                             f"PDF hasil pindaian.")
            notes.append("CV dari berkas tidak memiliki ground truth, sehingga "
                         "metrik akurasi tidak berlaku.")
            source = {"kind": "files", "folder": cv_folder,
                      "candidates": len(candidates), **report}
            return candidates, notes, True, source
        generator = SyntheticGenerator(settings.seed, today=settings.today)
        candidates = {candidate.candidate_id: candidate
                      for candidate in generator.make_candidates(synthetic_count, job)}
        notes.append(f"Kandidat sintetis: {len(candidates)} orang.")
        return candidates, notes, False, {"kind": "synthetic",
                                          "candidates": len(candidates)}

    # -- pengiriman pekerjaan ---------------------------------------------
    def submit(self, job_path: str, cv_folder: Optional[str] = None,
               settings: Optional[Settings] = None,
               synthetic_count: int = 150,
               approval_mode: str = "simulated",
               approval_timeout: float = 1800.0) -> RunRecord:
        """Daftarkan satu alur rekrutmen untuk dijalankan di latar belakang.

        Args:
            job_path: berkas atau folder job requirement (.txt).
            cv_folder: folder CV; None berarti memakai kandidat sintetis.
            settings: parameter sistem; None memakai nilai bawaan.
            synthetic_count: jumlah kandidat sintetis bila cv_folder kosong.
            approval_mode: "simulated" untuk approver otomatis, atau "manual"
                agar alur benar-benar menunggu keputusan lewat API.
            approval_timeout: batas tunggu keputusan pada mode manual.

        Returns:
            RunRecord dengan status QUEUED. Pemrosesan berjalan di thread lain,
            sehingga pemanggil langsung mendapat respons.

        Raises:
            FileNotFoundError, ValueError: bila berkas job requirement tidak
                valid. Validasi dilakukan di sini, bukan di dalam thread,
                supaya kesalahan input langsung terlihat oleh pemanggil.
        """
        jobs_list = load_jobs(job_path)
        job = jobs_list[0]
        job.validate()
        run_id = "run-" + uuid.uuid4().hex[:10]
        record = RunRecord(run_id=run_id, job_id=job.job_id, job_title=job.title,
                           job=job, approval_mode=approval_mode)
        with self._lock:
            self._runs[run_id] = record
        record.future = self._executor.submit(
            self._execute, record, jobs_list, cv_folder,
            settings or Settings(), synthetic_count, approval_mode,
            approval_timeout)
        return record

    def _execute(self, record: RunRecord, jobs_list, cv_folder: Optional[str],
                 settings: Settings, synthetic_count: int, approval_mode: str,
                 approval_timeout: float) -> None:
        """Jalankan satu alur rekrutmen penuh di dalam thread pekerja."""
        from .recruitment_workflow import RecruitmentSystem
        from .database import Database

        if record.cancelled:                     # dibatalkan sebelum sempat mulai
            record.status = "CANCELLED"
            record.finished_at = _now()
            return
        record.status = "RUNNING"
        record.started_at = _now()
        try:
            jobs = {job.job_id: job for job in jobs_list}
            job = jobs_list[0]
            candidates, notes, from_files, source = self._load_candidates(
                job, cv_folder, synthetic_count, settings)
            record.notes = notes
            record.source_report = source

            approver = (QueuedApprover(timeout_seconds=approval_timeout)
                        if approval_mode == "manual"
                        else SimulatedApprover(seed=settings.seed))
            system = RecruitmentSystem(
                settings, jobs, candidates, approver, scorer=self.scorer,
                database=Database(self.database_path),
                source_all_channels=from_files)
            record.system = system
            record.job = job
            # Pembatalan bisa datang saat data masih dimuat, sebelum approver
            # ada untuk menerimanya.
            if record.cancelled and isinstance(approver, QueuedApprover):
                approver.cancel()

            if approval_mode == "manual":
                record.status = "WAITING_HUMAN"
            outcome = system.run_job(job.job_id,
                                     target_count=max(len(candidates), 1),
                                     conversation_id=record.run_id)
            record.outcome = outcome
            record.rows = build_candidate_report(system, job, outcome)
            record.status = "CANCELLED" if record.cancelled else "DONE"
        except Exception as error:               # kegagalan satu run tidak
            record.status = "FAILED"             # boleh menjatuhkan layanan
            record.error = f"{type(error).__name__}: {error}"
            record.notes.append(traceback.format_exc(limit=3))
        finally:
            record.finished_at = _now()

    # -- gerbang persetujuan ----------------------------------------------
    def pending_approvals(self, run_id: Optional[str] = None) -> List[dict]:
        """Daftar gerbang yang sedang menunggu keputusan manusia."""
        with self._lock:
            records = ([self._runs[run_id]] if run_id and run_id in self._runs
                       else list(self._runs.values()))
        pending: List[dict] = []
        for record in records:
            approver = getattr(record.system, "context", None)
            approver = approver.approver if approver else None
            if isinstance(approver, QueuedApprover):
                for gate in approver.pending():
                    pending.append({"run_id": record.run_id, **gate})
        return pending

    def decide(self, run_id: str, gate_id: str, decision: str, approver_id: str,
               reason: str = "") -> bool:
        """Kirim keputusan manusia untuk satu gerbang pada sebuah run."""
        record = self.get(run_id)
        if record is None or record.system is None:
            return False
        approver = record.system.context.approver
        if not isinstance(approver, QueuedApprover):
            return False
        return approver.submit(gate_id, decision, approver_id, reason)

    def cancel(self, run_id: str) -> bool:
        """Batalkan satu run yang belum selesai dan bebaskan slot pekerjanya.

        Gerbang yang sedang dan akan menunggu langsung ditolak dengan
        penyetuju SYSTEM-CANCELLED, sehingga alur cepat mencapai akhirnya dan
        jejak auditnya tetap jujur: tidak ada manusia yang memutuskannya.

        Returns:
            True bila run ditemukan dan masih berjalan saat dibatalkan.
        """
        record = self.get(run_id)
        if record is None or record.status in ("DONE", "FAILED", "CANCELLED"):
            return False
        record.cancelled = True
        if record.future is not None and record.future.cancel():
            record.status = "CANCELLED"          # belum sempat mulai
            record.finished_at = _now()
            return True
        approver = getattr(getattr(record.system, "context", None), "approver", None)
        if isinstance(approver, QueuedApprover):
            approver.cancel()
        return True

    def active_runs(self, exclude: Optional[str] = None) -> List[RunRecord]:
        """Run yang belum selesai, selain `exclude`; dipakai untuk antrean.

        Run yang sudah dibatalkan tidak dihitung meski thread-nya belum
        selesai: ia akan segera membebaskan slotnya sendiri.
        """
        with self._lock:
            records = list(self._runs.values())
        return [r for r in records if r.run_id != exclude and not r.cancelled
                and r.status in ("QUEUED", "RUNNING", "WAITING_HUMAN")]

    # -- pembacaan hasil ---------------------------------------------------
    def candidate_rows(self, run_id: str) -> Optional[List[dict]]:
        """Baris hasil per kandidat untuk satu run, terurut menurut fit score."""
        record = self.get(run_id)
        return record.rows if record else None

    def candidate_detail(self, run_id: str, candidate_id: str) -> Optional[dict]:
        """Detail satu kandidat, termasuk sumbu radar dan alasan gugur."""
        record = self.get(run_id)
        if record is None:
            return None
        row = next((r for r in record.rows if r["candidate_id"] == candidate_id), None)
        if row is None:
            return None
        return {**row, "alasan_gugur": rejection_reason(row),
                "radar": radar_axes(row, record.job, canonical_name)}

    def candidate_narrative(self, run_id: str, candidate_id: str) -> Optional[dict]:
        """Narasi LLM (DeepSeek) yang menjelaskan hasil satu kandidat."""
        record = self.get(run_id)
        if record is None:
            return None
        row = next((r for r in record.rows if r["candidate_id"] == candidate_id), None)
        if row is None:
            return None
        text, source = narrate_candidate(row, record.job)
        return {"candidate_id": candidate_id, "narasi": text, "sumber": source}

    def run_narrative(self, run_id: str) -> Optional[dict]:
        """Ringkasan naratif LLM (DeepSeek) untuk seluruh hasil satu run."""
        record = self.get(run_id)
        if record is None or record.job is None:
            return None
        text, source = narrate_run(record.job, record.rows, record.outcome)
        return {"run_id": run_id, "narasi": text, "sumber": source}

    def messages(self, run_id: str, limit: int = 200) -> List[dict]:
        """Jejak pesan antar-agen untuk satu run (message trace)."""
        record = self.get(run_id)
        if record is None or record.system is None:
            return []
        rows = record.system.database.query(
            """SELECT sender, receiver, performative, schema, risk, size_bytes,
                      created_at FROM agent_messages
               WHERE conversation_id=? ORDER BY rowid LIMIT ?""",
            (run_id, limit))
        return [dict(row) for row in rows]

    def governance(self, run_id: str) -> Optional[dict]:
        """Ringkasan tata kelola: otonomi, audit, keamanan, gerbang manusia."""
        record = self.get(run_id)
        if record is None or record.system is None:
            return None
        system = record.system
        chain_ok, broken_at = system.audit.verify()
        return {"run_id": run_id,
                "autonomy": system.policy.summary(),
                "approvals": system.supervisor.approvals,
                "bypassed_gates": system.supervisor.bypassed_gates,
                "conflicts": system.supervisor.conflicts,
                "audit_records": system.audit.count,
                "audit_chain_valid": chain_ok,
                "audit_broken_at": broken_at,
                "messages": system.bus.sent_count,
                "payload_bytes": system.bus.sent_bytes,
                "schema_rejections": len(system.bus.rejected),
                "injection_attempts": len(system.screening.injection_attempts),
                "injection_successes": len(system.screening.injection_successes),
                "blocked_unauthorized_actions": system.policy.blocked_attempts}

    def shutdown(self, wait: bool = False) -> None:
        """Hentikan pool pekerja; dipanggil saat server dimatikan."""
        self._executor.shutdown(wait=wait)
