"""Gerbang persetujuan manusia (laporan 5.9, 8.2 fitur 5).

Checkpoint di sini adalah STATE pada mesin status, bukan notifikasi: proses
benar-benar berhenti sampai ada keputusan manusia.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import random
import threading
import time
import uuid

# Peran yang berwenang pada tiap gerbang (sejalan dengan matriks RBAC laporan).
CHECKPOINT_ROLES: Dict[str, str] = {
    "HITL-1": "HR Manager",
    "HITL-2": "Compliance Officer",
    "HITL-3": "Recruiter",
    "HITL-4": "HR Manager",
    "HITL-5": "HR Manager",
    "HITL-6": "HR Manager",
}


@dataclass
class ApprovalRequest:
    """Permintaan keputusan yang diajukan Supervisor kepada manusia."""

    checkpoint_id: str
    decision_type: str
    evidence: dict
    job_id: Optional[str] = None
    candidate_id: Optional[str] = None


@dataclass
class ApprovalResult:
    """Keputusan manusia beserta jejak akuntabilitasnya."""

    decision: str          # APPROVED | REJECTED | REQUEST_REVISION
    approver_id: str
    approver_role: str
    reason: str
    latency_seconds: float


class SimulatedApprover:
    """Manusia tersimulasi untuk eksperimen otomatis.

    `accuracy` sengaja < 1.0: reviewer manusia juga bisa salah. Tanpa itu,
    arm dengan HITL akan tampak sempurna dan hasil evaluasi tidak kredibel.
    """

    def __init__(self, seed: int = 42, accuracy: float = 0.95,
                 latency_seconds: float = 0.002, reject_rate: float = 0.05,
                 revision_rate: float = 0.03):
        """Siapkan approver tersimulasi.

        Args:
            seed: penentu keacakan agar hasil dapat direproduksi.
            accuracy: peluang reviewer menilai dokumen dengan benar di HITL-2.
            latency_seconds: rata-rata waktu berpikir manusia.
            reject_rate: peluang penolakan pada gerbang non-compliance.
            revision_rate: peluang permintaan revisi.
        """
        self.rng = random.Random(seed)
        self.accuracy = accuracy
        self.latency_seconds = latency_seconds
        self.reject_rate = reject_rate
        self.revision_rate = revision_rate
        self.total_wait = 0.0
        self.log: List[dict] = []

    def _simulate_thinking(self) -> float:
        """Tunggu sejenak dan kembalikan durasinya, agar T_mesin bisa dipisah."""
        delay = max(0.0, self.rng.gauss(self.latency_seconds, self.latency_seconds * 0.3))
        time.sleep(min(delay, 0.02))
        self.total_wait += delay
        return delay

    def review(self, request: ApprovalRequest) -> ApprovalResult:
        """Putuskan sebuah permintaan persetujuan.

        Pada HITL-2 keputusan dibuat berdasar label emas dokumen (tersedia
        hanya di simulasi), dilemahkan oleh `accuracy`. Pada gerbang lain,
        keputusan mengikuti peluang penolakan/revisi.
        """
        delay = self._simulate_thinking()
        role = CHECKPOINT_ROLES.get(request.checkpoint_id, "Recruiter")

        if request.checkpoint_id == "HITL-2":
            truly_problematic = bool(request.evidence.get("document_problem_truth"))
            correct = self.rng.random() < self.accuracy
            found = truly_problematic if correct else not truly_problematic
            decision = "REJECTED" if found else "APPROVED"
            reason = ("temuan dokumen dikonfirmasi bermasalah" if found
                      else "dokumen dinyatakan sah setelah tinjauan manual")
        elif self.rng.random() < self.revision_rate:
            decision, reason = "REQUEST_REVISION", "minta penyesuaian kriteria"
        elif self.rng.random() < self.reject_rate:
            decision, reason = "REJECTED", "tidak sesuai kebutuhan klien"
        else:
            decision, reason = "APPROVED", "disetujui setelah tinjauan"

        role_slug = "".join(ch for ch in role.upper() if ch.isalpha())[:3]
        self.log.append({"checkpoint": request.checkpoint_id, "decision": decision,
                         "latency": delay, "role": role})
        return ApprovalResult(decision=decision, approver_id=f"USR-{role_slug}-01",
                              approver_role=role, reason=reason, latency_seconds=delay)


class TerminalApprover(SimulatedApprover):
    """Approver yang meminta keputusan dari terminal; dipakai saat presentasi."""

    def review(self, request: ApprovalRequest) -> ApprovalResult:
        """Tampilkan bukti dan tanyakan keputusan kepada pengguna.

        Label emas sengaja disembunyikan: manusia sungguhan tidak melihatnya.
        """
        started = time.time()
        role = CHECKPOINT_ROLES.get(request.checkpoint_id, "Recruiter")
        print(f"\n=== {request.checkpoint_id} | {request.decision_type} "
              f"| peran: {role} ===")
        for key, value in request.evidence.items():
            if key == "document_problem_truth":
                continue
            print(f"  {key}: {value}")
        answer = ""
        while answer not in ("a", "r", "v"):
            answer = input("  [a]pprove / [r]eject / re[v]ision > ").strip().lower()
        decision = {"a": "APPROVED", "r": "REJECTED", "v": "REQUEST_REVISION"}[answer]
        reason = input("  alasan > ").strip() or "-"
        delay = time.time() - started
        self.total_wait += delay
        self.log.append({"checkpoint": request.checkpoint_id, "decision": decision,
                         "latency": delay, "role": role})
        return ApprovalResult(decision, "USR-TERMINAL", role, reason, delay)


class QueuedApprover:
    """Approver yang MENUNGGU keputusan datang dari luar proses (mis. HTTP).

    Inilah bentuk human-in-the-loop yang sesungguhnya pada mode server: alur
    rekrutmen benar-benar berhenti pada gerbang, thread-nya memblokir, dan
    baru berlanjut setelah seseorang mengirim keputusan lewat API. Karena
    setiap alur berjalan di thread-nya sendiri, alur lain tetap jalan selama
    satu alur menunggu manusia.
    """

    def __init__(self, timeout_seconds: float = 1800.0,
                 on_timeout: str = "REJECTED"):
        """Siapkan antrean gerbang yang menunggu keputusan.

        Args:
            timeout_seconds: batas tunggu sebelum keputusan cadangan dipakai.
            on_timeout: keputusan yang dipakai bila batas tunggu terlampaui.
                Bawaannya REJECTED — pada domain teregulasi, diam bukan berarti
                setuju.
        """
        self.timeout_seconds = timeout_seconds
        self.on_timeout = on_timeout
        self.total_wait = 0.0
        self.log: List[dict] = []
        self._pending: Dict[str, dict] = {}
        self._lock = threading.RLock()
        self.cancelled = False

    def cancel(self, reason: str = "proses dibatalkan") -> None:
        """Tolak seluruh gerbang yang menunggu dan gerbang berikutnya.

        Dipakai saat sebuah run ditinggalkan. Tanpa ini thread alurnya tetap
        memblokir sampai batas tunggu habis dan menghabiskan slot pekerja,
        sehingga run baru tertahan di antrean.
        """
        with self._lock:
            self.cancelled = True
            waiting = [entry for entry in self._pending.values()
                       if entry["decision"] is None]
            for entry in waiting:
                entry["decision"] = ("REJECTED", "SYSTEM-CANCELLED", reason)
        for entry in waiting:
            entry["event"].set()

    def pending(self) -> List[dict]:
        """Daftar gerbang yang sedang menunggu keputusan manusia.

        Gerbang yang keputusannya sudah masuk tidak ikut didaftar, meski
        thread alurnya belum sempat menghapusnya. Tanpa penyaring ini,
        antarmuka bisa menampilkan lagi gerbang yang sudah diputuskan dan
        tombolnya tidak berbuat apa-apa.
        """
        with self._lock:
            return [{"gate_id": gate_id,
                     "checkpoint_id": entry["request"].checkpoint_id,
                     "decision_type": entry["request"].decision_type,
                     "job_id": entry["request"].job_id,
                     "candidate_id": entry["request"].candidate_id,
                     "evidence": {k: v for k, v in entry["request"].evidence.items()
                                  if k != "document_problem_truth"},
                     "requested_at": entry["requested_at"]}
                    for gate_id, entry in self._pending.items()
                    if entry["decision"] is None]

    def submit(self, gate_id: str, decision: str, approver_id: str,
               reason: str = "") -> bool:
        """Kirim keputusan untuk satu gerbang yang sedang menunggu.

        Returns:
            True bila gerbang ditemukan dan keputusan diterima.

        Raises:
            ValueError: bila keputusan bukan salah satu nilai yang sah.
        """
        if decision not in ("APPROVED", "REJECTED", "REQUEST_REVISION"):
            raise ValueError(f"keputusan tidak sah: {decision}")
        with self._lock:
            entry = self._pending.get(gate_id)
            # Keputusan pertama yang berlaku; kiriman ganda tidak menimpanya.
            if entry is None or entry["decision"] is not None:
                return False
            entry["decision"] = (decision, approver_id, reason)
        entry["event"].set()
        return True

    def review(self, request: ApprovalRequest) -> ApprovalResult:
        """Blokir sampai keputusan masuk, atau sampai batas tunggu terlampaui."""
        if self.cancelled:
            role = CHECKPOINT_ROLES.get(request.checkpoint_id, "Recruiter")
            self.log.append({"checkpoint": request.checkpoint_id,
                             "decision": "REJECTED", "latency": 0.0, "role": role})
            return ApprovalResult("REJECTED", "SYSTEM-CANCELLED", role,
                                  "proses dibatalkan", 0.0)
        gate_id =f"{request.checkpoint_id}:{request.job_id}:" \
                  f"{request.candidate_id or '-'}:{uuid.uuid4().hex[:6]}"
        event = threading.Event()
        started = time.time()
        with self._lock:
            self._pending[gate_id] = {"request": request, "event": event,
                                      "decision": None,
                                      "requested_at": datetime.now(
                                          timezone.utc).isoformat()}
        answered = event.wait(self.timeout_seconds)
        waited = time.time() - started
        self.total_wait += waited
        with self._lock:
            entry = self._pending.pop(gate_id, {})
        role = CHECKPOINT_ROLES.get(request.checkpoint_id, "Recruiter")

        if answered and entry.get("decision"):
            decision, approver_id, reason = entry["decision"]
        else:
            decision, approver_id = self.on_timeout, "SYSTEM-TIMEOUT"
            reason = (f"tidak ada keputusan dalam {self.timeout_seconds:.0f} detik; "
                      f"diam tidak dianggap persetujuan")
        self.log.append({"checkpoint": request.checkpoint_id, "decision": decision,
                         "latency": waited, "role": role})
        return ApprovalResult(decision=decision, approver_id=approver_id,
                              approver_role=role, reason=reason,
                              latency_seconds=waited)
