"""Mesin bounded autonomy (laporan 5.9.2).

    Autonomous(a) <=> R(a) <= tau  DAN  conf(a) >= gamma
                      DAN  a termasuk allowlist agen tersebut

Dipisahkan dari Supervisor supaya aturan otonomi dapat diubah dan diaudit
sebagai artefak tersendiri. Syarat ketiga bersifat statis: tidak ada prompt
yang dapat menegosiasikannya.
"""
from dataclasses import dataclass
from typing import List, Optional

from .settings import ACTIONS, AGENT_ACTIONS, Settings


class UnauthorizedAction(Exception):
    """Agen mencoba aksi di luar allowlist-nya; selalu diblokir dan dicatat."""


@dataclass
class Decision:
    """Hasil evaluasi satu permintaan aksi oleh PolicyEngine."""

    action: str
    agent: str
    autonomous: bool
    level: str
    risk: float
    confidence: float
    checkpoint: Optional[str]
    reason: str


class PolicyEngine:
    """Penjaga gerbang: memutuskan sebuah aksi boleh otonom atau harus manusia."""

    def __init__(self, settings: Settings, audit=None):
        """Siapkan mesin kebijakan.

        Args:
            settings: sumber nilai gamma, tau, dan saklar hitl_enabled.
            audit: AuditTrail opsional untuk mencatat percobaan aksi terlarang.
        """
        self.settings = settings
        self.audit = audit
        self.blocked_attempts: List[str] = []
        self.decisions: List[Decision] = []

    def evaluate(self, agent: str, action: str, confidence: float = 1.0) -> Decision:
        """Putuskan apakah `agent` boleh menjalankan `action` tanpa manusia.

        Args:
            agent: nama agen pemohon.
            action: kunci aksi pada settings.ACTIONS.
            confidence: keyakinan terkalibrasi atas keluaran aksi tersebut.

        Returns:
            Decision berisi status otonom, level, risiko, dan checkpoint.

        Raises:
            UnauthorizedAction: bila aksi tidak terdaftar atau di luar
                allowlist agen. Ini gerbang mekanis, bukan imbauan.
        """
        spec = ACTIONS.get(action)
        if spec is None:
            raise UnauthorizedAction(f"aksi tidak terdaftar: {action}")
        if action not in AGENT_ACTIONS.get(agent, set()):
            message = f"{agent} mencoba aksi terlarang '{action}'"
            self.blocked_attempts.append(message)
            if self.audit:
                self.audit.record("policy", action, "BLOCKED_UNAUTHORIZED_ACTION",
                                  "SYSTEM", "PolicyEngine", after={"agent": agent})
            raise UnauthorizedAction(message)

        level = spec["level"]
        # R(a) = L(a) x I(a), dinormalisasi dengan dampak maksimum 5.
        risk = (1.0 - max(0.0, min(1.0, confidence))) * spec["impact"] / 5.0

        if level == "A0":
            autonomous, reason = False, "aksi tidak terpulihkan: keputusan wajib manusia"
        elif level == "A1":
            autonomous, reason = False, "level A1: usulan agen menunggu persetujuan"
        elif confidence < self.settings.gamma:
            autonomous = False
            reason = f"keyakinan {confidence:.2f} < gamma {self.settings.gamma:.2f}"
        elif risk > self.settings.tau:
            autonomous = False
            reason = f"risiko {risk:.2f} > tau {self.settings.tau:.2f}"
        else:
            autonomous, reason = True, f"otonom pada level {level}"

        if not self.settings.hitl_enabled and not autonomous:
            autonomous = True
            reason = "HITL dimatikan (arm B2): dieksekusi tanpa persetujuan manusia"

        decision = Decision(action=action, agent=agent, autonomous=autonomous,
                            level=level, risk=risk, confidence=confidence,
                            checkpoint=spec["checkpoint"], reason=reason)
        self.decisions.append(decision)
        return decision

    def summary(self) -> dict:
        """Ringkas statistik keputusan untuk metrik Human Intervention Rate.

        HIR dipecah menjadi terencana (aksi A0/A1 yang memang wajib manusia)
        dan eskalasi (akibat ketidakpastian). Hanya HIR eskalasi yang layak
        dioptimasi menuju nol.
        """
        total = len(self.decisions)
        escalated = [d for d in self.decisions if not d.autonomous]
        planned = [d for d in escalated if d.level in ("A0", "A1")]
        uncertainty = [d for d in escalated if d.level in ("A2", "A3")]
        return {"total_decisions": total,
                "escalated": len(escalated),
                "hir": len(escalated) / total if total else 0.0,
                "hir_planned": len(planned) / total if total else 0.0,
                "hir_escalation": len(uncertainty) / total if total else 0.0,
                "blocked_unauthorized": len(self.blocked_attempts)}
