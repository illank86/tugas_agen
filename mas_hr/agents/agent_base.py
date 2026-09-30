"""Kelas dasar agen dan konteks sumber daya bersama.

Tiga hal ditegakkan di sini agar tidak bergantung pada disiplin masing-masing
agen: identitas agen, allowlist aksi, dan pencatatan audit.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..agent_messaging import Message, MessageBus, Performative, Risk
from ..audit_trail import AuditTrail
from ..autonomy_policy import Decision, PolicyEngine
from ..settings import Settings


@dataclass
class AgentContext:
    """Sumber daya yang dipakai bersama seluruh agen.

    Attributes:
        blackboard: konteks bersama per lowongan (laporan 5.8.3). Dipakai agar
            agen tidak perlu saling mengirim ulang data besar lewat pesan —
            mitigasi kelemahan communication overhead.
    """

    bus: MessageBus
    policy: PolicyEngine
    audit: AuditTrail
    database: Any
    settings: Settings
    embedder: Any
    candidates: Dict[str, Any]
    jobs: Dict[str, Any]
    approver: Any = None
    blackboard: Dict[str, dict] = field(default_factory=dict)

    def board(self, job_id: str) -> dict:
        """Ambil (atau buat) blackboard untuk satu lowongan."""
        return self.blackboard.setdefault(job_id, {})


class Agent:
    """Perilaku dasar yang dimiliki setiap agen."""

    name = "Agent"

    def __init__(self, context: AgentContext):
        """Daftarkan agen ke message bus dengan namanya sendiri."""
        self.context = context
        context.bus.register(self.name, self.handle)

    def request_permission(self, action: str, confidence: float = 1.0) -> Decision:
        """Minta izin PolicyEngine sebelum menjalankan sebuah aksi.

        Raises:
            UnauthorizedAction: bila aksi di luar allowlist agen ini.
        """
        return self.context.policy.evaluate(self.name, action, confidence)

    def record(self, entity_type: str, entity_id: str, action: str,
               after: Optional[dict] = None) -> None:
        """Catat kejadian ke jejak audit atas nama agen ini."""
        self.context.audit.record(entity_type, entity_id, action,
                                  "AGENT", self.name, after=after)

    def send(self, receiver: str, performative: Performative, schema: str,
             payload: dict, conversation_id: str, trace_id: str,
             risk: Risk = Risk.LOW,
             idempotency_key: Optional[str] = None) -> Optional[Message]:
        """Kirim pesan ke agen lain melalui bus dan kembalikan balasannya."""
        message = self.context.bus.compose(
            performative, self.name, receiver, schema, payload,
            conversation_id, trace_id, risk, idempotency_key)
        return self.context.bus.send(message)

    def reply(self, incoming: Message, performative: Performative, schema: str,
              payload: dict, risk: Risk = Risk.LOW) -> Message:
        """Bentuk balasan atas sebuah pesan, mewarisi conversation dan trace."""
        outgoing = self.context.bus.compose(
            performative, self.name, incoming.sender, schema, payload,
            incoming.conversation_id, incoming.trace_id, risk)
        return outgoing

    def handle(self, message: Message) -> Optional[Message]:
        """Tangani pesan masuk; wajib ditimpa oleh setiap agen turunan."""
        raise NotImplementedError(f"{self.name} tidak menangani {message.schema}")
