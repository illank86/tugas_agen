"""Amplop pesan bergaya FIPA-ACL dan broker antar-agen (laporan 5.6).

Yang ditegakkan broker ini, bukan sekadar disepakati antar-agen:
  - verifikasi tanda tangan   -> pesan tak terotentikasi ditolak
  - validasi skema payload    -> mencegah schema drift dan payload asing
  - idempotency               -> retry tidak menimbulkan efek ganda
  - batas hop                 -> mencegah siklus pesan antar-agen
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional
import hashlib
import hmac
import json
import threading
import uuid


class Performative(str, Enum):
    """Maksud komunikatif sebuah pesan, terpisah dari isinya."""
    REQUEST = "REQUEST"
    INFORM = "INFORM"
    PROPOSE = "PROPOSE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    FAILURE = "FAILURE"
    CFP = "CFP"


class Risk(str, Enum):
    """Tingkat risiko pesan; dibaca untuk menentukan perlunya gerbang HITL."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Registry skema: nama@versi -> daftar field wajib pada payload.
SCHEMAS: Dict[str, List[str]] = {
    "JobRequestRaw@1.0":            ["client_id", "job_id", "text"],
    "StructuredJobRequirement@1.0": ["job_id", "title", "required_skills",
                                     "min_experience_years", "required_documents"],
    "CandidateSearchRequest@1.0":   ["job_id", "target_count"],
    "SourcingBid@1.0":              ["channel", "expected_yield", "cost", "quality"],
    "CandidateProfileBatch@1.0":    ["batch_id", "job_id", "candidates"],
    "ScreeningResult@1.0":          ["job_id", "results"],
    "ComplianceStatus@1.0":         ["job_id", "statuses"],
    "ComplianceEscalation@1.0":     ["candidate_id", "reason", "confidence"],
    "ShortlistProposal@1.0":        ["job_id", "ranking"],
    "ApprovalRequest@1.0":          ["checkpoint_id", "decision_type", "evidence"],
    "InterviewRecommendation@1.0":  ["job_id", "candidates"],
    "InterviewFeedback@1.0":        ["job_id", "feedback"],
    "PlacementAuthorization@1.0":   ["job_id", "candidate_id", "approver_id"],
    "ContractDraft@1.0":            ["job_id", "candidate_id", "clauses"],
}


class MessageError(Exception):
    """Pesan ditolak broker: tanda tangan, skema, atau batas hop."""


@dataclass
class Message:
    """Satu pesan antar-agen beserta metadata tata kelolanya."""

    performative: Performative
    sender: str
    receiver: str
    schema: str
    payload: dict
    conversation_id: str
    trace_id: str
    risk: Risk = Risk.LOW
    message_id: str = field(default_factory=lambda: "msg-" + uuid.uuid4().hex[:8])
    reply_to: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    idempotency_key: Optional[str] = None
    signature: Optional[str] = None
    hop: int = 0

    def signable_bytes(self) -> bytes:
        """Serialkan bagian pesan yang ditandatangani, secara deterministik."""
        core = {"message_id": self.message_id, "conversation_id": self.conversation_id,
                "performative": self.performative.value, "sender": self.sender,
                "receiver": self.receiver, "schema": self.schema, "payload": self.payload}
        return json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")

    def size_bytes(self) -> int:
        """Ukuran payload dalam byte; dipakai sebagai communication cost."""
        return len(json.dumps(self.payload, ensure_ascii=False).encode("utf-8"))

    def as_record(self) -> dict:
        """Bentuk baris untuk tabel agent_messages."""
        return {"message_id": self.message_id, "conversation_id": self.conversation_id,
                "trace_id": self.trace_id, "sender": self.sender,
                "receiver": self.receiver, "performative": self.performative.value,
                "schema": self.schema,
                "payload_json": json.dumps(self.payload, ensure_ascii=False),
                "risk": self.risk.value, "size_bytes": self.size_bytes(),
                "created_at": self.timestamp}


class MessageBus:
    """Broker sinkron dan thread-safe untuk komunikasi antar-agen."""

    MAX_HOPS = 12

    def __init__(self, hmac_key: bytes, recorder: Optional[Callable] = None):
        """Siapkan broker.

        Args:
            hmac_key: kunci penandatanganan pesan antar-agen.
            recorder: callable opsional yang menyimpan setiap pesan (ke DB).
        """
        self._key = hmac_key
        self._handlers: Dict[str, Callable[[Message], Optional[Message]]] = {}
        self._replies: Dict[str, Optional[Message]] = {}
        self._lock = threading.RLock()
        self._recorder = recorder
        self.sent_count = 0
        self.sent_bytes = 0
        self.rejected: List[str] = []

    def register(self, name: str, handler: Callable[[Message], Optional[Message]]) -> None:
        """Daftarkan handler sebuah agen sebagai penerima pesan."""
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        """Lepaskan agen dari broker; dipakai skenario kegagalan agen (S5)."""
        self._handlers.pop(name, None)

    def _sign(self, message: Message) -> None:
        """Bubuhkan tanda tangan HMAC-SHA256 pada pesan."""
        digest = hmac.new(self._key, message.signable_bytes(), hashlib.sha256).hexdigest()
        message.signature = "hmac-sha256:" + digest

    def _verify(self, message: Message) -> None:
        """Pastikan tanda tangan cocok; lempar MessageError bila tidak."""
        if not message.signature:
            raise MessageError(f"{message.message_id}: pesan tanpa tanda tangan")
        expected = "hmac-sha256:" + hmac.new(
            self._key, message.signable_bytes(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, message.signature):
            raise MessageError(f"{message.message_id}: tanda tangan tidak valid")

    def _validate(self, message: Message) -> None:
        """Pastikan skema dikenal dan seluruh field wajib tersedia."""
        if message.schema not in SCHEMAS:
            raise MessageError(f"{message.message_id}: skema tak dikenal '{message.schema}'")
        missing = [f for f in SCHEMAS[message.schema] if f not in message.payload]
        if missing:
            raise MessageError(f"{message.message_id}: field wajib hilang {missing}")

    def send(self, message: Message) -> Optional[Message]:
        """Kirim pesan ke penerimanya dan kembalikan balasannya.

        Penerima "Human" sengaja tidak punya handler: gerbang HITL ditangani
        di luar bus, dan pesannya hanya dicatat untuk jejak audit.
        """
        if message.signature is None:
            self._sign(message)
        try:
            self._verify(message)
            self._validate(message)
        except MessageError as error:
            with self._lock:
                self.rejected.append(str(error))
            raise
        if message.hop > self.MAX_HOPS:
            raise MessageError(f"{message.message_id}: budget hop habis (dugaan siklus)")

        with self._lock:
            key = message.idempotency_key
            if key and key in self._replies:
                return self._replies[key]
            self.sent_count += 1
            self.sent_bytes += message.size_bytes()
            if self._recorder:
                self._recorder(message)

        handler = self._handlers.get(message.receiver)
        if handler is None:
            if message.receiver in ("Human", "Client"):
                return None
            raise MessageError(f"penerima tidak terdaftar: {message.receiver}")

        reply = handler(message)
        if reply is not None:
            reply.hop = message.hop + 1
            reply.reply_to = message.message_id
        with self._lock:
            if message.idempotency_key:
                self._replies[message.idempotency_key] = reply
        return reply

    def compose(self, performative: Performative, sender: str, receiver: str,
                schema: str, payload: dict, conversation_id: str, trace_id: str,
                risk: Risk = Risk.LOW,
                idempotency_key: Optional[str] = None) -> Message:
        """Bangun objek Message tanpa mengirimkannya."""
        return Message(performative=performative, sender=sender, receiver=receiver,
                       schema=schema, payload=payload,
                       conversation_id=conversation_id, trace_id=trace_id,
                       risk=risk, idempotency_key=idempotency_key)
