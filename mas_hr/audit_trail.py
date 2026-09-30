"""Jejak audit append-only dengan rantai hash (laporan 8.5, Lapisan 4).

    h_i = SHA256(h_{i-1} || payload_i)

Mengubah satu baris lama akan memutus rantai dan terdeteksi verify(). Inilah
alasan ledger terdistribusi tidak diperlukan untuk audit internal.
"""
from typing import Optional, Tuple
import hashlib
import json
import threading

GENESIS_HASH = "0" * 64


class AuditTrail:
    """Pencatat keputusan yang tidak dapat diubah secara diam-diam."""

    def __init__(self, database=None):
        """Siapkan rantai.

        Args:
            database: objek Database untuk persistensi; None berarti rantai
                hanya dihitung di memori (dipakai pada unit test).
        """
        self._database = database
        self._previous = GENESIS_HASH
        self._lock = threading.RLock()
        self.count = 0

    @staticmethod
    def _chain_hash(previous: str, payload: str) -> str:
        """Hitung hash berantai dari hash sebelumnya dan payload saat ini."""
        return hashlib.sha256((previous + "||" + payload).encode("utf-8")).hexdigest()

    def record(self, entity_type: str, entity_id: str, action: str,
               actor_type: str, actor_id: str,
               before: Optional[dict] = None, after: Optional[dict] = None) -> str:
        """Catat satu kejadian dan kembalikan hash barunya.

        Args:
            entity_type: jenis entitas, mis. "job" atau "candidate".
            entity_id: identitas entitas yang terpengaruh.
            action: nama kejadian, mis. "COMPLIANCE_FAIL".
            actor_type: "AGENT", "HUMAN", atau "SYSTEM".
            actor_id: identitas pelaku.
            before: keadaan sebelum perubahan, bila relevan.
            after: keadaan sesudah perubahan, bila relevan.
        """
        with self._lock:
            payload = json.dumps({
                "entity_type": entity_type, "entity_id": str(entity_id),
                "action": action, "actor_type": actor_type, "actor_id": actor_id,
                "before": before, "after": after,
            }, sort_keys=True, ensure_ascii=False)
            current = self._chain_hash(self._previous, payload)
            if self._database is not None:
                self._database.insert_audit(
                    entity_type=entity_type, entity_id=str(entity_id), action=action,
                    actor_type=actor_type, actor_id=actor_id,
                    before_json=json.dumps(before, ensure_ascii=False) if before else None,
                    after_json=json.dumps(after, ensure_ascii=False) if after else None,
                    previous_hash=self._previous, current_hash=current)
            self._previous = current
            self.count += 1
            return current

    def verify(self) -> Tuple[bool, Optional[int]]:
        """Periksa keutuhan seluruh rantai.

        Returns:
            (True, None) bila utuh, atau (False, log_id) pada baris pertama
            yang tidak konsisten.
        """
        if self._database is None:
            return True, None
        previous = GENESIS_HASH
        for row in self._database.all_audit_rows():
            payload = json.dumps({
                "entity_type": row["entity_type"], "entity_id": row["entity_id"],
                "action": row["action"], "actor_type": row["actor_type"],
                "actor_id": row["actor_id"],
                "before": json.loads(row["before_json"]) if row["before_json"] else None,
                "after": json.loads(row["after_json"]) if row["after_json"] else None,
            }, sort_keys=True, ensure_ascii=False)
            if row["previous_hash"] != previous or \
                    self._chain_hash(previous, payload) != row["current_hash"]:
                return False, row["log_id"]
            previous = row["current_hash"]
        return True, None
