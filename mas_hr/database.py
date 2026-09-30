"""Penyimpanan SQLite — implementasi skema laporan Bagian 7.2.

SQLite dipilih secara sadar supaya prototipe berjalan tanpa infrastruktur.
DDL-nya dibuat sedekat mungkin dengan versi PostgreSQL di laporan sehingga
migrasi hanya soal tipe data.

`audit_logs` bersifat append-only dan ditegakkan oleh trigger, bukan oleh
konvensi aplikasi.
"""
from typing import Iterable, List, Optional
import json
import sqlite3
import threading

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS clients (
  client_id TEXT PRIMARY KEY, name TEXT, industry TEXT, preferences_json TEXT);

CREATE TABLE IF NOT EXISTS skills (
  skill_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL,
  synonyms_json TEXT, parent_skill_id TEXT);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, client_id TEXT, title TEXT NOT NULL,
  headcount INTEGER, min_experience_years REAL, location TEXT, sla_days INTEGER,
  required_documents_json TEXT, current_state TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS job_skills (
  job_id TEXT, skill_id TEXT, importance REAL, min_level TEXT,
  PRIMARY KEY (job_id, skill_id));

CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY, name_encrypted TEXT, location TEXT,
  experience_years REAL, source TEXT, dedup_hash TEXT,
  consent_status TEXT DEFAULT 'GRANTED', retention_until TEXT);

CREATE TABLE IF NOT EXISTS candidate_skills (
  candidate_id TEXT, skill_id TEXT, level TEXT, evidence_source TEXT,
  PRIMARY KEY (candidate_id, skill_id));

CREATE TABLE IF NOT EXISTS applications (
  application_id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT, job_id TEXT, current_state TEXT, sourced_from TEXT,
  created_at TEXT, UNIQUE (candidate_id, job_id));

CREATE TABLE IF NOT EXISTS screening_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, job_id TEXT,
  score REAL, confidence REAL, features_json TEXT, model_version TEXT,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS compliance_documents (
  doc_id TEXT PRIMARY KEY, candidate_id TEXT, kind TEXT, sha256 TEXT,
  expires_at TEXT, encrypted INTEGER DEFAULT 1);

CREATE TABLE IF NOT EXISTS compliance_checks (
  check_id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, doc_kind TEXT,
  rule_id TEXT, result TEXT, ocr_confidence REAL, finding TEXT,
  checked_by_agent TEXT, checked_at TEXT);

CREATE TABLE IF NOT EXISTS matching_scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, job_id TEXT,
  fit_score REAL, components_json TEXT, theta_used REAL, rank INTEGER,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS interviews (
  interview_id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, job_id TEXT,
  slot TEXT, mode TEXT, status TEXT);

CREATE TABLE IF NOT EXISTS interview_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, interview_id INTEGER, interviewer_id TEXT,
  competency_json TEXT, recommendation TEXT, transcript_ref TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS placements (
  placement_id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, job_id TEXT,
  contract_ref TEXT, start_date TEXT, location TEXT, status TEXT, signed_at TEXT);

CREATE TABLE IF NOT EXISTS human_approvals (
  approval_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, candidate_id TEXT,
  checkpoint_id TEXT NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('APPROVED','REJECTED','REQUEST_REVISION')),
  approver_id TEXT NOT NULL, approver_role TEXT, reason TEXT,
  decided_at TEXT, latency_seconds REAL);

CREATE TABLE IF NOT EXISTS agents (
  agent_id TEXT PRIMARY KEY, agent_type TEXT, mobility TEXT,
  autonomy_level TEXT, allowed_actions_json TEXT, version TEXT);

CREATE TABLE IF NOT EXISTS agent_messages (
  message_id TEXT PRIMARY KEY, conversation_id TEXT, trace_id TEXT,
  sender TEXT, receiver TEXT, performative TEXT, schema TEXT,
  payload_json TEXT, risk TEXT, size_bytes INTEGER, created_at TEXT);

CREATE TABLE IF NOT EXISTS audit_logs (
  log_id INTEGER PRIMARY KEY AUTOINCREMENT, entity_type TEXT, entity_id TEXT,
  action TEXT, actor_type TEXT, actor_id TEXT, before_json TEXT, after_json TEXT,
  previous_hash TEXT, current_hash TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TRIGGER IF NOT EXISTS audit_block_update BEFORE UPDATE ON audit_logs
BEGIN SELECT RAISE(ABORT, 'audit_logs bersifat append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_block_delete BEFORE DELETE ON audit_logs
BEGIN SELECT RAISE(ABORT, 'audit_logs bersifat append-only'); END;

CREATE INDEX IF NOT EXISTS ix_messages_conversation ON agent_messages(conversation_id);
CREATE INDEX IF NOT EXISTS ix_approvals_job ON human_approvals(job_id);
"""


class Database:
    """Pembungkus tipis SQLite yang aman dipakai dari banyak thread."""

    def __init__(self, path: str = ":memory:", timeout: float = 30.0):
        """Buka koneksi dan pasang skema bila belum ada.

        Args:
            path: berkas SQLite, atau ":memory:" untuk basis data sementara.
            timeout: lama menunggu bila berkas sedang dikunci proses lain.

        Untuk basis data berbasis berkas, mode WAL diaktifkan agar beberapa
        alur rekrutmen dapat berjalan BERSAMAAN di atas berkas yang sama:
        pembaca tidak memblokir penulis dan sebaliknya.
        """
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False,
                                           timeout=timeout)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if path != ":memory:":
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        """Jalankan perintah tulis dan langsung commit."""
        with self._lock:
            cursor = self._connection.execute(sql, tuple(params))
            self._connection.commit()
            return cursor

    def query(self, sql: str, params: Iterable = ()) -> List[sqlite3.Row]:
        """Jalankan SELECT dan kembalikan seluruh barisnya."""
        with self._lock:
            return self._connection.execute(sql, tuple(params)).fetchall()

    def close(self) -> None:
        """Tutup koneksi database."""
        with self._lock:
            self._connection.close()

    def insert_message(self, message) -> None:
        """Simpan satu pesan antar-agen untuk keperluan message trace."""
        record = message.as_record()
        self.execute(
            """INSERT OR IGNORE INTO agent_messages
               (message_id, conversation_id, trace_id, sender, receiver,
                performative, schema, payload_json, risk, size_bytes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (record["message_id"], record["conversation_id"], record["trace_id"],
             record["sender"], record["receiver"], record["performative"],
             record["schema"], record["payload_json"], record["risk"],
             record["size_bytes"], record["created_at"]))

    def insert_audit(self, **fields) -> None:
        """Sisipkan satu baris audit; hanya dipanggil oleh AuditTrail."""
        self.execute(
            """INSERT INTO audit_logs
               (entity_type, entity_id, action, actor_type, actor_id,
                before_json, after_json, previous_hash, current_hash)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (fields["entity_type"], fields["entity_id"], fields["action"],
             fields["actor_type"], fields["actor_id"], fields.get("before_json"),
             fields.get("after_json"), fields["previous_hash"],
             fields["current_hash"]))

    def all_audit_rows(self) -> List[sqlite3.Row]:
        """Ambil seluruh baris audit berurutan; dipakai verifikasi rantai hash."""
        return self.query("SELECT * FROM audit_logs ORDER BY log_id ASC")

    def seed_reference_data(self) -> None:
        """Isi tabel `skills` dan `agents` dari taksonomi dan settings.

        Menyimpan allowlist agen sebagai DATA, bukan hanya kode, supaya dapat
        diaudit dan ditampilkan di dashboard.
        """
        from .settings import AGENT_ACTIONS, AGENT_PROFILE
        from .skill_taxonomy import SKILLS
        for skill_id, (name, synonyms, parent) in SKILLS.items():
            self.execute(
                """INSERT OR REPLACE INTO skills
                   (skill_id, canonical_name, synonyms_json, parent_skill_id)
                   VALUES (?,?,?,?)""",
                (skill_id, name, json.dumps(synonyms, ensure_ascii=False), parent))
        for agent, actions in AGENT_ACTIONS.items():
            agent_type, mobility, level = AGENT_PROFILE.get(
                agent, ("AI", "STATIC", "A3"))
            self.execute(
                """INSERT OR REPLACE INTO agents
                   (agent_id, agent_type, mobility, autonomy_level,
                    allowed_actions_json, version) VALUES (?,?,?,?,?,?)""",
                (agent, agent_type, mobility, level,
                 json.dumps(sorted(actions)), "2.0"))
