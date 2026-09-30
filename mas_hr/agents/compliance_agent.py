"""Compliance Agent — hibrida (OCR + rule engine), static. Fase 4: Compliance.

Keputusan desain terpenting di seluruh sistem ada di sini: PEMISAHAN PERSEPSI
DARI PENGHAKIMAN.

  - Persepsi (OCR, probabilistik)    -> boleh tidak pasti
  - Penghakiman (aturan, determinis) -> tidak boleh menebak

Akibatnya, ketidakpastian pembacaan SELALU menjadi eskalasi ke manusia, bukan
menjadi tebakan. Agen ini juga memegang hak veto: compliance FAIL mengalahkan
skor kompetensi setinggi apa pun (konflik K1).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Dict, List, Optional
import json

from ..agent_messaging import Message, Performative, Risk
from .agent_base import Agent

# Rule engine deklaratif: setiap aturan dapat dikutip pada penjelasan
# keputusan ("gagal karena R-014"), sesuatu yang tidak dapat dilakukan model
# probabilistik.
COMPLIANCE_RULES = [
    {"rule_id": "R-001", "document": "*", "level": "WAJIB",
     "description": "Dokumen wajib harus diserahkan"},
    {"rule_id": "R-002", "document": "*", "level": "WAJIB",
     "description": "Dokumen tidak boleh melewati masa berlaku"},
    {"rule_id": "R-003", "document": "*", "level": "WAJIB",
     "description": "Nama pada dokumen harus konsisten dengan data kandidat"},
    {"rule_id": "R-014", "document": "SKCK", "level": "WAJIB",
     "description": "SKCK diterbitkan maksimal 180 hari sebelum pengajuan "
                    "[ASUMSI: verifikasi ke kebijakan sebenarnya]"},
]


class ComplianceAgent(Agent):
    """Memverifikasi dokumen kandidat dengan aturan yang dapat dikutip."""

    name = "ComplianceAgent"

    def __init__(self, context):
        """Siapkan agen dan daftar eskalasi kosong."""
        super().__init__(context)
        self.today: date = context.settings.today
        self.escalations: List[dict] = []

    def read_document(self, document) -> dict:
        """Tahap PERSEPSI: baca field dokumen beserta keyakinannya.

        Pada prototipe, OCR disimulasikan dari metadata dokumen. Di produksi
        diganti model pra-latih dengan antarmuka yang sama.
        """
        self.request_permission("ocr_extract", confidence=document.ocr_confidence)
        return {"kind": document.kind, "issue_date": document.issue_date,
                "expires_at": document.expires_at, "present": document.present,
                "name_consistent": document.name_consistent,
                "ocr_confidence": document.ocr_confidence}

    def apply_rules(self, document, fields: dict) -> List[dict]:
        """Tahap PENGHAKIMAN: evaluasi aturan kepatuhan atas satu dokumen.

        Returns:
            Daftar temuan, masing-masing menyebut rule_id, hasil, dan bukti.
            Hasil UNCERTAIN berarti dokumen tidak terbaca — bukan berarti
            lolos, dan bukan berarti gagal.
        """
        settings = self.context.settings
        if not fields["present"]:
            return [{"rule_id": "R-001", "document": document.kind, "result": "FAIL",
                     "finding": "dokumen tidak diserahkan",
                     "ocr_confidence": fields["ocr_confidence"]}]

        if fields["ocr_confidence"] < settings.ocr_unreadable_threshold:
            return [{"rule_id": "R-002", "document": document.kind,
                     "result": "UNCERTAIN",
                     "finding": f"dokumen tidak terbaca "
                                f"(confidence {fields['ocr_confidence']:.2f} < "
                                f"{settings.ocr_unreadable_threshold:.2f})",
                     "ocr_confidence": fields["ocr_confidence"]}]

        findings: List[dict] = []
        if fields["expires_at"] and fields["expires_at"] < self.today:
            findings.append({"rule_id": "R-002", "document": document.kind,
                             "result": "FAIL",
                             "finding": f"kedaluwarsa pada {fields['expires_at']}",
                             "ocr_confidence": fields["ocr_confidence"]})
        if not fields["name_consistent"]:
            findings.append({"rule_id": "R-003", "document": document.kind,
                             "result": "FAIL",
                             "finding": "nama tidak konsisten antar-dokumen",
                             "ocr_confidence": fields["ocr_confidence"]})
        if document.kind == "SKCK":
            age_days = (self.today - fields["issue_date"]).days
            if age_days > settings.skck_max_age_days:
                findings.append({"rule_id": "R-014", "document": "SKCK",
                                 "result": "FAIL",
                                 "finding": f"usia SKCK {age_days} hari melebihi batas",
                                 "ocr_confidence": fields["ocr_confidence"]})
        if not findings:
            findings.append({"rule_id": "R-001", "document": document.kind,
                             "result": "PASS",
                             "finding": "lolos seluruh aturan wajib",
                             "ocr_confidence": fields["ocr_confidence"]})
        return findings

    def verify_candidate(self, candidate_id: str, required_documents: List[str]) -> dict:
        """Periksa seluruh dokumen wajib milik satu kandidat.

        Keyakinan verdict diambil dari keyakinan persepsi TERENDAH: sebuah
        kesimpulan tidak boleh lebih yakin daripada dokumen terlemahnya.
        Eskalasi ditentukan PolicyEngine (syarat conf >= gamma), bukan oleh
        ambang terpisah di dalam agen ini.
        """
        candidate = self.context.candidates[candidate_id]
        findings: List[dict] = []
        for document in candidate.documents:
            if document.kind not in required_documents:
                continue
            findings.extend(self.apply_rules(document, self.read_document(document)))

        has_failure = any(f["result"] == "FAIL" for f in findings)
        has_uncertainty = any(f["result"] == "UNCERTAIN" for f in findings)
        confidences = [f["ocr_confidence"] for f in findings if f.get("ocr_confidence")]
        overall_confidence = min(confidences) if confidences else 0.0
        status = "FAIL" if has_failure else ("UNCERTAIN" if has_uncertainty else "PASS")

        decision = self.request_permission("compliance_verdict",
                                           confidence=overall_confidence)
        for finding in findings:
            self.context.database.execute(
                """INSERT INTO compliance_checks
                   (candidate_id, doc_kind, rule_id, result, ocr_confidence,
                    finding, checked_by_agent, checked_at)
                   VALUES (?,?,?,?,?,?,?,datetime('now'))""",
                (candidate_id, finding["document"], finding["rule_id"],
                 finding["result"], finding.get("ocr_confidence"),
                 finding["finding"], self.name))
        self.record("candidate", candidate_id, f"COMPLIANCE_{status}",
                    {"rules": [f["rule_id"] for f in findings],
                     "confidence": overall_confidence})
        return {"candidate_id": candidate_id, "status": status,
                "confidence": round(overall_confidence, 3), "findings": findings,
                "needs_escalation": (not decision.autonomous and status != "FAIL"),
                "autonomy_level": decision.level,
                "checkpoint": decision.checkpoint if not decision.autonomous else None}

    def handle(self, message: Message) -> Optional[Message]:
        """Verifikasi seluruh kandidat yang lolos screening, secara paralel.

        Detail temuan dan label emas disimpan di blackboard, bukan dikirim
        lewat bus: payload tetap kecil, dan label emas bukan informasi yang
        boleh mengalir antar-agen.
        """
        if message.schema != "ScreeningResult@1.0":
            raise NotImplementedError(message.schema)

        job = self.context.jobs[message.payload["job_id"]]
        candidate_ids = [r["candidate_id"] for r in message.payload["results"]]
        required = job.required_documents

        with ThreadPoolExecutor(
                max_workers=self.context.settings.max_workers) as executor:
            statuses = list(executor.map(
                lambda candidate_id: self.verify_candidate(candidate_id, required),
                candidate_ids))

        board = self.context.board(job.job_id)
        board.setdefault("compliance_detail", {}).update(
            {s["candidate_id"]: s for s in statuses})
        board.setdefault("document_problem_truth", {}).update(
            {candidate_id: self.context.candidates[candidate_id].document_problem_truth
             for candidate_id in candidate_ids})

        for status in statuses:
            if status["needs_escalation"]:
                self.escalations.append(status)
                self.send("SupervisorAgent", Performative.REQUEST,
                          "ComplianceEscalation@1.0",
                          {"candidate_id": status["candidate_id"],
                           "reason": "keyakinan verdict di bawah ambang otonomi",
                           "confidence": status["confidence"]},
                          message.conversation_id, message.trace_id, Risk.HIGH)

        payload = {"job_id": job.job_id,
                   "statuses": [{"candidate_id": s["candidate_id"],
                                 "status": s["status"], "confidence": s["confidence"],
                                 "needs_escalation": s["needs_escalation"]}
                                for s in statuses],
                   "n_pass": sum(1 for s in statuses if s["status"] == "PASS"),
                   "n_fail": sum(1 for s in statuses if s["status"] == "FAIL")}
        return self.reply(message, Performative.INFORM, "ComplianceStatus@1.0",
                          payload, Risk.HIGH)
