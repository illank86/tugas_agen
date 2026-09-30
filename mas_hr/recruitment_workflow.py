"""Orkestrasi alur rekrutmen end-to-end (arm P dan B2).

Sifat aliran, yang sering ditanyakan: SEQUENTIAL pada dependensi antar-fase,
PARALEL di dalam fase (kanal sourcing, kandidat screening, dokumen
compliance), dan BARRIER pada fan-in Matching serta setiap gerbang HITL.

Streaming batch antara Sourcing dan Screening membuat kedua agen tumpang
tindih. Bila `batch_size` disetel >= jumlah kandidat, pipeline berubah menjadi
serial murni dan keunggulan paralelisme hilang tanpa terlihat dari luar.
"""
from typing import Any, Callable, Dict, List, Optional
import time
import uuid

from .agent_messaging import MessageBus, MessageError, Performative, Risk
from .agents import (AgentContext, ComplianceAgent, IntakeAgent, InterviewAgent,
                     MatchingAgent, PlacementAgent, ScreeningAgent, SourcingAgent,
                     SupervisorAgent)
from .audit_trail import AuditTrail
from .autonomy_policy import PolicyEngine, UnauthorizedAction
from .database import Database
from .settings import Settings
from .text_similarity import get_embedder

# Batas putaran "minta revisi" per permintaan persetujuan. Tanpa batas, alur
# bisa berputar selamanya; setelah batas tercapai revisi dianggap penolakan.
MAX_REVISIONS = 3


class RecruitmentSystem:
    """Rakitan lengkap: delapan agen, bus, kebijakan otonomi, dan audit."""

    def __init__(self, settings: Settings, jobs: Dict[str, Any],
                 candidates: Dict[str, Any], approver, scorer=None,
                 database: Optional[Database] = None,
                 prefer_neural_embedding: bool = False,
                 source_all_channels: bool = False):
        """Rakit sistem lengkap dan daftarkan seluruh agen ke bus.

        Args:
            settings: parameter sistem, termasuk saklar hitl_enabled.
            jobs: job_id -> JobRequirement.
            candidates: candidate_id -> Candidate.
            approver: objek dengan metode review(ApprovalRequest).
            scorer: model screening; None berarti RuleScorer.
            database: Database siap pakai; None berarti SQLite in-memory.
            prefer_neural_embedding: coba sentence-transformers bila tersedia.
            source_all_channels: True bila kandidat berasal dari berkas yang
                disediakan pengguna. Contract Net tetap berjalan, tetapi
                seluruh kanal dimenangkan sehingga tidak ada CV yang hilang
                hanya karena kanalnya kalah lelang.
        """
        settings.validate()
        self.settings = settings
        self.database = database or Database(":memory:")
        self.database.seed_reference_data()
        self.audit = AuditTrail(self.database)
        self.bus = MessageBus(settings.hmac_key, recorder=self.database.insert_message)
        self.policy = PolicyEngine(settings, self.audit)
        self.context = AgentContext(
            bus=self.bus, policy=self.policy, audit=self.audit,
            database=self.database, settings=settings,
            embedder=get_embedder(prefer_neural_embedding),
            candidates=candidates, jobs=jobs, approver=approver)

        self.supervisor = SupervisorAgent(self.context)
        self.intake = IntakeAgent(self.context)
        from .agents.sourcing_agent import SOURCING_CHANNELS
        self.sourcing = SourcingAgent(
            self.context, seed=settings.seed,
            max_winners=len(SOURCING_CHANNELS) if source_all_channels else 3)
        self.screening = ScreeningAgent(self.context, scorer=scorer)
        self.compliance = ComplianceAgent(self.context)
        self.matching = MatchingAgent(self.context)
        self.interview = InterviewAgent(self.context, seed=settings.seed)
        self.placement = PlacementAgent(self.context)
        self.task_failures = 0

    def _approve(self, checkpoint: str, decision_type: str, evidence: dict,
                 job_id: str, candidate_id: Optional[str] = None,
                 redo: Optional[Callable[[str], dict]] = None) -> dict:
        """Ajukan persetujuan; kerjakan ulang tahapnya bila manusia minta revisi.

        Minta revisi BUKAN penolakan. Versi sebelumnya memperlakukannya sama
        dengan tolak sehingga alur berhenti atau kandidat terbuang. Di sini
        agen mengerjakan ulang tahapnya (`redo`) lalu persetujuan yang sama
        diajukan kembali beserta catatan revisi dari penyetuju.

        Args:
            redo: fungsi yang menerima catatan revisi, mengerjakan ulang tahap,
                dan mengembalikan bukti baru. None berarti bukti diajukan
                ulang apa adanya bersama catatan tersebut.

        Returns:
            Keputusan akhir. Bila batas revisi tercapai, keputusan menjadi
            REJECTED dan alasannya menyebut batas tersebut.
        """
        revision = None
        for round_number in range(MAX_REVISIONS + 1):
            shown = dict(evidence, revisi=revision) if revision else evidence
            decision = self.supervisor.request_approval(
                checkpoint, decision_type, shown, job_id, candidate_id)
            if decision["decision"] != "REQUEST_REVISION":
                return decision
            if round_number == MAX_REVISIONS:
                break
            previous = evidence
            if redo is not None:
                evidence = redo(decision["reason"])
            revision = {"ke": round_number + 1, "maks": MAX_REVISIONS,
                        "catatan": decision["reason"],
                        "dikerjakan_ulang": redo is not None,
                        "berubah": evidence != previous}
        self.supervisor.record("job", job_id, f"{checkpoint}_REVISION_LIMIT",
                               {"limit": MAX_REVISIONS, "candidate_id": candidate_id})
        return {**decision, "decision": "REJECTED",
                "reason": f"batas {MAX_REVISIONS} kali revisi tercapai"}

    # -- fase-fase ---------------------------------------------------------
    def _run_intake(self, job, conversation: str, trace: str) -> dict:
        """Fase 1: ubah permintaan klien menjadi spesifikasi terstruktur."""
        reply = self.supervisor.send(
            "IntakeAgent", Performative.REQUEST, "JobRequestRaw@1.0",
            {"client_id": job.client_id, "job_id": job.job_id,
             "text": f"Butuh {job.headcount} {job.title}"},
            conversation, trace)
        return reply.payload

    def _run_sourcing(self, job, target_count: int,
                      conversation: str, trace: str) -> dict:
        """Fase 2: lelang kanal dan kumpulkan kandidat."""
        reply = self.supervisor.send(
            "SourcingAgent", Performative.REQUEST, "CandidateSearchRequest@1.0",
            {"job_id": job.job_id, "target_count": target_count},
            conversation, trace, idempotency_key=f"{job.job_id}:sourcing")
        return reply.payload

    def _run_screening_and_compliance(self, job, candidate_entries: List[dict],
                                      state_machine, conversation: str,
                                      trace: str) -> List[dict]:
        """Fase 3 dan 4 dengan streaming batch agar kedua agen tumpang tindih.

        Returns:
            Daftar status kepatuhan seluruh kandidat yang lolos screening.
        """
        batch_size = max(1, self.settings.batch_size)
        statuses: List[dict] = []
        entered_compliance = False
        for start in range(0, len(candidate_entries), batch_size):
            chunk = candidate_entries[start:start + batch_size]
            screening_reply = self.bus.send(self.bus.compose(
                Performative.INFORM, "SourcingAgent", "ScreeningAgent",
                "CandidateProfileBatch@1.0",
                {"batch_id": f"BATCH-{job.job_id}-{start}", "job_id": job.job_id,
                 "candidates": chunk},
                conversation, trace))
            if not entered_compliance:
                state_machine.to("COMPLIANCE_CHECK")
                entered_compliance = True
            compliance_reply = self.bus.send(self.bus.compose(
                Performative.INFORM, "ScreeningAgent", "ComplianceAgent",
                "ScreeningResult@1.0",
                {"job_id": job.job_id, "results": screening_reply.payload["results"],
                 "total_screened": screening_reply.payload["total_screened"]},
                conversation, trace, Risk.MEDIUM))
            statuses.extend(compliance_reply.payload["statuses"])
        if not entered_compliance:
            state_machine.to("COMPLIANCE_CHECK")
        return statuses

    def _review_uncertain_documents(self, job, statuses: List[dict],
                                    state_machine) -> None:
        """Gerbang HITL-2: mintakan tinjauan manusia untuk dokumen tak pasti.

        Mengubah status di tempat menjadi PASS atau FAIL sesuai keputusan.
        """
        pending = [s for s in statuses if s.get("needs_escalation")]
        if not pending:
            return
        state_machine.to("COMPLIANCE_REVIEW")
        board = self.context.board(job.job_id)
        details = board.get("compliance_detail", {})
        truths = board.get("document_problem_truth", {})
        for status in pending:
            candidate_id = status["candidate_id"]
            findings = [f["finding"]
                        for f in details.get(candidate_id, {}).get("findings", [])]
            decision = self.supervisor.request_approval(
                "HITL-2", "Tinjauan dokumen tidak pasti",
                {"candidate_id": candidate_id, "confidence": status["confidence"],
                 "findings": findings,
                 "document_problem_truth": truths.get(candidate_id, False)},
                job.job_id, candidate_id)
            # Revisi di sini berarti dokumen diminta ulang: dokumen BELUM sah,
            # sehingga tidak boleh diloloskan. Versi sebelumnya memakai
            # `!= "REJECTED"` dan tanpa sengaja meloloskannya.
            status["status"] = "PASS" if decision["decision"] == "APPROVED" else "FAIL"
            status["needs_escalation"] = False
            status["resolved_by_human"] = True
        state_machine.to("COMPLIANCE_CHECK")

    def _run_matching(self, job, statuses: List[dict],
                      conversation: str, trace: str) -> dict:
        """Fase 5: hitung skor akhir dan usulkan shortlist (titik BARRIER)."""
        reply = self.bus.send(self.bus.compose(
            Performative.INFORM, "ComplianceAgent", "MatchingAgent",
            "ComplianceStatus@1.0",
            {"job_id": job.job_id,
             "statuses": [{"candidate_id": s["candidate_id"], "status": s["status"],
                           "confidence": s["confidence"]} for s in statuses],
             "n_pass": sum(1 for s in statuses if s["status"] == "PASS"),
             "n_fail": sum(1 for s in statuses if s["status"] == "FAIL")},
            conversation, trace, Risk.HIGH))
        return reply.payload

    def _run_interviews(self, job, shortlist: List[dict],
                        conversation: str, trace: str) -> List[dict]:
        """Fase 6: jadwalkan dan jalankan wawancara untuk shortlist teratas."""
        quota = max(3, job.headcount * 3)
        reply = self.bus.send(self.bus.compose(
            Performative.INFORM, "MatchingAgent", "InterviewAgent",
            "InterviewRecommendation@1.0",
            {"job_id": job.job_id,
             "candidates": [{"candidate_id": r["candidate_id"],
                             "fit_score": r["fit_score"]} for r in shortlist[:quota]]},
            conversation, trace))
        return reply.payload["feedback"]

    def _run_placement(self, job, hired: List[str], approver_id: str,
                       state_machine, conversation: str, trace: str) -> List[dict]:
        """Fase 7: susun kontrak, lewati HITL-5 dan HITL-6, lalu tempatkan."""
        placed: List[dict] = []
        for candidate_id in hired:
            current: Dict[str, Any] = {}

            def draft_contract(note: Optional[str] = None,
                               candidate_id: str = candidate_id) -> dict:
                """Susun (ulang) draf kontrak dan kembalikan buktinya."""
                if state_machine.state == "CONTRACT_PENDING_APPROVAL":
                    state_machine.to("CONTRACT_DRAFTING")
                draft = self.bus.send(self.bus.compose(
                    Performative.REQUEST, "SupervisorAgent", "PlacementAgent",
                    "PlacementAuthorization@1.0",
                    {"job_id": job.job_id, "candidate_id": candidate_id,
                     "approver_id": approver_id},
                    conversation, trace, Risk.HIGH))
                if state_machine.state == "CONTRACT_DRAFTING":
                    state_machine.to("CONTRACT_PENDING_APPROVAL")
                current["draft"] = draft
                return {"candidate_id": candidate_id,
                        "clauses": draft.payload["clauses"]}

            contract_decision = self._approve(
                "HITL-5", "Persetujuan kontrak", draft_contract(),
                job.job_id, candidate_id, redo=draft_contract)
            if contract_decision["decision"] != "APPROVED":
                continue
            if state_machine.state == "CONTRACT_PENDING_APPROVAL":
                state_machine.to("PLACEMENT_AUTHORIZED")
            placement_decision = self._approve(
                "HITL-6", "Otorisasi penempatan",
                {"candidate_id": candidate_id,
                 "start_date": current["draft"].payload["start_date"],
                 "location": job.location}, job.job_id, candidate_id)
            if placement_decision["decision"] != "APPROVED":
                continue
            placed.append(self.placement.execute_placement(
                job, candidate_id, placement_decision["approver_id"]))
            if state_machine.state == "PLACEMENT_AUTHORIZED":
                state_machine.to("PLACED")
                state_machine.to("MONITORING")
        return placed

    # -- alur utama --------------------------------------------------------
    def run_job(self, job_id: str, target_count: int = 200,
                conversation_id: Optional[str] = None) -> dict:
        """Jalankan seluruh tujuh fase untuk satu lowongan.

        Args:
            job_id: lowongan yang diproses.
            target_count: jumlah kandidat yang diminta dari sourcing.
            conversation_id: pengenal percakapan untuk message trace. Wajib
                diisi bila beberapa alur untuk lowongan yang sama berjalan
                bersamaan, agar jejak pesannya tidak tercampur.

        Returns:
            Ringkasan hasil: outcome, shortlist, penempatan, riwayat state,
            dan status kepatuhan (dengan label emas dilampirkan dari
            blackboard, bukan dari pesan).
        """
        job = self.context.jobs[job_id]
        supervisor = self.supervisor
        state_machine = supervisor.state_machine(job_id)
        conversation = conversation_id or job_id
        trace = "trace-" + uuid.uuid4().hex[:8]
        started = time.time()
        result: Dict[str, Any] = {"job_id": job_id, "shortlist": [], "placed": [],
                                  "rejected_by_compliance": []}

        def finish(outcome: str, statuses: Optional[List[dict]] = None) -> dict:
            """Tutup permintaan, lampirkan label emas, dan kembalikan hasil."""
            state_machine.close_if_possible()
            truths = self.context.board(job_id).get("document_problem_truth", {})
            for status in (statuses or []):
                status["document_problem_truth"] = truths.get(
                    status["candidate_id"], False)
            result.update({"outcome": outcome, "elapsed": time.time() - started,
                           "states": state_machine.history,
                           "compliance_statuses": statuses or []})
            return result

        # Fase 1 + HITL-1
        def run_intake(note: Optional[str] = None) -> dict:
            """Susun (ulang) spesifikasi lowongan dan kembalikan buktinya."""
            state_machine.to("INTAKE_CLARIFYING")
            requirement = self._run_intake(job, conversation, trace)
            state_machine.to("REQ_PENDING_APPROVAL")
            return {"title": requirement["title"],
                    "headcount": requirement["headcount"],
                    "skills": [s["skill"] for s in requirement["required_skills"]],
                    "clarification_turns": requirement["clarification_turns"],
                    "unresolved": requirement["unresolved_ambiguities"],
                    "parsed_by": requirement.get("parsed_by", "format")}

        intake_decision = self._approve(
            "HITL-1", "Konfirmasi structured job requirement", run_intake(),
            job_id, redo=run_intake)
        if intake_decision["decision"] != "APPROVED":
            return finish(f"berhenti pada HITL-1 ({intake_decision['decision']})")

        # Fase 2
        state_machine.to("SOURCING")
        try:
            sourcing = self._run_sourcing(job, target_count, conversation, trace)
        except (MessageError, UnauthorizedAction) as error:
            self.task_failures += 1
            state_machine.to("ON_HOLD")
            return finish(f"sourcing gagal: {error}")
        result["sourced"] = len(sourcing["candidates"])
        result["channels"] = sourcing["channels_used"]

        # Fase 3 dan 4
        state_machine.to("SCREENING")
        statuses = self._run_screening_and_compliance(
            job, sourcing["candidates"], state_machine, conversation, trace)
        self._review_uncertain_documents(job, statuses, state_machine)

        screening_results = self.context.board(job_id).get("screening", {})
        for status in statuses:
            screening = screening_results.get(status["candidate_id"])
            if screening and supervisor.resolve_conflict(
                    status["candidate_id"], screening, status) == "REJECT":
                result["rejected_by_compliance"].append(status["candidate_id"])

        # Fase 5 + HITL-3
        def run_matching(note: Optional[str] = None) -> dict:
            """Hitung (ulang) shortlist dan kembalikan buktinya."""
            state_machine.to("MATCHING")
            matching = self._run_matching(job, statuses, conversation, trace)
            result["shortlist"] = matching["ranking"]
            result["n_scored"] = matching["n_scored"]
            state_machine.to("SHORTLIST_PENDING_APPROVAL")
            return {"n_shortlist": len(matching["ranking"]),
                    "theta": matching["theta"],
                    "top": [{"candidate_id": r["candidate_id"], "fit": r["fit_score"],
                             "contributions": r["contributions"],
                             "unmet": r["unmet_requirements"]}
                            for r in matching["ranking"][:5]]}

        shortlist_decision = self._approve(
            "HITL-3", "Finalisasi shortlist", run_matching(), job_id,
            redo=run_matching)
        shortlist = result["shortlist"]
        if shortlist_decision["decision"] != "APPROVED" or not shortlist:
            return finish(f"berhenti pada HITL-3 "
                          f"({shortlist_decision['decision']}, "
                          f"{len(shortlist)} kandidat)", statuses)

        # Fase 6 + HITL-4
        state_machine.to("INTERVIEW_SCHEDULING")
        feedback = self._run_interviews(job, shortlist, conversation, trace)
        state_machine.to("INTERVIEW_DONE")
        result["interviewed"] = len(feedback)
        state_machine.to("HIRE_PENDING_APPROVAL")

        # Keputusan hire adalah aksi A0: MANUSIA yang memutuskan. Rekomendasi
        # pewawancara adalah BUKTI, bukan penyaring. Versi sebelumnya hanya
        # membuka gerbang bagi kandidat ber-status RECOMMEND, sehingga kandidat
        # yang dinilai di bawah ambang tersingkir tanpa seorang pun ditanya —
        # itu keputusan A0 yang diambil mesin, bertentangan dengan bounded
        # autonomy, dan membuat alur berakhir tanpa HITL-4 sama sekali.
        ranked = sorted(feedback, key=lambda f: -f["competency"]["teknis"])
        considered = ranked[:max(job.headcount, min(len(ranked), job.headcount + 2))]
        fit_by_candidate = {r["candidate_id"]: r["fit_score"] for r in shortlist}
        hired, approver_id = [], "SYSTEM"

        def hire_evidence(entry: dict) -> dict:
            """Bukti keputusan hire untuk satu hasil wawancara."""
            return {"candidate_id": entry["candidate_id"],
                    "competency": entry["competency"],
                    "rekomendasi_pewawancara": entry["recommendation"],
                    "catatan": ("pewawancara TIDAK merekomendasikan; Anda tetap "
                                "berwenang memutuskan"
                                if entry["recommendation"] != "RECOMMEND"
                                else "pewawancara merekomendasikan"),
                    "posisi_terisi": f"{len(hired)}/{job.headcount}"}

        def reinterview(note: str, entry: dict) -> dict:
            """Revisi pada keputusan hire = wawancara ulang kandidat tersebut."""
            state_machine.to("INTERVIEW_SCHEDULING")
            fresh = self._run_interviews(
                job, [{"candidate_id": entry["candidate_id"],
                       "fit_score": fit_by_candidate.get(entry["candidate_id"], 0.0)}],
                conversation, trace)
            state_machine.to("INTERVIEW_DONE")
            state_machine.to("HIRE_PENDING_APPROVAL")
            # Bila kandidat tidak mendapat slot, hasil wawancara lama dipakai.
            return hire_evidence(fresh[0] if fresh else entry)

        for entry in considered:
            if len(hired) >= job.headcount:
                break
            hire_decision = self._approve(
                "HITL-4", "Keputusan hire final", hire_evidence(entry),
                job_id, entry["candidate_id"],
                redo=lambda note, entry=entry: reinterview(note, entry))
            if hire_decision["decision"] == "APPROVED":
                hired.append(entry["candidate_id"])
                approver_id = hire_decision["approver_id"]
        if not hired:
            reason = ("tidak ada kandidat yang lolos wawancara"
                      if not feedback
                      else "tidak ada kandidat yang disetujui manusia pada HITL-4")
            return finish(reason, statuses)

        # Fase 7 + HITL-5 dan HITL-6
        state_machine.to("CONTRACT_DRAFTING")
        result["placed"] = self._run_placement(job, hired, approver_id,
                                               state_machine, conversation, trace)
        return finish(f"{len(result['placed'])} kandidat ditempatkan", statuses)
