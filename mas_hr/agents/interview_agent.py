"""Interview Agent — AI, static. Fase 6: Interview & Decision.

Paralel antar-kandidat tetapi SERIAL per panel: satu pewawancara tidak dapat
memimpin dua wawancara pada slot yang sama. Karena itu penjadwalan dijalankan
sebagai negosiasi iteratif PROPOSE -> COUNTER -> ACCEPT dengan batas putaran,
dan kegagalan menjadwalkan dieskalasi ke manusia.
"""
from typing import List, Optional, Set, Tuple
import random

from ..agent_messaging import Message, Performative, Risk
from .agent_base import Agent

# Slot wawancara tersedia (tersimulasi): 8 hari kerja x 6 jam.
AVAILABLE_SLOTS = [f"{day:02d}-09-2026 {hour:02d}:00"
                   for day in range(16, 24) for hour in (9, 10, 11, 13, 14, 15)]


class InterviewAgent(Agent):
    """Menjadwalkan wawancara dan mengumpulkan umpan balik terstruktur."""

    name = "InterviewAgent"

    def __init__(self, context, seed: int = 42, max_rounds: int = 3):
        """Siapkan agen.

        Args:
            seed: penentu keacakan penerimaan slot dan derau penilaian.
            max_rounds: batas putaran negosiasi sebelum dieskalasi ke manusia.
        """
        super().__init__(context)
        self.rng = random.Random(seed)
        self.max_rounds = max_rounds
        self.booked_slots: Set[str] = set()
        self.negotiation_rounds: List[int] = []
        self.scheduling_escalations: List[str] = []

    def negotiate_slot(self, candidate_id: str) -> Tuple[Optional[str], int]:
        """Negosiasikan satu slot wawancara dengan kandidat.

        Returns:
            Pasangan (slot terpilih atau None, jumlah putaran yang dipakai).
            None berarti negosiasi gagal dan harus ditangani manusia.
        """
        available = [slot for slot in AVAILABLE_SLOTS if slot not in self.booked_slots]
        self.rng.shuffle(available)
        for round_number in range(1, self.max_rounds + 1):
            if not available:
                break
            proposal = available.pop()
            if self.rng.random() < 0.75:            # kandidat menerima
                self.booked_slots.add(proposal)
                return proposal, round_number
        return None, self.max_rounds

    def conduct_interview(self, candidate_id: str, slot: str, rounds: int) -> dict:
        """Hasilkan umpan balik wawancara terstruktur untuk satu kandidat.

        Penilaian disimulasikan dari `interview_quality` ditambah derau
        penilai. Manusia adalah sumber kebenaran pada tahap ini, sehingga
        hasilnya boleh berbeda dari skor mesin (konflik K5).
        """
        candidate = self.context.candidates[candidate_id]
        noise = self.rng.gauss(0, 0.12)
        competence = max(0.0, min(1.0, candidate.interview_quality + noise))

        def jitter() -> float:
            """Variasikan nilai per kompetensi agar tidak seragam."""
            return round(max(0.0, min(1.0, competence + self.rng.gauss(0, 0.1))), 3)

        return {"candidate_id": candidate_id, "slot": slot, "rounds": rounds,
                "competency": {"teknis": round(competence, 3),
                               "komunikasi": jitter(), "sikap": jitter()},
                "recommendation": "RECOMMEND" if competence >= 0.55 else "NOT_RECOMMEND",
                "interviewer_id": "USR-HRM-01",
                "transcript_ref": f"asr://transcripts/{candidate_id}.json"}

    def handle(self, message: Message) -> Optional[Message]:
        """Jadwalkan dan jalankan wawancara untuk kandidat yang direkomendasikan."""
        if message.schema != "InterviewRecommendation@1.0":
            raise NotImplementedError(message.schema)

        job = self.context.jobs[message.payload["job_id"]]
        feedback: List[dict] = []
        for entry in message.payload["candidates"]:
            candidate_id = entry["candidate_id"]
            self.request_permission("schedule_interview", confidence=1.0)
            slot, rounds = self.negotiate_slot(candidate_id)
            self.negotiation_rounds.append(rounds)
            if slot is None:
                self.scheduling_escalations.append(candidate_id)
                self.record("candidate", candidate_id, "SCHEDULING_ESCALATED",
                            {"rounds": rounds})
                continue
            self.request_permission("notify_candidate", confidence=0.99)
            self.context.database.execute(
                """INSERT INTO interviews (candidate_id, job_id, slot, mode, status)
                   VALUES (?,?,?,?,?)""",
                (candidate_id, job.job_id, slot, "onsite", "SCHEDULED"))
            result = self.conduct_interview(candidate_id, slot, rounds)
            feedback.append(result)
            self.record("candidate", candidate_id, "INTERVIEW_DONE",
                        {"slot": slot, "recommendation": result["recommendation"]})

        # Catat siapa saja yang benar-benar diwawancara, agar laporan per
        # kandidat dapat menunjukkan tahap terakhir yang dicapai.
        board = self.context.board(job.job_id)
        board.setdefault("interviewed", []).extend(
            entry["candidate_id"] for entry in feedback)
        board.setdefault("interview_feedback", {}).update(
            {entry["candidate_id"]: entry for entry in feedback})

        payload = {"job_id": job.job_id, "feedback": feedback,
                   "average_rounds": round(
                       sum(self.negotiation_rounds) / max(1, len(self.negotiation_rounds)), 2),
                   "scheduling_escalations": self.scheduling_escalations}
        return self.reply(message, Performative.INFORM, "InterviewFeedback@1.0",
                          payload, Risk.MEDIUM)
