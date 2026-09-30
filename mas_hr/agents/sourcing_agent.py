"""Sourcing Agent — AI, static. Fase 2: Candidate Sourcing.

Menjalankan Contract Net Protocol (laporan 5.7.2): kanal-kanal mengajukan
penawaran dan dimenangkan berdasarkan fungsi utilitas

    U(c) = a*yield + b*quality - g*cost - d*latency

`quality` diperbarui tiap siklus, sehingga modul ini juga menjadi titik masuk
alami untuk contextual bandit yang dinyatakan sebagai future work.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import hashlib
import random
import time

from ..agent_messaging import Message, Performative, Risk
from .agent_base import Agent

# Karakteristik kanal tersimulasi: biaya relatif, latensi, kualitas awal,
# dan porsi kandidat yang biasanya berasal dari kanal tersebut.
SOURCING_CHANNELS: Dict[str, dict] = {
    "portal-a":    {"cost": 0.60, "latency": 0.040, "quality": 0.72, "share": 0.45},
    "portal-b":    {"cost": 0.30, "latency": 0.025, "quality": 0.65, "share": 0.25},
    "db-internal": {"cost": 0.05, "latency": 0.002, "quality": 0.88, "share": 0.20},
    "referral":    {"cost": 0.10, "latency": 0.005, "quality": 0.90, "share": 0.10},
}

UTILITY_COEFFICIENTS = {"yield": 1.0, "quality": 30.0, "cost": 8.0, "latency": 20.0}


class SourcingAgent(Agent):
    """Melelang tugas pencarian ke beberapa kanal lalu menggabungkan hasilnya."""

    name = "SourcingAgent"

    def __init__(self, context, seed: int = 42, max_winners: int = 3):
        """Siapkan agen.

        Args:
            seed: penentu keacakan estimasi yield tiap kanal.
            max_winners: jumlah kanal yang dimenangkan per permintaan.
        """
        super().__init__(context)
        self.rng = random.Random(seed)
        self.max_winners = max_winners
        self.channel_quality = {name: spec["quality"]
                                for name, spec in SOURCING_CHANNELS.items()}
        self.awards: List[dict] = []

    def collect_bids(self, target_count: int) -> List[dict]:
        """Kumpulkan penawaran dari seluruh kanal untuk sebuah permintaan.

        Kanal yang tidak punya stok untuk peran ini menolak (REFUSE) dengan
        cara tidak mengajukan penawaran sama sekali.
        """
        bids = []
        for channel, spec in SOURCING_CHANNELS.items():
            # Minimal satu: pada permintaan kecil, pembulatan ke bawah akan
            # membuat semua kanal menolak dan sourcing pulang kosong.
            expected = max(1, int(target_count * spec["share"]
                                  * self.rng.uniform(0.7, 1.2)))
            bids.append({"channel": channel, "expected_yield": expected,
                         "cost": spec["cost"], "latency": spec["latency"],
                         "quality": self.channel_quality[channel]})
        return bids

    @staticmethod
    def utility(bid: dict) -> float:
        """Hitung nilai utilitas sebuah penawaran kanal."""
        coefficient = UTILITY_COEFFICIENTS
        return (coefficient["yield"] * bid["expected_yield"]
                + coefficient["quality"] * bid["quality"]
                - coefficient["cost"] * bid["cost"]
                - coefficient["latency"] * bid["latency"])

    def update_channel_quality(self, observed: Dict[str, float],
                               learning_rate: float = 0.2) -> None:
        """Perbarui kualitas kanal dari umpan balik hasil shortlist.

        Ini langkah pertama menuju contextual bandit: kanal yang terbukti
        menghasilkan kandidat lolos akan lebih sering dimenangkan.
        """
        for channel, value in observed.items():
            if channel in self.channel_quality:
                self.channel_quality[channel] = (
                    (1 - learning_rate) * self.channel_quality[channel]
                    + learning_rate * value)

    def handle(self, message: Message) -> Optional[Message]:
        """Lelang tugas pencarian, ambil kandidat secara paralel, deduplikasi.

        Args:
            message: pesan berskema CandidateSearchRequest@1.0.

        Returns:
            Balasan berskema CandidateProfileBatch@1.0 berisi kandidat unik.
        """
        if message.schema != "CandidateSearchRequest@1.0":
            raise NotImplementedError(message.schema)

        self.request_permission("search_candidates", confidence=1.0)
        job_id = message.payload["job_id"]
        bids = self.collect_bids(message.payload["target_count"])
        for bid in bids:
            self.send("SupervisorAgent", Performative.PROPOSE, "SourcingBid@1.0",
                      bid, message.conversation_id, message.trace_id)

        winners = sorted(bids, key=self.utility, reverse=True)[:self.max_winners]
        winning_channels = {bid["channel"] for bid in winners}
        self.awards.append({"job_id": job_id, "winners": sorted(winning_channels)})
        self.record("job", job_id, "SOURCING_AWARD",
                    {"bids": len(bids), "winners": sorted(winning_channels)})

        pool = [candidate for candidate in self.context.candidates.values()
                if candidate.job_id_hint in ("", job_id)]

        def fetch_from(channel: str) -> List:
            """Ambil kandidat dari satu kanal, termasuk latensi tersimulasinya."""
            time.sleep(SOURCING_CHANNELS[channel]["latency"])
            return [candidate for candidate in pool if candidate.source == channel]

        fetched: List = []
        with ThreadPoolExecutor(max_workers=max(1, len(winning_channels))) as pool_exec:
            for batch in pool_exec.map(fetch_from, sorted(winning_channels)):
                fetched.extend(batch)

        self.request_permission("deduplicate", confidence=1.0)
        seen, unique = set(), []
        for candidate in fetched:
            fingerprint = hashlib.sha256(
                f"{candidate.name}|{candidate.location}|{candidate.cv_text[:80]}"
                .encode()).hexdigest()[:16]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidate.dedup_hash = fingerprint
            unique.append(candidate)

        payload = {
            "batch_id": f"BATCH-{job_id}", "job_id": job_id,
            "candidates": [{"candidate_id": c.candidate_id, "source": c.source}
                           for c in unique],
            "channels_used": sorted(winning_channels),
            "duplicates_removed": len(fetched) - len(unique),
        }
        self.record("job", job_id, "SOURCING_DONE",
                    {"found": len(fetched), "unique": len(unique)})
        return self.reply(message, Performative.INFORM,
                          "CandidateProfileBatch@1.0", payload, Risk.LOW)
