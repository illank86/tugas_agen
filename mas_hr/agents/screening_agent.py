"""Screening Agent — AI, static, stateless per CV. Fase 3: Screening.

Stateless adalah alasan agen ini dapat direplikasi horizontal: pemrosesan
antar-kandidat adalah data parallelism murni.

Pertahanan prompt injection ada di sini karena inilah titik masuk teks tak
terpercaya. Teks CV diperlakukan sebagai DATA, tidak pernah sebagai instruksi.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import re

from ..agent_messaging import Message, Performative, Risk
from ..screening_model import RuleScorer, build_features
from ..skill_taxonomy import normalize
from .agent_base import Agent

LEVEL_PATTERN = re.compile(r"\((basic|intermediate|advanced)\)", re.IGNORECASE)
EXPERIENCE_PATTERN = re.compile(r"pengalaman\s+kerja\s*:\s*([\d.,]+)", re.IGNORECASE)
LEVEL_ORDER = {"basic": 1, "intermediate": 2, "advanced": 3}

# Pola instruksi yang tidak punya alasan sah muncul di dalam sebuah CV.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"abaikan\s+(semua\s+)?instruksi",
    r"^\s*system\s*:",
    r"#{2,}\s*new\s+instruction",
    r"you\s+are\s+now\s+in\s+admin\s+mode",
    r"(set|beri)\s+(status|skor|score)\s+(pass|100)",
    r"catatan\s+untuk\s+ai",
    r"(approve|loloskan)\s+(contract|kandidat|immediately|langsung)",
    r"lewati\s+tahap",
]
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                       for p in INJECTION_PATTERNS]


def sanitize_cv(cv_text: str) -> Tuple[str, List[str]]:
    """Netralkan baris yang menyerupai instruksi di dalam CV.

    Baris yang cocok tidak dihapus diam-diam melainkan diganti penanda, dan
    polanya dikembalikan agar jejak upaya serangan tetap tercatat di audit.

    Returns:
        Pasangan (teks bersih, daftar pola yang terdeteksi).
    """
    findings, cleaned_lines = [], []
    for line in cv_text.splitlines():
        matched = next((p.pattern for p in _COMPILED_INJECTION if p.search(line)), None)
        if matched:
            findings.append(matched)
            cleaned_lines.append("[REDACTED: pola instruksi pada dokumen kandidat]")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines), findings


def contains_injection(cv_text: str) -> bool:
    """Periksa apakah sebuah CV memuat pola instruksi."""
    return any(pattern.search(cv_text) for pattern in _COMPILED_INJECTION)


def parse_cv(cv_text: str) -> Tuple[Dict[str, str], float]:
    """Ekstraksi skill kanonik dan masa kerja dari teks CV.

    Sengaja tidak sempurna: CV nyata memang tidak menyebut semua kompetensi,
    dan model screening harus belajar menghadapi itu.

    Returns:
        Pasangan (skill_id -> level, tahun pengalaman).
    """
    skills: Dict[str, str] = {}
    experience = 0.0
    for line in cv_text.splitlines():
        found = EXPERIENCE_PATTERN.search(line)
        if found:
            try:
                experience = float(found.group(1).replace(",", "."))
            except ValueError:
                pass
        if not line.strip().startswith("-"):
            continue
        entry = line.strip("- ").strip()
        level_match = LEVEL_PATTERN.search(entry)
        level = level_match.group(1).lower() if level_match else "basic"
        skill_id = normalize(LEVEL_PATTERN.sub("", entry).strip())
        if not skill_id:
            continue
        previous = skills.get(skill_id)
        if previous is None or LEVEL_ORDER[level] > LEVEL_ORDER[previous]:
            skills[skill_id] = level
    return skills, experience


class ScreeningAgent(Agent):
    """Memberi skor kelayakan awal untuk setiap kandidat dalam satu batch."""

    name = "ScreeningAgent"

    def __init__(self, context, scorer=None):
        """Siapkan agen dengan model skoring yang dapat ditukar."""
        super().__init__(context)
        self.scorer = scorer or RuleScorer()
        self.injection_attempts: List[str] = []
        self.injection_successes: List[str] = []

    def screen_one(self, candidate_id: str, job) -> dict:
        """Nilai satu kandidat terhadap satu lowongan.

        Skor dihitung HANYA dari fitur terstruktur. Sebagai bukti, skor dari
        CV mentah (yang masih memuat instruksi) dibandingkan dengan skor dari
        CV yang sudah dinetralkan; bila berbeda, injeksi dicatat berhasil.
        """
        candidate = self.context.candidates[candidate_id]
        cleaned_text, findings = sanitize_cv(candidate.cv_text)
        if findings:
            self.injection_attempts.append(candidate_id)
            self.record("candidate", candidate_id, "INJECTION_ATTEMPT_BLOCKED",
                        {"patterns": findings})

        self.request_permission("parse_cv", confidence=1.0)
        skills, parsed_experience = parse_cv(cleaned_text)
        experience = parsed_experience or candidate.experience_years
        features = build_features(skills, experience, job, cleaned_text)
        score, confidence = self.scorer.predict(features)

        if findings:
            raw_skills, raw_experience = parse_cv(candidate.cv_text)
            raw_features = build_features(raw_skills, raw_experience or experience,
                                          job, candidate.cv_text)
            raw_score, _ = self.scorer.predict(raw_features)
            if raw_score > score + 1e-6:
                self.injection_successes.append(candidate_id)
                self.record("candidate", candidate_id, "INJECTION_INFLUENCED_SCORE",
                            {"raw": raw_score, "sanitized": score})

        decision = self.request_permission("screen_candidate", confidence=confidence)
        self.context.database.execute(
            """INSERT INTO screening_results
               (candidate_id, job_id, score, confidence, model_version, created_at)
               VALUES (?,?,?,?,?,datetime('now'))""",
            (candidate_id, job.job_id, score, confidence, self.scorer.version))
        return {"candidate_id": candidate_id, "score": round(score, 4),
                "confidence": round(confidence, 4), "features": features,
                "extracted_skills": skills, "experience_years": experience,
                "injection_flag": bool(findings),
                "autonomy_level": decision.level,
                "model_version": self.scorer.version}

    def handle(self, message: Message) -> Optional[Message]:
        """Nilai seluruh kandidat dalam satu batch secara paralel.

        Hasil lengkap ditaruh di blackboard; pesan ke hilir hanya membawa
        ringkasan agar payload tetap kecil.
        """
        if message.schema != "CandidateProfileBatch@1.0":
            raise NotImplementedError(message.schema)

        job = self.context.jobs[message.payload["job_id"]]
        candidate_ids = [entry["candidate_id"] for entry in message.payload["candidates"]]
        settings = self.context.settings

        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            results = list(executor.map(
                lambda candidate_id: self.screen_one(candidate_id, job),
                candidate_ids))
        results.sort(key=lambda item: item["score"], reverse=True)
        passed = [r for r in results if r["score"] >= settings.screening_pass_threshold]

        # update(), bukan penugasan: dengan streaming batch, penugasan akan
        # menimpa hasil batch sebelumnya dan Matching hanya melihat batch akhir.
        board = self.context.board(job.job_id)
        board.setdefault("screening", {}).update(
            {r["candidate_id"]: r for r in results})

        self.record("job", job.job_id, "SCREENING_DONE",
                    {"total": len(results), "passed": len(passed),
                     "injection_blocked": len(self.injection_attempts)})
        payload = {"job_id": job.job_id,
                   "results": [{"candidate_id": r["candidate_id"],
                                "score": r["score"], "confidence": r["confidence"]}
                               for r in passed],
                   "total_screened": len(results)}
        return self.reply(message, Performative.INFORM, "ScreeningResult@1.0",
                          payload, Risk.MEDIUM)
