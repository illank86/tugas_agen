"""Penyusun laporan hasil per kandidat untuk antarmuka dan ekspor.

Data satu kandidat tersebar di beberapa tempat: hasil screening ada di
blackboard, temuan kepatuhan di blackboard lain, rincian skor di hasil
Matching, dan identitas di objek Candidate. Modul ini menyatukannya menjadi
satu baris per kandidat, termasuk untuk kandidat yang gugur di tengah jalan —
antarmuka harus dapat menjelaskan MENGAPA seseorang tidak masuk shortlist,
bukan hanya menampilkan yang lolos.
"""
from typing import Any, Dict, List, Optional

# Urutan tahap, dipakai untuk menentukan sejauh mana kandidat melangkah.
STAGES = ["SOURCED", "SCREENED", "SCREENING_FAILED", "COMPLIANCE_CHECKED",
          "COMPLIANCE_FAILED", "SCORED", "SHORTLISTED", "INTERVIEWED", "PLACED"]


def _stage_of(screened: bool, passed_screening: bool, compliance: Optional[str],
              scored: bool, shortlisted: bool, interviewed: bool,
              placed: bool) -> str:
    """Tentukan tahap terakhir yang dicapai seorang kandidat."""
    if placed:
        return "PLACED"
    if interviewed:
        return "INTERVIEWED"
    if shortlisted:
        return "SHORTLISTED"
    if compliance == "FAIL":
        return "COMPLIANCE_FAILED"
    if scored:
        return "SCORED"
    if compliance:
        return "COMPLIANCE_CHECKED"
    if screened and not passed_screening:
        return "SCREENING_FAILED"
    if screened:
        return "SCREENED"
    return "SOURCED"


def build_candidate_report(system, job, outcome: Dict[str, Any]) -> List[dict]:
    """Susun satu baris hasil untuk setiap kandidat yang tersentuh alur.

    Args:
        system: RecruitmentSystem yang sudah menjalankan lowongan ini.
        job: JobRequirement yang diproses.
        outcome: nilai kembalian RecruitmentSystem.run_job().

    Returns:
        Daftar dict terurut menurun berdasarkan fit score, lalu skor
        screening. Kandidat yang gugur tetap disertakan dengan alasannya.
    """
    board = system.context.board(job.job_id)
    screening_results = board.get("screening", {})
    compliance_detail = board.get("compliance_detail", {})
    ranking = {entry["candidate_id"]: entry for entry in board.get("ranking", [])}
    shortlist_ranks = {entry["candidate_id"]: entry.get("rank")
                       for entry in outcome.get("shortlist", [])}
    placed_ids = {record["candidate_id"] for record in outcome.get("placed", [])}
    interviewed_ids = set(board.get("interviewed", []))

    rows: List[dict] = []
    threshold = system.settings.screening_pass_threshold

    for candidate_id, screening in screening_results.items():
        candidate = system.context.candidates.get(candidate_id)
        compliance = compliance_detail.get(candidate_id)
        scored = ranking.get(candidate_id)
        passed_screening = screening["score"] >= threshold

        documents = []
        for document in (candidate.documents if candidate else []):
            documents.append({
                "jenis": document.kind,
                "ada": document.present,
                "ocr_confidence": document.ocr_confidence,
                "terbit": document.issue_date.isoformat() if document.present else "-",
                "kedaluwarsa": (document.expires_at.isoformat()
                                if document.expires_at else "-"),
            })

        rows.append({
            "candidate_id": candidate_id,
            "nama": candidate.name if candidate else candidate_id,
            "lokasi": candidate.location if candidate else "",
            "sumber": candidate.source if candidate else "",
            "pengalaman_tahun": screening.get("experience_years", 0.0),
            "skor_screening": screening["score"],
            "keyakinan_screening": screening["confidence"],
            "lolos_screening": passed_screening,
            "status_kepatuhan": compliance["status"] if compliance else "-",
            "keyakinan_kepatuhan": compliance["confidence"] if compliance else None,
            "temuan_kepatuhan": (compliance.get("findings", []) if compliance else []),
            "ditinjau_manusia": bool(compliance and compliance.get("resolved_by_human")),
            "fit_score": scored["fit_score"] if scored else 0.0,
            "komponen": scored["components"] if scored else {},
            "kontribusi": scored["contributions"] if scored else {},
            "per_syarat": scored["per_requirement"] if scored else {},
            "syarat_belum_terpenuhi": scored["unmet_requirements"] if scored else [],
            # Saat alur masih berjalan, outcome belum berisi shortlist. Nomor
            # peringkat diambil dari entri Matching yang sudah menyimpannya,
            # supaya tabel tetap informatif di tengah proses.
            "peringkat": shortlist_ranks.get(candidate_id,
                                             (scored or {}).get("rank")),
            "masuk_shortlist": (candidate_id in shortlist_ranks
                                or bool((scored or {}).get("rank"))),
            "ditempatkan": candidate_id in placed_ids,
            "skill_terbaca": screening.get("extracted_skills", {}),
            "ada_indikasi_injeksi": screening.get("injection_flag", False),
            "cv_text": candidate.cv_text if candidate else "",
            "dokumen": documents,
            "tahap": _stage_of(True, passed_screening,
                               compliance["status"] if compliance else None,
                               bool(scored), candidate_id in shortlist_ranks,
                               candidate_id in interviewed_ids,
                               candidate_id in placed_ids),
        })

    # Kandidat shortlist tampil sesuai PERINGKAT resmi dari Matching Agent.
    # Mengurutkan ulang hanya dengan fit score akan mengacak urutan ketika ada
    # nilai seri, sehingga nomor peringkat terlihat melompat-lompat.
    rows.sort(key=lambda row: (0 if row["peringkat"] else 1,
                               row["peringkat"] or 0,
                               -row["fit_score"], -row["skor_screening"],
                               row["candidate_id"]))
    return rows


def rejection_reason(row: dict) -> str:
    """Jelaskan dalam satu kalimat mengapa kandidat tidak masuk shortlist.

    Antarmuka wajib bisa menjawab ini: daftar yang hanya menampilkan pemenang
    tidak dapat diaudit.
    """
    if row["masuk_shortlist"]:
        return "-"
    if not row["lolos_screening"]:
        return (f"skor screening {row['skor_screening']:.2f} di bawah ambang "
                f"kelayakan awal")
    if row["status_kepatuhan"] == "FAIL":
        rules = ", ".join(sorted({f["rule_id"] for f in row["temuan_kepatuhan"]
                                  if f["result"] == "FAIL"}))
        return f"gagal kepatuhan (aturan {rules or 'tidak tercatat'})"
    if row["status_kepatuhan"] == "-":
        return "tidak diteruskan ke pemeriksaan dokumen"
    if row["fit_score"] == 0.0:
        return "tidak memperoleh skor kecocokan"
    return f"fit score {row['fit_score']:.2f} di bawah ambang theta"


def radar_axes(row: dict, job, canonical_name) -> Dict[str, float]:
    """Susun sumbu radar: kekuatan yang membentuk skor kandidat.

    Sumbu terdiri atas pemenuhan tiap syarat skill (nilai 0-1), ditambah
    pengalaman, kepatuhan, dan preferensi klien, sehingga terlihat komponen
    mana yang mengangkat atau menjatuhkan skor.
    """
    axes: Dict[str, float] = {}
    for requirement in job.required_skills:
        skill_id = requirement["skill_id"]
        axes[canonical_name(skill_id)] = round(
            row["per_syarat"].get(skill_id, 0.0), 4)
    experience_ratio = min(1.0, row["pengalaman_tahun"]
                           / max(0.5, job.min_experience_years))
    axes["Pengalaman"] = round(experience_ratio, 4)
    axes["Kepatuhan"] = 1.0 if row["status_kepatuhan"] == "PASS" else 0.0
    axes["Preferensi klien"] = 0.9
    return axes
