"""Antarmuka baris perintah.

    python -m mas_hr.cli demo                     jalankan satu lowongan end-to-end
    python -m mas_hr.cli demo --job data/job_requirements/staff_admin_gudang.txt
    python -m mas_hr.cli demo --cv data/cv        pakai CV dari folder (.pdf/.txt)
    python -m mas_hr.cli demo --interactive       keputusan HITL dari terminal
    python -m mas_hr.cli eval --scenario S2       eksperimen 4 arm
    python -m mas_hr.cli injection                uji ketahanan prompt injection
    python -m mas_hr.cli assign --job data/job_requirements
    python -m mas_hr.cli gamma                    kurva trade-off otonomi
    python -m mas_hr.cli skills                   daftar skill kanonik
    python -m mas_hr.cli ui                       buka antarmuka Streamlit
    python -m mas_hr.cli serve                    jalankan server FastAPI
    python -m mas_hr.cli start                    jalankan API + UI sekaligus
"""
from typing import Dict, List, Optional
import argparse
import json
import sys

from .assignment_optimizer import assign_candidates_to_slots, assign_greedy
from .cv_reader import read_cv_folder
from .evaluation_metrics import micro_average_compliance
from .experiment_runner import (SCENARIOS, build_world, run_arm, run_experiment,
                                sweep_gamma, train_screening_model)
from .human_approval import SimulatedApprover, TerminalApprover
from .job_requirement_reader import load_jobs
from .recruitment_workflow import RecruitmentSystem
from .settings import Settings
from .synthetic_data import SyntheticGenerator


def print_table(rows: List[dict], columns: List[str], title: str = "") -> None:
    """Cetak tabel teks dengan lebar kolom menyesuaikan isi."""
    if not rows:
        return
    if title:
        print(f"\n{title}")
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in columns}
    print(" | ".join(c.ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" | ".join(f"{row.get(c, '')}".ljust(widths[c]) for c in columns))


def prepare_world(args) -> tuple:
    """Siapkan lowongan dan kandidat sesuai opsi baris perintah.

    Prioritas sumber data:
      - lowongan : --job (berkas/folder .txt), jika tidak ada pakai sintetis
      - kandidat : --cv (folder .pdf/.txt), jika tidak ada pakai sintetis

    Returns:
        Tuple (jobs, candidates, catatan) di mana catatan berisi pesan yang
        harus ditampilkan ke pengguna tentang asal dan keterbatasan data.
    """
    spec = SCENARIOS[args.scenario]
    notes: List[str] = []

    if args.job:
        jobs_list = load_jobs(args.job)
        jobs = {job.job_id: job for job in jobs_list}
        notes.append(f"Lowongan : {args.job} ({len(jobs)} lowongan)")
    else:
        jobs, synthetic_candidates = build_world(args.scenario, args.seed)
        notes.append(f"Lowongan : katalog sintetis skenario {args.scenario}")

    if args.cv:
        primary_job = list(jobs.values())[0]
        candidates, report = read_cv_folder(
            args.cv, primary_job, Settings().today,
            assume_documents_valid=args.assume_documents_valid)
        for candidate in candidates.values():
            candidate.job_id_hint = primary_job.job_id
        notes.append(f"CV       : {args.cv} ({report['total']} berkas, "
                     f"metode {sorted(set(report['extraction_methods'].values()))})")
        if report.get("skipped_duplicates"):
            notes.append(f"           {report['skipped_duplicates']} dilewati "
                         f"(orang yang sama sudah terbaca dari format lain)")
        if report["empty_files"]:
            notes.append(f"PERINGATAN: tidak ada teks terbaca dari "
                         f"{report['empty_files']} — kemungkinan PDF hasil "
                         f"pindaian. Pasang pypdf atau sediakan versi teks.")
        notes.append("CATATAN  : CV dari berkas tidak punya ground truth, "
                     "sehingga metrik akurasi tidak berlaku.")
        notes.append("           Nilai wawancara memakai placeholder; pakai "
                     "--interactive agar manusia yang menilai.")
        if args.assume_documents_valid:
            notes.append("PERINGATAN: --assume-documents-valid aktif. Dokumen "
                         "tanpa metadata dianggap sah — ini mengarang bukti "
                         "kepatuhan dan bukan verifikasi nyata.")
    elif args.job:
        generator = SyntheticGenerator(args.seed)
        candidates = {}
        for job in jobs.values():
            for candidate in generator.make_candidates(
                    spec["candidates"], job,
                    problem_ratio=spec["problem_ratio"],
                    injection_ratio=spec["injection_ratio"]):
                candidates[candidate.candidate_id] = candidate
        notes.append(f"Kandidat : sintetis, {len(candidates)} orang")
    else:
        candidates = synthetic_candidates
        notes.append(f"Kandidat : sintetis, {len(candidates)} orang")
    return jobs, candidates, notes


def command_demo(args) -> None:
    """Jalankan satu lowongan end-to-end dan tampilkan jejaknya."""
    scorer, training = train_screening_model(args.seed)
    print(f"Model    : {getattr(scorer, 'version', '?')} "
          f"(ECE={training.get('ece', 'n/a')}, dilatih pada data sintetis)")
    jobs, candidates, notes = prepare_world(args)
    for note in notes:
        print(note)

    settings = Settings(seed=args.seed, hitl_enabled=not args.no_hitl)
    if args.top_k:
        settings.top_k = args.top_k
    approver = (TerminalApprover(seed=args.seed) if args.interactive
                else SimulatedApprover(seed=args.seed))
    system = RecruitmentSystem(settings, jobs, candidates, approver, scorer=scorer,
                               source_all_channels=bool(args.cv))

    job_id = list(jobs)[0]
    job = jobs[job_id]
    print(f"\n=== {job_id} — {job.title} ({job.headcount} posisi) ===")
    outcome = system.run_job(job_id, target_count=len(candidates) or 150)

    print(f"\nHasil    : {outcome['outcome']}")
    print(f"Sourced  : {outcome.get('sourced')} dari kanal {outcome.get('channels')}")
    print(f"Dinilai  : {outcome.get('n_scored')} | shortlist "
          f"{len(outcome['shortlist'])}")
    print(f"State    : {' -> '.join(outcome['states'])}")
    print(f"Waktu    : {outcome['elapsed']:.3f}s "
          f"(tunggu manusia {system.supervisor.total_human_wait:.3f}s)")

    if outcome["shortlist"]:
        print("\n--- Penjelasan skor (3 teratas) ---")
        for entry in outcome["shortlist"][:3]:
            print(f"  #{entry['rank']} {entry['candidate_id']} "
                  f"fit={entry['fit_score']}")
            print(f"      kontribusi  : {entry['contributions']}")
            print(f"      per-syarat  : {entry['per_requirement']}")
            print(f"      belum penuh : {entry['unmet_requirements'] or '-'}")

    print("\n--- Message trace ---")
    for row in system.database.query(
            """SELECT sender, receiver, performative, schema, risk, size_bytes
               FROM agent_messages WHERE conversation_id=? ORDER BY rowid LIMIT 40""",
            (job_id,)):
        print(f"  {row['sender']:>16} -> {row['receiver']:<16} "
              f"{row['performative']:<8} {row['schema']:<30} "
              f"{row['risk']:<6} {row['size_bytes']}B")
    print(f"  ... total {system.bus.sent_count} pesan, {system.bus.sent_bytes} byte")

    print("\n--- Gerbang HITL ---")
    for approval in system.supervisor.approvals:
        print(f"  {approval['checkpoint']:<8} {approval['decision']:<18} "
              f"oleh {approval['approver_id']:<14} "
              f"({approval['latency'] * 1000:.1f} ms)")

    summary = system.policy.summary()
    chain_ok, broken_at = system.audit.verify()
    print(f"\nOtonomi  : HIR={summary['hir']:.3f} "
          f"(terencana {summary['hir_planned']:.3f}, "
          f"eskalasi {summary['hir_escalation']:.3f})")
    print(f"Audit    : {system.audit.count} record | rantai hash "
          f"{'UTUH' if chain_ok else f'RUSAK di log_id {broken_at}'}")
    print(f"Keamanan : {len(system.screening.injection_attempts)} upaya injeksi "
          f"diblokir, {len(system.screening.injection_successes)} berhasil | "
          f"{len(system.policy.blocked_attempts)} aksi tak sah ditolak")
    system.database.close()


def command_eval(args) -> None:
    """Jalankan eksperimen perbandingan arm dan cetak hasilnya."""
    output = run_experiment(args.scenario, args.arms, args.repeats, args.seed,
                            verbose=not args.quiet)
    print(f"\n=== {output['scenario']} — {output['scenario_name']} "
          f"({output['repeats']} ulangan) ===")
    rows = [dict(arm=arm, **stats) for arm, stats in output["aggregate"].items()]
    print_table(rows, ["arm", "elapsed_s_mean", "machine_s_mean", "messages_mean",
                       "payload_bytes_mean", "ndcg_at_10_mean", "screening_f1_mean",
                       "hir_mean", "trace_completeness_mean",
                       "injection_resistance_mean", "shortlisted_mean",
                       "placed_mean"],
                "RATA-RATA PER ARM")

    micro = micro_average_compliance(output["rows"])
    print_table([dict(arm=arm, **values) for arm, values in micro.items()],
                ["arm", "tp", "fp", "fn", "positives", "recall", "fnr", "f_beta",
                 "enough_samples"],
                "KEPATUHAN — MICRO-AVERAGE (tp/fp/fn dijumlahkan lintas ulangan)")
    if any(not values["enough_samples"]
           for arm, values in micro.items() if arm != "B0"):
        print("  PERINGATAN: kasus positif < 10 pada sebagian arm. Naikkan "
              "jumlah kandidat atau problem_ratio sebelum menarik kesimpulan.")

    print("\n=== Uji hipotesis (Mann-Whitney U, non-parametrik) ===")
    for name, test in output["tests"].items():
        medians = {k: v for k, v in test.items() if k.startswith("median_")}
        print(f"  {name}: U={test['u']} z={test['z']} p~{test['p']} "
              f"effect={test['effect_size']}")
        print(f"      median {medians} | {test['direction']} -> {test['verdict']}")

    print("\nCATATAN: B0 adalah MODEL WAKTU ILUSTRATIF, bukan eksperimen.")
    print("Seluruh angka berasal dari data sintetis dan tidak dapat")
    print("digeneralisasi ke kinerja produksi.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
        print(f"\nHasil mentah disimpan: {args.out}")


def command_injection(args) -> None:
    """Bandingkan ketahanan prompt injection antara arm B1 dan arm P."""
    scorer, _ = train_screening_model(args.seed)
    print("=== S6: ketahanan prompt injection (metrik laporan 9.2 nomor 10) ===")
    for arm in ("B1", "P"):
        result = run_arm(arm, "S6", args.seed, scorer=scorer)
        print(f"  arm {arm}: {result.injection_attempts} upaya, "
              f"{result.injection_successes} berhasil, "
              f"resistance={result.injection_resistance:.3f}")
    print("\nB1 rentan karena teks dokumen dan instruksi tidak dipisahkan.")
    print("Arm P memperlakukan CV sebagai DATA: skor hanya dihitung dari fitur")
    print("terstruktur, dan baris instruksi dinetralkan serta dicatat.")


def command_assign(args) -> None:
    """Bandingkan penugasan greedy dengan penugasan optimal lintas lowongan."""
    scorer, _ = train_screening_model(args.seed)
    args.cv = args.cv or ""
    jobs, candidates, notes = prepare_world(args)
    for note in notes:
        print(note)
    system = RecruitmentSystem(Settings(seed=args.seed), jobs, candidates,
                              SimulatedApprover(seed=args.seed), scorer=scorer)
    from .agents.screening_agent import parse_cv, sanitize_cv

    slots = [f"{job_id}#slot{n + 1}"
             for job_id, job in jobs.items() for n in range(job.headcount)]
    matrix: Dict[str, Dict[str, float]] = {}
    for candidate in list(candidates.values())[:args.candidates]:
        cleaned, _ = sanitize_cv(candidate.cv_text)
        skills, _ = parse_cv(cleaned)
        row = {}
        for job_id, job in jobs.items():
            score, _ = system.matching.skill_score(skills, job)
            for n in range(job.headcount):
                row[f"{job_id}#slot{n + 1}"] = round(score, 4)
        matrix[candidate.candidate_id] = row

    greedy = assign_greedy(matrix, slots)
    optimal = assign_candidates_to_slots(matrix, slots)
    print(f"\n=== Optimasi penugasan lintas lowongan (konflik K4) ===")
    print(f"  kandidat={len(matrix)} slot={len(slots)}")
    print(f"  greedy first-come   : total fit {sum(v for _, _, v in greedy):.4f}")
    print(f"  Hungarian (optimal) : total fit {sum(v for _, _, v in optimal):.4f}")
    for candidate_id, slot, value in optimal[:10]:
        print(f"    {candidate_id} -> {slot} (fit {value})")
    system.database.close()


def command_gamma(args) -> None:
    """Tampilkan kurva trade-off antara beban manusia dan kesalahan kepatuhan."""
    rows = sweep_gamma(args.scenario, args.seed)
    print("=== H6: pengaruh gamma terhadap HIR dan kesalahan kepatuhan ===")
    print_table(rows, ["gamma", "hir", "hir_escalation", "fnr", "recall",
                       "approvals", "human_wait_s"])
    print("\nBaca sebagai kurva, bukan satu angka:")
    print("  gamma naik  -> lebih banyak eskalasi, beban manusia bertambah")
    print("  gamma turun -> lebih banyak verdict dieksekusi tanpa manusia")
    print("Bila FNR tetap 0 di seluruh rentang, laporkan apa adanya; jangan")
    print("mengklaim penurunan yang tidak terlihat pada data.")


def command_skills(args) -> None:
    """Cetak taksonomi skill kanonik untuk menyusun berkas job requirement."""
    from .skill_taxonomy import SKILLS
    print("Taksonomi skill kanonik (mas_hr/skill_taxonomy.py).")
    print("Pada berkas job requirement, tulis skill_id ATAU salah satu nama/sinonim.\n")
    print_table([{"skill_id": skill_id, "nama": name, "induk": parent or "-",
                  "sinonim": ", ".join(synonyms[:4])}
                 for skill_id, (name, synonyms, parent) in sorted(SKILLS.items())],
                ["skill_id", "nama", "induk", "sinonim"])
    print("\nSkill di luar daftar ini DITOLAK saat memuat berkas job requirement.")
    print("Untuk menambah: sunting SKILLS dan sertakan sinonimnya, agar parser")
    print("CV mengenali berbagai cara penulisan.")


def command_ui(args) -> None:
    """Jalankan antarmuka Streamlit pada berkas ui_app.py.

    Streamlit harus dijalankan lewat perintahnya sendiri (bukan sekadar
    mengimpor modul), sehingga fungsi ini memanggil `streamlit run` dengan
    jalur berkas aplikasi.
    """
    import subprocess
    from pathlib import Path

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit belum terpasang. Pasang dengan:")
        print("    pip install streamlit plotly pandas")
        print("Seluruh perintah lain tetap berjalan tanpa Streamlit.")
        raise SystemExit(1)

    app_path = Path(__file__).resolve().parent / "ui_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path),
               "--server.port", str(args.port), "--server.headless", "true"]
    print(f"Membuka antarmuka di http://localhost:{args.port}")
    subprocess.run(command, check=False)


def command_serve(args) -> None:
    """Jalankan server FastAPI sehingga sistem melayani banyak permintaan.

    Berbeda dengan subperintah lain yang sekali jalan lalu selesai, server ini
    hidup terus-menerus dan memproses beberapa lowongan bersamaan lewat pool
    pekerja.
    """
    try:
        import uvicorn
    except ImportError:
        print("FastAPI/uvicorn belum terpasang. Pasang dengan:")
        print("    pip install fastapi uvicorn python-multipart")
        raise SystemExit(1)

    from .api_server import create_app

    application = create_app(max_workers=args.workers,
                             database_path=args.database)
    print(f"Server berjalan di http://{args.host}:{args.port}")
    print(f"Dokumentasi interaktif: http://{args.host}:{args.port}/docs")
    print(f"Pekerja paralel: {args.workers} | basis data: {args.database}")
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")


def command_start(args) -> None:
    """Jalankan server API dan antarmuka Streamlit dalam satu perintah."""
    from .launcher import start_all

    code = start_all(api_host=args.host, api_port=args.api_port,
                     ui_port=args.ui_port, workers=args.workers,
                     database=args.database)
    raise SystemExit(code)


def build_parser() -> argparse.ArgumentParser:
    """Susun seluruh subperintah beserta opsinya."""
    parser = argparse.ArgumentParser(
        prog="mas_hr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42,
                        help="penentu keacakan agar hasil dapat direproduksi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_options(subparser) -> None:
        """Tambahkan opsi sumber data yang dipakai beberapa subperintah."""
        subparser.add_argument("--scenario", default="S1", choices=list(SCENARIOS))
        subparser.add_argument("--job", default="",
                               help="berkas .txt atau folder berisi job requirement")
        subparser.add_argument("--cv", default="",
                               help="folder berisi CV (.pdf, .txt, .md)")
        subparser.add_argument("--assume-documents-valid", action="store_true",
                               help="anggap dokumen tanpa metadata sebagai sah "
                                    "(eksplisit: ini mengarang bukti kepatuhan)")

    demo = subparsers.add_parser("demo", help="jalankan satu lowongan end-to-end")
    add_data_options(demo)
    demo.add_argument("--interactive", action="store_true",
                      help="keputusan HITL diminta dari terminal")
    demo.add_argument("--no-hitl", action="store_true", help="mode arm B2")
    demo.add_argument("--top-k", type=int, default=0,
                      help="ambil k teratas alih-alih memakai ambang theta")
    demo.set_defaults(function=command_demo)

    evaluate = subparsers.add_parser("eval", help="eksperimen perbandingan arm")
    evaluate.add_argument("--scenario", default="S1", choices=list(SCENARIOS))
    evaluate.add_argument("--repeats", type=int, default=5)
    evaluate.add_argument("--arms", nargs="+", default=["B0", "B1", "B2", "P"])
    evaluate.add_argument("--out", default="", help="simpan hasil mentah ke JSON")
    evaluate.add_argument("--quiet", action="store_true")
    evaluate.set_defaults(function=command_eval)

    injection = subparsers.add_parser("injection", help="uji prompt injection (S6)")
    injection.set_defaults(function=command_injection)

    assign = subparsers.add_parser("assign", help="optimasi penugasan lintas lowongan")
    add_data_options(assign)
    assign.add_argument("--candidates", type=int, default=12,
                        help="jumlah kandidat yang dimasukkan ke matriks")
    assign.set_defaults(function=command_assign)

    gamma = subparsers.add_parser("gamma", help="kurva trade-off otonomi (H6)")
    gamma.add_argument("--scenario", default="S2", choices=list(SCENARIOS))
    gamma.set_defaults(function=command_gamma)

    skills = subparsers.add_parser("skills", help="daftar skill kanonik")
    skills.set_defaults(function=command_skills)

    ui = subparsers.add_parser("ui", help="buka antarmuka Streamlit")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(function=command_ui)

    serve = subparsers.add_parser("serve", help="jalankan server FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=4,
                       help="jumlah alur rekrutmen yang berjalan bersamaan")
    serve.add_argument("--database", default=":memory:",
                       help="berkas SQLite bersama, mis. data/mas_hr.db")
    serve.set_defaults(function=command_serve)

    start = subparsers.add_parser(
        "start", help="jalankan server API dan antarmuka Streamlit sekaligus")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--api-port", type=int, default=8000)
    start.add_argument("--ui-port", type=int, default=8501)
    start.add_argument("--workers", type=int, default=4)
    start.add_argument("--database", default=":memory:")
    start.set_defaults(function=command_start)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Titik masuk baris perintah."""
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
