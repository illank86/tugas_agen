# mas_hr — Prototipe Multi-Agent System untuk Outsourcing HRD

Implementasi **Project #1: Intelligent Multi-Agent System Prototype** —
*Agentic AI untuk Perusahaan Outsourcing HRD: Multi-Agent System dengan
Bounded Autonomy & Human-in-the-Loop untuk Rekrutmen dan Penempatan Karyawan.*

Berjalan dengan Python 3.9+ tanpa dependensi wajib.

---

## Di mana peran manusianya?

Enam gerbang. Di setiap gerbang alur BERHENTI — bukan berjalan terus lalu
dikonfirmasi belakangan.

| Gerbang | Yang diputuskan manusia | Peran berwenang | Level |
|---|---|---|---|
| HITL-1 | Spesifikasi lowongan sebelum pencarian dimulai | HR Manager | A1 |
| HITL-2 | Dokumen yang tidak terbaca cukup yakin oleh sistem | Compliance Officer | A1 |
| HITL-3 | Finalisasi shortlist: siapa yang maju ke wawancara | Recruiter | A1 |
| HITL-4 | Keputusan hire — tidak dapat dibatalkan | HR Manager | **A0** |
| HITL-5 | Persetujuan isi kontrak kerja | HR Manager | **A0** |
| HITL-6 | Otorisasi penempatan di lokasi klien | HR Manager | **A0** |

Level A0 berarti agen hanya menyajikan bukti; keputusannya sepenuhnya manusia.
Agen secara mekanis tidak memiliki izin mengeksekusinya — `PolicyEngine`
menolak, dan penolakan itu tercatat di audit.

**Siapa yang menjadi "manusia" itu tergantung mode yang dipilih:**

| Mode | Siapa yang memutuskan | Dipakai untuk |
|---|---|---|
| `SimulatedApprover` | **Generator acak**, bukan manusia | Eksperimen otomatis (`eval`, `injection`, `gamma`) — agar ribuan keputusan dapat direproduksi |
| `TerminalApprover` | Anda, di terminal | `cli demo --interactive` |
| UI mode "Manual" | Anda, lewat tombol di Streamlit | Demo dan presentasi |
| `QueuedApprover` | Siapa pun lewat HTTP | Server API, `approval_mode: manual` |

Mode simulasi ada karena eksperimen 4 arm membutuhkan ribuan keputusan yang
dapat diulang. `accuracy` sengaja disetel di bawah 1.0: reviewer manusia juga
bisa salah, dan tanpa itu arm dengan HITL akan tampak sempurna sehingga hasil
evaluasinya tidak kredibel.

**Yang harus dikatakan apa adanya saat presentasi:** pada mode eksperimen,
tidak ada manusia sungguhan di dalam lingkaran — yang diukur adalah
*mekanisme* gerbangnya, bukan kualitas penilaian manusia. Untuk menunjukkan
peran manusia yang sebenarnya, pakai mode manual (UI atau API).

## Satu perintah untuk semuanya

```bash
pip install fastapi "uvicorn[standard]" python-multipart streamlit plotly pandas
python3 -m mas_hr.cli start
```

Menyalakan server API dan antarmuka Streamlit sekaligus:

```
API  : http://127.0.0.1:8000      (dokumentasi: /docs)
UI   : http://localhost:8501
```

Keluaran kedua proses ditampilkan di satu terminal dengan awalan `[API]` dan
`[UI]` berwarna. Ctrl+C mematikan keduanya; bila salah satu berhenti sendiri,
pasangannya ikut dimatikan agar tidak ada layanan yang jalan setengah-setengah
dan tidak ada proses yatim yang menahan port.

Opsi: `--api-port`, `--ui-port`, `--workers`, `--database`.

## Server API (FastAPI)

Mode ini membuat sistem hidup terus-menerus dan memproses **beberapa lowongan
bersamaan**, bukan menjalankan agen satu per satu untuk satu lowongan lalu
selesai.

```bash
pip install fastapi "uvicorn[standard]" python-multipart
python3 -m mas_hr.cli serve --workers 4        # http://127.0.0.1:8000
# dokumentasi interaktif: http://127.0.0.1:8000/docs
```

Setiap permintaan menjadi satu *run* yang dijalankan di thread pekerja
tersendiri. Endpoint mengembalikan `run_id` seketika (HTTP 202); klien
kemudian memantau statusnya.

| Metode | Endpoint | Fungsi |
|---|---|---|
| GET | `/health` | status layanan, jumlah pekerja, rekap run |
| GET | `/api/agents`, `/api/skills` | delapan agen dan taksonomi skill |
| GET | `/api/folders?path=&kind=` | periksa isi folder sebelum dipakai |
| POST | `/api/runs` | kirim lowongan dari jalur berkas/folder di server |
| POST | `/api/runs/upload` | unggah job requirement + CV lalu langsung proses |
| GET | `/api/runs` | daftar seluruh run |
| GET | `/api/runs/{run_id}` | status satu run |
| GET | `/api/runs/{run_id}/candidates` | hasil per kandidat, terurut |
| GET | `/api/runs/{run_id}/candidates/{id}` | detail + sumbu radar |
| GET | `/api/runs/{run_id}/messages` | jejak pesan antar-agen |
| GET | `/api/runs/{run_id}/governance` | otonomi, audit, keamanan, gerbang |
| GET | `/api/approvals` | gerbang yang menunggu keputusan manusia |
| POST | `/api/approvals/{run_id}/{gate_id}` | kirim keputusan manusia |

Contoh:

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"job_path":"data/job_requirements/staff_admin_gudang.txt","synthetic_count":150}'

curl -X POST http://127.0.0.1:8000/api/runs/upload \
  -F "job_file=@data/job_requirements/staff_admin_gudang.txt" \
  -F "cv_files=@data/cv/andi.pdf" -F "meta_files=@data/cv/andi.meta.txt"
```

### Human-in-the-loop lewat HTTP

Dengan `"approval_mode": "manual"`, gerbang HITL menjadi **barrier lintas
proses**: thread alur benar-benar berhenti di `HITL-1`, dan baru berlanjut
setelah seseorang mengirim keputusan ke `/api/approvals/{run_id}/{gate_id}`.
Alur lain tetap berjalan selama itu, karena masing-masing punya thread sendiri.

```bash
curl "http://127.0.0.1:8000/api/approvals?run_id=run-xxxx"
curl -X POST http://127.0.0.1:8000/api/approvals/run-xxxx/HITL-1:JOB-2026-0201:-:ab12cd \
  -H 'Content-Type: application/json' \
  -d '{"decision":"APPROVED","approver_id":"USR-HRM-01","reason":"sesuai kebutuhan klien"}'
```

Bila tidak ada keputusan sampai batas waktu, keputusan cadangan adalah
**REJECTED**, bukan APPROVED: pada domain teregulasi, diam tidak boleh
dianggap persetujuan.

Untuk menyimpan jejak lintas restart, jalankan dengan basis data berkas:
`--database data/mas_hr.db` (SQLite mode WAL, sehingga beberapa run dapat
menulis bergantian tanpa saling memblokir lama).

## Antarmuka grafis (Streamlit)

```bash
pip install streamlit plotly pandas
python3 -m mas_hr.cli ui              # membuka http://localhost:8501
```

Tiga layar:

1. **Panel kiri** — tentukan folder job requirement dan folder CV, unggah CV
   langsung ke folder tersebut, atur parameter (theta, gamma, top-k, nomor variasi data/seed), dan
   pilih **mode persetujuan**: manual (Anda yang memutuskan di enam titik
   persetujuan), otomatis (approver tersimulasi), atau tanpa persetujuan
   manusia (arm B2).
   Pada mode manual, halaman menampilkan satu gerbang pada satu waktu beserta
   buktinya, dan **alasan keputusan wajib diisi** — keputusan tanpa alasan
   tidak dapat diaudit.
   Di gerbang HITL-3 sampai HITL-6, tabel kandidat lengkap ikut ditampilkan
   dan dapat diklik untuk membuka detail beserta radar **sebelum** keputusan
   diambil. Linimasa enam gerbang di bagian atas menunjukkan posisi proses,
   dan tiap gerbang dapat disetel "Tanya saya" atau diputuskan otomatis
   (ditandai SIM-AUTO pada jejak audit) lewat panel *Simulasi per gerbang*.
2. **Daftar hasil** — satu baris per kandidat, terurut menurut fit score,
   lengkap dengan skor screening, pengalaman, status kepatuhan, tahap
   terakhir, dan **alasan gugur**. Kandidat yang tidak lolos tetap
   ditampilkan: daftar yang hanya memuat pemenang tidak dapat diaudit.
3. **Detail** — klik satu baris untuk melihat rincian pelamar: radar kekuatan
   pembentuk skor, sumbangan tiap komponen, skill yang berhasil dibaca dari
   CV, tabel dokumen, temuan kepatuhan per aturan, dan teks CV asli.

Radar chart memakai satu sumbu untuk setiap syarat skill pada lowongan,
ditambah pengalaman, kepatuhan, dan preferensi klien. Garis putus-putus adalah
target penuh, sehingga terlihat syarat mana yang mengangkat atau menjatuhkan
skor.

Catatan: ketika kandidat berasal dari folder CV, seluruh kanal sourcing
diaktifkan agar tidak ada berkas yang hilang hanya karena kanalnya kalah dalam
lelang Contract Net.

## Menjalankan lewat baris perintah

```bash
python3 -m mas_hr.cli demo                 # satu lowongan end-to-end + message trace
python3 -m mas_hr.cli demo --interactive   # keputusan HITL dari terminal (untuk presentasi)
python3 -m mas_hr.cli demo --no-hitl       # arm B2: gerbang manusia dimatikan
python3 -m mas_hr.cli eval --scenario S2 --repeats 5
python3 -m mas_hr.cli injection            # uji ketahanan prompt injection
python3 -m mas_hr.cli assign --job data/job_requirements
python3 -m mas_hr.cli gamma                # kurva trade-off otonomi (H6)
python3 -m mas_hr.cli skills               # daftar skill kanonik
python3 -m mas_hr.cli ui                   # antarmuka Streamlit
python3 -m mas_hr.cli serve                # server FastAPI saja
python3 -m mas_hr.cli start                # API + UI sekaligus
python3 -m pytest tests/ -q                # 30 uji invarian
```

### Memakai job requirement sendiri (.txt)

```bash
python3 -m mas_hr.cli demo --job data/job_requirements/staff_admin_gudang.txt
python3 -m mas_hr.cli demo --job data/job_requirements     # seluruh folder
```

Format berkas (`data/job_requirements/*.txt`):

```
job_id: JOB-2026-0201
judul: Staff Admin Gudang
klien: CLI-0031
jumlah: 3
pengalaman_minimal: 1
lokasi: Sleman, DIY
sla_hari: 14
dokumen: KTP, Ijazah, SKCK, Surat Keterangan Sehat

skill: Microsoft Excel | 0.40 | intermediate
skill: SOP Pergudangan | 0.30 | basic
skill: Komunikasi      | 0.30 | intermediate
```

Kolom skill: `nama | importance | min_level`. Dua aturan ditegakkan keras:

- **Jumlah importance harus 1.00.** Bobot itu adalah $\beta_j$ pada rumus
  coverage-aware matching; bila tidak berjumlah 1, skor keluar dari skala
  [0,1] dan tidak lagi sebanding antar-lowongan, padahal optimasi penugasan
  lintas lowongan mengandalkan itu.
- **Skill harus ada di taksonomi kanonik.** Entitas tak dikenal ditolak, bukan
  didiamkan — itu penegakan ontologi bersama antar-agen. Lihat daftarnya
  dengan `python3 -m mas_hr.cli skills`; untuk menambah, sunting
  `mas_hr/skill_taxonomy.py` beserta sinonimnya.

### Memakai CV sendiri (.pdf)

```bash
python3 -m mas_hr.cli demo --job data/job_requirements/staff_admin_gudang.txt --cv data/cv
```

Struktur folder:

```
data/cv/
  andi.pdf
  andi.meta.txt      (opsional: identitas + metadata dokumen)
  budi.pdf
  budi.meta.txt
```

Isi `.meta.txt`:

```
nama: Andi Santoso
lokasi: Sleman, DIY
pengalaman: 3
sumber: db-internal
dokumen: KTP                   | ada | 0.97
dokumen: SKCK                  | ada | 0.93 | 2026-06-01 | 2026-11-28
dokumen: Surat Keterangan Sehat| tidak
```

Kolom dokumen: `jenis | ada/tidak | ocr_confidence | tanggal_terbit | kedaluwarsa`.

**Tanpa metadata, dokumen dinyatakan HILANG, bukan sah.** Itu disengaja:
sistem kepatuhan tidak boleh mengarang bukti. Untuk demo cepat ada flag
`--assume-documents-valid`, dan CLI akan mencetak peringatan bahwa bukti
kepatuhan sedang diasumsikan.

Ekstraksi teks PDF dicoba berurutan: **pypdf** (bila terpasang) →
**pdftotext** (bila ada Poppler) → **parser bawaan** (zlib + ToUnicode CMap,
tanpa dependensi). Parser bawaan sudah menangani PDF hasil pengolah kata,
termasuk font subset. PDF hasil **pindaian** tidak menghasilkan teks sama
sekali — CLI melaporkannya agar tidak disalahartikan sebagai kesalahan sistem.

---

## Peta kode ke bagian laporan

Nama berkas mengikuti fungsinya.

| Berkas | Isi | Bagian laporan |
|---|---|---|
| `settings.py` | parameter, matriks risiko aksi, allowlist per agen | 5.9.3 |
| `domain_models.py` | JobRequirement, Candidate, Document | 7.2 |
| `skill_taxonomy.py` | ontologi skill kanonik dan normalisasi | 5.8.2 |
| `text_similarity.py` | embedding dan cosine similarity | 6.2 |
| `agent_messaging.py` | amplop FIPA-ACL, bus, tanda tangan, idempotency | 5.6 |
| `audit_trail.py` | rantai hash append-only | 8.5 |
| `autonomy_policy.py` | bounded autonomy: R(a), gamma, tau, allowlist | 5.9.2 |
| `job_state_machine.py` | 19 state dan invarian gerbang | 5.6.4 |
| `human_approval.py` | approver tersimulasi dan interaktif | 5.9 |
| `screening_model.py` | fitur, regresi logistik, kalibrasi Platt, ECE | 6.3 |
| `assignment_optimizer.py` | Hungarian/Jonker-Volgenant dan pembanding greedy | 6.5.3 |
| `database.py` | skema SQLite 18 tabel, trigger append-only | 7.2 |
| `synthetic_data.py` | generator kandidat dan lowongan + ground truth | 7.4 |
| `job_requirement_reader.py` | pembaca lowongan dari .txt | — |
| `cv_reader.py` | pembaca CV dari .pdf/.txt + ekstraksi PDF | — |
| `recruitment_workflow.py` | orkestrasi tujuh fase dan enam gerbang HITL | 5.7 |
| `comparison_baselines.py` | arm B0 dan B1 | 9.1 |
| `evaluation_metrics.py` | metrik, micro-average, Mann-Whitney U | 9.2 |
| `experiment_runner.py` | skenario S1–S7 dan eksperimen | 8.4, 9.3 |
| `result_reporting.py` | laporan per kandidat + sumbu radar + alasan gugur | 8.2 |
| `ui_app.py` | antarmuka Streamlit: daftar hasil, detail, radar | 8.2 |
| `recruitment_service.py` | antrean run, pool pekerja, registri hasil | 8.2 |
| `api_server.py` | server FastAPI dan gerbang HITL lewat HTTP | 8.2 |
| `launcher.py` | menjalankan API dan UI dalam satu perintah | 8.2 |
| `cli.py` | antarmuka baris perintah | 8.2 |
| `agents/*_agent.py` | satu berkas per agen | 5.4 |

---

## Yang benar-benar ada vs yang tidak

Ditulis eksplisit agar tidak ada klaim berlebih saat presentasi.

**Ada dan berjalan:** 8 agen dengan orkestrasi end-to-end; 6 gerbang HITL
sebagai barrier nyata; message bus dengan tanda tangan, validasi skema, dan
idempotency; rule engine kepatuhan yang keputusannya dapat dikutip per-aturan;
coverage-aware matching dengan penjelasan kontribusi skor; Hungarian untuk
penugasan lintas lowongan; audit rantai hash dengan `audit_logs` append-only;
regresi logistik dengan kalibrasi Platt dan ECE; pertahanan prompt injection
yang terukur; eksperimen 4 arm dengan uji non-parametrik; pembaca job
requirement .txt dan CV .pdf.

**Tidak ada — jangan diklaim:**

- **LLM sungguhan.** Intake dan Placement memakai logika deterministik dan
  template. Hook-nya tersedia; sampai ditukar, jangan tulis "memakai LLM".
- **Embedding neural.** Default `HashingEmbedder` berbasis karakter n-gram.
- **OCR sungguhan.** Disimulasikan dari metadata dokumen.
- **ASR / transkripsi wawancara.** Hanya referensi placeholder.
- **Mobile agent.** Seluruh agen statis; migrasi ke edge node tetap future work.
- **RL dan GNN.** Tidak diimplementasikan (laporan 6.7 dan 6.8).
- **Data nyata.** Seluruh angka dari data sintetis dan tidak dapat
  digeneralisasi ke kinerja produksi.

---

## Dua hal yang mudah ditanyakan penguji

**`hir` berbeda dari `approvals` — kenapa?** `hir` dihitung di tingkat
PolicyEngine: proporsi seluruh permintaan aksi yang tidak lolos syarat otonom,
termasuk aksi per-dokumen. `approvals` dihitung di tingkat gerbang: berapa kali
manusia benar-benar dimintai keputusan. Satu gerbang dapat mewakili banyak
keputusan aksi, jadi keduanya tidak harus sama dan sengaja tidak digabung.

**Kenapa `elapsed` dipisah dari `machine`?** Arm dengan HITL selalu lebih
lambat secara end-to-end karena manusia butuh waktu berpikir. Menyatukannya
akan membuat pengawasan manusia terlihat sebagai cacat kinerja, padahal itu
fiturnya. `machine = elapsed − waktu tunggu manusia`.

---

## Temuan dari eksekusi (jalankan sendiri, jangan kutip angka ini)

Skenario S2, 5 ulangan, data sintetis:

- **H3 (HITL menurunkan false negative kepatuhan): DIDUKUNG.** Arm P mencapai
  FNR 0; arm B2 dan B1 meninggalkan false negative karena ketidakpastian
  pembacaan dokumen diperlakukan sebagai lolos.
- **H4 (HITL menaikkan waktu end-to-end): DIDUKUNG.** Hipotesis yang
  memprediksi kerugian, dan memang terjadi.
- **H1 (MAS lebih cepat dari single-agent): TERBANTAH pada skala ini.**
  Overhead orkestrasi mendominasi pada ratusan kandidat. Laporkan apa adanya —
  mencari titik impasnya justru temuan yang bernilai.
- **H7 (MAS lebih tahan injeksi): DIDUKUNG.**

## Lisensi

MIT — silakan pakai untuk keperluan tugas.
