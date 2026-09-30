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
python3 -m mas_hr.cli llm-check            # cek LLM DeepSeek (lihat bagian LLM)
python3 -m mas_hr.cli ui                   # antarmuka Streamlit
python3 -m mas_hr.cli serve                # server FastAPI saja
python3 -m mas_hr.cli start                # API + UI sekaligus
python3 -m pytest tests/ -q                # 42 uji invarian
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

## LLM DeepSeek (opsional)

LLM DeepSeek dipakai untuk tiga hal: **parsing job description**, **parsing
CV**, dan **narasi** penjelasan hasil. Ketiganya opsional. Tanpa API key,
sistem berjalan seperti semula: deterministik, tanpa jaringan, dan tanpa
paket tambahan.

Prinsip rancangannya: **LLM mengekstrak dan menjelaskan, tidak memutuskan.**
Skor, status kepatuhan, peringkat, dan gerbang persetujuan manusia tetap
dihitung oleh kode yang sama seperti sebelumnya.

### Mengaktifkan

```bash
cp .env.example .env                  # PowerShell: Copy-Item .env.example .env
# buka .env lalu isi:  DEEPSEEK_API_KEY=sk-...
python -m mas_hr.cli llm-check        # pastikan semuanya [OK]
```

`.env` sudah ada di `.gitignore`, jadi kunci tidak ikut ter-commit. Jangan
menulis kunci di `.env.example`: berkas itu ikut ter-commit dan tidak dibaca
aplikasi. Setelah `.env` diubah, jalankan ulang aplikasi.

| Variabel | Wajib | Default | Keterangan |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | ya | — | Kunci dari platform.deepseek.com → API Keys |
| `DEEPSEEK_MODEL` | tidak | `deepseek-chat` | `deepseek-chat` cepat (±1 dtk per CV). Model reasoning seperti `deepseek-v4-pro` jauh lebih lambat (±15 dtk per CV). |
| `DEEPSEEK_BASE_URL` | tidak | `https://api.deepseek.com` | Ganti bila memakai proxy atau endpoint lain yang kompatibel |

Urutan pencarian: variabel yang disetel di terminal (misalnya
`$env:DEEPSEEK_API_KEY="sk-..."` atau `setx`) → `.env` di folder kerja →
`.env` di akar proyek. Yang ditemukan lebih dulu menang.

### Memeriksa apakah LLM bekerja

```
python -m mas_hr.cli llm-check
```

| Langkah | Yang diuji | Bila GAGAL, periksa |
|---|---|---|
| Berkas .env / API key | `.env` ditemukan dan kunci terbaca (ditampilkan tersamar) | Nama berkas harus `.env`, bukan `.env.example` |
| Koneksi & narasi | Satu pesan pendek lewat `chat()`, jalur yang sama dengan narasi | Kunci salah (HTTP 401), jaringan, atau nama model |
| Parsing CV | Satu CV contoh lewat `parse_cv_text()` (mode JSON) | Format balasan model, atau batas token |
| Parsing job description | Satu lowongan contoh lewat `parse_job_description()` | Skill tidak terpetakan ke taksonomi |

Perintah ini keluar dengan kode 1 bila ada langkah yang gagal, sehingga bisa
dipakai di skrip.

### Tiga fungsi LLM

| Fungsi | Kapan aktif | Batas yang dipaksakan kode |
|---|---|---|
| **Parsing job description** | Berkas lowongan (.txt/.md/.pdf) yang **bukan** format `kunci: nilai` | Berkas `kunci: nilai` tetap memakai parser ketat. Skill wajib ada di taksonomi; skill asing menjadi ambiguitas di gerbang Konfirmasi lowongan. Bobot dinormalkan ke 1.0 dan `validate()` tetap berlaku. |
| **Parsing CV** | Setiap CV dari folder, bila kunci tersedia | CV disanitasi (`sanitize_cv`) sebelum dikirim. Skill wajib ada di taksonomi, level wajib `basic`/`intermediate`/`advanced`, pengalaman dibatasi 0–50 tahun. `.meta.txt` selalu menang. Dokumen kepatuhan **tidak pernah** diambil dari LLM. |
| **Narasi** | Hanya saat tombol di UI ditekan atau endpoint `/narrative` dipanggil | Hanya fakta terstruktur yang dikirim, tanpa teks CV. Narasi tidak mengalir balik ke skor maupun keputusan. |

### Alur data

```
berkas lowongan ─► job_requirement_reader.parse_job_document
                     ├─ format "kunci: nilai" ─► parse_job_text (parser ketat, tanpa LLM)
                     └─ teks bebas ───────────► llm_parsing.parse_job_description ─► DeepSeek
                                                  └─ skill asing ─► job.unresolved_skills
                                                       └─► IntakeAgent ─► bukti gerbang HITL-1

folder CV ─► cv_reader.read_cv_folder
               ├─ ekstraksi teks (pypdf / pdftotext / bawaan)
               ├─ llm_parsing.parse_cv_text  (4 CV paralel, setelah sanitize_cv) ─► DeepSeek
               └─ Candidate(nama/lokasi/pengalaman: .meta.txt > LLM > default,
                            llm_skills = skill tervalidasi)
                    └─► ScreeningAgent: merge_skills(parser aturan, llm_skills)
                         └─► model screening menghitung skor (tanpa LLM)

hasil run ─► llm_narrator.narrate_candidate / narrate_run
               └─ candidate_facts(): hanya angka, status, temuan ─► DeepSeek ─► narasi
```

### Referensi modul

**`mas_hr/deepseek_client.py`**: klien HTTP DeepSeek berbasis `urllib`.
Endpoint DeepSeek kompatibel dengan format chat completions OpenAI.

| Nama | Keterangan |
|---|---|
| `is_available()` | `True` bila `DEEPSEEK_API_KEY` terisi |
| `model_name()` | Model aktif, dari `DEEPSEEK_MODEL` atau default |
| `chat(system, user, max_tokens=4000, json_mode=False, temperature=0.3)` | Satu panggilan; mengembalikan teks balasan |
| `chat_json(system, user, max_tokens=8000)` | Mode JSON, `temperature=0`, hasil di-cache per isi prompt supaya rerun Streamlit tidak memanggil API ulang |
| `DeepSeekError` | Galat kunci, HTTP, jaringan, balasan kosong, atau JSON rusak |
| `ENV_FILES` | Daftar berkas `.env` yang berhasil dimuat |

Batas token dibuat longgar (4.000 untuk teks, 8.000 untuk JSON) karena model
reasoning menghabiskan token untuk "berpikir" sebelum menjawab. Bila batas
habis, galatnya menyebut penyebab ini.

**`mas_hr/env_loader.py`**: pemuat `.env` tanpa paket tambahan.

| Nama | Keterangan |
|---|---|
| `load_env(paths=None)` | Muat `.env` ke `os.environ` tanpa menimpa variabel yang sudah ada |
| `parse_env_text(text)` | Urai `KUNCI=nilai`; mendukung komentar `#`, awalan `export`, dan tanda kutip |

**`mas_hr/llm_parsing.py`**: ekstraksi dengan validasi.

| Nama | Keterangan |
|---|---|
| `parse_job_description(text, fallback_job_id)` | Teks bebas → `JobRequirement` dengan `parsed_by="deepseek"`. Melempar `ValueError` bila gagal atau tidak ada skill yang terpetakan. |
| `parse_cv_text(cv_text)` | Teks CV → `{name, location, experience_years, skills}`. Melempar `DeepSeekError` bila gagal. |
| `looks_like_key_value(text, keys)` | `True` bila ≥60% baris berformat `kunci: nilai`; menentukan jalur parser ketat atau LLM |

Setiap `skill_id` dari LLM diperiksa ulang: bila tidak ada di taksonomi,
nama skill dinormalkan dengan `normalize()`, dan bila tetap tidak dikenal,
skill itu dibuang (CV) atau dicatat sebagai ambiguitas (lowongan). Teks
masukan dipotong di 30.000 karakter.

**`mas_hr/llm_narrator.py`**: narasi hasil.

| Nama | Keterangan |
|---|---|
| `candidate_facts(row, job)` | Menyaring baris hasil kandidat menjadi fakta yang aman dikirim; `cv_text` sengaja dibuang |
| `narrate_candidate(row, job)` | Mengembalikan `(narasi, sumber)` untuk satu kandidat |
| `narrate_run(job, rows, outcome)` | Mengembalikan `(narasi, sumber)` untuk seluruh run |

`sumber` bernilai `"deepseek"`, atau `"template (alasan)"` bila LLM tidak
tersedia atau gagal.

**Perubahan pada modul yang sudah ada**

| Berkas | Perubahan |
|---|---|
| `domain_models.py` | `JobRequirement.parsed_by`, `JobRequirement.unresolved_skills`, `Candidate.llm_skills` |
| `job_requirement_reader.py` | `parse_job_document()` memilih jalur; mendukung .md dan .pdf (`JOB_SUFFIXES`) |
| `cv_reader.py` | `read_cv_folder(..., use_llm=None)`; `None` berarti otomatis aktif bila kunci ada. Laporan menambah `llm_parsed` dan `llm_errors`. |
| `agents/screening_agent.py` | `merge_skills()` menggabungkan skill parser aturan dan LLM; level tertinggi yang dipakai |
| `agents/intake_agent.py` | `unresolved_skills` masuk `unresolved_ambiguities`; payload membawa `parsed_by` |
| `recruitment_workflow.py` | Bukti gerbang HITL-1 memuat `unresolved` dan `parsed_by` |
| `recruitment_service.py` | `candidate_narrative()` dan `run_narrative()` |
| `cli.py` | Perintah `llm-check` |

### Endpoint API

| Metode | Path | Hasil |
|---|---|---|
| GET | `/api/runs/{run_id}/narrative` | `{run_id, narasi, sumber}`: ringkasan seluruh run |
| GET | `/api/runs/{run_id}/candidates/{candidate_id}/narrative` | `{candidate_id, narasi, sumber}`: penjelasan satu kandidat |

`GET /api/folders?kind=job` dan `POST /api/runs/upload` juga menerima job
description .md dan .pdf.

### Tampilan di antarmuka Streamlit

- **Ringkasan hasil**: tombol *Buat ringkasan naratif*, dan keterangan
  *"N CV juga diekstrak LLM DeepSeek"* atau peringatan bila ekstraksi gagal.
- **Detail kandidat**: tombol *Buat narasi penjelasan*. Narasi dari LLM diberi
  label *"Ditulis oleh LLM DeepSeek…"*, sedangkan narasi cadangan diberi label
  *"Sumber: template (…)"*.
- **Gerbang Konfirmasi lowongan**: keterangan bila spesifikasi berasal dari
  LLM, dan daftar skill yang tidak terpetakan ke taksonomi.

Narasi baru dibuat saat tombol ditekan, lalu disimpan di sesi, supaya rerun
Streamlit tidak memanggil API berulang kali.

### Perilaku saat LLM tidak tersedia atau gagal

| Situasi | Parsing job description | Parsing CV | Narasi |
|---|---|---|---|
| Tanpa kunci | Format `kunci: nilai` jalan; teks bebas ditolak dengan pesan yang menyebut `.env` | Parser aturan saja | Template |
| Jaringan atau API gagal | `ValueError` dengan alasannya | Parser aturan; nama berkas masuk `llm_errors` | Template, dengan alasan di `sumber` |

Kandidat **sintetis** (eksperimen S1–S7) tidak pernah melewati LLM, sehingga
hasil eksperimen tetap dapat direproduksi.

### Keamanan dan data pribadi

- **Prompt injection**: CV adalah teks tak terpercaya. Baris yang menyerupai
  instruksi dinetralkan sebelum dikirim, prompt menegaskan bahwa isi `<cv>`
  adalah data, dan keluaran dibatasi ke taksonomi dan level yang sah. Deteksi
  injeksi di Screening Agent tetap berjalan pada teks mentah.
- **Tidak mengarang bukti**: dokumen kepatuhan hanya berasal dari
  `.meta.txt`, tidak pernah dari LLM.
- **Data pribadi**: parsing CV mengirim isi CV ke layanan DeepSeek. Untuk CV
  orang sungguhan, pastikan ada persetujuan pemiliknya. Narasi tidak mengirim
  teks CV.

### Pengujian

Uji LLM tidak pernah memanggil DeepSeek sungguhan. HTTP diganti balasan
palsu, dan fixture `_tanpa_deepseek` mengosongkan seluruh variabel
`DEEPSEEK_*`, termasuk yang dimuat dari `.env` lokal.

| Uji | Klaim |
|---|---|
| `test_narasi_tanpa_kunci_memakai_template` | Tanpa kunci, narasi jatuh ke template |
| `test_narasi_tidak_mengirim_teks_cv_ke_llm` | Teks CV tidak ikut terkirim |
| `test_narasi_gagal_jatuh_ke_template` | Gangguan jaringan tidak menggagalkan halaman |
| `test_job_description_bebas_tanpa_kunci_ditolak_jelas` | Pesan galat menunjuk solusinya |
| `test_job_description_bebas_diparsing_llm_dan_divalidasi` | Taksonomi, level, dan bobot dipaksakan; skill asing tercatat |
| `test_format_kunci_nilai_tidak_diserahkan_ke_llm` | Galat penulisan berkas terstruktur tetap terlihat |
| `test_ambiguitas_parsing_llm_sampai_ke_intake` | Skill asing sampai ke bukti HITL-1 |
| `test_cv_diparsing_llm_tersanitasi_dan_meta_diutamakan` | Sanitasi, prioritas `.meta.txt`, dokumen tidak dikarang |
| `test_cv_llm_gagal_tetap_terbaca_dengan_parser_aturan` | Kegagalan LLM tidak menggagalkan pembacaan folder |
| `test_env_dimuat_tanpa_menimpa_variabel_terminal` | `.env` terbaca; variabel terminal menang |
| `test_env_tidak_ada_tidak_galat` | Tanpa `.env` tetap berjalan |

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
| `job_requirement_reader.py` | pembaca lowongan dari .txt, atau teks bebas via LLM | — |
| `cv_reader.py` | pembaca CV dari .pdf/.txt + ekstraksi PDF | — |
| `deepseek_client.py` | klien HTTP DeepSeek (stdlib) + cache | — |
| `env_loader.py` | pemuat berkas `.env` (stdlib) | — |
| `llm_parsing.py` | parsing CV dan job description dengan DeepSeek | — |
| `llm_narrator.py` | narasi hasil kandidat dan run dengan DeepSeek | — |
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

- **LLM dalam pengambilan keputusan.** DeepSeek (opsional) hanya dipakai
  untuk parsing CV/job description dan narasi. Intake, Placement, skor, dan
  keputusan kepatuhan tetap deterministik, dan eksperimen S1–S7 tidak
  memakai LLM. Jangan klaim "agen berbasis LLM".
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
