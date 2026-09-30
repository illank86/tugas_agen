"""Klien minimal LLM DeepSeek, hanya memakai standard library.

Endpoint DeepSeek kompatibel dengan format chat completions OpenAI, sehingga
cukup satu POST HTTP; tidak perlu paket tambahan dan inti sistem tetap
berjalan tanpa dependensi.

Konfigurasi lewat variabel lingkungan:
    DEEPSEEK_API_KEY   kunci API (wajib untuk memakai LLM)
    DEEPSEEK_MODEL     default "deepseek-chat"
    DEEPSEEK_BASE_URL  default "https://api.deepseek.com"
"""
from typing import Any, Dict
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
TIMEOUT_SECONDS = 60

# Cache balasan JSON per isi prompt. Streamlit menjalankan ulang skrip pada
# setiap interaksi; tanpa cache, berkas yang sama diparsing berulang kali.
_json_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


class DeepSeekError(RuntimeError):
    """Panggilan DeepSeek gagal: kunci tidak ada, HTTP gagal, atau balasan rusak."""


def is_available() -> bool:
    """True bila DEEPSEEK_API_KEY terpasang sehingga LLM dapat dipanggil."""
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())


def model_name() -> str:
    """Nama model yang dipakai, untuk dicatat pada laporan dan audit."""
    return os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)


def chat(system_prompt: str, user_prompt: str, max_tokens: int = 600,
         json_mode: bool = False, temperature: float = 0.3) -> str:
    """Kirim satu percakapan ke DeepSeek dan kembalikan teks balasan.

    Args:
        json_mode: minta balasan berupa objek JSON (`response_format`).
            DeepSeek mensyaratkan kata "json" muncul di prompt.

    Raises:
        DeepSeekError: bila kunci tidak ada, HTTP gagal, atau balasan kosong.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise DeepSeekError("DEEPSEEK_API_KEY belum disetel")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    body: Dict[str, Any] = {
        "model": model_name(),
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json",
                                "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise DeepSeekError(f"DeepSeek HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise DeepSeekError(f"DeepSeek tidak dapat dihubungi: {error}") from error

    try:
        text = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise DeepSeekError(f"format balasan DeepSeek tidak dikenal: {payload}") from error
    if not text.strip():
        raise DeepSeekError("DeepSeek mengembalikan balasan kosong")
    return text.strip()


def chat_json(system_prompt: str, user_prompt: str,
              max_tokens: int = 1500) -> Dict[str, Any]:
    """Seperti `chat`, tetapi balasan diurai sebagai objek JSON dan di-cache.

    Temperature 0 dipakai karena ini ekstraksi, bukan tulisan kreatif.

    Raises:
        DeepSeekError: bila panggilan gagal atau balasan bukan objek JSON.
    """
    key = hashlib.sha256(
        f"{model_name()}\x00{system_prompt}\x00{user_prompt}".encode("utf-8")).hexdigest()
    with _cache_lock:
        if key in _json_cache:
            return _json_cache[key]
    text = chat(system_prompt, user_prompt, max_tokens=max_tokens,
                json_mode=True, temperature=0.0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise DeepSeekError(f"balasan DeepSeek bukan JSON valid: {text[:200]}") from error
    if not isinstance(data, dict):
        raise DeepSeekError("balasan DeepSeek bukan objek JSON")
    with _cache_lock:
        _json_cache[key] = data
    return data
