"""Representasi teks dan pengukuran kemiripan semantik.

Default `HashingEmbedder` memakai standard library saja supaya repositori
dapat dijalankan tanpa unduhan model. Antarmuka sengaja dibuat agar dapat
ditukar dengan embedding neural tanpa mengubah agen mana pun.

KEJUJURAN: ini BUKAN embedding neural. Jangan tulis "memakai Sentence-BERT"
di laporan kecuali `SentenceTransformerEmbedder` benar-benar dipakai.
"""
from typing import Dict, List
import math
import re
import zlib

Vector = Dict[int, float]


def _tokenize(text: str, ngram: int = 3) -> List[str]:
    """Pecah teks menjadi token kata + karakter n-gram.

    Karakter n-gram membuat "Excel" dan "excell" tetap berdekatan, sesuatu
    yang tidak ditangkap oleh pencocokan kata saja.
    """
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    words = re.sub(r"\s+", " ", cleaned).strip().split()
    grams = list(words)
    for word in words:
        padded = f" {word} "
        grams += [padded[i:i + ngram] for i in range(len(padded) - ngram + 1)]
    return grams


class HashingEmbedder:
    """Vektor TF ter-hash dan ter-normalisasi L2, tanpa dependensi eksternal."""

    def __init__(self, dim: int = 4096, ngram: int = 3):
        """Siapkan embedder.

        Args:
            dim: dimensi ruang hash. Makin besar makin kecil tabrakan.
            ngram: panjang karakter n-gram.
        """
        self.dim = dim
        self.ngram = ngram
        self._cache: Dict[str, Vector] = {}

    @property
    def name(self) -> str:
        """Identitas model, dicatat bersama setiap keputusan untuk audit."""
        return f"hashing-{self.dim}d-{self.ngram}gram"

    def embed(self, text: str) -> Vector:
        """Ubah teks menjadi vektor jarang ternormalisasi.

        Memakai zlib.crc32, bukan hash() bawaan, karena hash() string diacak
        oleh PYTHONHASHSEED dan akan merusak reproduktibilitas antar-proses.
        """
        if text in self._cache:
            return self._cache[text]
        vector: Vector = {}
        for gram in _tokenize(text, self.ngram):
            index = zlib.crc32(gram.encode("utf-8")) % self.dim
            vector[index] = vector.get(index, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        vector = {k: v / norm for k, v in vector.items()}
        if len(self._cache) < 20000:
            self._cache[text] = vector
        return vector


class SentenceTransformerEmbedder:
    """Pembungkus sentence-transformers; aktif hanya bila paketnya terpasang."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Muat model; melempar ImportError bila paket tidak tersedia."""
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def name(self) -> str:
        """Identitas model untuk audit."""
        return f"sentence-transformers:{self._model_name}"

    def embed(self, text: str) -> Vector:
        """Kembalikan embedding neural ternormalisasi sebagai dict jarang."""
        values = self._model.encode(text, normalize_embeddings=True)
        return {i: float(x) for i, x in enumerate(values)}


def cosine(a: Vector, b: Vector) -> float:
    """Hitung cosine similarity dua vektor yang sudah ternormalisasi L2."""
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


def get_embedder(prefer_neural: bool = False):
    """Pilih embedder yang tersedia.

    Args:
        prefer_neural: bila True, coba sentence-transformers lebih dulu dan
            jatuh kembali ke HashingEmbedder bila paketnya tidak ada.
    """
    if prefer_neural:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashingEmbedder()
