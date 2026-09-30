"""Fitur screening dan model klasifikasi kelayakan kandidat.

KEJUJURAN (laporan 6.3): kelompok tidak memiliki data historis berlabel dari
recruiter. Model di sini dilatih pada DATA SINTETIS yang label emasnya kita
bangkitkan sendiri. Itu sah untuk membuktikan pipeline dan mengukur
kalibrasi, tetapi angka akurasinya tidak boleh diklaim berlaku pada CV nyata.

`RuleScorer` adalah fallback bila pelatihan tidak mungkin. Bila fallback yang
dipakai, laporan harus menyebutnya "skoring berbasis aturan + similarity",
bukan "model machine learning".
"""
from typing import Dict, List, Sequence, Tuple
import math

from .settings import LEVELS

FEATURE_NAMES = ["skill_coverage", "experience_ratio", "required_skill_hits",
                 "skill_count", "cv_richness"]


def _sigmoid(z: float) -> float:
    """Fungsi logistik yang stabil secara numerik untuk z negatif besar."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exponent = math.exp(z)
    return exponent / (1.0 + exponent)


def build_features(extracted_skills: Dict[str, str], experience_years: float,
                   job, cv_text: str) -> Dict[str, float]:
    """Susun vektor fitur dari hasil parsing CV terhadap satu lowongan.

    Args:
        extracted_skills: skill_id -> level, hasil parsing CV.
        experience_years: masa kerja yang berhasil dibaca.
        job: objek JobRequirement yang sedang dilamar.
        cv_text: teks CV yang sudah dinetralkan.

    Returns:
        Dict fitur bernilai [0,1]; semuanya ternormalisasi agar bobot model
        dapat dibandingkan langsung.
    """
    total_importance = sum(r["importance"] for r in job.required_skills) or 1.0
    covered, hits = 0.0, 0
    for requirement in job.required_skills:
        level = extracted_skills.get(requirement["skill_id"])
        if not level:
            continue
        hits += 1
        if LEVELS[level] >= LEVELS[requirement["min_level"]]:
            covered += requirement["importance"]
        else:
            covered += requirement["importance"] * 0.5     # kredit parsial
    return {
        "skill_coverage": covered / total_importance,
        "experience_ratio": min(1.0, experience_years
                                / max(0.5, job.min_experience_years)),
        "required_skill_hits": hits / max(1, len(job.required_skills)),
        "skill_count": min(1.0, len(extracted_skills) / 6.0),
        "cv_richness": min(1.0, len(cv_text) / 600.0),
    }


class RuleScorer:
    """Skoring heuristik transparan tanpa pelatihan.

    Keyakinan sengaja dibatasi <= 0.80 sehingga tidak pernah melewati
    gamma = 0.85. Artinya, tanpa model terlatih sistem otomatis lebih sering
    bertanya kepada manusia — dan itu perilaku yang benar.
    """

    version = "rule-v1"
    trained = False

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Kembalikan (skor, keyakinan) dari bobot heuristik tetap."""
        score = (0.50 * features["skill_coverage"]
                 + 0.25 * features["experience_ratio"]
                 + 0.15 * features["required_skill_hits"]
                 + 0.10 * features["cv_richness"])
        confidence = 0.55 + 0.25 * abs(score - 0.5) * 2
        return max(0.0, min(1.0, score)), min(0.80, confidence)


class LogisticScorer:
    """Regresi logistik murni Python dengan kalibrasi Platt."""

    version = "logreg-v1"

    def __init__(self, learning_rate: float = 0.5, epochs: int = 400,
                 l2: float = 1e-3):
        """Siapkan model kosong.

        Args:
            learning_rate: laju gradient descent.
            epochs: jumlah iterasi pelatihan.
            l2: koefisien regularisasi.
        """
        self.weights = {name: 0.0 for name in FEATURE_NAMES}
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.trained = False
        self._platt = (1.0, 0.0)
        self.expected_calibration_error = 1.0

    def _raw_score(self, features: Dict[str, float]) -> float:
        """Hitung logit mentah sebelum kalibrasi."""
        return sum(self.weights[k] * features[k] for k in FEATURE_NAMES) + self.bias

    def fit(self, X: Sequence[Dict[str, float]], y: Sequence[int],
            validation_fraction: float = 0.25) -> dict:
        """Latih model dan kalibrasi keluarannya.

        Kalibrasi wajib: tanpa itu, "confidence" hanyalah angka dan ambang
        gamma pada bounded autonomy menjadi tidak bermakna.

        Returns:
            Ringkasan pelatihan berisi ukuran data, bobot, dan ECE.

        Raises:
            ValueError: bila data latih kurang dari 20 sampel.
        """
        if len(X) < 20:
            raise ValueError("data latih terlalu sedikit untuk dilatih")
        split = int(len(X) * (1 - validation_fraction))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        for _ in range(self.epochs):
            gradients = {k: 0.0 for k in FEATURE_NAMES}
            bias_gradient = 0.0
            for features, label in zip(X_train, y_train):
                error = _sigmoid(self._raw_score(features)) - label
                for k in FEATURE_NAMES:
                    gradients[k] += error * features[k]
                bias_gradient += error
            n = len(X_train)
            for k in FEATURE_NAMES:
                self.weights[k] -= self.learning_rate * (
                    gradients[k] / n + self.l2 * self.weights[k])
            self.bias -= self.learning_rate * bias_gradient / n

        self.trained = True
        self._fit_platt(X_val, y_val)
        self.expected_calibration_error = self._compute_ece(X_val, y_val)
        return {"n_train": len(X_train), "n_validation": len(X_val),
                "weights": dict(self.weights), "bias": round(self.bias, 4),
                "ece": self.expected_calibration_error,
                "positive_rate": round(sum(y) / len(y), 3)}

    def _fit_platt(self, X_val, y_val, epochs: int = 300, lr: float = 0.3) -> None:
        """Cocokkan p = sigmoid(a*z + b) pada data validasi (Platt scaling)."""
        a, b = 1.0, 0.0
        for _ in range(epochs):
            grad_a = grad_b = 0.0
            for features, label in zip(X_val, y_val):
                z = self._raw_score(features)
                error = _sigmoid(a * z + b) - label
                grad_a += error * z
                grad_b += error
            n = max(1, len(X_val))
            a -= lr * grad_a / n
            b -= lr * grad_b / n
        self._platt = (a, b)

    def _compute_ece(self, X_val, y_val, bins: int = 5) -> float:
        """Hitung Expected Calibration Error: bukti confidence = probabilitas."""
        buckets: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
        for features, label in zip(X_val, y_val):
            probability, _ = self.predict(features)
            buckets[min(bins - 1, int(probability * bins))].append((probability, label))
        total = max(1, len(X_val))
        error = 0.0
        for bucket in buckets:
            if not bucket:
                continue
            mean_probability = sum(p for p, _ in bucket) / len(bucket)
            observed = sum(label for _, label in bucket) / len(bucket)
            error += (len(bucket) / total) * abs(mean_probability - observed)
        return round(error, 4)

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Kembalikan (probabilitas layak, keyakinan) yang sudah terkalibrasi.

        Keyakinan diturunkan dari jarak terhadap titik ragu 0.5, dipetakan ke
        rentang [0.5, 1.0].
        """
        a, b = self._platt
        probability = _sigmoid(a * self._raw_score(features) + b)
        confidence = 0.5 + abs(probability - 0.5)
        return probability, confidence
