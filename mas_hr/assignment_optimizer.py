"""Optimasi penugasan kandidat ke slot lowongan (laporan 6.5.3).

Perusahaan outsourcing melayani banyak klien bersamaan. Penugasan greedy
per-lowongan ("siapa cepat dia dapat") menghasilkan solusi sub-optimal secara
global: kandidat terbaik diambil lowongan pertama padahal jauh lebih bernilai
di lowongan kedua. Ini konflik K4 pada laporan, dan ia punya solusi eksak
berkompleksitas polinomial — jauh lebih tepat daripada menyerahkannya ke LLM.
"""
from typing import Dict, List, Sequence, Tuple


def solve_min_cost_assignment(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """Cari penugasan berbiaya minimum dengan algoritma Jonker-Volgenant.

    Args:
        cost: matriks biaya berukuran n x m dengan n <= m.

    Returns:
        Daftar pasangan (indeks_baris, indeks_kolom) yang meminimumkan total
        biaya.

    Raises:
        ValueError: bila jumlah baris melebihi jumlah kolom; pemanggil yang
            bertanggung jawab melakukan padding.
    """
    n_rows, n_cols = len(cost), len(cost[0])
    if n_rows > n_cols:
        raise ValueError("jumlah baris harus <= jumlah kolom (lakukan padding)")

    infinity = float("inf")
    potential_row = [0.0] * (n_rows + 1)
    potential_col = [0.0] * (n_cols + 1)
    match_of_col = [0] * (n_cols + 1)
    path = [0] * (n_cols + 1)

    for row in range(1, n_rows + 1):
        match_of_col[0] = row
        current_col = 0
        minima = [infinity] * (n_cols + 1)
        visited = [False] * (n_cols + 1)
        while True:
            visited[current_col] = True
            current_row = match_of_col[current_col]
            delta, next_col = infinity, -1
            for col in range(1, n_cols + 1):
                if visited[col]:
                    continue
                reduced = (cost[current_row - 1][col - 1]
                           - potential_row[current_row] - potential_col[col])
                if reduced < minima[col]:
                    minima[col], path[col] = reduced, current_col
                if minima[col] < delta:
                    delta, next_col = minima[col], col
            for col in range(n_cols + 1):
                if visited[col]:
                    potential_row[match_of_col[col]] += delta
                    potential_col[col] -= delta
                else:
                    minima[col] -= delta
            current_col = next_col
            if match_of_col[current_col] == 0:
                break
        while True:
            previous = path[current_col]
            match_of_col[current_col] = match_of_col[previous]
            current_col = previous
            if current_col == 0:
                break
    return [(match_of_col[col] - 1, col - 1)
            for col in range(1, n_cols + 1) if match_of_col[col] != 0]


def assign_candidates_to_slots(score_matrix: Dict[str, Dict[str, float]],
                               slots: List[str]) -> List[Tuple[str, str, float]]:
    """Tugaskan kandidat ke slot lowongan sehingga total kecocokan maksimum.

    Args:
        score_matrix: score_matrix[candidate_id][slot_id] -> fit score.
        slots: daftar slot, satu entri per posisi yang harus diisi.

    Returns:
        Daftar (candidate_id, slot_id, fit) terurut menurun. Slot dummy hasil
        padding dan pasangan berskor nol dibuang.
    """
    candidates = sorted(score_matrix)
    if not candidates or not slots:
        return []
    columns = list(slots)
    if len(candidates) > len(columns):
        columns += [f"__dummy-{i}" for i in range(len(candidates) - len(columns))]

    # Maksimasi fit = minimasi biaya negatifnya.
    cost = [[0.0 if column.startswith("__dummy")
             else -score_matrix[candidate].get(column, 0.0)
             for column in columns] for candidate in candidates]

    assignments: List[Tuple[str, str, float]] = []
    for row_index, column_index in solve_min_cost_assignment(cost):
        if row_index >= len(candidates) or columns[column_index].startswith("__dummy"):
            continue
        candidate, slot = candidates[row_index], columns[column_index]
        fit = score_matrix[candidate].get(slot, 0.0)
        if fit > 0:
            assignments.append((candidate, slot, round(fit, 4)))
    return sorted(assignments, key=lambda item: -item[2])


def assign_greedy(score_matrix: Dict[str, Dict[str, float]],
                  slots: List[str]) -> List[Tuple[str, str, float]]:
    """Penugasan greedy first-come; disediakan sebagai pembanding.

    Dipakai untuk menunjukkan selisih terhadap solusi optimal, bukan untuk
    dipakai di alur produksi.
    """
    taken, result = set(), []
    for candidate in sorted(score_matrix):
        available = [(slot, score_matrix[candidate].get(slot, 0.0))
                     for slot in slots if slot not in taken]
        if not available:
            break
        slot, fit = max(available, key=lambda item: item[1])
        if fit > 0:
            taken.add(slot)
            result.append((candidate, slot, round(fit, 4)))
    return result
