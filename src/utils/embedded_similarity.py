"""Embedded clinical similarity: the IDF-weighted best match (BM) of two code sets.

    BM(a -> b) = sum_{i in a} idf(i) * max_{j in b} cos(e_i, e_j) / sum_{i in a} idf(i),   BM(a, b) = (BM(a -> b) + BM(b -> a)) / 2
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import DEVICE, SPLIT
from src.utils.clinical_codes import BANDS, CATEGORIES, PERCENTILES, CodeSets, idf_weighted_codes

JOBS = 1_000            # later records of one history record that form one piece
CELLS = 2 ** 21         # codes of a x codes of b compared in one batch on the device (the fastest on an RTX 5060 Ti)
B_CODES = 2 ** 22       # codes of the later records in one batch, a bound for batches of history records without codes
CACHE_SIZE = 2          # pair sets whose scores are kept: the candidate pairs are scored for their cuts and their levels


def _to_device(array: np.ndarray, dtype) -> torch.Tensor:
    return torch.as_tensor(array.astype(dtype), device=DEVICE)


@dataclass(frozen=True)
class _Rows:
    """The rows of a CSR matrix on the device, so the codes of many records are gathered without going back to the CPU."""
    indptr: torch.Tensor
    indices: torch.Tensor
    data: torch.Tensor
    lengths: np.ndarray     # codes per row, kept on the CPU so the size of a gather is known without a device round trip

    @classmethod
    def of(cls, matrix) -> "_Rows":
        return cls(_to_device(matrix.indptr, np.int64), _to_device(matrix.indices, np.int64),
                   _to_device(matrix.data, np.float32), np.diff(matrix.indptr))

    def gather(self, rows: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(column, IDF weight, owner) of the `total` codes of `rows`; owner = position of the code's row in `rows`."""
        total = int(total)
        lengths = self.indptr[rows + 1] - self.indptr[rows]
        owner = torch.repeat_interleave(torch.arange(len(rows), device=DEVICE), lengths, output_size=total)
        # position of each code inside its own row, added to where that row starts
        row_offset = (torch.cumsum(lengths, 0) - lengths)[owner]
        position = self.indptr[rows][owner] + torch.arange(total, device=DEVICE) - row_offset
        return self.indices[position], self.data[position], owner


def _sum_per_pair(values: torch.Tensor, pair: torch.Tensor, n_pairs: int) -> torch.Tensor:
    return torch.zeros(n_pairs, device=DEVICE).index_add_(0, pair, values)


@dataclass(frozen=True)
class EmbeddedSimilarity(CodeSets):
    """history and later are IDF-weighted (not normalised). cosine is the codes x codes cosine of the code vectors,
    computed once, so comparing two codes is a lookup. A code without a vector is similar only to itself."""
    LEVEL_NAMES = ("missing",) + BANDS
    cosine: torch.Tensor
    history_rows: _Rows
    later_rows: _Rows
    cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_vectors(cls, subject_ids, modalities, tokens, vectors, split_dir=SPLIT) -> "EmbeddedSimilarity":
        """tokens, vectors: the code vectors (row k of `vectors` belongs to tokens[k]); only this category's codes are used."""
        columns, history, later = idf_weighted_codes(subject_ids, modalities, split_dir)
        table = np.zeros((len(columns), vectors.shape[1]), np.float32)
        positions = columns.get_indexer(tokens)
        in_category = positions >= 0
        table[positions[in_category]] = vectors[in_category]
        table = torch.as_tensor(table, device=DEVICE)
        cosine = table @ table.T
        return cls(history, later, cosine, _Rows.of(history), _Rows.of(later))

    def scores(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """BM of history record i and later record j, for every pair. The last CACHE_SIZE results are kept."""
        key = hash((i.tobytes(), j.tobytes()))
        if key not in self.cache:
            if len(self.cache) >= CACHE_SIZE:
                oldest_key = next(iter(self.cache))  # a dict keeps insertion order
                self.cache.pop(oldest_key)
            self.cache[key] = self._scores(i, j)
        return self.cache[key]

    def cuts(self, i: np.ndarray, j: np.ndarray) -> tuple[float, ...]:
        """The PERCENTILES of the BM of the pairs that have codes on both sides."""
        scores = self.scores(i, j)
        with_codes = ~self.missing(i, j)
        return tuple(np.percentile(scores[with_codes], PERCENTILES))

    def _scores(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        # The pairs are sorted by history record and cut into pieces of one history record and at most JOBS later
        # records. As many consecutive pieces as fit into CELLS (and B_CODES) are scored in one batch, so the device
        # does a few large operations instead of a small one per record.
        order = np.argsort(i, kind="stable")
        history = i[order]
        later = j[order]
        n_pairs = len(history)

        run_start = np.flatnonzero(np.concatenate(([True], history[1:] != history[:-1])))
        run_length = np.diff(np.concatenate((run_start, [n_pairs])))
        position_in_run = np.arange(n_pairs) - np.repeat(run_start, run_length)
        piece_start = np.flatnonzero(position_in_run % JOBS == 0)
        piece_stop = np.concatenate((piece_start[1:], [n_pairs]))
        n_pieces = len(piece_start)

        # cumulative number of codes: a_codes and b_codes per piece, later_codes per pair
        a_codes = np.concatenate(([0], np.cumsum(self.history_rows.lengths[history[piece_start]])))
        later_codes = np.concatenate(([0], np.cumsum(self.later_rows.lengths[later])))
        b_codes = later_codes[np.concatenate((piece_start, [n_pairs]))]

        piece_records = torch.as_tensor(history[piece_start], device=DEVICE)
        later_on_device = torch.as_tensor(later, device=DEVICE)
        piece_of_pair = torch.as_tensor(np.repeat(np.arange(n_pieces), piece_stop - piece_start), device=DEVICE)

        parts = []
        first = 0
        while first < n_pieces:
            last = first + 1
            while last < n_pieces:
                a_total = a_codes[last + 1] - a_codes[first]
                b_total = b_codes[last + 1] - b_codes[first]
                if b_total > B_CODES or a_total * b_total > CELLS:
                    break
                last += 1
            pairs = slice(piece_start[first], piece_stop[last - 1])
            parts.append(self._best_match(piece_records[first:last], a_codes[last] - a_codes[first],
                                          later_on_device[pairs], b_codes[last] - b_codes[first],
                                          piece_of_pair[pairs] - first))
            first = last

        result = np.empty(len(i))
        if parts:
            result[order] = torch.cat(parts).cpu().numpy()  # one transfer back at the end
        return result

    def _best_match(self, records, a_total: int, later, b_total: int, piece_of_pair) -> torch.Tensor:
        """BM of every pair of one batch. `records` are the pieces' history records, `later` the pairs' later records,
        piece_of_pair the piece of each pair. All codes are compared at once and codes of different pieces are masked."""
        a_codes, a_idf, a_piece = self.history_rows.gather(records, a_total)
        b_codes, b_idf, b_pair = self.later_rows.gather(later, b_total)
        n_pairs = len(later)

        # codes of a x codes of b; a code is always fully similar to itself, even without a vector
        similarity = self.cosine[a_codes][:, b_codes]
        similarity = torch.maximum(similarity, (a_codes[:, None] == b_codes[None, :]).float())
        same_piece = a_piece[:, None] == piece_of_pair[b_pair][None, :]
        similarity = torch.where(same_piece, similarity, -torch.inf)

        # a -> b: for each code of a, its best match within each pair's b
        own = a_piece[:, None] == piece_of_pair[None, :]  # codes of a x pairs
        best_in_b = torch.zeros(len(a_codes), n_pairs, device=DEVICE).scatter_reduce(
            1, b_pair.expand(len(a_codes), -1), similarity, "amax", include_self=False)
        a_to_b = ((a_idf[:, None] * torch.where(own, best_in_b, 0)).sum(dim=0)
                  / (a_idf[:, None] * own).sum(dim=0).clamp(min=1e-9))

        # b -> a: for each code of b, its best match in a
        if len(a_codes):
            best_in_a = similarity.max(dim=0).values
        else:
            best_in_a = torch.zeros(len(b_codes), device=DEVICE)
        best_in_a = torch.where(torch.isneginf(best_in_a), 0, best_in_a)  # a history record without codes
        b_to_a = _sum_per_pair(b_idf * best_in_a, b_pair, n_pairs) / _sum_per_pair(b_idf, b_pair, n_pairs).clamp(min=1e-9)
        return (a_to_b + b_to_a) / 2


def embedded_similarities(subject_ids, vectors_file, split_dir=SPLIT) -> dict[str, EmbeddedSimilarity]:
    """One EmbeddedSimilarity per category, from a file written by preprocessing/code_vectors.py."""
    similarities = {}
    with np.load(vectors_file) as stored:
        tokens = stored["tokens"]
        vectors = stored["vectors"]
        for category, modalities in CATEGORIES.items():
            similarities[category] = EmbeddedSimilarity.from_vectors(subject_ids, modalities, tokens, vectors, split_dir)
    return similarities
