"""Builds the two code-vector files of the embedded clinical evidence: E-stat, GloVe on the co-occurrence of
codes in one patient's side, and E-ktx, the mean Bio_ClinicalBERT token embedding of the code's title."""
import duckdb
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

from src.config import DEVICE, SEED, SPLIT
from src.preprocessing.code_titles import CODE_TITLES_FILE
from src.utils.clinical_codes import ALL_MODALITIES, HISTORY, LATER, TOKEN, code_sets
from src.utils.experiment_data import load_patient_ids

STAT_FILE = SPLIT / "code_vectors_stat.npz"
KTX_FILE = SPLIT / "code_vectors_ktx.npz"
MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BATCH = 128
MIN_RECORDS = 3     # a code seen in fewer records gets no GloVe vector
N_TOP = 3           # principal directions removed by the postprocessing


def cooccurrence(records: sparse.csr_matrix, min_records: int = MIN_RECORDS):
    """(kept columns, i, j, count) of every pair i < j of codes that share a record; records is a binary records x codes matrix."""
    record_counts = np.asarray(records.sum(axis=0)).ravel()
    kept = np.flatnonzero(record_counts >= min_records)
    kept_records = records[:, kept]
    shared = sparse.triu(kept_records.T @ kept_records, k=1).tocoo()
    return kept, shared.row, shared.col, shared.data


def train_glove(n_codes: int, i: np.ndarray, j: np.ndarray, count: np.ndarray, dim: int = 100, epochs: int = 15,
                x_max: float = 50.0, alpha: float = 0.75, lr: float = 0.05, batch: int = 262_144) -> np.ndarray:
    """GloVe (Pennington et al. 2014) with AdaGrad: the sum of every code's word and context vector."""
    torch.manual_seed(SEED)
    word = torch.nn.Parameter(torch.randn(n_codes, dim) / dim)
    context = torch.nn.Parameter(torch.randn(n_codes, dim) / dim)
    word_bias = torch.nn.Parameter(torch.zeros(n_codes))
    context_bias = torch.nn.Parameter(torch.zeros(n_codes))
    optimiser = torch.optim.Adagrad([word, context, word_bias, context_bias], lr=lr)

    i = torch.as_tensor(i)
    j = torch.as_tensor(j)
    count = torch.as_tensor(count.astype(np.float32))
    weight = torch.clamp(count / x_max, max=1) ** alpha
    target = torch.log(count)
    for _ in range(epochs):
        for rows in torch.randperm(len(count)).split(batch):
            a = i[rows]
            b = j[rows]
            error = (word[a] * context[b]).sum(dim=1) + word_bias[a] + context_bias[b] - target[rows]
            optimiser.zero_grad()
            (weight[rows] * error ** 2).mean().backward()
            optimiser.step()
    return (word + context).detach().numpy()


def all_but_the_top(vectors: np.ndarray, n_top: int = N_TOP) -> np.ndarray:
    """Centre, remove the top principal directions (Mu & Viswanath 2018), L2-normalise each row. Without it the cosines
    of unrelated codes fall in a narrow band far above zero."""
    centred = vectors - vectors.mean(axis=0)
    _, _, directions = np.linalg.svd(centred, full_matrices=False)
    top = directions[:n_top]
    return normalize(centred - centred @ top.T @ top)


def stat_vectors(subject_ids) -> tuple[np.ndarray, np.ndarray]:
    """(tokens, vectors) of the codes seen in at least MIN_RECORDS records."""
    history = code_sets(HISTORY, subject_ids, ALL_MODALITIES, SPLIT)
    later = code_sets(LATER, subject_ids, ALL_MODALITIES, SPLIT)
    # one record per patient and side: the later side's rows come after the history rows
    records = pd.concat([history, later.assign(row=later["row"] + len(subject_ids))])
    tokens, codes = np.unique(records["token"], return_inverse=True)
    values = np.ones(len(records))
    matrix = sparse.csr_matrix((values, (records["row"], codes)), shape=(2 * len(subject_ids), len(tokens)))
    kept, i, j, count = cooccurrence(matrix)
    return tokens[kept], all_but_the_top(train_glove(len(kept), i, j, count))


def titles() -> pd.DataFrame:
    """(token, title) of every code of the two sides; the code itself where there is no title."""
    selects = []
    for side in (HISTORY, LATER):
        selects.append(f"SELECT DISTINCT modality, vocabulary, code FROM '{(SPLIT / f'{side}.parquet').as_posix()}'")
    sides = " UNION ".join(selects)
    return duckdb.sql(f"""
        SELECT {TOKEN} AS token, coalesce(title, code) AS title
        FROM ({sides}) LEFT JOIN '{CODE_TITLES_FILE.as_posix()}' USING (modality, vocabulary, code) ORDER BY token
    """).df()


@torch.no_grad()
def ktx_vectors() -> tuple[np.ndarray, np.ndarray]:
    """(tokens, vectors) of every code: the mean over the title's tokens of the last hidden layer."""
    codes = titles()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    model.to(DEVICE)
    batches = []
    for start in range(0, len(codes), BATCH):
        batch_titles = codes["title"][start:start + BATCH].tolist()
        inputs = tokenizer(batch_titles, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        mask = inputs.attention_mask.unsqueeze(-1)
        hidden = model(**inputs).last_hidden_state
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean over the real tokens, padding left out
        batches.append(mean.cpu().numpy())
    return codes["token"].to_numpy(), all_but_the_top(np.concatenate(batches))


def _save(file, tokens: np.ndarray, vectors: np.ndarray) -> None:
    np.savez(file, tokens=tokens.astype(str), vectors=vectors)
    print(f"{file.name}: {len(tokens):,} codes, dimension {vectors.shape[1]}")


def main() -> None:
    tokens, vectors = stat_vectors(load_patient_ids())
    _save(STAT_FILE, tokens, vectors)
    tokens, vectors = ktx_vectors()
    _save(KTX_FILE, tokens, vectors)


if __name__ == "__main__":
    main()
