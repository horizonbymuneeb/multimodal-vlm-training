"""A word-level vocabulary with explicit special tokens.

A pretrained tokenizer would mean a download, so the project uses a word-level
vocabulary built from the generated captions.  The vocabulary is an object with
stable ids rather than a dictionary floating around, because the padding id has
to be identical at training, evaluation and inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


@dataclass
class Vocab:
    """Bidirectional token <-> id mapping with the four special tokens first."""

    itos: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))

    def __post_init__(self) -> None:
        if len(self.itos) < len(SPECIAL_TOKENS):
            raise ValueError("vocabulary must contain the special tokens")
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("vocabulary contains duplicate tokens")
        self._stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self._stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._stoi[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return self.size

    def token_to_id(self, token: str) -> int:
        return self._stoi.get(token, self.unk_id)

    def id_to_token(self, index: int) -> str:
        if not 0 <= index < self.size:
            raise IndexError(f"token id {index} out of range for vocabulary of size {self.size}")
        return self.itos[index]

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        """Turn a caption into ids, wrapping it in ``<bos>``/``<eos>``."""
        tokens = [t for t in text.lower().strip().split() if t]
        ids = [self.token_to_id(token) for token in tokens]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Turn ids back into text, dropping padding and special tokens."""
        tokens = []
        for index in ids:
            token = self.id_to_token(int(index))
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def to_dict(self) -> dict:
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocab":
        if "itos" not in payload:
            raise ValueError("vocabulary payload needs an 'itos' list")
        return cls(itos=[str(t) for t in payload["itos"]])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no vocabulary at {path}")
        return cls.from_dict(json.loads(path.read_text()))


def build_vocab(captions: Iterable[str], *, min_freq: int = 1) -> Vocab:
    """Build a vocabulary from captions, ordered by descending frequency."""
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1")
    counts: dict[str, int] = {}
    for caption in captions:
        for token in caption.lower().split():
            counts[token] = counts.get(token, 0) + 1
    kept = [(token, n) for token, n in counts.items() if n >= min_freq]
    kept.sort(key=lambda pair: (-pair[1], pair[0]))
    return Vocab(itos=[*SPECIAL_TOKENS, *[token for token, _ in kept]])


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int = PAD_ID,
) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad id sequences and return ``(padded, mask)``."""
    if not sequences:
        raise ValueError("no sequences to pad")
    width = max(len(seq) for seq in sequences)
    padded: list[list[int]] = []
    mask: list[list[int]] = []
    for seq in sequences:
        if len(seq) > width:
            raise ValueError("sequence longer than the padded width")
        padded.append([*seq, *([pad_id] * (width - len(seq)))])
        mask.append([*([1] * len(seq)), *([0] * (width - len(seq)))])
    return padded, mask


def truncate(ids: Sequence[int], max_len: int, *, eos_id: int = EOS_ID) -> list[int]:
    """Keep ``max_len`` ids but always preserve the final ``<eos>``."""
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    if len(ids) <= max_len:
        return list(ids)
    return [*ids[: max_len - 1], eos_id]


def decode_batch(vocab: Vocab, batch: Sequence[Sequence[int]]) -> list[str]:
    return [vocab.decode(row) for row in batch]
