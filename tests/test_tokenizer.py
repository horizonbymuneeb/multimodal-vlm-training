"""Vocabulary tests: ordering, special tokens, padding and persistence."""

import pytest

from src.vlm.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    Vocab,
    build_vocab,
    decode_batch,
    pad_sequences,
    truncate,
)


def test_build_vocab_orders_by_frequency_then_alphabetically():
    captions = ["a red circle", "a red square", "a blue circle"]
    vocab = build_vocab(captions)
    assert vocab.itos[: len(SPECIAL_TOKENS)] == list(SPECIAL_TOKENS)
    assert vocab.itos[len(SPECIAL_TOKENS) :] == ["a", "circle", "red", "blue", "square"]
    assert len(vocab) == 4 + 5


def test_build_vocab_min_freq_filters_rare_words():
    vocab = build_vocab(["a a b c"], min_freq=2)
    assert vocab.itos[len(SPECIAL_TOKENS) :] == ["a"]
    with pytest.raises(ValueError):
        build_vocab(["a"], min_freq=0)


def test_special_ids_are_stable():
    vocab = build_vocab(["a b"])
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.unk_id == 3
    assert vocab.token_to_id(PAD_TOKEN) == 0
    assert vocab.token_to_id(UNK_TOKEN) == 3


def test_encode_decode_roundtrip_and_unknown_words():
    vocab = build_vocab(["a red circle"])
    ids = vocab.encode("A red circle")
    assert ids[0] == vocab.bos_id and ids[-1] == vocab.eos_id
    assert vocab.decode(ids) == "a red circle"
    assert vocab.token_to_id("zzz") == vocab.unk_id
    assert vocab.decode(vocab.encode("a zzz"), skip_special=False) == f"{BOS_TOKEN} a {UNK_TOKEN} {EOS_TOKEN}"


def test_encode_without_special_tokens():
    vocab = build_vocab(["a red circle"])
    assert vocab.encode("a red circle", add_special=False) == [
        vocab.token_to_id("a"),
        vocab.token_to_id("red"),
        vocab.token_to_id("circle"),
    ]


def test_vocab_validation():
    with pytest.raises(ValueError):
        Vocab(itos=["<pad>"])
    with pytest.raises(ValueError):
        Vocab(itos=[*SPECIAL_TOKENS, "a", "a"])
    with pytest.raises(IndexError):
        Vocab().id_to_token(99)


def test_pad_sequences_returns_mask():
    padded, mask = pad_sequences([[1, 2, 3], [4]], pad_id=0)
    assert padded == [[1, 2, 3], [4, 0, 0]]
    assert mask == [[1, 1, 1], [1, 0, 0]]
    with pytest.raises(ValueError):
        pad_sequences([])


def test_truncate_keeps_eos():
    ids = [1, 5, 6, 7, 8, 2]
    assert truncate(ids, 4) == [1, 5, 6, 2]
    assert truncate(ids, 10) == ids
    with pytest.raises(ValueError):
        truncate(ids, 1)


def test_save_and_load_roundtrip(tmp_path):
    vocab = build_vocab(["a red circle at center"])
    path = vocab.save(tmp_path / "vocab.json")
    loaded = Vocab.load(path)
    assert loaded.itos == vocab.itos
    with pytest.raises(FileNotFoundError):
        Vocab.load(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        Vocab.from_dict({"nope": []})
