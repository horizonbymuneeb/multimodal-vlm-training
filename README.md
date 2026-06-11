# multimodal-vlm-training

A small vision-language model that trains end to end on a laptop CPU: a frozen
convolutional vision encoder, a projection head that maps image features into
the text embedding space, a one-layer text decoder adapted with LoRA, and two
objectives trained together — a hand-written InfoNCE contrastive loss between
images and captions, and a teacher-forced captioning cross-entropy. The data is
a synthetic scene grammar (a shape, a colour, a position, sometimes a second
shape) rendered as small RGB images with generated captions, so nothing is
downloaded and the whole suite runs in seconds.

The project exists to make the *seam* between the two modalities explicit and
testable: which parameters are frozen, where the two embedding spaces meet, what
the adapter actually changes, and what retrieval@k and token accuracy look like
when those choices are right versus wrong.

## What is in the box

| Module | Job |
| --- | --- |
| `src/vlm/tokenizer.py` | Word-level `Vocab` with stable special-token ids, encode/decode, right-padding with a mask, truncation that keeps `<eos>`, JSON persistence. |
| `src/vlm/dataset.py` | Synthetic scene grammar and renderer, caption text generator, `SyntheticCaptionDataset`, caption collation with padding, seeded `DataLoader` factory. |
| `src/vlm/vision.py` | `ConvStem` (three stride-2 blocks), `VisionEncoder` with pooled projection and LayerNorm, freezing/unfreezing helpers, gradient-free `encode_images`. |
| `src/vlm/text.py` | Causal mask, sinusoidal positions, explicit scaled dot-product attention, self- plus cross-attention `DecoderBlock`, `TextDecoder` returning logits and attention weights. |
| `src/vlm/projection.py` | Two-layer `ProjectionHead` with L2-normalised output, cosine similarity matrix, alignment term, parameter accounting. |
| `src/vlm/lora.py` | `LoRALinear` (frozen base + zero-initialised low-rank update), `apply_lora` by module name, trainability masking, merge/unmerge, adapter state dict. |
| `src/vlm/losses.py` | Temperature-scaled InfoNCE, symmetric InfoNCE, padding-aware caption cross-entropy, token accuracy, loss mixture and per-step report. |
| `src/vlm/model.py` | `VLMConfig` and `SmallVLM`: frozen vision → projection → cross-attention memory → LoRA decoder; parameter summary, checkpoint save/load. |
| `src/vlm/evaluate.py` | Rank matrix with explicit targets, retrieval@k, MRR, token accuracy, loader embedding pass, bootstrap confidence interval for tiny validation sets. |
| `src/vlm/train.py` | `TrainConfig`, seeded experiment builder, one optimisation step (contrastive + captioning), epoch loop, config-driven CLI. |
| `src/vlm/demo.py` | One-command report: data sample, parameter split, untrained losses, short training run, validation metrics. |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

make test                                    # pytest -q
python -m src.vlm.demo --config config.yaml  # dataset + params + a short run
python -m src.vlm.train --config config.yaml --out artifacts/run.json
```

`config.yaml` holds the real tunables: image size and sample counts, vision and
text widths, decoder depth and head count, LoRA rank/alpha/targets, and the
training schedule (epochs, batch size, lr, temperature, caption weight).

## Design notes

**The projection head is the only bridge, so it is the only thing that has to be
right.** It maps the pooled vision feature into the *text embedding width* and
L2-normalises its output. That one decision means the contrastive logits, the
retrieval metric and the cross-attention memory all live in the same geometry:
there is no second "alignment" space to keep in sync. The loss also re-normalises
defensively, so a caller cannot accidentally train against unnormalised vectors.

**LoRA targets `q_proj` and `v_proj`, on both self- and cross-attention.** Query
and value projections are where the adapter changes *what the decoder looks at*
and *what it copies*; adapting the output projection as well mostly adds
parameters. Both attention paths are adapted because the captioning objective
needs the cross-attention to change, not just the language modelling. `lora_b` is
initialised to zeros, so step 0 is byte-identical to the frozen model — the tests
assert that equality, because a non-zero init would silently move the baseline
that every experiment is compared against. `merge_lora` folds `B @ A` into the
frozen weight and `unmerge` puts it back; both are tested against the unmerged
forward pass.

**The toy config is CPU-trainable because only two small parameter groups are
trained.** The vision tower and the decoder's own weights never receive
gradients: the optimiser sees the projection head and the adapters (tens of
thousands of parameters, not millions). The decoder is one block wide with a
16-wide head for tests and 48-wide for the shipped config, images are 16×16, and
the vocabulary is a few dozen words. A full train + evaluate pass in the demo
takes a couple of seconds, which is what makes it reasonable to run the training
loop inside the test suite.

**Retrieval counts ties as hits.** `rank_matrix` counts gallery items with a
*strictly* higher similarity than the correct one, and accepts explicit targets
so a gallery may be larger than the batch. With duplicates in a gallery (or a
bootstrap resample, which duplicates by construction), a correct answer that
ties for first should not be reported as a miss; the tests pin both the
distractor case and the identity mapping case.

## Tests

`python -m pytest -q` runs **103 tests in about 5 s** on CPU. Coverage by file:

- `tests/test_tokenizer.py` — frequency-then-alphabetical vocabulary order,
  special-token ids, encode/decode round-trip, unknown words, padding masks,
  truncation that preserves `<eos>`, JSON round-trip.
- `tests/test_dataset.py` — pixels land in the specified colour/position for
  every shape and position, second-shape changes the render, caption grammar,
  spec validation, item shapes, collation padding/mask, vocabulary coverage (no
  unknown tokens), caption length budget.
- `tests/test_vision.py` — stride-2 downsampling, output width, frozen-by-default
  parameter counts, unfreeze behaviour, gradient-free inference, seeded
  reproducibility.
- `tests/test_text.py` — causal mask is lower-triangular, positional encoding
  values, attention weights are zero for future positions, cross-attention
  consumes the memory, logits shape, `max_len` and config validation.
- `tests/test_projection.py` — unit-norm outputs, zero-row safety, similarity
  matrix properties, alignment loss exact values, seeded reproducibility,
  parameter counts.
- `tests/test_lora.py` — zero-init equals the base layer, only adapters are
  trainable, the update matches the hand formula, merge/unmerge preserve
  outputs, replacement count, adapter state dict.
- `tests/test_losses.py` — aligned pairs give near-zero InfoNCE while shuffled
  pairs do not, symmetric loss equals the average of both directions, caption
  cross-entropy matches a manual shift, padding is ignored, smoothing changes the
  value, loss mixture arithmetic.
- `tests/test_model.py` — forward shapes, vision tower frozen, only projection
  and adapters trainable, parameter summary adds up, checkpoint round-trip
  reproduces logits exactly, padding-aware text pooling.
- `tests/test_evaluate.py` — identity ranking, distractor ranking, retrieval@k,
  MRR, token accuracy with padding, `evaluate_all` key set, bootstrap interval
  brackets the estimate.
- `tests/test_train.py` — config validation, seeded determinism, loss components
  sum correctly, frozen weights do not change during a step, a two-epoch run
  lowers the loss, experiment builder is reproducible, no-trainable-parameters
  error.
- `tests/test_demo.py` — every demo section runs, the CLI writes JSON, the
  repository config produces the expected data shape.

## Limitations

- The scene grammar is tiny and synthetic: five colours, three shapes, five
  positions. A model trained here learns this grammar, not real image-text data.
- The vocabulary is word-level with no subword handling; unseen words map to
  `<unk>` and captions are restricted to the template.
- The decoder is one layer in the shipped config and two at most in tests; there
  is no generation loop, only teacher-forced scoring (no beam search, no
  sampling).
- Retrieval is evaluated on small validation sets, so point estimates are noisy;
  the bootstrap helper reports a wide interval rather than a single number.
- LoRA targets are a fixed list of module names, not a regex or a learned
  allocation.
- CPU-only by design: there is no distributed, mixed-precision or checkpointing
  path.
