# enwik8-scaling-laws

Character-level language models on enwik8, compared under equal FLOP budgets
rather than equal epochs.

**Status:** in progress. Baselines done, training pipeline validated.

## Question

At a fixed compute budget, which architecture reaches the lowest
bits-per-character, and how does that change as the budget grows? Then fit a
power law, predict an untrained model size, train it, report the error.

## Method

- Standard enwik8 90/5/5 split, byte-level, vocab 205.
- Compute-matched comparisons using an explicit FLOP model per architecture.
- Learning rate swept independently at every model size.
- Numbers are means over multiple seeds, spread shown.
- No pretrained weights, no tokenizer library, no training framework.

Metric is bits per character, i.e. cross-entropy in base 2. PyTorch reports
nats, so everything is divided by ln(2).

## Results

Validation split, fixed evaluation seed.

| Model | val BPC |
|---|---|
| Uniform | 7.6795 |
| Unigram (counted) | 5.0658 |
| Bigram (counted) | _pending_ |
| Trigram (counted) | _pending_ |
| Bias-only (trained) | 5.0668 |

The uniform model matches log2(205) exactly, confirming the vocab remapping
and log base. The bias-only model has no inputs, so its optimum is the
training-set character frequencies; it converges to within 3.9e-3 of them,
validating the data pipeline, loss and training loop against a known answer.

## Reproducing

Requires `numpy` and `torch`.

```bash
mkdir -p data && cd data
curl -O http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip && cd ..

python src/prepare.py
python src/baseline_unigram.py
python src/baseline_ngram.py
python src/train_bias.py
```

## Conventions

- `test.bin` is sealed. All tuning happens on validation.
- Every measurement goes in `results/log.md`, including the ones that lost.
- Evaluation uses a fixed seed so the same windows are scored every time.