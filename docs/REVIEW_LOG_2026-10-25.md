# MAVIS Review Log — 2026-10-25

Record of a full-codebase review: what was found, what was measured, what was changed.

**Ground rules this log was written under.** Every number below was read out of a file or
produced by a script in the session that wrote this log. Where the documentation and the code
disagree, the code is treated as correct and the disagreement is written down. Nothing was
committed or pushed.

**All three primary findings have now been fixed**, on request, later the same day. Part 1
describes each bug as it was found and measured; Part 1b records the patch that was applied and
the verification it passed. The secondary issues listed at the end of Part 1 were left alone.

Because the fixes change the model, the published M3 and M4 result files no longer describe the
code in this tree and have to be re-run. See "Retraining implications" at the end.

---

## Part 1 — Issues that are holding performance down

Three findings are primary, in the sense that each one was traced to a specific line and
measured. Everything after them is secondary: real, but smaller or not yet quantified.

### Finding A — Gaussian prototype replay is training on noise

**Severity: this is the largest single performance loss found.**

What happens:

```
  +---------------------------------------------------------------+
  |  stored prototype          length 1.00   (unit vector)        |
  |  noise added to it         length 9.35                        |
  |                                                               |
  |  prototype + noise  ->  normalise  ->  a direction that is    |
  |                                        90% noise              |
  +---------------------------------------------------------------+
```

The memory bank stores prototypes as **unit vectors**: the K-means helper ends with a
normalisation at `models/memory.py:43`, and the result is written straight into the bank at
`models/memory.py:124`.

The per-class variances stored alongside them are computed one line later, at
`models/memory.py:126`, from the **raw, un-normalised** encoder features. Those features leave
the encoder through a LayerNorm over 512 dimensions, so they have length about 22.6, not 1.

Both loss terms that replay prototypes draw their noise from those variances:

- the replay term inside the main training loop, `continual/mavis_trainer.py:265`
- the post-task balanced head finetune, `continual/mavis_trainer.py:449`

So noise sized for length-22.6 features is added to length-1.0 prototypes.

Measured, from the task-3 checkpoint of the width-32 review run (600 past prototypes):

| Quantity | Measured |
|---|---|
| Stored prototype length | 1.0000 |
| Replay noise length | 9.3519 |
| Ratio | noise is 9.4x the prototype |
| Cosine between a draw and its own prototype | 0.1019 |
| Nearest-prototype oracle accuracy on the draws | 2.67% (chance 1.67%) |
| Same oracle, noise rescaled into the prototypes' own space | 97.83% |

The oracle line is the important one. A nearest-prototype lookup is the most generous
classifier possible for this data, and on the draws as currently generated it scores 2.67%
against a chance level of 1.67%. The class label is almost entirely gone. Rescale the noise so
it lives in the same space as the prototypes and the same oracle scores 97.83%.

Consequences that were also measured:

- **The replay loss is pinned at the guessing floor.** Running the trained head on these draws
  gives a cross-entropy of 3.9856. For 60 classes, pure guessing is 4.0943. The term has
  essentially no reducible signal in it, and it carries a weight of 2.0, so it contributes
  about 7.97 to a total training loss that was logged at 17.09 at the start of task 4. Close to
  half the loss being minimised is a quantity that cannot be minimised.
- **The balanced head finetune cannot work.** It reported 3.97% rising to 5.30% over 25 epochs.
  The head's accuracy on the draws measures 5.17%, so 5.30% was not a training failure; it was
  the ceiling of the data it was given.
- **The head never learns to tell tasks apart.** The main cross-entropy is task-masked, by
  design, so cross-task discrimination is supposed to come from exactly two places: the replay
  term and the balanced finetune. Both are the ones above.

One consistency check, not a proof: in the published results the nearest-class-mean predictor
is the only one that never touches the head or the replay path, and it is also the only
predictor EWC did not move (−0.80 points, interval spanning zero) while the head moved +9.83.
That fits the picture of a head being held up by regularisation rather than by learning, but
the mechanism is not measured and should not be claimed.

Why this is not an artefact of the small review run: the encoder ends in a LayerNorm over 512
dimensions at every width, so raw features sit near length 22.6 regardless, while the K-means
helper always returns unit vectors. The mismatch is structural. No balanced-finetune accuracy
was logged for the published full-width runs, so there is no direct published-scale evidence;
the argument above is why it transfers.

**Proposed fix.** Compute the variance in the same space the prototypes live in — measure it on
normalised features at `models/memory.py:126` rather than on raw ones. Dividing the existing
numbers by a constant is *not* the fix; it happens to work out to roughly the right size here,
but it leaves the two quantities defined in different spaces.

Also note `continual/mavis_trainer.py:233`: the live task's prototypes get a hard-coded
variance of 0.01, which gives a noise length of about 1.13 against a unit prototype. Even the
deliberately small default is signal-sized.

### Finding B — the attention reconstruction loss switches off the memory path

The attention self-reconstruction term compares two things that are not on the same scale.

- Its target is a set of normalised prototypes, length 1.0.
- Its prediction is the attention module's output, which leaves through a LayerNorm over 512
  dimensions (`models/attention.py:135` and `models/attention.py:146`), so at the initial gain
  it has length 22.6.

The loss is a plain mean-squared error between them at `continual/mavis_trainer.py:218`, with a
hard-coded weight of 1.0.

Measured on the task-3 checkpoint:

| Quantity | Measured |
|---|---|
| Target length | 1.0000 |
| Prediction length, LayerNorm gain reset to its initial 1.0 | 22.6274 |
| The loss at that initial gain | 1.0028 |
| The loss at the trained gain | 0.0052 |
| Ratio | 193x cheaper after shrinking the gain |

The cheapest way to reduce this loss is therefore to shrink the LayerNorm gain, and that is
what training does. Read straight out of the three saved checkpoints:

| After task | LayerNorm gain, mean absolute | Retrieved length at evaluation | Gate | Retrieved share of the fused vector |
|---|---|---|---|---|
| 1 | 0.9045 | 36.24 | 0.686 | 68.7% |
| 2 | 0.6164 | 1.72 | 0.502 | 0.70% |
| 3 | 0.4413 | 2.11 | 0.469 | 1.76% |

That same LayerNorm sits on the retrieval path used in the ordinary forward pass. So the term
whose purpose was to keep retrieval faithful is the thing that turns retrieval off. The fusion
gate actually opens *towards* memory as training proceeds, from 0.686 down to 0.469, and it
makes no difference, because what arrives has length 2 against a live feature of length 24.

Caveat kept deliberately: the task-1 row shows a retrieved length of 36.24, which is larger
than a uniform gain of 0.90 on a 512-dimensional LayerNorm would give, around 20. The gain
decay is the direction of the effect but not a complete account of the magnitude, and the full
decomposition was not done.

**A second scale mismatch hidden behind the first.** `fuse_features` in `models/mavis.py` is the
path replayed prototypes take to the classifier, and it was being handed unit-norm prototypes
while the attention and the fusion gate had only ever been trained on live encoder features of
length about 22.6. The gate is a per-dimension sigmoid over the query and the retrieved vector
together, so a query 22 times shorter than it expects produces a different gate. This was
already wrong before any fix; it was invisible because the retrieved branch had itself been
crushed to length 2, so the two wrong scales were roughly in proportion. Repairing the
reconstruction loss restores the retrieved branch to length 22.6 and would therefore have made
this mismatch acute, with retrieval swamping the query outright.

**Proposed fix.** Compare the two at matched scale — normalise the prediction before the
mean-squared error at `continual/mavis_trainer.py:218`, or use a cosine loss, so the term
measures direction rather than length. Either way the gain stops being the cheapest thing to
sacrifice. The query scale in `fuse_features` has to be corrected in the same change, not after
it.

### Finding C — MAVIS trains with heavy dropout, the baseline trains with none

`models/mavis.py:53` passes the configured dropout rate into the encoder. That rate is 0.5
(`config.py:84`, and `configs/mavis.yaml:48`). `models/baseline.py:22` passes nothing, so the
baseline gets the default of 0.0.

The dropout in question is channel dropout (`models/encoder.py:57`), which zeroes whole feature
maps, and it sits inside all eight residual blocks.

Measured on the trained width-32 encoder with 128 real validation images:

| Quantity | Measured |
|---|---|
| Relative length change, training mode against evaluation mode | 0.140 mean, 0.303 max |
| Cosine between the two | 0.989 |

This is a real asymmetry but a modest one in size; an earlier estimate of 0.374 in this session
came from untrained weights on random input and is superseded by the figures above.

The more important consequence is not the size of the perturbation, it is that the comparison
is **confounded**. M1 and M2 run with no dropout; M3 and M4 run with channel dropout at 0.5.
Any statement of the form "memory helps" or "memory hurts" measured across that boundary is
also measuring a regularisation difference.

**Proposed fix.** Make the two sides match, by passing the same dropout rate to both models or
by setting it to 0.0 for both, and state in the results which was chosen.

### Secondary issues

Each of these is verified in code, but either smaller or not yet quantified.

- **Weight aligning does nothing.** `models/classifier.py:66` sets the per-class scale buffer to
  1.0. That buffer is created as ones at `models/classifier.py:29` and is written nowhere else.
  The operation is logged on every task as a published CVPR 2020 method. It is a no-op.
- **The learned no-memory embedding never trains.** `models/attention.py:104` creates it and
  `models/attention.py:121` returns it when the bank is empty. Its length is 0.220 in all three
  saved checkpoints, which is its initial value, so it receives no effective gradient. The
  reason is that training retrieval uses the current task's running prototypes as well as past
  ones, so the bank is never actually empty and that branch is never taken. It is dead weight.
- **The EWC strength is tied to the batch size.** The Fisher estimate in `continual/ewc.py`
  squares the gradient of a batch-*mean* loss and then averages over batches, so its magnitude
  falls as the batch grows. That makes the EWC strength non-transferable between runs with
  different batch sizes, which is consistent with 50000 having been needed at batch 256 while
  the default is 5000 at batch 128. Direction only; the factor was not measured.
- **Prototype variances include augmentation jitter.** Prototypes are built from the training
  loader, which applies random crops, flips and colour jitter. The variance at
  `models/memory.py:126` is therefore a mixture of genuine class spread and augmentation noise,
  and it is this variance that shapes the replay draws. Not quantified.
- **The documented number of prototypes per class does not match the code.** The code uses 10
  (`config.py:126`). The comment at the top of `models/memory.py:5` says 3. The treatise says
  3 growing to 7. The walkthrough says 3.
- **The initialisation is not what it is called.** `models/memory.py:26` is labelled
  "K-Means++ style" but spaces the starting centroids evenly through the samples, which is not
  what K-Means++ does.
- **The learning-rate schedule may not finish.** The one-cycle schedule
  (`continual/mavis_trainer.py:99`) is advanced only on optimiser steps that mixed precision
  actually accepted. Skipped steps therefore leave the schedule short of its final low learning
  rate by the end of the task. Not quantified.
- **Throughput.** Data loading runs single-process and attention is hand-rolled in 32-bit with
  batched matrix multiplies. Both are deliberate, to work around cluster crashes, and are
  documented as such. Decoding one task's images cost about 235 seconds per process before this
  session; a cache now makes it 0.1 seconds after the first build.

---

## Part 1b — The fixes that were applied

Four code changes close findings A, B and C. Each was verified by measurement, and each is now
guarded by a test that fails if it is reverted.

### Fix A — measure the variance where the prototypes live

`models/memory.py`, in `update_memory`: the per-class variance is now measured on normalised
features instead of raw ones, with the numerical floor reduced from 1e-4 to 1e-8 to suit the
smaller numbers that result.

Verified by rebuilding task-1 prototypes from all 10,000 real training images with the trained
width-32 encoder, then drawing replay samples both ways:

| | Noise length | Cosine to own prototype | Nearest-prototype oracle |
|---|---|---|---|
| Before | 9.0902 | 0.1276 | 10.50% |
| After | 0.3767 | 0.9371 | **99.50%** |

Chance on this task is 5.00%. The prototypes are unit vectors in both rows; only the noise
scale changed.

A companion change at the same finding: the live task's placeholder variance in
`continual/mavis_trainer.py` was a hard-coded 0.01, a raw-space number that produced noise of
length 1.13 against a unit prototype. It now borrows the mean of the past classes' variances,
which are measured in the right space.

### Fix B — compare directions, not lengths, in the reconstruction loss

`continual/mavis_trainer.py`: the attention output is normalised before the mean-squared error
against the already-normalised target.

Verified by sweeping the attention LayerNorm's gain, which is what the old loss was quietly
paying training to destroy:

| LayerNorm gain | Retrieved length | Loss before | Loss after |
|---|---|---|---|
| 1.00, its initial value | 22.627 | 1.002821 | 0.003945 |
| 0.44, where training had taken it | 9.995 | 0.197435 | 0.003945 |
| 0.05 | 1.131 | 0.004497 | 0.003945 |

The old loss falls by a factor of 223 as the gain collapses, which is why training collapsed it.
The new loss is identical at all three gains, so the incentive is gone.

### Fix B2 — hand the fusion gate a query on the scale it was trained for

`models/mavis.py`, in `fuse_features`: replayed prototypes arrive as unit vectors, but the
attention and the gate were trained on live encoder features of length about 22.6. The query is
now rescaled to that length before fusion. Without this, fix B would have swapped one scale
mismatch for another: a restored retrieval branch of length 22.6 blended against a query of
length 1.0 would simply have swamped the query.

### Fix C — one dropout setting, wired into both models

The shipped channel-dropout rate is now 0.0 instead of 0.5 (`config.py` and all four files in
`configs/`), and `models/baseline.py` accepts the rate and passes it to its encoder, with both
construction sites in `main.py` supplying it from the same config field. Either model can still
be given dropout, but neither can be given it without the other.

### End-to-end check

A two-task, one-epoch run at width 32 completes cleanly with the patched code. The diagnostic
shows the memory path alive again: the retrieved branch is 68.3% of the fused vector after task
2, against 2.36% measured at the same settings before the fixes, with retrieved and live feature
lengths now comparable at 25.78 and 24.28.

The balanced finetune reports 2.30% in that run, which is chance for 40 classes and is *not*
evidence about fix A. After a single epoch the encoder is untrained, so its prototypes carry no
class information no matter how the noise is scaled. Fix A was verified on trained features, in
the table above.

### Tests added

`tests/test_review_sanity.py` grew from six checks to nine. All nine pass.

| Check | Fails if |
|---|---|
| 7 | Replay noise is longer than the prototype it perturbs, or draws stop carrying their label |
| 8 | The reconstruction loss depends on the attention LayerNorm gain |
| 9 | The two models' channel dropout can differ, or the shipped default is not 0.0 |

---

## Part 2 — Measurements that landed

### Two valid baseline cells, for the first time

These are the first M1 and M2 numbers in the project produced with correct label offsets and
with predictions restricted to the classes seen so far. Width 32, five tasks, ten epochs per
task, seed 1.

| Cell | Accuracy on each task when first learned | Final row | Overall | Forgetting | Noise floor |
|---|---|---|---|---|---|
| M1 baseline | 45.5, 41.2, 52.5, 61.6, 64.9 | 0.0, 0.0, 0.0, 0.0, 64.9 | 12.98% | 50.20% | 0.30 |
| M2 EWC | 48.7, 39.3, 51.2, 53.9, 54.6 | 0.0, 0.0, 0.0, 0.2, 54.6 | 10.96% | 48.22% | 0.32 |

EWC came out 2.02 points **below** the baseline. This is a single seed, so it is a point
estimate and not a finding; the seed-to-seed spread is still unmeasured. The direction matches
the published literature on this benchmark, where online EWC also scores below plain
fine-tuning.

The whole of the gap is the task being learned: 64.9 against 54.6. Past-task accuracy is 0.0
either way. At this setting EWC is paying plasticity and buying nothing.

Two notes so the zeros are not misread. They are not a masking bug: M2's fourth row shows
0.2%, so old classes are reachable. And M1's 12.98% is its final-task accuracy divided by five
tasks, which is the expected shape for a model that has forgotten completely.

**Caveat on M2.** The result file records an EWC strength of 5000, which is the default
calibrated for M4 where distillation also acts. The published M2 used 50000. The driver has a
flag for this and it was not passed. This cell is therefore "EWC at the M4 strength", not a
reproduction of the published M2.

### The run that did not finish

Three of fifteen runs were attempted. M1 seed 1 finished in 24.5 minutes, M2 seed 1 in 20.3
minutes. M3 seed 1 was stopped four epochs into task 4, with tasks 1 to 3 fully checkpointed.

The stop was not a failure of the job. Claude Code killed the background shell because the
machine was critically low on memory while the session was idle.

---

## Part 3 — Every change made

### Existing files modified

| File | Change | Changes behaviour? |
|---|---|---|
| `models/encoder.py` | Added a base width setting so the encoder can be narrowed for local runs. Defaults to 128, the published width. Rejects widths that are not multiples of 32, because the normalisation layers require it. | No, at the default |
| `models/baseline.py` | Accepts and forwards the base width. | No, at the default |
| `models/mavis.py` | Added a switch that bypasses attention retrieval and gated fusion, for ablation. Added a per-forward diagnostic recording the length of the live feature, the length of the retrieved feature, the gate value, and the retrieved branch's share of the result. | No, at the default |
| `config.py` | Added the two settings above, and a note that the existing channels field is not read by anything. | No |
| `continual/trainer.py` | The EWC decay is now taken from the configuration instead of silently using the class default. Evaluation now predicts among the classes seen so far, matching the MAVIS trainer. | **Yes, for M1 and M2** |
| `continual/mavis_trainer.py` | Guarded the attention reconstruction loss behind the ablation switch. Saved the task-routing accuracy, which was previously printed and discarded. Accumulated and saved the diagnostics above. | No, at the default |
| `data/tiny_imagenet.py` | Added a per-task dataset cache, and a disk cache of decoded images. Verified image-for-image identical to decoding from scratch. | No |
| `main.py` | Added seeding of all random number generators. Added flags for seed, task count, width, batch size, attention off, and a run tag. Tagged output paths. Recorded run settings in every result file. | **Yes: training is now seeded** |

### New files

| File | Purpose |
|---|---|
| `tests/test_review_sanity.py` | Six checks that would fail if the review experiment were broken: label arithmetic, the ablation bypass, seeding reproducibility, the EWC penalty against its closed form, the diagnostic, and the narrow encoder. All six pass. |
| `tests/test_decode_cache.py` | Four checks that the image cache is faithful: identical pixels, labels and ordering; identical tensors under both transforms; and the expected file type and shape. All four pass. |
| `experiments/run_factorial.py` | Runs all five cells through one identical code path with matched settings, one process per cell and seed, skipping any cell already finished. |
| `experiments/analyze_factorial.py` | Reads the tagged result files and reports accuracy, spread, intervals, effect sizes, paired contrasts, the interaction term, and how many seeds a given effect would need. |
| `scripts/run_factorial.sh` | SLURM job array, one task per cell and seed, following the cluster paths and the Triton compiler workaround already used by the Phase 5 scripts. Defaults to published scale; `SCALE=quick` runs the reduced sweep. Skips any cell whose result file exists, and resumes rather than restarting, so it survives a wall-clock limit. |

### Naming

These started out called "the review factorial", which said nothing useful: every reader
reasonably asked what was being reviewed. The experiment is the project's own 2x2 factorial, so
the vague half of the name was dropped and the precise half kept. "Factorial" is load-bearing,
because it is the design that licenses an interaction term, which is what a synergy claim is.

| Was | Now |
|---|---|
| `experiments/run_review_factorial.py` | `experiments/run_factorial.py` |
| `experiments/analyze_review_factorial.py` | `experiments/analyze_factorial.py` |
| `scripts/run_review_factorial.sh` | `scripts/run_factorial.sh` |
| run tag `_rev_w32_t5_e10_s1` | `_fac_w32_t5_e10_s1` |
| `results/review_logs/` | `results/factorial_logs/` |
| banner "REVIEW FACTORIAL" | "FACTORIAL STUDY" |

The two finished result files, their checkpoint directories and the three per-run logs were
renamed to match, so the analyser still finds them. `tests/test_review_sanity.py` kept its name:
there "review" means the code review that produced it, which is accurate and is a different
thing from the factorial.

The grid the name refers to:

```
                         EWC off           EWC on
                    +----------------+----------------+
      memory off    |       M1       |       M2       |
                    |    baseline    |   CNN + EWC    |
                    +----------------+----------------+
      memory on     |       M3       |       M4       |
                    |  memory+attn   |   full MAVIS   |
                    +----------------+----------------+

      plus one cell outside the grid:
        M4noA  =  M4 with the forward-pass retrieval bypassed
```

| Contrast | What it isolates |
|---|---|
| M2 - M1 | EWC on its own |
| M3 - M1 | the memory pipeline on its own |
| M4 - M3 | what EWC adds on top of memory |
| (M4 - M3) - (M2 - M1) | the interaction, i.e. whether the two together beat the sum of their parts |
| M4 - M4noA | what the attention retrieval itself is worth |

The last row is the project's actual novelty question and had never been run. The fourth row is
the one the "super-additive 1.8x" claim rests on, and it needs all four cells to exist.

### Changes made while writing this log

- **The analyzer was stating a conclusion it had no data for.** With one seed there is no
  spread, so the interval came out undefined, and an undefined comparison fell through to
  printing "null not rejected". It now reports the point estimate and says the spread is
  unmeasured. The same guard was added to the interaction term, and single-seed rows now say so
  instead of printing undefined numbers.
- The analyzer now prints the EWC strength each run actually used, so the mismatch described in
  Part 2 cannot pass unnoticed again.

### A bug found in this session's own work

The image cache wrote through a temporary file and then renamed it, but the save call appends
its own file extension, so the rename looked for a name that did not exist. The failure was
swallowed by an exception handler, so the cache silently never wrote while the equivalence
checks still passed. Fixed by writing through an open file handle, and the test now asserts that
the file exists and has the expected type and shape, which is what caught it.

---

## Part 4 — What was not done

- **The secondary issues were left alone**: the no-op weight aligning, the dead no-memory
  embedding, the batch-size dependence of the EWC strength, augmentation jitter in the
  variances, the prototype-count mismatch between code and documentation, the mislabelled
  initialisation, and the schedule that may not reach its final learning rate.
- **Nothing was committed or pushed.** All of the above is in the working tree.
- **The three MAVIS cells were not run**: memory only, full, and full with attention bypassed.
  Seeds 2 and 3 were not run for any cell.
- **The seed-to-seed spread is still unmeasured**, which is the main thing the interrupted run
  existed to produce.
- **The EWC strength question is open.** Matching it across M2 and M4 keeps the comparison
  clean; using 50000 reproduces the published M2. These answer different questions, and running
  both as separate cells costs about 20 minutes per seed.

---

## Retraining implications

**All four published result files now need re-running.** This is a change from the position
earlier the same day, when the fixes had not yet been applied.

- **M3 and M4 must be re-run.** Fixes A, B, B2 and C all change how MAVIS trains. The published
  figures, including the 22.24% headline, describe a model in which the replay signal was noise
  and the retrieval branch was suppressed to under 2% of the fused feature. That model no longer
  exists in this tree.
- **M1 and M2 must be re-run**, but for reasons that predate this work: they were produced by a
  different encoder with a label bug, a month before the fix landed. Fix C also now gives them
  the same dropout setting as MAVIS, which is the point of it.
- The parameter counts are unchanged, at 45.53M for the encoder, 45.63M for the baseline and
  47.21M for MAVIS, and so are the channel widths, the output filenames and the checkpoint
  directories. Nothing about the shape of the model moved; what moved is how three of its loss
  terms and one of its regularisers behave.
- Training is now seeded, where previously no seed was set at all, so runs from here are
  reproducible in distribution. The old files are single unseeded draws and cannot be reproduced
  exactly in any case.

All four changes act on MAVIS. **Fix C does not change the baseline at all**: it was already
training at 0.0, because the old code passed no dropout and the encoder's own default is 0.0.
What fix C does is take the 0.5 away from MAVIS and make the shared setting visible, so the two
can no longer drift apart.

The direction of the result is still not predictable, for a reason worth stating plainly. The
gain collapse that fix B prevents may have been the optimiser correctly discovering that
retrieval was not helping, taking the cheapest route available to switch it off. That route is
now closed. If retrieval genuinely does not help on this benchmark, forcing it to stay live
could cost accuracy rather than gain it. Which of those is true is the question this
architecture was built to answer, and it has never been measured, because until now the loss
made the answer unobservable.

Fix C carries its own uncertainty in the opposite direction. The encoder has 45.5M parameters
and sees 10,000 images per task, so channel dropout at 0.5 may have been load-bearing
regularisation rather than a mistake. The asymmetry with the baseline was definitely a flaw; the
right *value* is an empirical question, and 0.0 versus 0.5 is worth running as a two-cell
comparison rather than assumed.
