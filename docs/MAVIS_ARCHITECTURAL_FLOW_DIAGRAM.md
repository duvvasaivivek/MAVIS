# MAVIS Architectural Flow Diagram (Every Block, In Plain Words)

As of: 2026-10-24. This document draws the complete MAVIS system block by block so it can be
redrawn by hand and explained to someone with no machine-learning background. Every number here
was checked against the running code, but the code itself is not quoted. A short technical name
for each block is given once in brackets so it can be found in the source later if needed.

Companion documents: theory and history in `MAVIS_MASTER_ARCHITECTURAL_TREATISE.md`, a reviewer
walkthrough in `MAVIS_REVIEW_WALKTHROUGH.md`, a Canva layout in `MAVIS_WHITEBOARD_BLUEPRINT.md`,
results in `../PROJECT_LOG.md`.

## Contents

0. [How to read the diagrams, and the words used](#0-how-to-read-the-diagrams-and-the-words-used)
1. [The big picture and the four models](#1-the-big-picture-and-the-four-models)
2. [From image files to training batches](#2-from-image-files-to-training-batches)
3. [The Eye: the feature extractor](#3-the-eye-the-feature-extractor)
4. [The Judge: the classifier head](#4-the-judge-the-classifier-head)
5. [Models 1 and 2: Eye plus Judge, with or without weight anchoring](#5-models-1-and-2-eye-plus-judge-with-or-without-weight-anchoring)
6. [The Library: the memory bank](#6-the-library-the-memory-bank)
7. [The Librarian: attention retrieval](#7-the-librarian-attention-retrieval)
8. [MAVIS end to end: one image in, one answer out](#8-mavis-end-to-end-one-image-in-one-answer-out)
9. [One training step and the eight lessons (loss terms)](#9-one-training-step-and-the-eight-lessons-loss-terms)
10. [How fast each part learns](#10-how-fast-each-part-learns)
11. [What happens after a task is finished](#11-what-happens-after-a-task-is-finished)
12. [How the model is graded: four ways to answer](#12-how-the-model-is-graded-four-ways-to-answer)
13. [The run from start to finish](#13-the-run-from-start-to-finish)
14. [Facts about the code that are easy to get wrong](#14-facts-about-the-code-that-are-easy-to-get-wrong)

---

## 0. How to read the diagrams, and the words used

### Box shapes

```
┌────────────────────────────────┐   single-line box = a step that transforms data.
│ step name              [learns]│   "[learns]" means the step has adjustable numbers
└────────────────────────────────┘   (weights) that training changes.

╔════════════════════════════════╗   double-line box = a LESSON (a loss term): a score that
║ lesson name · weight · idea    ║   says how wrong the model is in one particular way.
╚════════════════════════════════╝   Training pushes every lesson's score toward zero.

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   shaded box = stored, frozen information. It is written
▓ stored thing  [size]         ▓   at specific moments but never changed by the learning
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   step itself (the library, snapshots, anchors).

──▶  (name [size])                  solid arrow  = data flows AND the lesson can teach the
                                    step behind it (gradient flows back).
╌╌▶  (name [size])                  dashed arrow = data flows but teaching is blocked here.
◇ question? ◇                       a decision; its YES and NO arms are labelled.
```

### Words used throughout

| Word used here | What it means | Technical name |
| :--- | :--- | :--- |
| **image** | one 64 x 64 colour picture | input tensor `3 × 64 × 64` |
| **batch** | 128 images processed together | `B = 128` |
| **task** | a group of 20 new classes learned together; there are 10 tasks in a row | `task t`, 0..9 |
| **class** | one object category (200 in total, 20 per task) | class id 0..199 |
| **feature vector** | a list of 512 numbers that summarises one image | `D = 512` features |
| **feature map** | a grid of features that still keeps "where in the picture" | spatial tensor |
| **the Eye** | the part that turns an image into a feature vector | CNN encoder |
| **the Judge** | the part that turns a feature vector into 200 class scores | cosine classifier |
| **prototype** | a stored "typical feature vector" for a class; 10 are kept per class | K = 10 |
| **the Library** | all stored prototypes and their spread | memory bank |
| **the Librarian** | looks up the stored prototypes most similar to what the Eye sees now | attention retrieval |
| **the Gate** | a learned dial that mixes "what I see" with "what I recall" | gated fusion |
| **lesson** | one component of the training loss | loss term |
| **snapshot** | a frozen copy of the Eye taken at a specific moment | `E_0`, `E_{t-1}` |
| **anchor** | a reference the model is pulled toward so it does not drift | distillation / EWC |
| **seen classes** | all classes of tasks finished so far, 20 × (t+1) | `seen` |
| **past prototypes** | the 200 × t prototypes of finished tasks | `N_past` |

### Key numbers (all from the configuration or fixed in code)

| Quantity | Value |
| :--- | :--- |
| images per class, training / validation | 500 / 50 |
| images per task, training / validation | 10,000 / 1,000 |
| batches per epoch (training) | 79 (the last one has 16 images) |
| epochs per task | 50 |
| feature vector length | 512 |
| prototypes per class, per task, total | 10, 200, 2,000 |
| attention heads | 4 heads, each looking at 128 of the 512 numbers |
| adjustable numbers (weights) in MAVIS | Eye 45.5 M, Librarian 1.05 M, Gate 0.52 M, Judge 0.10 M, total 47.2 M |
| learning rate | 0.008 at the start of every task, one-cycle shape |
| lesson weights (MAVIS) | classification 1, feature anchoring 3.0, layout anchoring 1.5, prototype pull 0.5, push-away 2.0, memory fidelity 1.0, rehearsal 2.0, weight anchoring strength 5000 |

---

## 1. The big picture and the four models

MAVIS learns 200 object classes in 10 instalments of 20. After each instalment it must still
recognise all earlier classes without ever seeing their images again. The danger is **catastrophic
forgetting**: learning new classes overwrites old knowledge. MAVIS fights this with two ideas that
can be switched on independently, which gives four models to compare.

```
                       weight anchoring OFF           weight anchoring ON
                  ┌──────────────────────────────┬──────────────────────────────┐
  no Library      │ M1  Eye + Judge              │ M2  Eye + Judge + anchoring  │
                  │     the plain baseline       │     (EWC only)               │
                  ├──────────────────────────────┼──────────────────────────────┤
  Library +       │ M3  MAVIS without anchoring  │ M4  full MAVIS               │
  Librarian       │     (memory only)            │     (memory + anchoring)     │
                  └──────────────────────────────┴──────────────────────────────┘
```

| Component | M1 | M2 | M3 | M4 |
| :--- | :---: | :---: | :---: | :---: |
| the Eye (feature extractor) | yes | yes | yes, with 50% channel dropout | same as M3 |
| the Judge (cosine classifier) | yes, no margin | yes, no margin | yes, margin 0.20 | same as M3 |
| the Library (prototypes) | no | no | yes | yes |
| the Librarian (attention) | no | no | yes | yes |
| the Gate (mixing dial) | no | no | yes | yes |
| lesson: classification | over all 200 classes | over all 200 classes | only the 20 current classes | same as M3 |
| lesson: feature anchoring | no | no | yes | yes |
| lesson: layout anchoring | no | no | yes | yes |
| lesson: prototype pull | no | no | yes | yes |
| lesson: push away from old memories | no | no | yes | yes |
| lesson: memory fidelity | no | no | yes | yes |
| lesson: memory rehearsal | no | no | yes | yes |
| lesson: weight anchoring (EWC) | no | yes (strength 5000, decay 0.95) | no | yes (strength 5000, decay 0.88) |
| after each task: store prototypes, refresh old ones, re-tune the Judge | no | no | yes | yes |
| after each task: measure which weights matter (Fisher) | no | yes | no | yes |
| grading | one way (Judge only) | one way | four ways | four ways |
| results file | `baseline_results.json` | `ewc_results.json` | `mavis_m3_results.json` | `mavis_results.json` |

The published M2 result used anchoring strength 50,000, not the default 5,000.

---

## 2. From image files to training batches

Shared by all four models. The output of this stage is one batch of images with their labels.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Image files on disk: 200 folders, one per class, 64x64 pictures inside   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ the 200 class names, sorted
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Task splitter   [fixed random seed 42]                                   │
│   shuffles the 200 class names once, then cuts the list into 10 groups   │
│   of 20. Group k is "task k". The order never changes between runs.      │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ the 20 class names of the current task
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Task dataset                                                             │
│   lists every picture of those 20 classes, loads them all into RAM once  │
│   gives each picture a LOCAL label 0..19 (its position inside the task)  │
│   training split: 10,000 pictures ; validation split: 1,000 pictures     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ one picture + local label (0..19)
               ┌─────────────────────┴─────────────────────┐
               ▼ training pictures                         ▼ validation pictures
┌─────────────────────────────────┐          ┌─────────────────────────────────┐
│ Training-time variations        │          │ No variations                   │
│ 1 random shift up to 4 pixels   │          │ 1 convert to numbers (0..1)     │
│ 2 random mirror (50%)           │          │ 2 standardise colours           │
│ 3 random tilt up to ±10°        │          └───────────────┬─────────────────┘
│ 4 random brightness/contrast/   │                          │
│   saturation ±20%, hue ±10%     │                          │
│ 5 convert to numbers (0..1)     │                          │
│ 6 random small patch blacked    │                          │
│   out (25% of pictures)         │                          │
│ 7 standardise colours           │                          │
└───────────────┬─────────────────┘                          │
                ▼                                            ▼
┌─────────────────────────────────┐          ┌─────────────────────────────────┐
│ Batcher: 128 pictures, shuffled │          │ Batcher: 128 pictures, in order │
│ 79 batches per epoch            │          │ 8 batches                       │
└───────────────┬─────────────────┘          └───────────────┬─────────────────┘
                │ (batch: 128 × 3 × 64 × 64, 128 local labels)│
                ▼                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Label translation (inside the trainer, every batch)                      │
│   global label = local label + 20 × task number     -> 0..199            │
│   MAVIS keeps BOTH labels: local for the "current task" lesson,          │
│   global for everything that spans all seen classes                      │
└──────────────────────────────────────────────────────────────────────────┘
```

| Block | What it does | In | Out |
| :--- | :--- | :--- | :--- |
| Task splitter | fixes which 20 classes belong to which task, same every run | 200 names | 10 lists of 20 |
| Task dataset | loads one task's pictures into RAM, numbers them 0..19 | 20 names | pictures + local labels |
| Training-time variations | makes each picture slightly different each epoch so the Eye cannot memorise pixels | picture | altered picture |
| Batcher | groups 128 pictures | pictures | one batch |
| Label translation | turns 0..19 into the global class id | local label | global label |

In plain words: the pictures are dealt out in ten hands of twenty classes. Each training picture is
jiggled a little every time it is shown. The model only ever sees the hand it is currently learning.

---

## 3. The Eye: the feature extractor

The Eye is a deep stack of image filters. It takes a 64 x 64 picture and produces a feature vector
of 512 numbers that describe *what* is in the picture. All four models use the same Eye. It is
built from scratch; nothing is pretrained.

### 3a. The Eye, stage by stage

```
 (picture: 3 colour layers of 64 x 64)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0  "look closely"                                     [learns]│
│   128 small 3x3 filters scan the picture; results are normalised    │
│   and negatives are cut off. The picture is NOT shrunk here.        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 128 layers of 64 x 64
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1  two Residual Units, 128 layers each               [learns] │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 128 layers of 64 x 64
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2  two Residual Units, 256 layers, picture halved    [learns] │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 256 layers of 32 x 32
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3  two Residual Units, 512 layers, picture halved    [learns] │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 512 layers of 16 x 16  ───────▶ tap A (kept for a lesson, §9b)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 4  two Residual Units, 1024 layers, picture halved   [learns] │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 1024 layers of 8 x 8  ────────▶ tap B (kept for a lesson, §9b)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Squash   average each of the 1024 layers down to one number         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ 1024 numbers
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Summarise   mix the 1024 numbers into 512 and normalise them [learns]│
└─────────────────────────────────┬────────────────────────────────────┘
                                  ▼
                   (feature vector: 512 numbers per picture)
```

| Stage | Picture size after | Layers (channels) after | Note |
| :--- | :--- | :--- | :--- |
| Stage 0 | 64 x 64 | 128 | 3x3 filters, no shrinking, so small objects survive |
| Stage 1 | 64 x 64 | 128 | two Residual Units |
| Stage 2 | 32 x 32 | 256 | first unit halves the picture |
| Stage 3 | 16 x 16 | 512 | output is "tap A", used by the layout-anchoring lesson |
| Stage 4 | 8 x 8 | 1024 | output is "tap B", used by the layout-anchoring lesson |
| Squash | 1 x 1 | 1024 | plain average |
| Summarise | - | 512 | the feature vector every other part consumes |

### 3b. One Residual Unit (there are 8 of them)

A Residual Unit refines its input and then **adds the input back**, so each unit only has to learn
a correction, which keeps very deep stacks trainable.

```
 (input feature map)
   │
   ├────────────────────────────────────────────────────────────────┐
   ▼                                                                ▼
┌─────────────────────────────────────────┐        ┌─────────────────────────────────────┐
│ Filter A   3x3 filters          [learns]│        │ Shortcut                            │
│   may halve the picture (stride 2)      │        │  if size and layer count are the    │
└───────────────────┬─────────────────────┘        │  same: pass the input through as is │
                    ▼                              │  else: a 1x1 filter that matches    │
┌─────────────────────────────────────────┐        │  the size and layer count  [learns] │
│ Normalise + cut negatives               │        └─────────────────┬───────────────────┘
└───────────────────┬─────────────────────┘                          │
                    ▼                                                │
┌─────────────────────────────────────────┐                          │
│ Channel dropout (MAVIS only, 50%)       │                          │
│   randomly silences whole layers during │                          │
│   training so no single layer is relied │                          │
│   on; the baseline has 0% here          │                          │
└───────────────────┬─────────────────────┘                          │
                    ▼                                                │
┌─────────────────────────────────────────┐                          │
│ Filter B   3x3 filters          [learns]│                          │
└───────────────────┬─────────────────────┘                          │
                    ▼                                                │
┌─────────────────────────────────────────┐                          │
│ Normalise                               │                          │
└───────────────────┬─────────────────────┘                          │
                    ▼                                                │
┌──────────────────────────────────────────┐                          │
│ Channel attention (SE block, §3c)[learns]│                          │
└───────────────────┬──────────────────────┘                          │
                    ▼                                                ▼
              ┌───────────────────────────────────────────────────────────┐
              │ Add the two paths together                                │
              └────────────────────────────┬──────────────────────────────┘
                                           ▼
              ┌───────────────────────────────────────────────────────────┐
              │ Cut negatives                                             │
              └────────────────────────────┬──────────────────────────────┘
                                           ▼
                                   (refined feature map)
```

Only the first unit of Stages 2, 3 and 4 needs the 1x1 matching shortcut (3 units out of 8). The
other 5 pass the input straight through.

### 3c. Channel attention (the "SE block") inside every Residual Unit

It lets the unit turn its own layers up or down depending on the picture.

```
 (feature map with c layers)
   │
   ├──────────────────────────────────────────────────────────────┐
   ▼                                                              │
┌────────────────────────────────────────────┐                    │
│ Average each layer to a single number      │  c numbers         │
└───────────────────┬────────────────────────┘                    │
                    ▼                                             │
┌────────────────────────────────────────────┐                    │
│ Shrink to a few numbers, cut negs [learns] │                    │
│   c=128 -> 8   256 -> 16   512 -> 32       │                    │
│   1024 -> 64                               │                    │
└───────────────────┬────────────────────────┘                    │
                    ▼                                             │
┌────────────────────────────────────────────┐                    │
│ Expand back to c numbers, squash   [learns]│                    │
│ each to between 0 and 1                    │                    │
└───────────────────┬────────────────────────┘                    │
                    │ one "volume knob" between 0 and 1 per layer │
                    ▼                                             ▼
              ┌──────────────────────────────────────────────────────────┐
              │ Multiply every layer by its knob                         │
              └────────────────────────────┬─────────────────────────────┘
                                           ▼
                                  (re-weighted feature map)
```

In plain words: the Eye is a tower of filters. Each floor looks for more abstract patterns than the
floor below and shrinks the picture. At the top the whole picture is boiled down to 512 numbers.
Two intermediate floors (taps A and B) are also kept because a later lesson checks that the Eye has
not forgotten *where* things used to be in the picture.

---

## 4. The Judge: the classifier head

The Judge holds one "direction" (a 512-number template) for each of the 200 classes and scores a
feature vector by how well its direction matches each template. Only directions matter, not lengths,
so new and old classes are always compared on an equal footing. All four models use it; it exists
for all 200 classes from the very first step. **Which of the 200 scores are allowed to count is
decided outside the Judge**, by the trainer.

```
 (feature vector, 512 numbers)                 Class templates [learns]  200 × 512
   │                                              │
   ▼                                              ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ Keep only the direction      │    │ Keep only each template's    │
│ (scale to length 1)          │    │ direction (length 1)         │
└──────────────┬───────────────┘    └──────────────┬───────────────┘
               │ unit feature                      │ 200 unit templates
               └──────────────┬────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Match score = how aligned the feature is with each template      │
│   one number per class, between −1 (opposite) and +1 (identical) │
└──────────────────────────────┬───────────────────────────────────┘
                              ▼
   ◇ training, AND a margin is set, AND true labels were handed in? ◇
       │ YES (only during MAVIS live training)                       │ NO
       ▼                                                             │
┌──────────────────────────────────────────────┐                     │
│ Handicap the true class: subtract 0.20 from  │                     │
│ its score, so the model must win by a margin │                     │
└──────────────────────────┬───────────────────┘                     │
                           └──────────────┬──────────────────────────┘
                                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ Per-class volume (all set to 1.0)                                │
│   a "weight aligning" step after each task re-writes the seen    │
│   classes' volume to 1.0, which is already their value: no effect│
└──────────────────────────────┬───────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Multiply all scores by 24 (sharpens the final probabilities)     │
└──────────────────────────────┬───────────────────────────────────┘
                              ▼
                   (200 class scores, each between −24 and +24)
```

| Who asks the Judge | Handicap applied? | Which scores are allowed to count |
| :--- | :--- | :--- |
| M1 / M2 training and grading | never (margin is 0) | all 200, including classes not yet learned |
| MAVIS live training step | yes | only the 20 classes of the current task |
| MAVIS memory rehearsal (§9d) | no | all classes seen so far |
| MAVIS Judge re-tuning after a task (§11c) | no | all classes seen so far |
| MAVIS grading (§12) | no (not training) | all classes seen so far |
| MAVIS importance measurement (§11d) | no | all classes seen so far |

In plain words: the Judge is a compass with 200 needles. It asks "which needle does this feature
point along best?" During MAVIS training the true needle is handicapped so the model learns to
point at it decisively, not just barely.

---

## 5. Models 1 and 2: Eye plus Judge, with or without weight anchoring

```
 (batch of pictures)                 (global labels)
        │                                  │
        ▼                                  │
┌──────────────────────────────┐           │
│ the Eye (no dropout) [learns]│           │
└──────────────┬───────────────┘           │
               │ feature vectors (512)     │
               ▼                           │
┌──────────────────────────────┐           │
│ the Judge (no margin)[learns]│           │
└──────────────┬───────────────┘           │
               │ 200 class scores          │
               ▼                           ▼
╔══════════════════════════════════════════════════════════════════╗
║ Lesson: Classification · weight 1                                ║
║   "the true class should have the highest of ALL 200 scores"     ║
║   (cross-entropy, 10% label smoothing). Because all 200 scores   ║
║   count, every new picture also pushes DOWN every old class.     ║
╚══════════════════════════════╤═══════════════════════════════════╝
                               │
   ◇ M2, and at least one task already finished? ◇
                               │ YES
                               ▼
╔══════════════════════════════════════════════════════════════════╗
║ Lesson: Weight anchoring (EWC) · strength 5000                   ║
║   "weights that mattered for old tasks should stay where they    ║
║    were". Each weight is pulled back to its saved value, harder  ║
║    the more it mattered (its Fisher importance).                 ║
╚══════════════════════════════╤═══════════════════════════════════╝
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Learning step: add the lessons, compute how each weight should   │
│ move, move it (AdamW, learning rate 0.008 in a one-cycle shape)  │
└──────────────────────────────────────────────────────────────────┘

 after the 50 epochs of a task (M2 only):
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓ Importance update (§11d): measure how much each weight matters   ▓
▓   for the task just learned (2,000 pictures), blend it into the  ▓
▓   running importance (old × 0.95 + new), and save a copy of all  ▓
▓   weights as the new anchor point.                               ▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

 grading (M1 and M2): the answer is the highest of all 200 scores, even for classes never taught yet
```

In plain words: M1 is a student who only ever studies the current chapter and is tested on the whole
book. M2 is the same student with a rule: "do not move the brain cells that the old chapters
depended on". Notice there is no notebook of old examples at all in these two models.

Three details that differ from MAVIS: the classification lesson covers all 200 classes, label
smoothing is 10% rather than 2%, and the anchoring decay is 0.95 rather than 0.88 because the M2
trainer never passes its own value.

---

## 6. The Library: the memory bank

The Library stores, for every class learned so far, **10 prototypes** (typical feature vectors) and
one **spread** (how much each of the 512 numbers varies within that class). It is pure storage: it
has no adjustable weights. It is saved and reloaded with the model.

### 6a. What is stored

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓ Prototype shelves   10 tasks × 20 classes × 10 prototypes × 512 numbers      ▓
▓ Class spreads       10 tasks × 20 classes × 512 numbers (variance per number)▓
▓ Shelf filled?       10 yes/no flags, one per task                            ▓
▓ Live sketches       20 × 512: running average feature of each CURRENT class  ▓
▓ Sketch counters     20 counts: how many batches have fed each live sketch    ▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

Total capacity: 2,000 prototypes of 512 numbers each, plus 200 spreads.

### 6b. Live sketches: keeping a rough picture of the classes being learned right now

The shelf for the current task is empty until the task ends, so during training the Library keeps a
cheap running average per class instead. It is refreshed on every batch and teaching is blocked
through it.

```
 (feature vectors of the batch) ╌╌▶ copy, no teaching       (local labels 0..19)
                          │                                        │
                          ▼                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│ For each of the 20 classes present in the batch:                 │
│   average the features of that class in this batch               │
│   first time: sketch = that average                              │
│   afterwards: sketch = 92% old sketch + 8% new average           │
│   counter + 1                                                    │
└──────────────────────────────────────────────────────────────────┘
          writes ▓ Live sketches ▓  and  ▓ Sketch counters ▓
```

### 6c. Ways to read from the Library

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Current sketches  -> 20 × 512                                                │
│   each sketch scaled to length 1; a class not yet seen gives all zeros       │
├──────────────────────────────────────────────────────────────────────────────┤
│ Past prototypes  -> (200 × t) × 512, or "none" during the first task         │
│   every prototype of every finished task, stacked                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Training memories  -> past prototypes + current sketches                     │
│   what the Librarian may consult while training task t                       │
│   (on the very first batch of task 1 this is "none")                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ All prototypes  -> (200 × tasks finished) × 512                              │
│   what the Librarian consults during grading and rehearsal                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Past prototypes WITH labels and spreads                                      │
│   the same past prototypes, each tagged with its class id and its class's    │
│   spread, ready for rehearsal (§9d)                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Nearest-prototype scores  (used in grading, §12)                             │
│   compare a feature vector with every stored prototype, keep the BEST of     │
│   the 10 per class  -> one score per seen class                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6d. Filling a shelf: turning a finished task into prototypes

Done once for the task just finished, and then **again for every earlier task** (see §11b), each
time passing that task's training pictures through the Eye as it is *now*.

```
 (the task's 10,000 training pictures)              the Eye, in "no learning" mode
        │                                                   │
        ▼                                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Run every picture through the Eye; sort the 10,000 feature vectors into  │
│ 20 piles, one per class (about 500 vectors each)                         │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼  for each pile (class)
┌──────────────────────────────────────────────────────────────────────────┐
│ Find 10 cluster centres (k-means, 10 rounds)                             │
│   start: 10 vectors picked at evenly spaced positions in the pile        │
│   repeat 10 times: assign each vector to its nearest centre, move each   │
│   centre to the middle of its members                                    │
│   result: 10 prototypes, each scaled to length 1                         │
│   (10 instead of 1 so that a class with several "looks" keeps them all)  │
├──────────────────────────────────────────────────────────────────────────┤
│ Measure the spread: the variance of each of the 512 numbers across the   │
│ pile, plus a tiny floor so it is never zero                              │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓ write the 10 prototypes and the spread onto the shelf; mark the shelf    ▓
▓ "filled"; wipe the live sketches and counters                            ▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

In plain words: when a chapter is finished, the model writes ten "typical examples" per class into
its notebook, plus a note on how varied each class is. The pictures themselves are thrown away; only
these summaries survive. The summaries are rewritten every time the Eye changes so they never go
stale.

---

## 7. The Librarian: attention retrieval

Given what the Eye sees now, the Librarian finds the most similar stored prototypes and returns a
blend of them: "this is what the past says about pictures like this one". It uses the standard
attention mechanism with 4 heads, written out by hand in full precision for cluster stability.
It has about one million adjustable numbers.

### 7a. The Librarian's three situations

```
 (current feature vectors)         (memories: N prototypes × 512, or "none")
        │                                  │
        ▼                                  ▼
   ◇ no memories at all? ◇
        │ YES  (first batches of task 1)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ Return a learned "nothing to recall" vector            [learns]  │
│   the same 512 numbers for every picture, so task 1 is not       │
│   trained on exact zeros                                         │
└──────────────────────────────────────────────────────────────────┘
        │ NO
        ▼
   ◇ more than 64 pictures in the batch? ◇
        │ YES: handle 64 pictures at a time, then stitch the results back
        │ NO : handle all at once
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ Give every picture its own copy of the memory list to consult    │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4-head lookup (§7b)                                     [learns] │
└──────────────────────────────┬───────────────────────────────────┘
                               │ one recalled vector (512) per picture
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Normalise the recalled vector                           [learns] │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                 (recalled feature vector, 512 numbers per picture)
```

### 7b. The 4-head lookup

```
 everything below runs in full precision (a cluster requirement)
 (question: current feature, 512)   (memory keys: N × 512)   (memory contents: N × 512)
        │                                  │                         │
        ▼                                  ▼                         ▼
┌──────────────────┐             ┌──────────────────┐     ┌──────────────────┐
│ Re-express the   │             │ Re-express the   │     │ Re-express the   │
│ question [learns]│             │ keys     [learns]│     │ contents [learns]│
└────────┬─────────┘             └────────┬─────────┘     └────────┬─────────┘
         │ split into 4 heads of 128      │ 4 heads of 128         │ 4 heads of 128
         └──────────────┬─────────────────┘                        │
                        ▼                                          │
┌──────────────────────────────────────────────────┐               │
│ Similarity of the question to every key, per head│               │
│   divided by √128 (standard) and by 0.5          │               │
│   (makes the choice sharper, less "averaging")   │               │
└────────────────────────┬─────────────────────────┘               │
                         ▼                                         │
┌──────────────────────────────────────────────────┐               │
│ Turn similarities into percentages that sum to 1 │               │
│ (softmax); randomly drop 10% during training     │               │
└────────────────────────┬─────────────────────────┘               │
                         └──────────────┬──────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│ Weighted blend of the memory contents using those percentages,   │
│ per head; then glue the 4 heads back into 512 numbers            │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Final re-expression                                      [learns]│
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                    (recalled vector, 512 numbers per picture)
```

In plain words: the Librarian asks "which of my stored examples look like this?", gives each a
percentage of attention, and hands back a weighted average of them. Four heads means four
librarians each paying attention to a different quarter of the description, then pooling their
answers. The sharpening (÷0.5) stops the answer from being a bland average of hundreds of memories.

---

## 8. MAVIS end to end: one image in, one answer out

The only difference between training and grading is **which memories the Librarian may consult**
and whether the Judge applies its handicap.

```
 (batch of pictures)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ the Eye (with 50% channel dropout during training)              [learns] │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ "what I see": 512 per picture (+ taps A, B when asked)
         ┌──────────────────────────┼─────────────────────────────┐
         │                          ▼                             │
         │     ◇ are we training?  ◇                              │
         │       │ YES  (task t)                                  │ NO (grading)
         │       ▼                                                ▼
         │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
         │  ▓ Training memories                   ▓    ▓ All prototypes                  ▓
         │  ▓  prototypes of finished tasks       ▓    ▓  every filled shelf             ▓
         │  ▓  + live sketches of this task       ▓    ▓  (200 per finished task)        ▓
         │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
         │       │                                                │
         │       └───────────────────────┬────────────────────────┘
         │                               ▼ (memories)
         │  ┌──────────────────────────────────────────────────────────────┐
         ├─▶│ the Librarian (§7)                                  [learns] │
         │  └──────────────────────────────┬───────────────────────────────┘
         │                                 │ "what I recall": 512 per picture
         ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ the Gate                                                        [learns] │
│   looks at both vectors side by side (1024 numbers) and sets, for each   │
│   of the 512 positions, a dial between 0.05 and 0.95                     │
│   fused = dial × "what I see" + (1 − dial) × "what I recall"             │
│   starts at exactly 0.5 (a 50/50 mix) and learns from there              │
│   the dial can never fully silence either source (5% floor each)         │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ fused feature: 512 per picture
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ the Judge (§4)                                                  [learns] │
│   training: handicap of 0.20 on the true class ; grading: no handicap    │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    ▼
                  (200 class scores per picture, plus the feature vectors if asked)
```

### 8b. The same path for a feature vector instead of a picture

Rehearsal (§9d) and Judge re-tuning (§11c) feed **stored prototypes**, not pictures, through the
Librarian and the Gate. This is the identical Librarian and Gate; the only difference is that the
result is scaled to length 1 at the end. It exists so that prototypes are judged in the same
"fused" space as real pictures.

```
 (prototype-like vectors) ──▶ the Librarian (all prototypes as memory) ──▶ (recalled)
   both ──▶ the Gate ──▶ fused ──▶ scale to length 1 ──▶ (fused prototype vectors)
```

In plain words: the Eye says what it sees, the Librarian says what the past recalls about similar
things, the Gate decides how much to trust each, and the Judge names the class. During training the
Librarian may only consult finished chapters plus rough sketches of the current one; during the exam
it may consult everything.

---

## 9. One training step and the eight lessons (loss terms)

This is the heart of the system. One step = one batch of 128 pictures of task t, in epoch e.
Eight lessons are computed and added up; the sum is what the weights are moved to reduce.

### 9.0 The step as a whole

```
 (batch of pictures)   (labels: local 0..19 and global)   (task t, epoch e)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ 1  Full MAVIS pass (§8) with the Librarian in training mode               │
│    -> 200 scores, "what I see", fused feature, taps A and B               │
├───────────────────────────────────────────────────────────────────────────┤
│ 2  Refresh the live sketches with this batch (§6b, teaching blocked)      │
├───────────────────────────────────────────────────────────────────────────┤
│ 3  Compute the eight lessons        (§9a classification, §9b anchoring,   │
│                                      §9c prototypes, §9d rehearsal,       │
│                                      §9e weight anchoring)                │
├───────────────────────────────────────────────────────────────────────────┤
│ 4  Warm-up factor s = 0.5 at epoch 0, rising to 1.0 by epoch 5 (tasks 2+) │
│    s = 0 throughout task 1 (nothing to anchor to yet)                     │
│    TOTAL = classification                                                 │
│          + 3.0·s × feature anchoring   + 1.5·s × layout anchoring         │
│          + 0.5 × prototype pull        + 2.0 × push-away                  │
│          + 1.0 × memory fidelity       + 2.0 × rehearsal                  │
│          + weight anchoring (its strength 5000 is already inside it)      │
├───────────────────────────────────────────────────────────────────────────┤
│ 5  Work out how every weight should move to reduce TOTAL and move it      │
│    (mixed precision with a safety scaler; the learning-rate schedule      │
│    advances only when the step was numerically healthy)                   │
├───────────────────────────────────────────────────────────────────────────┤
│ 6  Book-keeping: accuracy within the current 20 classes, and accuracy     │
│    among all classes seen so far                                          │
└───────────────────────────────────────────────────────────────────────────┘
```

### 9a. The live pass and the Classification lesson

```
 (pictures) ──▶ the Eye ──▶ ("what I see" 512) ──┬──▶ the Librarian ──▶ ("what I recall" 512)
                                                  │        ▲ memories: finished tasks + sketches
                                                  ▼        │ (storage, teaching blocked)
                                          ┌───────────────────────────────┐
                                          │ the Gate  (see + recall)      │
                                          └───────────────┬───────────────┘
                                                          │ fused feature (512)
                                                          ▼
                                          ┌───────────────────────────────┐
                                          │ the Judge, with handicap 0.20 │
                                          └───────────────┬───────────────┘
                                                          │ 200 scores
                                                          ▼
                                          ┌───────────────────────────────┐
                                          │ keep ONLY the 20 scores of    │
                                          │ the current task              │
                                          └───────────────┬───────────────┘
                                                          │ 20 scores
                                                          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 1: Classification · weight 1                                          ║
║   "among the 20 classes of this task, the true one should score highest"     ║
║   (cross-entropy, 2% label smoothing, against the LOCAL label)               ║
║   teaches: the Eye, the Librarian, the Gate, and the 20 current templates    ║
║   old classes' templates are untouched by this lesson (that is the point)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 9b. Lessons 2 and 3: anchoring to the past Eyes (tasks 2 onward only)

Two frozen snapshots of the Eye are kept: **Eye-0**, taken after task 1, and **Eye-previous**, taken
after the task just before this one. The live Eye must keep describing pictures the way they did.

```
                               (the same batch of pictures)
            ┌─────────────────────────┼──────────────────────────┐
            ▼ teaching allowed        ▼ teaching blocked         ▼ teaching blocked
┌──────────────────────┐  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
│ live Eye     [learns]│  ▓ Eye-previous (snapshot)  ▓  ▓ Eye-0 (snapshot)       ▓
│ "see", tap A, tap B  │  ▓ "see", tap A, tap B      ▓  ▓ "see"                  ▓
└──────────┬───────────┘  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
           │                          ╌                          ╌
           ▼                          ▼                          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 2: Feature anchoring · weight 3.0 × warm-up                           ║
║   "describe each picture in the same DIRECTION as Eye-previous and Eye-0"    ║
║   half the lesson compares with Eye-previous, half with Eye-0                ║
║   (1 − cosine similarity, averaged over the batch)                           ║
║   teaches: the Eye only                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

 taps A (512 layers of 16x16) and B (1024 layers of 8x8), live versus Eye-previous
                                      │
                                      ▼  for each tap
┌──────────────────────────────────────────────────────────────────────────────┐
│ Column profile: average each layer top-to-bottom  -> "what is where, left    │
│   to right" (8,192 numbers for either tap)                                   │
│ Row profile: average each layer left-to-right  -> "what is where, top to     │
│   bottom" (8,192 numbers)                                                    │
│ compare live and snapshot profiles by direction; average the two profiles    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 3: Layout anchoring (PODNet) · weight 1.5 × warm-up                   ║
║   "keep noticing things in the same PLACES of the picture as before"         ║
║   averaged over taps A and B; direction-based, not squared distance          ║
║   teaches: the Eye only                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

In the first task there is nothing to anchor to, so both lessons are zero. If Eye-0 is missing (only
possible on a broken resume), lesson 2 uses Eye-previous alone.

### 9c. Lessons 4, 5, 6: shaping features around the prototypes

```
 ("what I see" 512) ──▶ scale to length 1 ──▶ (unit feature)
                                        │
     ▓ Live sketches ▓ ╌╌▶ the 20 current sketches, length 1 each (teaching blocked)
                                        │                                   ╌
                                        ▼                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 4: Prototype pull (contrastive) · weight 0.5                          ║
║   "a picture's feature should sit closest to ITS class's sketch and away     ║
║    from the other 19 sketches"  (similarities ÷ 0.20, then cross-entropy)    ║
║   switched on once at least 2 sketches exist                                 ║
║   teaches: the Eye                                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

     ▓ Prototype shelves ▓ ╌╌▶ all past prototypes, length 1 each (teaching blocked)
                                        │                                   ╌
                                        ▼                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 5: Push away from old memories · weight 2.0                           ║
║   "a NEW picture must not look like an OLD class"                            ║
║   for each picture take its 32 most similar past prototypes; any similarity  ║
║   above 0.20 is penalised (squared). Only the 32 nearest count so that       ║
║   hundreds of irrelevant memories do not drown the signal.                   ║
║   tasks 2 onward ; teaches: the Eye                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

     (past prototypes) ╌╌▶ pick up to 128 of them at random as test questions
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Ask the Librarian about each chosen prototype, with ALL past prototypes as   │
│ the memory it may consult                                           [learns] │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 6: Memory fidelity · weight 1.0                                       ║
║   "asked about a stored memory, the Librarian should hand that memory back"  ║
║   (mean squared difference between what came back and what was asked)        ║
║   tasks 2 onward ; teaches: the Librarian only                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 9d. Lesson 7: Memory rehearsal for the Judge (tasks 2 onward only)

No old pictures exist, so the Judge is rehearsed on **imitation features**: stored prototypes with a
little noise added, shaped by each class's recorded spread.

```
 ▓ Prototype shelves + Class spreads ▓ ╌╌▶ all past prototypes with their class ids and spreads
 ▓ Live sketches ▓                    ╌╌▶ current sketches with their class ids, small fixed spread
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Put them in one list  (past: 200 per finished task ; current: up to 20)      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
   ◇ more than 256 in the list? ◇
        │ YES: keep a fair sample: 256 ÷ (tasks seen) per task, chosen at random,
        │      so every task is rehearsed equally and the step stays cheap
        │ NO : keep them all
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Jiggle: add random noise to each prototype, half the size of that class's    │
│ spread in each of the 512 directions; scale back to length 1                 │
│   -> an imitation of a real feature from that class                          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ╌ teaching blocked from here to the Judge
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Pass the imitations through the Librarian and the Gate (§8b) so they live    │
│ in the same fused space as real pictures (memory = finished tasks)           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ the Judge, no handicap ; keep the scores of all classes seen so far          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 7: Memory rehearsal · weight 2.0                                      ║
║   "the Judge must still name OLD classes correctly from imitation features"  ║
║   (cross-entropy over all seen classes)                                      ║
║   teaches: the Judge's class templates ONLY; this is the only lesson during  ║
║   live training that keeps old templates in shape                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 9e. Lesson 8: Weight anchoring (EWC), M4 only

```
 ▓ Importance of every weight (Fisher) ▓          ▓ Saved copy of every weight ▓
                      ╌                                        ╌
                      ▼                                        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ Lesson 8: Weight anchoring (EWC) · strength 5000                             ║
║   "do not move the weights that old tasks relied on"                         ║
║   for every weight: importance × (current value − saved value)², summed,     ║
║   × 5000 / 2                                                                 ║
║   covers every adjustable number: Eye, Librarian, Gate, Judge                ║
║   tasks 2 onward (after the first importance measurement) ; absent in M3     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 9f. The eight lessons at a glance

| # | Lesson | Compares | Pushes the model to | Weight | Active | Teaches |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Classification | 20 current scores vs true local label | name the current class decisively | 1 | always | Eye, Librarian, Gate, 20 templates |
| 2 | Feature anchoring | live feature vs Eye-previous and Eye-0 | describe pictures as the old Eyes did | 3.0 × warm-up | tasks 2+ | Eye |
| 3 | Layout anchoring | taps A/B row and column profiles vs Eye-previous | keep noticing things in the same places | 1.5 × warm-up | tasks 2+ | Eye |
| 4 | Prototype pull | feature vs the 20 live sketches | cluster tightly around its own class | 0.5 | ≥ 2 sketches | Eye |
| 5 | Push away | feature vs its 32 nearest past prototypes | not resemble old classes | 2.0 | tasks 2+ | Eye |
| 6 | Memory fidelity | Librarian's answer vs the memory asked about | recall memories faithfully | 1.0 | tasks 2+ | Librarian |
| 7 | Memory rehearsal | Judge's scores on imitation features vs their class | keep old templates sharp | 2.0 | tasks 2+ | Judge templates |
| 8 | Weight anchoring | every weight vs its saved value, scaled by importance | leave important weights alone | 5000 inside | M4, tasks 2+ | everything |

Warm-up: 0.5 at epoch 0, 0.6, 0.7, 0.8, 0.9, then 1.0 from epoch 5 on; zero during task 1.

In plain words: lesson 1 teaches the new chapter. Lessons 2 and 3 say "keep seeing the world the
way you used to". Lessons 4 and 5 arrange new knowledge so it does not collide with old knowledge.
Lesson 6 keeps the Librarian honest. Lesson 7 keeps the Judge practised on old classes using
imitation examples. Lesson 8 nails down the brain cells that mattered before.

---

## 10. How fast each part learns

The optimiser is rebuilt at the start of every task, so nothing about step sizes carries over.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Four groups of weights, each with its own learning rate (AdamW, decay 0.01)  │
│                                                                              │
│   group            task 1                 tasks 2..10                        │
│   the Eye          0.008                  0.008 × β   (slowed down)          │
│   the Librarian    0.008                  0.008                              │
│   the Gate         0.008                  0.008                              │
│   the Judge        0.008                  0.008                              │
│                                                                              │
│   β shrinks a little every task: 0.200, 0.193, 0.186, 0.179, 0.172,          │
│   0.165, 0.158, 0.151, then held at the floor 0.150                          │
│   (the Eye becomes more cautious as more knowledge accumulates)              │
├──────────────────────────────────────────────────────────────────────────────┤
│ One-cycle shape within each task (3,950 steps = 79 batches × 50 epochs)      │
│   starts at 1/10 of the rate, peaks after 15% of the steps (about step 590), │
│   then eases down to 1/1000 of the rate by the end                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Mixed precision (16-bit arithmetic where safe) with a safety scaler; the     │
│ schedule only advances on numerically healthy steps                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

The Library's storage is in none of these groups; it is never changed by the optimiser.

---

## 11. What happens after a task is finished

After the 50 epochs of task t, the following runs once, in exactly this order.

### 11a. Overview

```
 end of epoch 50 of task t
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ A  Fill the shelf for task t (§6d): 10 prototypes + spread per class;        │
│    wipe the live sketches                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ B  Refresh every OLDER shelf (§11b): re-run each earlier task's training     │
│    pictures through the Eye as it is NOW and rewrite their prototypes        │
├──────────────────────────────────────────────────────────────────────────────┤
│ C  "Weight aligning" on the Judge: sets the per-class volume of all seen     │
│    classes to 1.0, which it already is. Changes nothing (§4). Tasks 2+.      │
├──────────────────────────────────────────────────────────────────────────────┤
│ D  Re-tune the Judge on imitation features of ALL seen classes (§11c).       │
│    Tasks 2+.                                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ E  After task 1 only: freeze a copy of the Eye as Eye-0                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ F  Freeze a copy of the Eye as Eye-previous for the next task                │
├──────────────────────────────────────────────────────────────────────────────┤
│ G  M4 only: measure weight importance and save the anchor copy (§11d)        │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 grade the model on every task so far (§12), save a checkpoint
```

Because G runs last, the saved anchor already includes the re-tuned Judge.

### 11b. Refreshing older shelves: what actually changes

```
 before:  shelf τ holds prototypes written by an OLDER version of the Eye (stale)
 after :  shelf τ holds prototypes of the SAME 10,000 pictures seen through TODAY's Eye
 cost  :  one pass over every earlier task's pictures (t passes after task t)
 why   :  the Eye drifts a little each task; without this, nearest-prototype
          grading and the Librarian would compare apples with old oranges
```

### 11c. Re-tuning the Judge (step D)

```
 ▓ all filled shelves + spreads ▓
        ╌  every prototype becomes one training item with its class id and spread
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 200 items per finished task (400 after task 2 ... 2,000 after task 10)       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Freeze the Eye, the Librarian and the Gate. Only the Judge may change.       │
│ Small learning rate (0.005), easing down over 25 epochs                      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼  repeat for 25 epochs
┌──────────────────────────────────────────────────────────────────────────────┐
│ shuffle; take 64 items at a time                                             │
│   jiggle each item 5 different ways (noise = half its class's spread)        │
│   -> 320 imitation features, scaled to length 1                              │
│   pass them through the Librarian and the Gate (§8b, teaching blocked;       │
│   the memory now INCLUDES the task just finished)                            │
│   the Judge scores them WITHOUT handicap; keep the scores of seen classes    │
│   lesson: "name the right class" (cross-entropy) ; move the Judge's weights  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Unfreeze everything                                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

Only the Judge's class templates change here. This step exists because during the task the Judge
heard mostly about the 20 new classes and would otherwise favour them.

### 11d. Measuring weight importance (step G, used by M4 and M2)

```
 (the finished task's training pictures)      the whole model, in "no learning" mode
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ For 2,000 pictures (16 batches):                                             │
│   run the full model (Librarian consults every filled shelf)                 │
│   score the true class among all seen classes                                │
│   ask "how much would each weight have to change to alter this answer?"      │
│   square that sensitivity and add it up per weight                           │
│ divide by the number of batches  -> one importance number per weight         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓ running importance = new importance            after the first task           ▓
▓                    = 0.88 × old + new           afterwards (M2 uses 0.95)     ▓
▓ anchor copy        = every weight's current value                             ▓
▓ (saved to disk with the checkpoint in M4)                                     ▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

In plain words: after each chapter the model writes its summaries, refreshes the old summaries so
they match its current way of seeing, re-drills the Judge on all chapters using imitation examples,
takes two photographs of the Eye for next time, and notes which brain cells mattered most.

---

## 12. How the model is graded: four ways to answer

After every task, MAVIS is tested on the validation pictures of **every task so far**. Nothing
learns during grading. Four different answering strategies are scored side by side.

```
 (validation pictures of task τ)   true global labels   "seen" = 20 × (tasks finished)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Full MAVIS pass (§8), Librarian consults ALL prototypes                      │
│ -> 200 scores and "what I see" per picture                                   │
└───────┬──────────────────────────────────────────────────┬───────────────────┘
        │ scores                                           │ "what I see"
        ▼                                                  ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────────┐
│ Answer 1: the JUDGE              │   │ Answer 2: NEAREST PROTOTYPE          │
│ keep the scores of seen classes, │   │ compare the feature with all stored  │
│ pick the highest (also top-5)    │   │ prototypes, best-of-10 per class,    │
└──────────────┬───────────────────┘   │ pick the highest                     │
               │                       └──────────────┬───────────────────────┘
               │                                      │
               │     ◇ no prototypes yet? ◇ YES (a safety net; never true after a task):
               │        all other answers copy answer 1
               │                                      │ NO
               └──────────────────┬───────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Answer 3: the BLEND                                                          │
│   turn both score lists into probabilities (Judge softened by 0.90,          │
│   prototypes sharpened by 0.05)                                              │
│   for each class, how old is it? newest task = 0, one task older = 1, ...    │
│   trust in the prototypes rises with age: 19% for the newest task, 50% one   │
│   task back, 81% two tasks back, almost 100% for the oldest                  │
│   blend = (1 − trust) × Judge + trust × prototypes ; rescale to sum to 1     │
│   pick the highest (also top-5)                                              │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Answer 4: TWO-STAGE (first guess the task, then the class)                   │
│   stage 1  which task is this picture from?                                  │
│     clue A: for each task, average its 3 best prototype scores, sharpen ÷0.18│
│     clue B: for each task, add up the blend's probabilities of its classes   │
│     task belief = 50% clue A + 50% clue B  (one number per task, sums to 1)  │
│     (the top task belief is also scored as "task routing accuracy")          │
│   stage 2  within each task, share out the task belief among its 20 classes  │
│     in exact proportion to the blend: class = task belief × (blend of class  │
│     ÷ total blend of that task). This is an exact probability split, NOT a   │
│     second softmax; a unit test guards it.                                   │
│   pick the highest (also top-5)                                              │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Recorded per task: 7 accuracies (Judge, Judge top-5, prototype, blend,       │
│   blend top-5, two-stage, two-stage top-5) -> one new row in 7 tables;       │
│   the "primary" table is a copy of the two-stage table, so 8 tables are kept │
│   task routing accuracy is printed but not stored                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Saved to disk: the full model (Library included), the importance/anchor      │
│   state (M4), and the 8 accuracy tables                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Forgetting** for a table: for each earlier task, take its best accuracy ever recorded and subtract
its accuracy now; average over the earlier tasks.

In plain words: the Judge is best at the newest material, the prototype lookup is best at old
material, so the blend trusts each where it is strong. The two-stage answer first asks "which
chapter is this from?" and only then "which class within that chapter?", which is how the headline
MAVIS numbers are produced.

---

## 13. The run from start to finish

```
 python main.py <command>  with optional: --config, --epochs, --ewc-lambda, --no-ewc, --fresh
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Load settings (a YAML file or the built-in defaults) and apply the overrides │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
   setup / verify   ─▶ check PyTorch and the GPU, download the dataset if missing
   config           ─▶ print the settings
   visualize        ─▶ save a grid of sample pictures
   train_baseline   ─▶ M1                                   -> baseline_results.json
   train_ewc        ─▶ M2                                   -> ewc_results.json
   train_mavis      ─▶ M4 (or M3 with --no-ewc)  -> mavis_results.json / mavis_m3_results.json
                                       │
                                       ▼  (MAVIS only) resume rules
┌──────────────────────────────────────────────────────────────────────────────┐
│ --fresh: delete old checkpoints, start at task 1                             │
│ otherwise: find the latest finished task's checkpoint and continue from the  │
│   next task, restoring the model, the Library, Eye-previous, Eye-0 (from     │
│   the task-1 checkpoint), the 8 accuracy tables and the anchoring state      │
│ if all 10 tasks are already done: archive that folder with a timestamp and   │
│   start a brand-new run from task 1                                          │
│ if the checkpoint does not fit the current architecture: start from task 1   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ for each remaining task:                                                     │
│   train it (§9, §10) ; finish it (§11) ; grade everything so far (§12)       │
│   print forgetting                                                           │
│ at the end: write all tables and forgetting numbers to results/raw/          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Facts about the code that are easy to get wrong

Draw what actually runs, not what the settings file or old comments say:

- The settings file lists Eye channel counts of 64/128/256/512 and a "batch norm" switch; neither
  is used. The Eye really has 128/128/256/512/1024 channels and uses group normalisation.
- The settings file lists attention key/query sizes of 256; they are not used. All four Librarian
  re-expressions are 512 → 512, with 4 heads of 128.
- The settings' "fusion dim" and "memory capacity" fields are not read anywhere.
- Prototypes per class is 10 (from the settings). Old comments in the memory code still say 3.
- The "smoke test" settings and the mixup/cutmix switches are not read. Only `--epochs` shortens a
  run, and it still runs all 10 tasks.
- The k-means starting points are evenly spaced samples, not the "k-means++" the comment claims.
- "Weight aligning" after each task sets values that are already 1.0 to 1.0. It changes nothing.
- M2 uses an importance decay of 0.95, not the 0.88 in the settings, because its trainer never
  passes that value along.
- M1 and M2 grade and learn over all 200 classes, including ones not yet taught. MAVIS restricts
  learning to the current 20 and grading to the seen classes.
- Label smoothing is 10% in M1/M2 and 2% in MAVIS.
- Prototypes are built from training pictures **with** the random variations switched on.
- The Judge re-tuning step has a default of 10 jiggles per prototype in its code, but the trainer
  asks for 5.
- Two helper functions (a combined validation loader and a mean-prototype reader) exist but are
  never called.
- The published M2 number used anchoring strength 50,000; the default is 5,000.
