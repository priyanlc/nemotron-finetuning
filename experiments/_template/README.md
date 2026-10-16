<!--
Experiment template. Start a new experiment with:  cp -r experiments/_template experiments/<slug>
Slug: <model>-<method>-<data>-<hardware>, for example lightning-30b-lora-hellaswag-dgx-spark.
Folder: README.md · configs/ · results/ (CSVs plus a README) · assets/ · data/ (scripts and notes only; data is git-ignored).
Delete these comments as you fill the sections in. Keep every number traceable to a log or a committed results file.
-->

# <Model>: <method> on <hardware>, measured with <evaluation>

*Part of [nemotron-finetuning](../../README.md), a series of fine-tuning experiments on NVIDIA's Nemotron models.*

**One sentence on what was done and what it showed.** Status: planned | in progress | measured | published.

> **Tested setup:** <hardware, count and memory> · <container images with tags> · <model id> · <date>. Anything untested is marked as such.

## The question

<!-- What this experiment is meant to find out, and why the setup is what it is: model, method, data, hardware. Two or three paragraphs at most. -->

## Data

<!-- Source and licence. How one raw example becomes training text (show it). How much was used, and what is held out for validation and for evaluation. -->

## Steps

<!-- Numbered, in order. Each step says where it runs (host, which container, which terminal), what to type and what to expect. Pin container tags and recipe commits. Say how long each step took you. -->

### Step 1: <title>

### Step 2: <title>

## Results

<!-- What was measured and how. The headline numbers with their uncertainty, a before/after example, and what the numbers do and don't show. Point at results/ and its README. -->

## Under the hood (optional)

<!-- What the recipe's settings do, what one training step looks like, memory and precision. -->

## If it fails

<!-- Symptom → cause → fix, as a table. -->

## Files in this experiment

| Path | What it is |
|---|---|
| `README.md` | This guide. |
| `configs/…` | |
| `results/…` | |
| `assets/…` | |

## Third-party material and licences

<!-- Model, dataset, container images, upstream recipes, photos: name each with its licence. Add any file redistributed here to ../../LICENSES.md. -->

## References
