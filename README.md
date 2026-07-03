# Continuous PHQ-9 Regression from Mandarin Speech

A task- and valence-stratified analysis on the MODMA clinical corpus. We predict
**PHQ-9 depression severity as a continuous score (0-27)** from speech, rather
than the binary depressed-vs-healthy classification that all prior published work
on this dataset has used. Graduate research project for **CSCI 567 (Machine
Learning) at USC**.

**Authors:** Ashwini Athreya, Alicia Danielle, Rasika Ramanan, Arianna Yuan.

## Motivation

MODMA (Multi-modal Open Dataset for Mental-disorder Analysis) pairs speech
recordings with clinician-verified diagnoses and self-reported PHQ-9 scores. Of
the 48 published MODMA papers we surveyed, none frame the task as PHQ-9
regression; all treat it as binary classification. But a clinician choosing
between watchful waiting, brief intervention, and urgent referral cares about the
gradient of symptom severity, not just whether someone crosses a diagnostic
threshold. We treat that gap as the problem: a calibrated, honest report of how
well continuous PHQ-9 prediction can be done at n = 52, with a defensible
evaluation protocol and a fine-grained look at which speech elicitations carry
the most signal.

## Approach

- **Data.** Audio modality only, n = 52 subjects (23 MDD, 29 HC), 29 recordings
  per subject across four elicitation tasks (clinical interview, word reading,
  passage reading, picture description).
- **Features.** 88 **eGeMAPSv02** acoustic functionals via openSMILE; 16
  **Whisper**-derived linguistic features (5 lexical, 3 syntactic, 8 sentiment)
  from Whisper-large-v3 Mandarin transcripts; plus demographic covariates.
- **Model.** Support Vector Regression (RBF kernel) with **Elastic-Net feature
  selection**, under **nested leave-one-subject-out (LOSO)** cross-validation so
  no speaker appears in both train and test.
- **Uncertainty.** Leave-one-out **conformal 95% prediction intervals** on every
  held-out prediction.
- **Fairness.** A post-hoc **multicalibration** step to reduce group-conditional
  prediction bias.

## Key results

- **Regression.** The best configuration (eGeMAPS + Whisper + demographics)
  reaches **RMSE 6.81** PHQ-9 points (R^2 = 0.34), a 1.75-point improvement over
  the mean-predictor baseline of 8.56. To our knowledge this is the first
  continuous PHQ-9 regression on MODMA.
- **Task stratification.** No single-task model matches the all-tasks model,
  indicating the depression signal is distributed across the elicitation set
  rather than concentrated in one task. Shortening the screening protocol by
  dropping elicitations would lose signal.
- **Valence stratification.** Positive-valence prompts are the only setting where
  linguistic features beat acoustic ones, consistent with the anhedonia signal
  those prompts are designed to elicit.
- **Calibration.** Conformal intervals attain 92.3-94.2% empirical marginal
  coverage, but MDD subjects are systematically under-covered relative to HC, a
  clinically important asymmetry that motivates the multicalibration step.
- **Multicalibration.** Post-hoc correction shrinks the MDD and HC mean
  prediction gaps from 5.79 to 0.42 and from 3.56 to 0.51 PHQ-9 points
  respectively.

## Repository structure

```
report.pdf   the final paper
scripts/     feature extraction, training, calibration, and analysis code
notes/       technical documentation written during the project
```

### `scripts/`

| File | Purpose |
| --- | --- |
| `extract_opensmile_features.py` | eGeMAPSv02 acoustic functionals via openSMILE |
| `agg_egemaps.py` | aggregate per-recording eGeMAPS to per-subject vectors |
| `extract_whisper_features.py` | Whisper transcription + downstream features |
| `extract_text_features.py` | lexical / syntactic / sentiment features |
| `extract_wavlm_features.py` | WavLM speech-representation features |
| `preprocess_transcripts.py`, `audit_whisper_transcripts.py` | transcript cleaning and QA |
| `get_wav_stats.py` | audio duration / sample-rate diagnostics |
| `train_svr.py`, `helpers_svr.py` | SVR + Elastic-Net training under nested LOSO |
| `multicalibration.py`, `run_multicalibration.py`, `run_svr_multicalibration.py` | multicalibration and runners |
| `groupwise_pi_coverage.py`, `analyze_intervals_and_groupings.py` | prediction-interval and subgroup analysis |
| `helpers_viz.py` | plotting utilities |
| `analysis.ipynb`, `analysis_intervals_and_groupings.ipynb` | analysis notebooks |

### `notes/`

Technical write-ups produced during the project: a MODMA dataset guide, a
literature index over the surveyed papers, the feature-pipeline explainer, the
Elastic-Net justification, an eGeMAPS feature-pruning review, and the Whisper
transcription pipeline.

## Data

**No MODMA data is included in this repository.** MODMA is restricted-access from
Lanzhou University; access must be requested from the dataset authors. The
scripts expect locally downloaded audio and metadata.

> Cai, H., Yuan, Z., Gao, Y., et al. A multi-modal open dataset for
> mental-disorder analysis. *Scientific Data* **9**, 178 (2022).

## My role

I led the **held-out experiments and uncertainty quantification**: the
leave-one-out conformal prediction intervals, the per-valence and per-task SVR
experiments, and the analysis of which feature combinations narrow the prediction
intervals.
