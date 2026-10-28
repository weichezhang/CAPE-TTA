# CAPE-TTA

**Counterfactual-Calibrated Progress Evidence for Reliable Test-Time Adaptation of Vision–Language–Action Policies**

This repository contains the CAPE-TTA research code and the LIBERO Oracle-CAPE experiment pipeline.

## Immediate experiment
The first gate is a real LIBERO infrastructure smoke test:
1. construct LIBERO-Spatial task 0;
2. restore the exact same MuJoCo state;
3. replay the same action chunk and verify determinism;
4. measure BDDL goal-predicate progress;
5. branch K candidate chunks from the same state;
6. save the raw JSON artifact.

This smoke test validates the simulator-side oracle machinery only. It is not a VLA benchmark result and does not by itself establish parameter-update benefit.

## Run
The GitHub Actions workflow is in `.github/workflows/libero-oracle-smoke.yml`.

## Free Colab OpenVLA pilot

For users without paid GPU access, run:

`colab/CAPE_TTA_OpenVLA_Oracle_Pilot_V3.ipynb`

The notebook builds an isolated Python 3.10 environment, installs the OpenVLA/LIBERO stack, loads the official LIBERO-Spatial OpenVLA checkpoint in 8-bit FP16 mode for a free T4-class GPU, and runs a real K=4 fresh-replay Oracle-CAPE pilot. The output is `openvla_oracle_pilot_colab.json`.

This is a pilot / infrastructure result, not the final paper benchmark. Final results must report the quantization/runtime setting and be scaled to the prespecified task/init-state evaluation grid.

## V3 evidence and current limits

[Open V3 in Colab](https://colab.research.google.com/github/weichezhang/CAPE-TTA/blob/main/colab/CAPE_TTA_OpenVLA_Oracle_Pilot_V3.ipynb).
V3 uses fixed source revision `86f393928ae264eb000c15e9a1bf2924c32fb47d`, an isolated Python 3.10 environment,
MuJoCo 2.3.7, parsed LIBERO paths, and visible subprocess logs. It downloads one ZIP
containing the manifest, complete log, environment versions, validation and any pilot result.
A failed run also produces a bundle. `pilot_validated` means pilot integrity only.

CPU verification: `python -m unittest discover -s tests -v` (no GPU or model required).
These tests do not validate OpenVLA loading or task performance. No successful GPU
run or CAPE-TTA effectiveness result is claimed by this revision.

The deployable evidence model, calibration protocol and parameter-update algorithm
are not implemented by the Oracle pilot. See [paper completion protocol](docs/PAPER_COMPLETION_PLAN.md)
for the staged work and remaining scientific requirements.
