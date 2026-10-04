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

`colab/CAPE_TTA_OpenVLA_Oracle_Pilot.ipynb`

The notebook builds an isolated Python 3.10 environment, installs the OpenVLA/LIBERO stack, loads the official LIBERO-Spatial OpenVLA checkpoint in 8-bit FP16 mode for a free T4-class GPU, and runs a real K=4 fresh-replay Oracle-CAPE pilot. The output is `openvla_oracle_pilot_colab.json`.

This is a pilot / infrastructure result, not the final paper benchmark. Final results must report the quantization/runtime setting and be scaled to the prespecified task/init-state evaluation grid.
