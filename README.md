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
