# Real historical evidence, re-audited 2026-10-30

These raw JSON files came from six user-uploaded experiment ZIPs. They are NOT
new executions. `reaudit.json` and each `provenance.json` preserve original ZIP
names, ZIP SHA256, raw-file SHA256, source revisions where available, and counts
of verified original manifest entries (46 total). Raw experiment JSON is copied
byte-for-byte. Full original ZIPs remain the source for dependency/log manifests;
this directory is a raw-result subset, not a replacement of the complete bundles.

| Folder | Actual evidence | Interpretation |
|---|---|---|
| v3 | Real OpenVLA + LIBERO pilot | Infrastructure/action-consequence feasibility |
| v4 | Action-token sampling diagnostic | Candidate diversity, not task effectiveness |
| v5 | Task0, init0/1/2 paired H12 rollouts | Privileged action-consequence Oracle, no parameter update |
| v6-init0 | Task0 init0 complete-episode comparison | First-action replacement only |
| v6-init12 | Task0 init1/2 complete-episode comparison | First-action replacement only |
| v8-task1-init0 | Baseline and independent replay | Both succeed at116; candidate branches remain unrun |

Task0 init0/1/2 are development data. The V5 temperature1.5 Oracle soft margins
were 0.007800576937356363, 0.006211963980045976 and 0.0015273991197268244.
V6 frozen vs privileged-first-action outcomes: init0 success85 vs success74;
init1 budget failure220 vs success143; init2 success80 vs success90.
The failed220 rollout is censored at budget, not a measured time to success.
These three selected states do not estimate population performance or establish
that short-horizon potential predicts long-horizon gains reliably.

None of these six bundles contains an executed parameter-adaptation experiment,
a learned benefit predictor, independent conformal calibration, or a CAPE gate
evaluation. They cannot support claims of adaptation gains, HAR control, or
CCF A readiness. Unit-test fixtures are never placed in this directory.
