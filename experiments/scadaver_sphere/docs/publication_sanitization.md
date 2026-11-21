# Public source snapshot and revalidation

Private filesystem paths and deployment coordinates were replaced for publication.
The production runtime manifest and enforced B2-R1 input fingerprints now identify
the sanitized source files. The checks still compare exact source bytes and reject
a mismatch; updating fingerprints does not qualify a deployment.

The original baseline commit and tag references are intentionally unset in current
study manifests. New runs record their actual implementation commit and input hashes.
The original deployment manifest digest was removed from the B2-R1 constants because
it was historical provenance, not an enforced check of the current manifest.

Existing reports and evidence bundles remain historical records. They have not been
rerun against the sanitized inputs and must not be presented as current readiness,
new model results, or validation of a new infrastructure reservation.

From the repository root, use the configured Python 3.10 environment to check the
public source baseline without loading a model or connecting to a testbed:

```bash
python -m pytest -q tests/test_reasoning_ablation.py tests/test_b2_r1.py tests/test_b2_r1_campaign.py
python -m experiments.scadaver_sphere.deployment_parity
```

The parity command checks the pinned Python and dependency versions as well as source
hashes. Before live research, configure a fresh authorized isolated deployment and
repeat its preflight, reset, and acceptance procedures. Capture new evidence for any
subsequent comparison; do not carry a prior acceptance result across source changes.
