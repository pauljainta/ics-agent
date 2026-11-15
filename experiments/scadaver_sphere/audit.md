# Audit Status

Audited on 2026-11-12 from the implementation workstation:

- `mrg whoami --json` authenticated as `researcher`.
- The account can see `cpsproj`, `lgarcia`, and `sphereai` projects. No new
  SCADAver experiment has been created, realized, materialized, or modified.
- Existing PLCBench XDCs are visible and remain out of scope.
- `scadaver` is not installed locally. The upstream HEAD observed during audit is
  pinned in `config/scadaver_source.json`; deployment must verify the checked-out
  commit and binary help before using it.
- The local environment has a PLCBench-named OpenRouter secret but no generic
  `OPENROUTER_API_KEY`. The implementation reads only the configured variable
  name at runtime and never stores a value in source, metadata, or results.
- The legacy SPHERE controller/simulator runtime was removed from this minimal
  branch during cleanup. Its identity and bridge location therefore remain
  unverified for this new experiment.

Live deployment is blocked pending operator review of a new realization model,
current node images/resources, the runtime-backed controller/simulator staging
procedure, active enforcement of access-node egress isolation, and a private
configuration with actual lab addresses.
