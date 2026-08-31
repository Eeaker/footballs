# Third-party notices

## Tryolabs Soccer Video Analytics

- Upstream: <https://github.com/tryolabs/soccer-video-analytics>
- Reviewed/migrated commit: `b00c7c75a1d52bd5ca183521a35fc76fbd89cc0e`
- Source file: `soccer/pass_event.py`
- License: MIT; full text at `licenses/TRYOLABS_MIT.txt`
- Migrated code: `analysis_lib/vendor/tryolabs_pass_rules.py`

Only the two validation rules from `PassEvent.validate_pass` are ported: the players must differ and their teams must match. Data classes and surrounding pipeline are local adapters because the upstream implementation uses Norfair/PIL domain objects and pixel thresholds, while this task requires tracking CSV inputs and a strict meter threshold.

## Local tracking delivery

The following project-owned modules were copied verbatim so the match analysis implementation uses the already reviewed feature and projection semantics instead of regenerating them:


Their byte/code equivalence is covered by `tests/test_source_migration.py` after removing the migration provenance module docstring.

## Football Player Identification — JerseyOCR

- Upstream: <https://github.com/Cappetti99/football-player-identification>
- Pinned commit: `b1bd36428ba55ed970ebda01d17559b9cd044bb6`
- License: MIT; full text at `licenses/FOOTBALL_PLAYER_IDENTIFICATION_MIT.txt`
- Directly used code: `ft/features/jersey_ocr.py` and its upstream support modules
- Local adapter: `analysis_lib/jersey_numbers.py`

The upstream implementation supplies crop-level augmentation deduplication,
multi-frame voting, numeric filtering and abstention diagnostics. The local
adapter only converts MOT tracks to crop rows, adds the stricter player-card
identity gate, detects simultaneous same-team/number conflicts and emits the
`clip_eligibility.json` contract.

## First-party shared helpers

Tracking actor, homography and team-feature helpers are imported from `engine/tracking/tracking_lib` through `analysis_lib/tracking_adapter.py`; copied first-party implementations are not bundled here.
