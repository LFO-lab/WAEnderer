# Phase 5 — implementation and remaining qualification

Status: implementation delivered and review fixes applied on 4 October 2026. This plan reflects the current code. Evidence completion remains separate from tool delivery; see [the usage and evidence guide](MULTI_VAE_ONNX_PHASE5.md).

## Delivered

- Shared source/artifact/corpus/policy identities, finite float32 PCM checks and exact model geometry.
- Identical real and synthetic inputs, two calls per engine, production overlap-add and hashed local listening pairs for all four VAEs.
- Three numerical scopes: quick diagnostics, compact qualification and full coverage. The manifest explicitly selects the required scope; a full report can satisfy compact coverage. Tolerances remain frozen in the original protocol.
- EAR source-bound fixture inputs, checked against the selected artifact before decoding. The navigation corpus retains its original identity and historical provenance label.
- Runtime profiles with explicit windows, navigation modes and optional adaptive bounds. The benchmark uses production transport and counts steady PCM generations; mixed transitions, unattributed blocks and underruns are excluded.
- Independent numerical, lifecycle, listening and physical runtime statuses. A separate listening report preserves approval of the exact earlier WAVs when numerical coverage is expanded.
- Stopped engine changes and fresh-process offline reload checks for all four VAEs, including MPS; CPU listening feedback recorded for each model.
- Stable Audio Open native CPU/MPS characterization and a separately frozen GPU policy. Historical SAME-S qualification remains separate.
- All eight CPU claims pass numerical/lifecycle/listening gates: three compact campaigns and the existing full EAR 48k report. 29 focused tests pass. A four-second restricted software run preserves EAR 48k CPU underruns as diagnostic evidence.

## Numerical qualification contract

| Scope | Windows | Real positions | Synthetic inputs | OLA |
| --- | --- | --- | --- | --- |
| Quick diagnostic | T2/T8/T32 | One | One seed/window | One 32-frame excerpt |
| Compact qualification | All even T2–T32 | One shared position | One seed/window for deterministic VAEs; eight for SAME-S | One 32-frame excerpt |
| Full | All even T2–T32 | Four positions per eligible file | Eight seeds/window | Up to 128 frames per eligible file |

Every probe and OLA render retains two calls per engine and the same numerical tolerances. Compact qualification proves a narrower input/content sample; it does not assert broad audio coverage. A quick report cannot qualify either campaign. EAR real inputs require recorded checkpoint provenance or a source-bound fixture. Existing full EAR 48k CPU evidence is reused.

## Physical runtime contract

A claim may restrict its fixed windows and modes. Adaptive playback, when included, uses all supported even windows between the profile's minimum and maximum fixed windows. The manifest and report must contain identical profiles.

The ten-minute requirement applies to the combined model/backend/device/hardware profile. Require at least 600 seconds of attributed steady playback, at least one steady second per declared member, settled transitions and decode observations for all fixed windows, finite PCM, zero separate buffer/device underruns and no runtime error. This does not establish ten independent minutes per window or navigation mode. An adaptive segment starts with Stop/Start so previously buffered fixed PCM cannot be credited to adaptive intent.

Physical qualification remains a serial, manual campaign. Short replay/software runs are diagnostics; missing physical measurements remain `not_tested`. Functional CPU results survive measured audio deadline failures. CUDA remains untested on this host.

## Remaining shared work

1. Choose the GPU/backend/window profiles that warrant physical qualification; complete numerical coverage and listening for those exact GPU pairs.
2. Run physical screening, then ten-minute sessions only for candidates carrying a real-time claim. Keep failures and narrow profile limits explicit.
3. Verify browser preference retention through refresh/reconnect and physical stop/start behavior with the user.
4. Update roadmap evidence checkboxes only when the corresponding matrix gates pass. Packaging remains Phase 6.

Validation is focused on changed evidence gates and PCM attribution. No full regression sweep or long physical session is needed merely to deliver these tooling changes.
