"""Evidence gates for one explicitly scoped dual-inference release campaign.

A passing GPU report never qualifies another device. Historical diagnostic
failures should remain in the release notes even after a new scenario passes.
"""
import json
import math
from pathlib import Path


def _at_least(value, minimum):
    return isinstance(value, (int,float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def evaluate_transport(report):
    reasons = []
    if not report.get('audio_device') or 'physical' not in report.get('clock', ''):
        reasons.append('No physical audio device evidence')
    if report.get('navigation') != 'production':
        reasons.append('Production navigation policies were not exercised')
    if not _at_least(report.get('requested_seconds'), 600) or not _at_least(report.get('duration_seconds'), 600):
        reasons.append('Less than ten minutes requested or completed')
    if report.get('error') or report.get('underruns') != 0:
        reasons.append('Runtime error or nonzero/missing underruns')
    if not _at_least(report.get('rendered_blocks'), 20000):
        reasons.append('Insufficient rendered PCM blocks')
    expected = {str(w) for w in range(2,33,2)}
    if not expected.issubset(report.get('windows', {})):
        reasons.append('Not all T2..T32 windows were decoded')
    else:
        if any(report['windows'][w].get('decode_ms', {}).get('count', 0) < 1 for w in expected):
            reasons.append('Missing decode observations')
    transitions = report.get('transitions', [])
    for mode in ('random','manual','reorganized'):
        steps = {row.get('step') for row in transitions if row.get('mode') == mode and row.get('settled_ms') is not None}
        if not set(range(17)).issubset(steps):
            reasons.append(f'Incomplete fixed/adaptive transitions for {mode}')
    return {'passed':not reasons, 'reasons':reasons}


def evaluate_campaign(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    results = {}
    corpus_hashes = set()
    for name, relative in manifest['transport_reports'].items():
        report = json.loads((manifest_path.parent / relative).read_text())
        result = evaluate_transport(report)
        result.update(backend=report.get('backend'), device=report.get('device'), report=relative)
        results[name] = result
        corpus_hashes.add(report.get('source', {}).get('sha256'))
    numeric = json.loads((manifest_path.parent / manifest['corpus_report']).read_text())
    numeric_ok = bool(numeric.get('numeric_passed')) and bool(numeric.get('comparisons')) and all(
        row.get('passed') and row['cross']['rmse'] <= row['rmse_limit'] for row in numeric.get('comparisons', []))
    numeric_ok = numeric_ok and {r.get('window') for r in numeric.get('comparisons', [])} == set(range(2,33,2))
    both_backends = {r['backend'] for r in results.values()} == {'onnxruntime','pytorch'}
    same_corpus = corpus_hashes == {numeric.get('corpus_sha256')} and None not in corpus_hashes
    listening = manifest.get('listening', {})
    listening_recorded = all(listening.get(field) for field in ('reviewer','date','observation'))
    # No inference from similar-sounding text: an audible-regression verdict is explicit.
    listening_passed = listening_recorded and listening.get('new_engine_regression') is False
    return {'transport':results, 'same_corpus':same_corpus, 'both_backends':both_backends, 'numeric_passed':numeric_ok,
            'listening_recorded':listening_recorded, 'listening_passed':listening_passed,
            'unqualified_devices':manifest.get('unqualified_devices', []),
            'passed':both_backends and bool(results) and all(r['passed'] for r in results.values()) and numeric_ok and same_corpus and listening_passed}
