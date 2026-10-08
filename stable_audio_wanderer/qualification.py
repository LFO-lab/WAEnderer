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
    for mode in ('wander','manual','reorganized'):
        steps = {row.get('step') for row in transitions if ('wander' if row.get('mode') == 'random' else row.get('mode')) == mode and row.get('settled_ms') is not None}
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


def evaluate_multi_vae_campaign(manifest_path):
    """Check independent Phase 5 claims. Never reinterpret historical reports.

    A claim names its expected model/backend/device and report paths. Missing
    reports remain pending; functionally valid CPU stays valid after underruns.
    """
    from eval_scripts.multi_vae_validation_common import digest, numerical_gate
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported multi-VAE campaign schema')
    root = manifest_path.parent
    policy_path = root / manifest['protocol']
    policy = json.loads(policy_path.read_text())
    policy_hash = digest(policy_path)
    default_policy_path = policy_path
    default_policy = policy

    def read(relative):
        if not relative:
            return None
        path = root / relative
        return json.loads(path.read_text()) if path.is_file() else None

    def bound(report, claim, baseline=None, *, check_policy=True):
        if not report or report.get('schema_version') != 1:
            return False
        if report.get('vae_id') != claim['vae_id'] or check_policy and report.get('policy_sha256') != policy_hash:
            return False
        if not report.get('artifact_identity') or not report.get('corpus_sha256'):
            return False
        if baseline:
            return all(report.get(k) == baseline.get(k) for k in
                       ('artifact_identity','artifact_files','model_source','corpus_sha256'))
        return True

    def metric_valid(metric):
        return isinstance(metric, dict) and all(_at_least(metric.get(k), 0) for k in ('rmse','max_abs_error'))

    def numeric_passed(report, claim):
        if not bound(report, claim) or report.get('error') or report.get('numeric_passed') is not True:
            return False
        if report.get('validation_scope') == 'quick_diagnostic':
            return False
        limits = policy['models'][claim['vae_id']] if str(report.get('device','')).startswith('cpu') else policy['native_gpu'][claim['vae_id']]
        if limits is None:
            return False
        reference_receipt = policy.get('native_gpu_reference',{})
        if claim['device'] != 'cpu' and reference_receipt.get('vae_id') == claim['vae_id']:
            # Native characterization receipts use project-relative paths,
            # like the source/audio receipts generated by the audit scripts.
            reference_path = Path(__file__).resolve().parents[1] / reference_receipt['report']
            if not reference_path.is_file() or digest(reference_path) != reference_receipt.get('sha256'):
                return False
            reference = json.loads(reference_path.read_text())
            if reference.get('passed') is not True or reference.get('error') or reference.get('candidate_limits') != limits:
                return False
            if any(reference.get(k) != report.get(k) for k in ('vae_id','artifact_identity','artifact_files','model_source','corpus_sha256','device')):
                return False
            expected_reference = {(w,k) for w in policy['windows'] for k in ('real','synthetic')}
            if {(r.get('window'),r.get('kind')) for r in reference.get('rows',[])} != expected_reference:
                return False
            for r in reference['rows']:
                metrics = (r.get('cross'),r.get('within_cpu'),r.get('within_gpu'))
                if not all(metric_valid(m) for m in metrics) or not numerical_gate(limits,*metrics)['passed']:
                    return False
        rows = report.get('comparisons', [])
        synthetic = report.get('synthetic', [])
        clips = report.get('listening_pairs', [])
        if not rows or not clips or {r.get('window') for r in rows} != set(policy['windows']):
            return False
        # Every real probe position must appear at every window.
        coverage = {(r.get('file_index'),r.get('start')) for r in rows}
        if any({(r.get('file_index'),r.get('start')) for r in rows if r.get('window') == w} != coverage for w in policy['windows']):
            return False
        compact = claim.get('qualification_profile') == 'compact'
        if claim.get('qualification_profile', 'full') not in ('full','compact'):
            return False
        if report.get('validation_scope') == 'compact_qualification' and not compact:
            return False
        if claim['vae_id'].startswith('ear_'):
            fixture = report.get('real_input_fixture')
            recorded = report.get('corpus_spec',{}).get('weights_sha256')
            if not (recorded and recorded == report.get('model_source',{}).get('weights_sha256') or
                    fixture and fixture.get('sha256') and fixture.get('audio_sha256') and
                    fixture.get('source') == report.get('model_source')):
                return False
        if len(coverage) < (1 if compact else len(policy['positions_per_file'])):
            return False
        seeds = 1 if compact and claim['vae_id'] != 'same_s' else policy['synthetic_seeds']
        expected = {(w,s) for w in policy['windows'] for s in range(seeds)}
        if not expected.issubset({(r.get('window'),r.get('seed')) for r in synthetic}):
            return False
        for group, is_synthetic in ((rows,False),(synthetic,True),(clips,False)):
            for row in group:
                cross = row.get('comparison_after_ola') if group is clips else row.get('cross')
                a, b = row.get('within_onnx'), row.get('within_native')
                if not row.get('input_sha256') or not all(metric_valid(m) for m in (cross,a,b)):
                    return False
                if not numerical_gate(limits, cross, a, b, synthetic=is_synthetic)['passed']:
                    return False
        return True

    matrix = []
    for claim in manifest['claims']:
        # A GPU policy can be frozen later without invalidating earlier CPU
        # evidence. Each claim still requires its exact protocol digest.
        policy_path = root / claim['protocol'] if claim.get('protocol') else default_policy_path
        policy = json.loads(policy_path.read_text())
        policy_hash = digest(policy_path)
        row = {k:claim[k] for k in ('vae_id','backend','device')}
        row.update(availability=claim.get('availability','installed'), functional='pending',
                   numerical='pending', realtime='not_tested', listening='pending', lifecycle='pending', reasons=[])
        row['reports'] = {k:claim.get(k) for k in ('numerical_report','runtime_report','lifecycle_report','listening_report')}
        row['protocol'] = str(policy_path)
        row['windows'] = list(policy['windows'])
        row['qualification_profile'] = claim.get('qualification_profile','full')
        numeric = read(claim.get('numerical_report'))
        correct_device = numeric and (claim['backend'] == 'onnxruntime' and claim['device'] == 'cpu' and str(numeric.get('device','')).startswith('cpu') or
            claim['backend'] == 'pytorch' and claim['device'] == numeric.get('device'))
        numeric_ok = bool(correct_device and numeric_passed(numeric, claim))
        if numeric:
            if numeric_ok:
                row['functional'] = row['numerical'] = 'passed'
            elif numeric.get('validation_scope') == 'quick_diagnostic' and correct_device and bound(numeric,claim) and not numeric.get('error') and numeric.get('numeric_passed') is True:
                row['functional'] = 'observed'
                row['numerical'] = 'diagnostic_passed'
                row['reasons'].append('Full numerical coverage pending; quick diagnostic passed')
            else:
                row['functional'] = row['numerical'] = 'failed'
        runtime = read(claim.get('runtime_report'))
        if runtime:
            reasons = []
            if not numeric_ok:
                reasons.append('Numerical evidence missing, failed or pending')
            if not bound(runtime, claim, numeric) or runtime.get('backend') != claim['backend'] or runtime.get('device') != claim['device']:
                reasons.append('Runtime identity does not match numerical evidence and claim')
            if not runtime.get('audio_device') or not runtime.get('audio_device_info') or 'physical' not in runtime.get('clock','') or runtime.get('navigation') != 'production':
                reasons.append('Physical device and production navigation required')
            if not _at_least(runtime.get('active_audio_seconds'), policy['minimum_active_seconds']):
                reasons.append('Less than ten minutes of rendered physical audio')
            sr = runtime.get('sample_rate')
            if not _at_least(sr, 1) or not _at_least(runtime.get('rendered_samples'), policy['minimum_active_seconds'] * (sr or 1)):
                reasons.append('Insufficient rendered samples')
            if runtime.get('error') or runtime.get('nonfinite_pcm') is not False or any(type(runtime.get(k)) is not int or runtime[k] != 0 for k in ('underruns','buffer_underruns','device_underruns')):
                reasons.append('Runtime error, nonfinite PCM or underruns')
            profile = claim.get('realtime_profile', dict(windows=policy['windows'],
                modes=['wander','manual','reorganized'], adaptive=True))
            windows, modes = profile.get('windows',[]), profile.get('modes',[])
            valid_profile = (bool(windows) and len(set(windows)) == len(windows) and
                set(windows).issubset(policy['windows']) and bool(modes) and
                len(set(modes)) == len(modes) and set(modes).issubset({'wander','random','manual','reorganized'}) and
                type(profile.get('adaptive')) is bool)
            if not valid_profile or runtime.get('realtime_profile') != profile:
                reasons.append('Runtime profile does not match explicit claim')
            expected = {f'{mode}:{setting}' for mode in modes
                for setting in [*[f'T{w}' for w in windows], *(['adaptive'] if profile.get('adaptive') else [])]}
            counts = runtime.get('scenario_samples',{})
            if set(runtime.get('declared_scenarios',[])) != expected or any(not _at_least(counts.get(s), sr or 1) for s in expected):
                reasons.append('Incomplete scenario coverage (one steady second per member required)')
            if (runtime.get('sample_attribution') != 'stable_pcm_generation_without_underrun' or
                    not all(_at_least(v,0) for v in counts.values()) or
                    not _at_least(sum(v for v in counts.values() if _at_least(v,0)), policy['minimum_active_seconds'] * (sr or 1)) or
                    sum(v for v in counts.values() if _at_least(v,0)) > runtime.get('rendered_samples',0)):
                reasons.append('Insufficient or invalid steady PCM generation attribution')
            settled = {r.get('scenario') for r in runtime.get('transitions',[]) if _at_least(r.get('settled_ms'),0)}
            if not expected.issubset(settled):
                reasons.append('Unsettled command transitions')
            row['realtime_profile'] = profile
            for w in windows:
                timing = runtime.get('windows',{}).get(str(w),{}).get('decode_ms',{})
                if not _at_least(timing.get('count'),1) or any(not _at_least(timing.get(k),0) for k in ('median','p95','p99')):
                    reasons.append(f'Missing T{w} decode timings')
            if any(not _at_least(runtime.get(k),0) for k in ('preparation_ms','process_cpu_percent','peak_rss_bytes')):
                reasons.append('Missing preparation/CPU/memory measurements')
            row['realtime'] = 'passed' if not reasons and numeric_ok else 'failed'
            row['reasons'].extend(reasons)
            row['window_measurements'] = runtime.get('windows',{})
        lifecycle = read(claim.get('lifecycle_report'))
        if lifecycle:
            # Lifecycle checks source/artifact ownership, not numerical limits.
            ok = bound(lifecycle,claim,numeric,check_policy=False) and not lifecycle.get('error') and lifecycle.get('passed') is True and lifecycle.get('offline_reload') is True
            choices = {(r.get('backend'),r.get('device')) for r in lifecycle.get('cycles',[]) if r.get('stopped') is True}
            ok = ok and (claim['backend'],claim['device']) in choices and len(choices) >= 2
            row['lifecycle'] = 'passed' if ok else 'failed'
        reviews = claim.get('listening', [])
        if reviews:
            row['listening'] = 'failed'
        # Listening remains independent of full numerical coverage: a reviewed
        # diagnostic pair is useful evidence, but cannot qualify the campaign.
        listening_report = read(claim.get('listening_report')) if claim.get('listening_report') else numeric
        listening_device = listening_report and (claim['device'] == listening_report.get('device'))
        if reviews and listening_report and listening_device and bound(listening_report,claim,numeric) and not listening_report.get('error'):
            pairs = {r['file_index']:r['files'] for r in listening_report['listening_pairs']}
            ok = all(r.get('reviewer') and r.get('date') and r.get('observation') and
                r.get('new_engine_regression') is False and r.get('files') == pairs.get(r.get('file_index')) for r in reviews)
            row['listening'] = 'passed' if ok else 'failed'
        row['realtime_claimed'] = claim.get('claim_realtime',False)
        row['passed'] = numeric_ok and row['listening'] == 'passed' and row['lifecycle'] == 'passed' and (not row['realtime_claimed'] or row['realtime'] == 'passed')
        matrix.append(row)
    models = set(default_policy['models'])
    required = {(v,b,'cpu') for v in models for b in ('onnxruntime','pytorch')}
    covered = {(r['vae_id'],r['backend'],r['device']) for r in matrix}
    return dict(schema_version=1, matrix=matrix, required_cpu_coverage=required.issubset(covered),
                untested_hardware=manifest.get('untested_hardware',[]),
                passed=bool(matrix) and required.issubset(covered) and all(r['passed'] for r in matrix))
