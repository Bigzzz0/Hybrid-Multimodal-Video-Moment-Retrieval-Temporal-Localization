"""Frozen cross-dataset pilot with the official, strictly loaded UCF VadCLIP.

This is not a reproduction of the published UCF benchmark. Raw-video feature
extraction is explicit because the upstream repository distributes precomputed
features without an unambiguous end-to-end extraction recipe.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import psutil
import requests
import torch

from .open_anomaly_eval import audit, bootstrap, metrics


LABELS = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
          'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
          'Shoplifting', 'Stealing', 'Vandalism']
STEP = 16


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def blocks(frame_count, fps):
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid video timing')
    return [(i, min(i + STEP, frame_count), i / fps,
             min(i + STEP, frame_count) / fps) for i in range(0, frame_count, STEP)]


def threshold_events(bounds, scores, threshold, duration):
    scores = np.asarray(scores)
    if len(bounds) != len(scores) or not np.isfinite(scores).all() or not math.isfinite(threshold):
        raise ValueError('Invalid scores or threshold')
    result = []
    for bound, score in zip(bounds, scores):
        if score < threshold:
            continue
        a, b = max(0., float(bound[2])), min(duration, float(bound[3]))
        if b <= a:
            continue
        if result and a <= result[-1]['t_end'] + 1e-7:
            result[-1]['t_end'] = b
        else:
            result.append(dict(t_start=a, t_end=b, assessment='suspicious'))
    return result


def calibration_threshold(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Missing/nonfinite normal calibration scores')
    # Fixed before inspecting evaluation scores; no positive labels are fitted.
    return float(np.quantile(values, .95, method='higher'))


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss = 0
        self.gpu_used_mib = []
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        tick = 0
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            if tick % 10 == 0:
                try:
                    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=3)
                    if r.returncode == 0:
                        self.gpu_used_mib.append(float(r.stdout.strip().splitlines()[0]))
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            tick += 1
            self.stop.wait(.1)


def import_model(repo, weights):
    sys.path.insert(0, str((repo / 'src').resolve()))
    model_module = importlib.import_module('model')
    clip_module = model_module.clip
    # The released state_dict includes the complete frozen CLIP backbone. Build
    # it directly rather than downloading another potentially different base.
    clip_state = {k[len('clipmodel.'):]: v for k, v in weights.items() if k.startswith('clipmodel.')}
    backbone = importlib.import_module('clip.model').build_model(clip_state)
    preprocess = clip_module._transform(224)
    with patch.object(clip_module, 'load', return_value=(backbone, preprocess)):
        model = model_module.CLIPVAD(14, 512, 256, 512, 1, 2, 8, 10, 10, 'cuda')
    model.load_state_dict(weights, strict=True)
    return model.float().cuda().eval(), preprocess, importlib.import_module('crop').image_crop


def features(row, model, preprocess, crop, sampling):
    cap = cv2.VideoCapture(row['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bounds = blocks(expected, fps)
    if abs(expected / fps - row['duration']) > .08:
        raise ValueError('Decoded video timing differs from manifest')
    output, batch, ids = [], [], []
    decoded = 0
    encoded = 0
    def flush():
        nonlocal encoded
        if not batch:
            return
        with torch.inference_mode():
            value = model.clipmodel.encode_image(torch.stack(batch).cuda()).float().cpu().numpy()
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite CLIP features')
        output.extend(zip(ids, value)); encoded += len(batch)
        batch.clear(); ids.clear()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = decoded; decoded += 1
            if sampling == 'stride16' and index % STEP:
                continue
            batch.append(preprocess(Image.fromarray(crop(frame, 0))))
            ids.append(index)
            if len(batch) == 32:
                flush()
        flush()
    finally:
        cap.release()
    if decoded != expected:
        raise ValueError(f'Incomplete decode: {decoded}/{expected}')
    groups = [[] for _ in bounds]
    for index, value in output:
        groups[index // STEP].append(value)
    if any(not g for g in groups):
        raise ValueError('Missing feature block')
    # No additional feature normalization: upstream consumes raw CLIP vectors.
    values = np.stack([np.mean(g, axis=0) for g in groups])
    return values, bounds, {'decoded_frames': decoded, 'encoded_images': encoded,
                            'fps': fps, 'frame_count': expected}


def score(model, values):
    real = len(values)
    if not real:
        raise ValueError('Empty features')
    chunks = []
    lengths = []
    for offset in range(0, real, 256):
        value = values[offset:offset + 256]
        lengths.append(len(value))
        chunks.append(np.pad(value, ((0, 256 - len(value)), (0, 0))))
    x = torch.from_numpy(np.stack(chunks)).cuda()
    lengths = torch.tensor(lengths, dtype=torch.int64)
    mask = torch.arange(256).unsqueeze(0) >= lengths.unsqueeze(1)
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode():
        _, binary, semantic = model(x, mask.cuda(), LABELS, lengths)
        b = torch.sigmoid(binary.squeeze(-1))
        s = 1 - semantic.softmax(-1)[:, :, 0]
        b = torch.cat([v[:n] for v, n in zip(b, lengths)]).cpu().numpy()
        s = torch.cat([v[:n] for v, n in zip(s, lengths)]).cpu().numpy()
    torch.cuda.synchronize()
    if len(b) != real or not np.isfinite(b).all() or not np.isfinite(s).all():
        raise ValueError('Invalid VadCLIP output')
    return {'binary': b, 'semantic': s}, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--repo', default='scratch/vadclip-official/repo')
    p.add_argument('--checkpoint', default='scratch/vadclip-official/model_ucf.pth')
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest['videos']; audit(rows)
    development = [r for r in rows if r['split'] == 'development']
    validation = [r for r in rows if r['split'] == 'validation']
    if not development or not validation or any(r['events'] or r['official_split'] != 'training' for r in development):
        raise ValueError('Only official training-normal videos may calibrate')
    repo = Path(args.repo)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Official checkout modified')
    protocol = dict(version='vadclip-cross-dataset-pilot-v1', manifest_sha256=digest(manifest_path),
        upstream='https://github.com/nwpu-zxr/VadCLIP', revision=revision,
        checkpoint_sha256=digest(args.checkpoint), checkpoint_training='UCF-Crime',
        code_sha256=digest(__file__), baseline_sha256=digest(args.baseline),
        sampling=['stride16', 'mean16'], primary='stride16/binary/fixed_0.5',
        crop='Unmodified upstream image_crop(type=0): resize340x256, center224x224; RGB then upstream CLIP transform',
        normalization='raw CLIP features, no extra L2 normalization', precision='FP32, batch32 images',
        temporal='upstream256 feature context, zero-pad tails; original actual FPS, 16-frame score blocks',
        thresholds=['fixed_0.5', 'training_normal_q95_higher'], threshold_fitting='training normal scores only; no validation labels',
        heads=['binary', 'semantic'], semantic_definition='1_minus_normal_softmax', inference='strict full state_dict load, unmodified upstream forward',
        repeats=3, smoothing='none, merge only adjacent positive blocks',
        limitation='UBnormal synthetic cross-dataset pilot, not published UCF reproduction; raw feature recipe ambiguity tested as two prespecified variants',
        test_opened=False, production_activation=False)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required by unchanged upstream DistanceAdj')
    jobs = requests.get('http://127.0.0.1:8002/api/v1/events/jobs', timeout=10).json()
    gemma = requests.get('http://127.0.0.1:8012/health', timeout=10).json()
    qwen = requests.get('http://127.0.0.1:8013/health', timeout=10).json()
    if any(j['status'] in ('running', 'queued', 'cancelling') for j in jobs) or gemma.get('resident') or qwen['models']['current_model']:
        raise RuntimeError('Demo GPU is busy; preserve protocol and stop benchmark')
    torch.set_num_threads(4); torch.manual_seed(234)
    torch.backends.cuda.matmul.allow_tf32 = False
    monitor = Monitor(); monitor.thread.start()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model, preprocess, crop = import_model(repo, weights)
        del weights
        torch.cuda.synchronize(); load_seconds = time.perf_counter() - started
        raw = {}; timing = {}; predictions = {}; thresholds = {}; measured = {}
        for sampling in protocol['sampling']:
            raw[sampling] = {}; timing[sampling] = {}
            for row in development + validation:
                tick = time.perf_counter()
                values, bounds, media = features(row, model, preprocess, crop, sampling)
                torch.cuda.synchronize(); extraction = time.perf_counter() - tick
                scored, first = score(model, values)
                repeated = [score(model, values)[1] for _ in range(3)]
                np.savez_compressed(root / f"{sampling}-{row['id']}.npz", features=values,
                                    bounds=np.asarray(bounds), **scored)
                raw[sampling][row['id']] = {'bounds': bounds, **{k: v.tolist() for k, v in scored.items()}}
                timing[sampling][row['id']] = dict(extraction_seconds=extraction, first_score_seconds=first,
                    repeated_score_seconds=repeated, **media)
                print('SCORED', sampling, row['id'], 'images', media['encoded_images'],
                      'seconds', round(extraction + first, 2), flush=True)
                (root / 'partial.json').write_text(json.dumps({'timing': timing, 'scores': raw}, indent=2), encoding='utf-8')
            for head in protocol['heads']:
                normal_scores = [v for r in development for v in raw[sampling][r['id']][head]]
                for name, threshold in [('fixed_0.5', .5), ('training_normal_q95_higher', calibration_threshold(normal_scores))]:
                    key = f'{sampling}/{head}/{name}'; thresholds[key] = threshold
                    pred = {r['id']: threshold_events(raw[sampling][r['id']]['bounds'], raw[sampling][r['id']][head], threshold, r['duration']) for r in validation}
                    predictions[key] = pred
                    measured[key] = {'events': metrics(validation, pred), 'video_bootstrap_95ci': bootstrap(validation, pred)}
            from sklearn.metrics import average_precision_score, roc_auc_score
            labels = []; frame_scores = {h: [] for h in protocol['heads']}
            for row in validation:
                info = timing[sampling][row['id']]; count = info['frame_count']; fps = info['fps']
                times = np.arange(count) / fps
                labels.extend(np.logical_or.reduce([(times >= e['t_start']) & (times < e['t_end']) for e in row['events']]) if row['events'] else np.zeros(count, bool))
                for head in frame_scores:
                    frame_scores[head].extend(np.repeat(raw[sampling][row['id']][head], STEP)[:count])
            for head, values in frame_scores.items():
                measured[f'{sampling}/{head}/frame_ranking'] = dict(auc=roc_auc_score(labels, values), ap=average_precision_score(labels, values))
        baseline = json.loads(Path(args.baseline).read_text(encoding='utf-8'))
        if baseline['protocol']['manifest_sha256'] != protocol['manifest_sha256']:
            raise ValueError('Baseline dataset differs')
        result = dict(status='complete', protocol=protocol, thresholds=thresholds, metrics=measured,
            baseline_metrics=baseline['metrics'], predictions=predictions, raw_scores=raw, timing=timing,
            load_seconds=load_seconds, total_seconds=time.perf_counter()-started,
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            code_unchanged=digest(__file__)==protocol['code_sha256'], test_opened=False, production_activation=False)
        del model; torch.cuda.empty_cache()
    finally:
        monitor.stop.set(); monitor.thread.join(timeout=5)
    result['peak_process_rss_bytes'] = monitor.peak_rss
    result['whole_gpu_peak_mib'] = max(monitor.gpu_used_mib, default=None)
    (root / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETE', json.dumps(measured), flush=True)


if __name__ == '__main__':
    main()
