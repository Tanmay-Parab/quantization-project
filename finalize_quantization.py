"""Locked-model CIFAR-10 test evaluation and CPU batch-1 latency benchmark.
Keep beside train_baseline.py. No training, calibration, or model selection.
Latency excludes preprocessing, loading, and transfers; it is NOT phone latency.
"""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from train_baseline import make_transforms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_name():
    if platform.system() == 'Windows':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


@torch.inference_mode()
def evaluate(model, loader, name):
    truth, predictions, top5_correct = [], [], 0
    for images, labels in tqdm(loader, desc=f'{name} held-out test'):
        logits = model(images)
        if logits.shape != (labels.numel(), 10) or not torch.isfinite(logits).all():
            raise RuntimeError(f'Invalid predictions: {name}')
        truth.extend(labels.tolist())
        predictions.extend(logits.argmax(1).tolist())
        top5_correct += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
    truth, predictions = np.array(truth), np.array(predictions)
    return truth, predictions, {
        'test_top1_pct': float(100 * (truth == predictions).mean()),
        'test_top5_pct': 100 * top5_correct / len(truth),
        'correct_top1': int((truth == predictions).sum()), 'test_images': len(truth)
    }


@torch.inference_mode()
def benchmark(models, inputs, out):
    # Every model occupies every position equally often across six blocks.
    orders = list(itertools.permutations(models))
    random.Random(42).shuffle(orders)
    samples = {name: [] for name in models}
    per_block = {name: [] for name in models}
    with (out / 'latency_samples.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['block', 'position', 'model', 'iteration', 'input_index', 'latency_ms'])
        for block, order in enumerate(orders, 1):
            print(f'Latency block {block}/6: {", ".join(order)}', flush=True)
            for position, name in enumerate(order):
                model = models[name]
                for index in range(50):
                    model(inputs[index % len(inputs)])
                block_samples = []
                rows = []
                for index in range(100):
                    input_index = index % len(inputs)
                    image = inputs[input_index]
                    start = time.perf_counter_ns()
                    output = model(image)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    del output
                    block_samples.append(elapsed)
                    rows.append([block, position, name, index, input_index, elapsed])
                writer.writerows(rows)
                handle.flush()
                samples[name].extend(block_samples)
                per_block[name].append(float(np.median(block_samples)))
    return {name: {
        'latency_p50_ms': float(np.median(values)),
        'latency_p95_ms': float(np.percentile(values, 95)),
        'latency_mean_ms': float(np.mean(values)),
        'block_medians_ms': per_block[name], 'timed_calls': len(values)
    } for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--ptq', required=True, type=Path)
    parser.add_argument('--qat', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if not str(torch.__version__).startswith('2.7.1'):
        raise RuntimeError('Use the torch 2.7.1 project environment.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.quantized.engine = 'x86'
    base, ptq_dir, qat_dir = [p.resolve() for p in (args.baseline, args.ptq, args.qat)]
    ptq = json.loads((ptq_dir / 'summary.json').read_text())
    qat_config = json.loads((qat_dir / 'config.json').read_text())
    config = json.loads((base / 'config.json').read_text())
    for key, file in [('baseline_checkpoint_sha256', 'best_fp32.pt'), ('splits_sha256', 'splits.json')]:
        if not (ptq[key] == qat_config[key] == sha(base / file)):
            raise RuntimeError(f'Mismatched experiment fingerprint: {key}')
    if config['input_size'] != 128 or ptq['backend'] != 'x86' or qat_config['backend'] != 'x86':
        raise RuntimeError('Expected original 128px/x86 experiment.')
    files = {'FP32': ptq_dir / 'fp32.ts', 'PTQ': ptq_dir / 'ptq_int8.ts',
             'QAT': qat_dir / 'qat_int8.ts'}
    models = {name: torch.jit.load(str(path), map_location='cpu').eval()
              for name, path in files.items()}
    root = Path(__file__).resolve().parent
    _, transform = make_transforms()
    test = CIFAR10(root / 'data', train=False, download=False, transform=transform)
    if len(test) != 10000:
        raise RuntimeError('Expected 10,000 held-out test images.')
    loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    out = base / datetime.now().strftime('final_%Y%m%d_%H%M%S_%f')
    out.mkdir(exist_ok=False)
    (out / 'finalize_quantization.py').write_text(Path(__file__).read_text())
    metadata = {
        'cpu': cpu_name(), 'platform': platform.platform(), 'python': platform.python_version(),
        'torch': str(torch.__version__), 'cpu_threads': args.threads, 'interop_threads': 1,
        'backend': 'x86', 'input_shape': [1, 3, 128, 128], 'latency_batch_size': 1,
        'accuracy_batch_size': 32, 'warmups_per_block': 50, 'blocks': 6,
        'timed_calls_per_model': 600, 'timer': 'perf_counter_ns',
        'scope': 'CPU TorchScript forward call only; excludes preprocessing/loading/transfers.',
        'phone_benchmark': False, 'test_used_for_model_selection': False,
        'models': {name: {'path': str(path), 'sha256': sha(path)} for name, path in files.items()},
        'preprocessing': {'resize': [128, 128], 'mean': config['mean'], 'std': config['std']}
    }
    (out / 'environment.json').write_text(json.dumps(metadata, indent=2))
    print(f'Output: {out}\nCPU: {metadata["cpu"]}; threads: {args.threads}', flush=True)
    results, predictions, truth = {}, {}, None
    for name, model in models.items():
        labels, pred, metrics = evaluate(model, loader, name)
        if truth is not None and not np.array_equal(labels, truth):
            raise RuntimeError('Test-label ordering differs between models.')
        truth, predictions[name] = labels, pred
        results[name] = {**metrics, 'torchscript_bytes': files[name].stat().st_size}
    with (out / 'test_predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['test_index', 'true_label', 'fp32_prediction', 'ptq_prediction', 'qat_prediction'])
        writer.writerows((i, int(truth[i]), *(int(predictions[n][i]) for n in models)) for i in range(len(truth)))
    # Save accuracy before timing, so a later interruption does not lose it.
    (out / 'test_accuracy.json').write_text(json.dumps(results, indent=2))
    inputs = [test[index][0].unsqueeze(0).contiguous() for index in range(64)]
    timings = benchmark(models, inputs, out)
    for name in models:
        results[name].update(timings[name])
    for name in models:
        results[name]['size_reduction_vs_fp32'] = results['FP32']['torchscript_bytes'] / results[name]['torchscript_bytes']
        results[name]['p50_speedup_vs_fp32'] = results['FP32']['latency_p50_ms'] / results[name]['latency_p50_ms']
    summary = {
        'models': results,
        'ptq_test_drop_percentage_points': results['FP32']['test_top1_pct'] - results['PTQ']['test_top1_pct'],
        'qat_test_recovery_percentage_points': results['QAT']['test_top1_pct'] - results['PTQ']['test_top1_pct'],
        'qat_test_gap_to_fp32_percentage_points': results['FP32']['test_top1_pct'] - results['QAT']['test_top1_pct'],
        'cpu': metadata['cpu'], 'cpu_threads': args.threads, 'latency_batch_size': 1,
        'latency_scope': metadata['scope'], 'phone_benchmark': False
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (out / 'results.csv').open('w', newline='') as handle:
        columns = ['model', 'test_top1_pct', 'test_top5_pct', 'torchscript_bytes',
                   'latency_p50_ms', 'latency_p95_ms', 'size_reduction_vs_fp32', 'p50_speedup_vs_fp32']
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({'model': name, **values} for name, values in results.items())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    specifications = [('test_top1_pct', 'Test top-1 accuracy (%)', 1),
                      ('torchscript_bytes', 'TorchScript size (MB)', 1e6),
                      ('latency_p50_ms', 'CPU median latency (ms)', 1)]
    for ax, (key, title, divisor) in zip(axes, specifications):
        values = [results[name][key] / divisor for name in models]
        bars = ax.bar(list(models), values, color=['#64748b', '#d97706', '#0f766e'])
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set_title(title)
        ax.set_ylim(0, 105 if key == 'test_top1_pct' else max(values) * 1.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('MobileNetV2 on CIFAR-10 | 128 x 128 input', fontweight='bold')
    fig.text(0.5, 0.02, f'10,000 test images | CPU batch 1, {args.threads} threads | forward-only latency; no phone benchmark',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    print('\nFINAL EVALUATION COMPLETE\n' + json.dumps(summary, indent=2))
    print(f'Saved to: {out}')


if __name__ == '__main__':
    main()
