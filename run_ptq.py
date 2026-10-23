"""INT8 PTQ for the saved CIFAR-10 MobileNetV2 baseline (torch 2.7.1).
Run beside train_baseline.py. Uses calibration and validation only, never test.
Exports comparable FP32/INT8 TorchScript files for later CPU benchmarking.
"""
import argparse
import copy
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from train_baseline import make_model, make_transforms


@torch.inference_mode()
def accuracy(model, loader, label):
    correct = total = 0
    for images, targets in tqdm(loader, desc=label):
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f'Nonfinite predictions in {label}')
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.numel()
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--threads', default=4, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error('threads and batch-size must be positive')
    baseline = args.baseline.resolve()
    config = json.loads((baseline / 'config.json').read_text())
    splits = json.loads((baseline / 'splits.json').read_text())
    previous = json.loads((baseline / 'summary.json').read_text())
    if config['architecture'] != 'quantizable_mobilenet_v2' or config['input_size'] != 128:
        raise ValueError('This script requires the original 128px MobileNetV2 baseline.')
    if not str(torch.__version__).startswith('2.7.1'):
        raise RuntimeError('Run in the torch 2.7.1 project environment.')
    groups = [set(splits[k]) for k in ('train', 'validation', 'calibration')]
    if any(len(g) != len(splits[k]) for g, k in zip(groups, ('train', 'validation', 'calibration'))):
        raise ValueError('Duplicate split indices')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Data split overlap')
    if [len(g) for g in groups] != [44000, 5000, 1000] or set.union(*groups) != set(range(50000)):
        raise ValueError('Unexpected dataset split')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(args.threads)
    backend = 'x86'
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f'x86 backend unavailable: {torch.backends.quantized.supported_engines}')
    torch.backends.quantized.engine = backend
    _, transform = make_transforms()
    root = Path(__file__).resolve().parent
    dataset = CIFAR10(root / 'data', train=True, download=False, transform=transform)
    generator = torch.Generator().manual_seed(config['seed'])
    calibration = DataLoader(Subset(dataset, splits['calibration']),
                             batch_size=args.batch_size, shuffle=True,
                             generator=generator, num_workers=0)
    validation = DataLoader(Subset(dataset, splits['validation']),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = baseline / 'best_fp32.pt'
    fp32 = make_model(pretrained=False).cpu().eval()
    fp32.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    out = baseline / datetime.now().strftime('ptq_%Y%m%d_%H%M%S_%f')
    out.mkdir(exist_ok=False)
    (out / 'run_ptq.py').write_text(Path(__file__).read_text())
    print(f'Output: {out}\nCPU threads: {args.threads}; backend: {backend}', flush=True)

    # Save both variants in the SAME deployment-file format for size comparisons.
    float_file = out / 'fp32.ts'
    int8_file = out / 'ptq_int8.ts'
    torch.jit.save(torch.jit.script(fp32), str(float_file))
    float_export = torch.jit.load(str(float_file), map_location='cpu').eval()
    fp32_acc = accuracy(float_export, validation, 'FP32 validation')
    expected = previous['best_validation_top1_pct']
    if abs(fp32_acc - expected) > 0.1:
        raise RuntimeError(f'FP32 recheck {fp32_acc:.2f}% differs from saved {expected:.2f}%. Stop and inspect.')

    model = copy.deepcopy(fp32).eval()
    model.fuse_model(is_qat=False)
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    prepared = torch.ao.quantization.prepare(model, inplace=False)
    with torch.inference_mode():
        for images, _ in tqdm(calibration, desc='Calibrate (1,000 images)'):
            prepared(images)
    quantized = torch.ao.quantization.convert(prepared, inplace=False).eval()
    quantized_layers = sum(isinstance(m, (torch.ao.nn.quantized.Conv2d,
                                          torch.ao.nn.quantized.Linear))
                           for m in quantized.modules())
    if not quantized_layers:
        raise RuntimeError('Conversion produced no quantized Conv2d/Linear modules.')
    torch.jit.save(torch.jit.script(quantized), str(int8_file))
    restored = torch.jit.load(str(int8_file), map_location='cpu').eval()
    sample, _ = next(iter(validation))
    with torch.inference_mode():
        torch.testing.assert_close(restored(sample), quantized(sample), rtol=0, atol=0)
    ptq_acc = accuracy(restored, validation, 'INT8 PTQ validation')
    summary = {
        'fp32_validation_top1_pct': fp32_acc,
        'ptq_validation_top1_pct': ptq_acc,
        'accuracy_drop_percentage_points': fp32_acc - ptq_acc,
        'fp32_torchscript_bytes': float_file.stat().st_size,
        'ptq_torchscript_bytes': int8_file.stat().st_size,
        'torchscript_size_reduction_factor': float_file.stat().st_size / int8_file.stat().st_size,
        'quantized_conv_linear_count': quantized_layers,
        'calibration_images': 1000, 'validation_images': 5000,
        'test_set_used': False, 'backend': backend, 'cpu_threads': args.threads,
        'batch_size': args.batch_size, 'seed': config['seed'],
        'torch': str(torch.__version__), 'platform': platform.platform(),
        'baseline_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'splits_sha256': hashlib.sha256((baseline / 'splits.json').read_bytes()).hexdigest(),
        'qconfig': str(model.qconfig),
        'note': 'INT8 weight/activation CPU PTQ. No phone latency or test accuracy measured.'
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nPTQ COMPLETE\n' + json.dumps(summary, indent=2))
    print(f'Saved to: {out}')


if __name__ == '__main__':
    main()
