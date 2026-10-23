"""Five-epoch MobileNetV2 QAT experiment; select by converted INT8 validation.
Requires train_baseline.py and run_ptq.py beside this file. Never uses test data.
Training is FP32 with fake quantization on CUDA; actual INT8 evaluation is on CPU.
"""
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from train_baseline import make_model, make_transforms, seed_worker
from run_ptq import accuracy


def train_epoch(model, loader, optimizer, epoch):
    model.train()
    # Retain the well-trained FP32 running statistics throughout fine-tuning.
    model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
    if epoch >= 3:
        model.apply(torch.ao.quantization.disable_observer)
    total = correct = 0
    loss_sum = 0.0
    for images, labels in tqdm(loader, desc=f'QAT train {epoch}'):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss; stop and inspect.')
        loss.backward()
        optimizer.step()
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return loss_sum / total, 100 * correct / total


def converted(model):
    result = copy.deepcopy(model).cpu().eval()
    result.apply(torch.ao.quantization.disable_observer)
    return torch.ao.quantization.convert(result, inplace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--ptq', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 2 or args.workers < 0 or args.threads < 1:
        parser.error('batch-size >= 2, workers >= 0, threads >= 1 required')
    if not torch.cuda.is_available() or not str(torch.__version__).startswith('2.7.1'):
        raise RuntimeError('Use the CUDA-enabled torch 2.7.1 project environment.')
    base, ptq_dir = args.baseline.resolve(), args.ptq.resolve()
    config = json.loads((base / 'config.json').read_text())
    splits = json.loads((base / 'splits.json').read_text())
    ptq = json.loads((ptq_dir / 'summary.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(base / 'best_fp32.pt') != ptq['baseline_checkpoint_sha256'] or sha(base / 'splits.json') != ptq['splits_sha256']:
        raise ValueError('PTQ and baseline checkpoint/split fingerprints differ.')
    groups = [set(splits[key]) for key in ('train', 'validation', 'calibration')]
    if [len(x) for x in groups] != [44000, 5000, 1000] or len(set.union(*groups)) != 50000:
        raise ValueError('Invalid or overlapping splits.')
    if config['input_size'] != 128 or ptq['backend'] != 'x86':
        raise ValueError('Expected original 128px/x86 experiment.')
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.quantized.engine = 'x86'
    model = make_model(pretrained=False)
    model.load_state_dict(torch.load(base / 'best_fp32.pt', map_location='cpu', weights_only=True))
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    qconfig_text = str(model.qconfig)
    model = torch.ao.quantization.prepare_qat(model, inplace=False).cuda()
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)
    train_transform, eval_transform = make_transforms()
    if args.smoke_test:
        fake = datasets.FakeData(size=8, image_size=(3, 32, 32),
                                 num_classes=10, transform=eval_transform)
        loader = DataLoader(fake, batch_size=4)
        train_epoch(model, loader, optimizer, 1)
        quantized = converted(model)
        scripted = torch.jit.script(quantized)
        accuracy(scripted, loader, 'Synthetic INT8 check')
        print('QAT SMOKE TEST PASSED. Synthetic metrics are NOT project results.')
        return
    root = Path(__file__).resolve().parent
    training = datasets.CIFAR10(root / 'data', train=True, download=False, transform=train_transform)
    validation = datasets.CIFAR10(root / 'data', train=True, download=False, transform=eval_transform)
    train_loader = DataLoader(Subset(training, splits['train']), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(Subset(validation, splits['validation']), batch_size=32,
                            shuffle=False, num_workers=0)
    out = base / datetime.now().strftime('qat_%Y%m%d_%H%M%S_%f')
    out.mkdir(exist_ok=False)
    run_config = {
        'epochs': 5, 'seed': seed, 'batch_size': args.batch_size,
        'workers': args.workers, 'cpu_threads': args.threads,
        'backbone_lr': 1e-5, 'head_lr': 1e-4, 'weight_decay': 1e-4,
        'optimizer': 'AdamW', 'bn_stats_frozen_from_epoch': 1,
        'observers_frozen_from_epoch': 3, 'backend': 'x86', 'qconfig': qconfig_text,
        'baseline_checkpoint_sha256': ptq['baseline_checkpoint_sha256'],
        'splits_sha256': ptq['splits_sha256'], 'ptq_directory': str(ptq_dir),
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
        'platform': platform.platform(), 'test_set_used': False,
        'selection_metric': 'converted INT8 validation top-1',
        'note': 'Standard PTQ and QAT recipes use different observer estimators; not an isolated observer-controlled ablation.'
    }
    (out / 'config.json').write_text(json.dumps(run_config, indent=2))
    (out / 'run_qat.py').write_text(Path(__file__).read_text())
    print(f'Output: {out}\nFive QAT epochs; test set untouched.', flush=True)
    best, best_epoch = -1.0, 0
    start = time.perf_counter()
    with (out / 'history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'train_fakequant_top1_pct', 'val_int8_top1_pct'])
        for epoch in range(1, 6):
            loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
            quantized = converted(model)
            scripted = torch.jit.script(quantized)
            val_acc = accuracy(scripted, val_loader, f'INT8 validation {epoch}')
            writer.writerow([epoch, loss, train_acc, val_acc])
            handle.flush()
            if val_acc > best:
                best, best_epoch = val_acc, epoch
                torch.jit.save(scripted, str(out / 'qat_int8.ts'))
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                           out / 'best_qat_state.pt')
                sample, _ = next(iter(val_loader))
                restored = torch.jit.load(str(out / 'qat_int8.ts')).eval()
                with torch.inference_mode():
                    torch.testing.assert_close(restored(sample), scripted(sample), rtol=0, atol=0)
            print(f'Epoch {epoch}: INT8 validation {val_acc:.2f}%; best {best:.2f}%', flush=True)
    result = {
        'fp32_validation_top1_pct': ptq['fp32_validation_top1_pct'],
        'ptq_validation_top1_pct': ptq['ptq_validation_top1_pct'],
        'qat_validation_top1_pct': best, 'best_epoch': best_epoch,
        'recovered_vs_ptq_percentage_points': best - ptq['ptq_validation_top1_pct'],
        'remaining_gap_to_fp32_percentage_points': ptq['fp32_validation_top1_pct'] - best,
        'qat_torchscript_bytes': (out / 'qat_int8.ts').stat().st_size,
        'torchscript_size_reduction_factor': ptq['fp32_torchscript_bytes'] / (out / 'qat_int8.ts').stat().st_size,
        'experiment_seconds_including_validation': time.perf_counter() - start,
        'test_set_used': False
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('\nQAT COMPLETE\n' + json.dumps(result, indent=2))
    print(f'Saved to: {out}')


if __name__ == '__main__':
    main()
