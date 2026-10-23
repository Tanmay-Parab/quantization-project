"""MobileNetV2 FP32 baseline. Run --smoke-test first; then run without it.

Python 3.11, torch 2.7.1, torchvision 0.22.1. No test-set evaluation here.
Input: CIFAR-10 resized to 128x128, ImageNet normalization.
Output: timestamped run with best weights, exact splits, config, and history.
Future PTQ/QAT must reuse this architecture, preprocessing, and split manifest.
"""
import argparse
import csv
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.quantization import mobilenet_v2
from tqdm import tqdm

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_model(pretrained=True):
    # Quantizable architecture now avoids changing residual operations for PTQ.
    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None,
        quantize=False,
    )
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    return model


def make_transforms():
    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128), antialias=True), *normalize,
    ])
    evaluate = transforms.Compose([
        transforms.Resize((128, 128), antialias=True), *normalize,
    ])
    return train, evaluate


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = correct = count = 0
    progress = tqdm(loader, desc='Train' if optimizer else 'Validate')
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        progress.set_postfix(loss=f'{total_loss/count:.3f}', acc=f'{100*correct/count:.2f}%')
    return total_loss / count, 100 * correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error('epochs >= 1, batch-size >= 2, workers >= 0 required')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: use the project virtual environment.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    train_transform, eval_transform = make_transforms()
    model = make_model(pretrained=not args.smoke_test).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)

    if args.smoke_test:
        # Synthetic data only: verifies forward/backward/evaluation, not accuracy.
        fake = datasets.FakeData(size=8, image_size=(3, 32, 32),
                                 num_classes=10, transform=eval_transform)
        loader = DataLoader(fake, batch_size=4, num_workers=0)
        run_epoch(model, loader, device, optimizer)
        run_epoch(model, loader, device)
        print('SMOKE TEST PASSED. Synthetic metrics are NOT project results.')
        return

    root = Path(__file__).resolve().parent
    raw_train = datasets.CIFAR10(root / 'data', train=True, download=True,
                                transform=train_transform)
    raw_eval = datasets.CIFAR10(root / 'data', train=True, download=False,
                               transform=eval_transform)
    rng = np.random.default_rng(args.seed)
    labels = np.asarray(raw_train.targets)
    splits = {'train': [], 'validation': [], 'calibration': []}
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label)).tolist()
        splits['validation'].extend(indices[:500])
        splits['calibration'].extend(indices[500:600])
        splits['train'].extend(indices[600:])
    assert len(set(splits['train']) | set(splits['validation']) |
               set(splits['calibration'])) == 50000
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=True, worker_init_fn=seed_worker,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(Subset(raw_train, splits['train']), shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(Subset(raw_eval, splits['validation']), shuffle=False,
                            **common)
    out = root / 'runs' / datetime.now().strftime('baseline_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'architecture': 'quantizable_mobilenet_v2',
              'initial_weights': 'IMAGENET1K_V1', 'input_size': 128,
              'mean': MEAN, 'std': STD, 'classes': raw_train.classes,
              'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
              'python': platform.python_version(), 'platform': platform.platform(),
              'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
              'optimizer': 'AdamW', 'backbone_lr': 1e-4, 'head_lr': 1e-3,
              'weight_decay': 1e-4, 'precision': 'FP32, TF32 disabled',
              'test_set_used': False}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'splits.json').write_text(json.dumps(splits))
    (out / 'train_baseline.py').write_text(Path(__file__).read_text())
    print(f'GPU: {config["gpu"]}\nOutput: {out}', flush=True)
    print('44,000 training / 5,000 validation / 1,000 calibration; test untouched.')
    best = -1.0
    best_epoch = 0
    start = time.perf_counter()
    with (out / 'history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'train_top1_pct', 'val_loss', 'val_top1_pct'])
        for epoch in range(1, args.epochs + 1):
            print(f'\nEpoch {epoch}/{args.epochs}', flush=True)
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            handle.flush()
            if val_acc > best:
                best, best_epoch = val_acc, epoch
                weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(weights, out / 'best_fp32.pt')
            print(f'Best validation top-1: {best:.2f}% (epoch {best_epoch})', flush=True)
    summary = {'best_validation_top1_pct': best, 'best_epoch': best_epoch,
               'training_seconds': time.perf_counter() - start,
               'state_dict_bytes': (out / 'best_fp32.pt').stat().st_size,
               'test_set_used': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nBASELINE COMPLETE\n' + json.dumps(summary, indent=2))
    print(f'Saved to: {out}\nKeep this entire folder for PTQ/QAT.')


if __name__ == '__main__':
    main()
