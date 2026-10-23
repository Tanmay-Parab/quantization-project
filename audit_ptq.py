"""Identify uncalibrated activation observers without altering saved results."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.ao.quantization.observer import ObserverBase
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from train_baseline import make_model, make_transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    base = args.baseline.resolve()
    config = json.loads((base / 'config.json').read_text())
    splits = json.loads((base / 'splits.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.backends.quantized.engine = 'x86'
    model = make_model(pretrained=False).cpu().eval()
    model.load_state_dict(torch.load(base / 'best_fp32.pt', map_location='cpu', weights_only=True))
    model.fuse_model(is_qat=False)
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    model = torch.ao.quantization.prepare(model, inplace=False)
    observers = {name: module for name, module in model.named_modules()
                 if isinstance(module, ObserverBase)}
    calls = dict.fromkeys(observers, 0)
    handles = []
    for name, observer in observers.items():
        def record(module, inputs, output, key=name):
            calls[key] += 1
        handles.append(observer.register_forward_hook(record))
    _, transform = make_transforms()
    data = CIFAR10(Path(__file__).resolve().parent / 'data', train=True,
                   download=False, transform=transform)
    loader = DataLoader(Subset(data, splits['calibration']), batch_size=32,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed']))
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc='Audit calibration'):
            model(images)
    for handle in handles:
        handle.remove()
    unused_residual = []
    unexpected = []
    for name, observer in observers.items():
        minimum = getattr(observer, 'min_val', None)
        maximum = getattr(observer, 'max_val', None)
        valid = (minimum is not None and maximum is not None and
                 minimum.numel() > 0 and maximum.numel() > 0 and
                 bool(torch.isfinite(minimum).all()) and
                 bool(torch.isfinite(maximum).all()))
        if calls[name] and valid:
            continue
        suffix = '.skip_add.activation_post_process'
        block = model.get_submodule(name[:-len(suffix)]) if name.endswith(suffix) else None
        if block is not None and getattr(block, 'use_res_connect', None) is False and calls[name] == 0:
            unused_residual.append(name)
        else:
            unexpected.append({'name': name, 'calls': calls[name], 'finite_range': valid})
    result = {'status': 'PASS' if not unexpected else 'INVESTIGATE',
              'observer_count': len(observers),
              'inactive_residual_observers': unused_residual,
              'unexpected_observers': unexpected,
              'test_set_used': False}
    output = base / datetime.now().strftime('observer_audit_%Y%m%d_%H%M%S_%f.json')
    output.write_text(json.dumps(result, indent=2))
    print('\nOBSERVER AUDIT\n' + json.dumps(result, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
