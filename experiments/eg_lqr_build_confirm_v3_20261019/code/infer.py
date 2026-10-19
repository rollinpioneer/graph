"""Shared scoring for frozen caches and actual EconomicGrasp inference."""
import argparse, os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scoring import FrozenScorer, cache_inputs, sha256


def sample_hash(sample):
    digest = hashlib.sha256()
    for key, value in sorted(sample.items()):
        value = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def setup_original(code, dataset_root):
    sys.path[:0] = [str(code), str(code/'libs/pointnet2')]
    # Original EG config parses argv while importing modules.
    sys.argv = [sys.argv[0], '--dataset_root', str(dataset_root), '--camera', 'kinect', '--model', 'economicgrasp', '--batch_size', '1']


class EconomicExtractor:
    def __init__(self, code, dataset_root, backbone, expected_sha256, device):
        setup_original(code, dataset_root)
        import torch
        from models.economicgrasp import economicgrasp, pred_decode
        from dataset.graspnet_dataset import GraspNetDataset, collate_fn
        self.torch, self.decode, self.collate_fn = torch, pred_decode, collate_fn
        torch.set_num_threads(1)
        torch.manual_seed(20261018)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(20261018)
        self.device = torch.device(device)
        assert sha256(backbone) == expected_sha256
        self.net = economicgrasp(seed_feat_dim=512, is_training=False).to(self.device).eval()
        self.net.load_state_dict(torch.load(backbone, map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
        self.net.requires_grad_(False)
        self.cap = {}
        self.handles = []
        self.install_hooks()
        self.data = GraspNetDataset(str(dataset_root), split='train', camera='kinect', num_points=20000, voxel_size=.005, remove_outlier=True, augment=False, load_label=False)
        self.dataset_root = Path(dataset_root)

    def install_hooks(self):
        def group(module, inputs, output):
            self.cap['group'] = output.detach().clone()
        def token(module, inputs):
            self.cap['token'] = inputs[0].detach().clone()
        self.handles = [self.net.cy_group.register_forward_hook(group), self.net.grasp_head.conv_score.register_forward_pre_hook(token)]

    def sample(self, scene, frame, input_sample=None):
        if input_sample:
            with np.load(input_sample, allow_pickle=False) as source:
                return {k: source[k].copy() for k in ['point_clouds', 'cloud_colors', 'coordinates_for_voxel', 'seg']}
        seed = int.from_bytes(hashlib.sha256('|'.join(map(str, [20261018, 'input', scene, frame])).encode()).digest()[:4], 'little')
        np.random.seed(seed)
        for attr, sub, ext in [('colorpath','rgb','png'), ('depthpath','depth','png'), ('labelpath','label','png'), ('metapath','meta','mat')]:
            getattr(self.data, attr)[0] = str(self.dataset_root/f'scenes/scene_{scene:04d}/kinect/{sub}/{frame:04d}.{ext}')
        self.data.scenename[0], self.data.frameid[0] = f'scene_{scene:04d}', frame
        return self.data.get_data(0)

    def extract(self, sample, verify_hook=False):
        torch = self.torch
        batch = {k: value.to(self.device) for k, value in self.collate_fn([sample]).items()}
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            endpoints = self.net(dict(batch))
            pred = self.decode(endpoints)[0].cpu().numpy()
        feature = torch.cat([self.cap['group'], self.cap['token']], dim=1)[0].T.cpu().numpy()
        angle_index = endpoints['grasp_angle_pred'][0, :12].cpu().numpy().T.argmax(1)
        theta = angle_index*np.pi/12
        action = np.column_stack([np.sin(2*theta), np.cos(2*theta), pred[:,3]/.04, pred[:,1]/.1]).astype(np.float32)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter()-started
        if verify_hook:
            for handle in self.handles:
                handle.remove()
            with torch.no_grad():
                plain = self.decode(self.net(dict(batch)))[0].cpu().numpy()
            assert np.array_equal(pred, plain), 'Hook changed baseline output'
            self.install_hooks()
        assert pred.shape == (1024,17) and feature.shape == (1024,320)
        return pred, feature, action, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'frozen_config.json')
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--family', choices=['R0','S','F05'], default='S')
    p.add_argument('--cache', type=Path)
    p.add_argument('--layout', choices=['nqs_v1','lqr_v3'], default='nqs_v1')
    p.add_argument('--eg-code', type=Path)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--backbone', type=Path)
    p.add_argument('--scene', type=int)
    p.add_argument('--frame', type=int)
    p.add_argument('--input-sample', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--verify-hook', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.nice(10) if hasattr(os, 'nice') else None
    config = json.loads(args.config.read_text(encoding='utf-8'))
    head = next(h for h in config['heads'] if h['seed'] == args.seed)
    scorer = FrozenScorer(args.head, head['sha256'])
    if args.cache:
        original, feature, action, identity = cache_inputs(args.cache, args.layout)
        mode, seconds = 'cache', None
    else:
        if None in [args.eg_code, args.dataset_root, args.backbone, args.scene, args.frame]:
            raise ValueError('Online mode requires original code, data, backbone, scene and frame')
        extractor = EconomicExtractor(args.eg_code, args.dataset_root, args.backbone, config['backbone_sha256'], args.device)
        sample = extractor.sample(args.scene, args.frame, args.input_sample)
        original, feature, action, seconds = extractor.extract(sample, args.verify_hook)
        identity = sha256(args.input_sample) if args.input_sample else sample_hash(sample)
        mode = 'original-EG-data-entry'
    updated, quality, receipt = scorer.apply(original, feature, action, args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), 'Do not overwrite prior prediction'
    geometry_sha = hashlib.sha256(np.ascontiguousarray(original[:, 1:]).tobytes()).hexdigest()
    np.savez_compressed(args.output, original_pred=original, pred=updated, feature320=feature, original_action=action, Q_original=quality, candidate_ids=np.arange(1024), input_sha256=np.array(identity or 'unrecorded'), geometry_sha256=np.array(geometry_sha), backbone_sha256=np.array(config['backbone_sha256']), receipt=np.array(json.dumps(dict(**receipt, mode=mode, EG_extract_seconds=seconds))))
    print(json.dumps(dict(output=str(args.output), family=args.family, seed=args.seed, mode=mode, geometry_unchanged=True)))


if __name__ == '__main__':
    main()
