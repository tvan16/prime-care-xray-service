#!/usr/bin/env python
"""Dental-lesion detection on panoramic X-rays (OralXrays-9, 9 classes).

Two models, same output format:
  v4_lite  -- accuracy of the reference MOQAM (73.2 AP), ~62 ms/image on RTX 5880
  v3_fast  -- 3.8x faster than MOQAM (~26 ms/image), 0.9 AP lower, mainly on decay

Use from Python (load once at server start, then call predict per image):

    from infer import load_model, predict
    model = load_model('v4_lite')                 # or 'v3_fast'
    result = predict(model, 'xray.jpg', score_thr=0.3)   # path or numpy BGR image

Or from the command line:

    python infer.py samples/026330.jpg --model v4_lite --score-thr 0.3

`score` is a confidence for ranking boxes, NOT a calibrated probability of
disease. bbox_xyxy is in pixels of the original image (left, top, right, bottom).
"""
import argparse
import json
import os.path as osp
import sys
import time

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)  # use the bundled mmdet / projects, not any installed copy

import logging  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings('ignore')

import mmcv  # noqa: E402
import torch  # noqa: E402
from mmcv.utils import get_logger  # noqa: E402

get_logger('mmcv').setLevel(logging.ERROR)  # hide weight-init messages
from mmdet.apis import inference_detector, init_detector  # noqa: E402
import projects  # noqa: F401,E402  (registers the custom model components)

CLASSES = ('Apical Periodontitis', 'Decay', 'Wisdom Tooth', 'Missing Tooth',
           'Dental Filling', 'Root Canal Filling', 'Implant',
           'Porcelain Crown', 'Ceramic Bridge')
MODELS = {
    'v4_lite': ('configs/v4_lite.py', 'weights/v4_lite.pth'),
    'v3_fast': ('configs/v3_fast.py', 'weights/v3_fast.pth'),
}


def load_model(name='v4_lite', device=None, warmup=True):
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg, ckpt = MODELS[name]
    model = init_detector(osp.join(HERE, cfg), osp.join(HERE, ckpt), device=device)
    model.CLASSES = CLASSES
    if warmup:  # the first call pays for CUDA/cuDNN setup (~several seconds)
        import numpy as np
        inference_detector(model, np.zeros((1127, 1991, 3), dtype=np.uint8))
    return model


def predict(model, image, score_thr=0.3):
    """image: file path or numpy array (BGR, as read by cv2.imread)."""
    name = osp.basename(image) if isinstance(image, str) else None
    # Always feed a numpy array: mmdet 2.x switches the model's loading step
    # in place the first time it sees an array, after which file paths fail.
    img = mmcv.imread(image) if isinstance(image, str) else image
    if img is None:
        raise ValueError(f'cannot read image: {image}')
    t0 = time.time()
    result = inference_detector(model, img)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000

    dets = []
    for cls_id, boxes in enumerate(result):
        for x1, y1, x2, y2, s in boxes.tolist():
            if s >= score_thr:
                dets.append({'class': CLASSES[cls_id], 'class_id': cls_id,
                             'score': round(s, 4),
                             'bbox_xyxy': [round(v, 1) for v in (x1, y1, x2, y2)]})
    dets.sort(key=lambda d: -d['score'])
    summary = {c: {'count': 0, 'max_score': 0.0} for c in CLASSES}
    for d in dets:
        e = summary[d['class']]
        e['count'] += 1
        e['max_score'] = max(e['max_score'], d['score'])
    return {'image': name,
            'score_thr': score_thr, 'inference_ms': round(ms, 1),
            'num_detections': len(dets), 'summary': summary, 'detections': dets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('images', nargs='+')
    p.add_argument('--model', default='v4_lite', choices=sorted(MODELS))
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    model = load_model(args.model, args.device)
    out = [predict(model, img, args.score_thr) for img in args.images]
    print(json.dumps(out if len(out) > 1 else out[0], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
