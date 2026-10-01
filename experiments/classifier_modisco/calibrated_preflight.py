"""CPU-only test suite and real-checkpoint replay before CUDA execution."""
import argparse
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from classifier_motifs.attribution import ensemble, one_hot
from classifier_motifs.calibrated_attribution import CalibratedTargets, integrate, load_classifier
from .common import digest, write_json


def run(root,checkpoint,report):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU preflight must hide CUDA')
    suite=unittest.TestLoader().discover(str(root/'tests'),pattern='test_calibrated_attribution.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun!=9:
        raise ValueError('Nine passing no-skip focused tests required')
    torch.set_num_threads(2)
    calibration=json.loads((root/'calibrators.json').read_text())['enhancers_only']
    model=load_classifier(checkpoint,'cpu')
    target=CalibratedTargets(model,calibration)
    with np.load(root/'endpoint_replay.npz',allow_pickle=False) as saved: replay=dict(saved)
    predictions=[]
    with torch.no_grad():
        for first in range(0,len(replay['sequence']),4):
            x=one_hot(replay['sequence'][first:first+4],'cpu')
            z,p=ensemble(model,x); value=target.endpoints(x)
            torch.testing.assert_close(z,value['logits'],atol=3e-5,rtol=2e-3)
            expected=1/(1+np.exp(-(np.log(np.clip(p.numpy(),1e-7,1-1e-7)/
                (1-np.clip(p.numpy(),1e-7,1-1e-7)))*np.asarray(calibration['a'])+np.asarray(calibration['b']))))
            np.testing.assert_allclose(value['calibrated_probabilities'],expected,atol=1e-6,rtol=1e-5)
            predictions.append(p.numpy())
    error=float(np.max(np.abs(np.concatenate(predictions)-replay['probabilities'])))
    if error>.005: raise ValueError('Wrong FP32 checkpoint replay')
    x=one_hot(replay['sequence'][:2],'cpu'); baseline=x.flip(2)
    labels=torch.as_tensor(replay['labels'][:2])
    small=integrate(target,x,baseline,labels,8,4,1)
    maximum={}
    for targets in (1,3,9):
        large=integrate(target,x,baseline,labels,8,16,targets)
        for key in small:
            torch.testing.assert_close(large[key],small[key],atol=3e-5,rtol=2e-3)
            maximum[f'{targets}_{key}']=float((large[key]-small[key]).abs().max())
    write_json(report,dict(status='passed',tests=result.testsRun,skips=[],
        manifest_sha256=digest(root/'MANIFEST.sha256'),checkpoint_sha256=digest(checkpoint),
        real_endpoint_examples=len(predictions)*4,maximum_fp32_vs_cached_fp16_error=error,
        real_model_batch_equivalence_maximum=maximum,torch_version=torch.__version__,no_cuda=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args(); run(args.root,args.checkpoint,args.report)
