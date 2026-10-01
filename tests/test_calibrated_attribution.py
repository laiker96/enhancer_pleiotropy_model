"""CPU numerical contracts; no CUDA required."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from classifier_motifs.attribution import active_weights, ensemble, integrated_gradients, one_hot
from classifier_motifs.calibrated_attribution import (CHECKPOINT_SHA, CONTEXTS,
    CalibratedTargets, integrate, quality)
from classifier_modisco import calibrated_context as driver


def calibration():
    return dict(checkpoint_sha256=CHECKPOINT_SHA, contexts=list(CONTEXTS),
        fitting_population='enhancers_only', method='sigmoid', probability_clip=1e-7,
        a=np.linspace(.7, 1.2, 8).tolist(), b=np.linspace(-.3, .4, 8).tolist())


class Toy(torch.nn.Module):
    def __init__(self, length=20):
        super().__init__()
        self.register_buffer('w', torch.randn(8, 4, length,
            generator=torch.Generator().manual_seed(93))*.1)

    def forward(self, x):
        v = torch.einsum('nbl,cbl->nc', x, self.w)
        return v+.05*v**3


class CalibratedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = Toy().eval()
        self.target = CalibratedTargets(self.model, calibration())
        rng = np.random.default_rng(51)
        codes = rng.integers(0, 4, (3, 20)); codes[1] = codes[0]
        self.x = one_hot(codes, 'cpu')
        self.b = one_hot(rng.integers(0, 4, (3, 20)), 'cpu')
        self.labels = torch.tensor([[1,0,1,0,0,0,0,0], [1]*8, [0,1,0,0,0,0,0,0]])

    def test_calibration_order_and_mean_logit(self):
        z, p = ensemble(self.model, self.x)
        got = self.target(self.x, active_weights(self.labels))
        cal = calibration()
        expected = torch.sigmoid(torch.logit(p)*torch.tensor(cal['a'])+torch.tensor(cal['b']))
        torch.testing.assert_close(got[:, :8], expected)
        torch.testing.assert_close(got[:, 8], (z*active_weights(self.labels)).sum(1))
        self.assertGreater(float((p-z.sigmoid()).abs().max()), .001)
        changed = self.target(self.x, active_weights(torch.ones_like(self.labels)))
        torch.testing.assert_close(got[:, :8], changed[:, :8])

    def test_pair_node_and_context_batches_match(self):
        expected = [integrate(self.target, self.x[i:i+1], self.b[i:i+1],
            self.labels[i:i+1], steps=17, internal_batch=4) for i in range(3)]
        for targets in (1, 3, 9):
            got = integrate(self.target, self.x, self.b, self.labels, 17, 23, targets)
            for key in got:
                torch.testing.assert_close(got[key], torch.cat([r[key] for r in expected]), atol=2e-6, rtol=2e-5)

    def test_breadth_is_sum_of_eight_ig_maps(self):
        got = integrate(self.target, self.x, self.b, self.labels, 32)
        nodes, weights = np.polynomial.legendre.leggauss(32)
        gradient = torch.zeros_like(self.x)
        for a, w in zip((nodes+1)/2, weights/2):
            points = (self.b+float(a)*(self.x-self.b)).requires_grad_(True)
            scores = self.target(points, active_weights(self.labels))[:, :8].sum()
            gradient += float(w)*torch.autograd.grad(scores, points)[0]
        hyp = gradient-(gradient*self.b).sum(1, keepdim=True)
        torch.testing.assert_close(got['hypothetical'][:, :8].sum(1), hyp, atol=2e-6, rtol=2e-5)
        old = integrated_gradients(self.model, self.x, self.b, active_weights(self.labels), 32, 64)
        for key in old:
            torch.testing.assert_close(got[key][:, 8], old[key], atol=2e-6, rtol=2e-5)

    def test_rc_projection_and_completeness(self):
        got = integrate(self.target, self.x, self.b, self.labels, 32)
        rc = integrate(self.target, self.x.flip((1,2)), self.b.flip((1,2)), self.labels, 32)
        torch.testing.assert_close(got['hypothetical'], rc['hypothetical'].flip((2,3)), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(got['actual'], (got['hypothetical']*self.x[:,None]).sum(2))
        torch.testing.assert_close(got['delta'], torch.zeros(3,9), atol=2e-6, rtol=0)
        zero = integrate(self.target, self.x, self.x, self.labels)
        torch.testing.assert_close(zero['actual'], torch.zeros_like(zero['actual']))

    def test_invalid_contracts_and_stop(self):
        for key, value in [('contexts', list(reversed(CONTEXTS))),
                           ('checkpoint_sha256', 'wrong'), ('fitting_population', 'mixed')]:
            cal = copy.deepcopy(calibration()); cal[key] = value
            with self.assertRaises(ValueError): CalibratedTargets(self.model, cal)
        with self.assertRaises(ValueError):
            integrate(self.target, self.x*.5, self.b, self.labels)
        with self.assertRaises(ValueError):
            integrate(self.target, self.x, self.b, torch.zeros_like(self.labels))
        with self.assertRaises(TimeoutError):
            integrate(self.target, self.x, self.b, self.labels, should_stop=lambda: True)

    def test_probability_and_logit_quality_scales(self):
        d = np.full((3,9), .003)
        passed = quality(d, np.zeros_like(d))
        self.assertFalse(passed[:, :8].any())
        self.assertTrue(passed[:, 8].all())

    def test_reference_resume_is_identical_and_does_not_double_count(self):
        rng=np.random.default_rng(17)
        data=dict(ids=np.asarray(['a','b']),sequence=rng.integers(0,4,(2,2048),dtype=np.uint8),
            labels=np.ones((2,8),np.uint8),split=np.asarray(['train','validation']),
            chrom=np.asarray(['chr2R','chr2L']),summit=np.asarray([2000,4000]))
        intervals=dict(start=np.asarray([1900,3900]),end=np.asarray([2100,4100]),
            offset=np.asarray([924,924]),length=np.asarray([200,200]))
        target=CalibratedTargets(Toy(2048).eval(),calibration())
        config=dict(references=4,steps=8,reference_block=2,seed=20260916)
        settings=dict(pair_batch=4,internal_batch=16,target_batch=3)
        indices=np.arange(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            full=driver.score_chunk(target,data,intervals,indices,config,settings,
                root/'full.npz','test_signature',None,lambda:False,'cpu')
            calls=[0]
            original=driver.integrate
            def wrapped(*args,**kwargs):
                out=original(*args,**kwargs); calls[0]+=1; return out
            with mock.patch.object(driver,'integrate',side_effect=wrapped):
                with self.assertRaises(TimeoutError):
                    driver.score_chunk(target,data,intervals,indices,config,settings,
                        root/'resume.npz','test_signature',None,lambda:calls[0]>=1,'cpu')
            with np.load(root/'resume.state.npz') as saved: self.assertEqual(int(saved['count']),2)
            resumed=driver.score_chunk(target,data,intervals,indices,config,settings,
                root/'resume.npz','test_signature',None,lambda:False,'cpu')
            for key in full: np.testing.assert_array_equal(full[key],resumed[key])
            with self.assertRaises(ValueError):
                driver.score_chunk(target,data,intervals,indices,config,settings,
                    root/'resume.npz','wrong_signature',None,lambda:False,'cpu')

    def test_reference_moments_match_direct_calculation(self):
        rng=np.random.default_rng(19); n,r,length=2,4,20
        state=driver.new_state(n,length,r,'s',np.arange(n))
        hyp=rng.normal(size=(n,r,9,4,length))
        actual=hyp[:,:,:,0]
        rows=np.repeat(np.arange(n),r); refs=np.tile(np.arange(r),n)
        values=dict(hypothetical=hyp.reshape(n*r,9,4,length),actual=actual.reshape(n*r,9,length),
            delta=np.zeros((n*r,9)),target_difference=np.zeros((n*r,9)),
            reference_orientation_logits=np.zeros((n*r,2,8)),
            reference_probabilities=np.zeros((n*r,8)),reference_calibrated_probabilities=np.zeros((n*r,8)))
        driver.update_state(state,values,rows,refs,2)
        np.testing.assert_allclose(state['sum_hyp']/r,hyp.mean(1))
        var=(state['sum_sq_hyp']-state['sum_hyp']**2/r)/(r-1)
        np.testing.assert_allclose(var,hyp.var(1,ddof=1),atol=1e-14)
        np.testing.assert_allclose(state['sum_half_actual'][:,0]/2,actual[:,:2].mean(1))
        np.testing.assert_allclose(state['sum_sq_breadth'],(actual[:,:,:8].sum(2)**2).sum(1))

    def test_reference_shuffle_and_allocation_guard(self):
        codes=np.random.default_rng(5).integers(0,4,100,dtype=np.uint8)
        a,ha=driver.shuffled((codes,'example',0,20260916))
        b,hb=driver.shuffled((codes,'example',0,20260916))
        np.testing.assert_array_equal(a,b); self.assertEqual(ha,hb)
        np.testing.assert_array_equal(np.bincount(4*codes[:-1]+codes[1:],minlength=16),
                                     np.bincount(4*a[:-1]+a[1:],minlength=16))
        for host,env in [('odin-adm',dict(SLURM_JOB_ID='1',SLURM_JOB_NODELIST='a100')),('a100',{})]:
            with mock.patch.dict('os.environ',env,clear=True),mock.patch('socket.gethostname',return_value=host):
                with self.assertRaises(RuntimeError): driver.require_gpu()


if __name__ == '__main__': unittest.main()
