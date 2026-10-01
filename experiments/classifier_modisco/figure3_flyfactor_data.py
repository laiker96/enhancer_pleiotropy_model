"""FlyFactorSurvey robustness scan and saved-map export for GAF illustrations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .figure3_flank05_data import active_mean, summarize, ASSETS, SOURCE
from .jaspar_importance import scan_shard
from .tomtom_atlas import read_meme


def run(project, root):
    require_allocation('cpu')
    config=json.loads((root/'config.json').read_text())
    for name,sha in config['source_hashes'].items():
        if digest(project/name)!=sha:raise ValueError('Changed input '+name)
    prepared=json.loads((project/'experiments/classifier_calibrated_motifs_20260927/prepared.json').read_text())
    source=project/SOURCE/'native'
    for name in ('metadata.npz','actual.npy'):
        if digest(source/name)!=prepared['files']['native/'+name]:raise ValueError('Changed native '+name)
    with np.load(source/'metadata.npz',allow_pickle=False) as z:meta=dict(z)
    actual=np.load(source/'actual.npy',mmap_mode='r',allow_pickle=False)
    out=root/'output';out.mkdir(exist_ok=False)
    proposals=json.loads((root/'proposals.json').read_text());examples=[]
    for e in proposals['examples']:
        i=e['index'];length=int(meta['length'][i]);site=e['sites'][0]
        assert str(meta['ids'][i])==e['id']
        np.testing.assert_array_equal(meta['labels'][i],e['labels'])
        a=actual[i,:,:length];values=active_mean(a[None],meta['labels'][i:i+1])[0]
        left,right=site['start'],site['end'];strength=float(values[left:right].mean())
        np.testing.assert_allclose(strength,e['core_strength'],atol=1e-8)
        outside=np.ones(length,bool);outside[max(0,left-2):right+2]=False
        contrast=strength/max(float(np.abs(values[outside]).mean()),1e-6)
        if contrast<2:continue
        examples.append(dict(e,sequence=meta['sequence'][i,:length].tolist(),actual=a.tolist(),
                             contrast=contrast,expected_WT=meta['calibrated_probabilities'][i].tolist()))
    write_json(out/'candidates.json',dict(examples=examples,selection=proposals['selection'],
        pwm=proposals['pwm'],source_motif=proposals['source_motif'],
        proposals_sha256=digest(root/'proposals.json')))
    write_json(out/'candidates_complete.json',dict(status='complete',examples=len(examples),
        sha256=digest(out/'candidates.json'),manifest_sha256=digest(root/'MANIFEST.sha256')))
    event('gaf_candidates_exported',examples=len(examples))

    prior=project/'experiments/figure3_context_flank05_20260929/output'
    prior_done=json.loads((prior/'complete.json').read_text())
    fasta=prior/'importance/enhancers.fa'
    assert digest(fasta)==prior_done['files']['importance/enhancers.fa']
    eligible=meta['quality_pass'].all(1)
    assert eligible.sum()==40309 and len(meta['ids'])==40338
    mean_maps=np.empty((len(meta['ids']),actual.shape[-1]),dtype=np.float32)
    for start in range(0,len(meta['ids']),512):
        mean_maps[start:start+512]=active_mean(actual[start:start+512],meta['labels'][start:start+512])
    database=project/ASSETS/'references/motif_databases/FLY/fly_factor_survey.meme'
    profiles=list(read_meme(database).values());ids=[p['id'] for p in profiles]
    assert len(ids)==656 and next(p for p in profiles if p['id']=='FBgn0013263')['name']=='Trl_FlyReg'
    scan_meta=dict(meta,split=np.where(eligible,'train','excluded'))
    workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))
    importance=out/'importance';importance.mkdir()
    def scan(j):
        return scan_shard(j,ids[j::workers],project/ASSETS/'bin/fimo',database,
                          root/'training_background.txt',fasta,scan_meta,mean_maps,importance,
                          database_key='flyfactorsurvey')
    scores=np.full((656,len(meta['ids'])),np.nan,dtype=np.float32)
    covered=np.zeros(scores.shape,dtype=np.uint16);commands=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for names,values,bases,counts,command in pool.map(scan,range(workers)):
            for j,name in enumerate(names):scores[ids.index(name)]=values[j];covered[ids.index(name)]=bases[j]
            commands.append(command)
    np.testing.assert_array_equal(np.isfinite(scores),covered>0)
    assert not np.isfinite(scores[:,~eligible]).any()
    breadth=meta['labels'].sum(1)
    summary=dict(profiles=656,database='FlyFactorSurvey (MEME archive 12.27)',gaf_id='FBgn0013263',
        target='Mean observed-active calibrated probabilities',references=100,steps=64,
        population='All 40309 common-QC enhancers; all splits exploratory',
        definition='Signed IG per union of FIMO hit bases, equally averaged across motif-containing '
                   'enhancers. Noncarriers missing. Untrimmed reference PWMs; FIMO p<=1e-4.',
        exact=summarize(scores,breadth,eligible,profiles),
        cumulative=summarize(scores,breadth,eligible,profiles,True),commands=commands,
        database_sha256=digest(database),background_sha256=digest(root/'training_background.txt'),
        native_hashes={name:prepared['files']['native/'+name] for name in ('actual.npy','metadata.npz')},
        fasta_sha256=digest(fasta),independent_biological_validation=False)
    write_json(importance/'summary.json',summary)
    np.savez_compressed(importance/'scores.npz',scores=scores,covered_bases=covered,ids=meta['ids'],
                        motif_ids=np.asarray(ids),breadth=breadth,split=meta['split'],eligible=eligible)
    write_json(out/'complete.json',dict(status='complete',manifest_sha256=digest(root/'MANIFEST.sha256'),
        summary=dict(profiles=656,enhancers=int(eligible.sum()),candidates=len(examples),new_attributions=0),
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}))
    event('flyfactor_current_maps_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.project,a.root)
