"""Fixed filtered TF-MoDISco PWM scans on saved native IG64/100 maps; CPU only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .dual_motif_pipeline import informative_core
from .figure3_flank05_data import active_mean, summarize, ASSETS, SOURCE
from .jaspar_importance import scan_shard
from .tomtom_atlas import query_id, read_meme

NAME='figure3_denovo_importance_20260930'
PRIOR='experiments/figure3_context_flank05_20260929/output'


def catalogue(analysis, database):
    """Keep every passing positive discovery separately, including repeated TF matches."""
    profiles=[]
    for group in analysis['context']:
        for row in group['rows']:
            if row['sign']!='positive':raise ValueError('Expected the positive-motif catalogue')
            quality=informative_core(row['full_pwm'],flank_threshold=.5)
            if not quality['passed'] or any(quality[k]!=row['quality'][k] for k in ('start','end','width','passed','reason')):
                raise ValueError('Information filter changed')
            for k in ('mean_bits','total_bits'):
                np.testing.assert_allclose(quality[k],row['quality'][k],atol=1e-12,rtol=0)
            ident=query_id(row);pwm=np.asarray(row['trimmed_pwm'])
            np.testing.assert_allclose(pwm,np.asarray(row['full_pwm'])[quality['start']:quality['end']],atol=1e-12)
            np.testing.assert_allclose(database[ident]['pwm'],pwm,atol=1e-10)
            match=row['match'];name=match['reference']['name']
            profiles.append(dict(id=ident,name=name,source_id=row['id'],
                discovery_group=group['task']['group'],discovery_rank=row['rank'],
                width=len(pwm),tomtom_q=match['q'],
                gaf_highlight=name=='Trl' and match['q']<=.05,
                information=quality,discovery_seqlets=row['seqlets']))
    ids=[p['id'] for p in profiles]
    if len(ids)!=66 or len(set(ids))!=66 or set(ids)!=set(database):
        raise ValueError('Expected exactly 66 unchanged, distinct filtered PWM IDs')
    return profiles


def run(project,root):
    require_allocation('cpu')
    config=json.loads((root/'config.json').read_text())
    for name,sha in config['source_hashes'].items():
        if digest(project/name)!=sha:raise ValueError('Changed source '+name)
    prepared=json.loads((project/'experiments/classifier_calibrated_motifs_20260927/prepared.json').read_text())
    if (prepared['references'],prepared['steps'])!=(100,64):raise ValueError('Wrong attribution protocol')
    source=project/SOURCE/'native'
    for name in ('metadata.npz','actual.npy'):
        if digest(source/name)!=prepared['files']['native/'+name]:raise ValueError('Changed native '+name)
    prior=project/PRIOR;done=json.loads((prior/'complete.json').read_text())
    names=('analysis.json','annotation/quality_passing_queries.meme','importance/enhancers.fa')
    for name in names:
        if digest(prior/name)!=done['files'][name]:raise ValueError('Changed prior '+name)
    database=prior/'annotation/quality_passing_queries.meme'
    analysis=json.loads((prior/'analysis.json').read_text())
    profiles=catalogue(analysis,read_meme(database));ids=[p['id'] for p in profiles]
    binary=project/ASSETS/'bin/fimo'
    if subprocess.check_output([str(binary),'--version'],text=True).strip()!='5.5.9':
        raise ValueError('Expected FIMO 5.5.9')
    with np.load(source/'metadata.npz',allow_pickle=False) as z:meta=dict(z)
    actual=np.load(source/'actual.npy',mmap_mode='r',allow_pickle=False)
    eligible=meta['quality_pass'].all(1)
    if len(meta['ids'])!=40338 or len(np.unique(meta['ids']))!=40338 or eligible.sum()!=40309:
        raise ValueError('Changed cohort')
    mean_maps=np.empty((len(meta['ids']),actual.shape[-1]),dtype=np.float32)
    for start in range(0,len(meta['ids']),512):
        mean_maps[start:start+512]=active_mean(actual[start:start+512],meta['labels'][start:start+512])
    out=root/'output';out.mkdir(exist_ok=False)
    importance=out/'importance';importance.mkdir()
    scan_meta=dict(meta,split=np.where(eligible,'train','excluded'))
    workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))
    scores=np.full((66,len(meta['ids'])),np.nan,dtype=np.float32)
    covered=np.zeros(scores.shape,dtype=np.uint16);commands=[];hit_counts={}
    lookup={ident:j for j,ident in enumerate(ids)}
    def scan(j):
        return scan_shard(j,ids[j::workers],binary,database,root/'training_background.txt',
                          prior/'importance/enhancers.fa',scan_meta,mean_maps,importance,
                          database_key='denovo')
    event('denovo_scan_ready',profiles=66,enhancers=int(eligible.sum()),workers=workers,
          gaf_highlighted=sum(p['gaf_highlight'] for p in profiles),new_attributions=0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for names,values,bases,counts,command in pool.map(scan,range(workers)):
            for j,name in enumerate(names):scores[lookup[name]]=values[j];covered[lookup[name]]=bases[j]
            commands.append(command);hit_counts.update(counts)
    np.testing.assert_array_equal(np.isfinite(scores),covered>0)
    if np.isfinite(scores[:,~eligible]).any():raise ValueError('Excluded enhancer received a score')
    breadth=meta['labels'].sum(1)
    summary=dict(profiles=66,database='Information-filtered TF-MoDISco positive PWMs',
        catalogue=profiles,filter=analysis['rule'],gaf_ids=[p['id'] for p in profiles if p['gaf_highlight']],
        target='Mean observed-active calibrated probabilities',references=100,steps=64,
        population='All 40309 common-QC native enhancers; all splits exploratory',
        definition='Signed IG per union of FIMO hit bases, equally averaged across motif-containing '
                   'enhancers. Noncarriers missing. Fixed filtered PWMs across degrees; FIMO p<=1e-4.',
        exact=summarize(scores,breadth,eligible,profiles),
        cumulative=summarize(scores,breadth,eligible,profiles,True),commands=commands,hit_counts=hit_counts,
        no_hit_profiles=[ids[j] for j in range(66) if not np.isfinite(scores[j]).any()],
        database_sha256=digest(database),background_sha256=digest(root/'training_background.txt'),
        fasta_sha256=digest(prior/'importance/enhancers.fa'),
        native_hashes={name:prepared['files']['native/'+name] for name in ('actual.npy','metadata.npz')},
        independent_biological_validation=False,no_reclustering=True,
        caveat='PWMs were discovered from these same maps and overlapping groups; descriptive, '
               'not independent validation. Similar PWMs remain separate, not independent TFs. '
               'Short low-information PWMs can have no sites at the unchanged scan threshold.')
    write_json(importance/'summary.json',summary)
    np.savez_compressed(importance/'scores.npz',scores=scores,covered_bases=covered,ids=meta['ids'],
                        motif_ids=np.asarray(ids),breadth=breadth,split=meta['split'],eligible=eligible)
    write_json(out/'complete.json',dict(status='complete',manifest_sha256=digest(root/'MANIFEST.sha256'),
        summary=dict(profiles=66,enhancers=int(eligible.sum()),visible_profiles=int(np.isfinite(scores).any(1).sum()),
                     new_attributions=0,no_hit_profiles=summary['no_hit_profiles']),
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}))
    event('denovo_importance_complete',visible_profiles=int(np.isfinite(scores).any(1).sum()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.project,a.root)
