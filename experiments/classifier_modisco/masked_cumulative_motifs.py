"""Cumulative family breadth: retain 14 identical fits and discover only >=2/>=3."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from . import calibrated_motifs as native
from . import cumulative_motifs as cumulative
from . import masked_motifs as masked
from .common import digest,event,require_allocation,write_json

NAME='classifier_masked_cumulative_motifs_20260929'
PARENT='experiments/'+masked.NAME
NEW_TASKS=(13,14)


def tasks():
    result=masked.tasks()
    for number in NEW_TASKS:
        result[number].update(group=f'at_least_{result[number]["low"]}_active_families',high=4)
    return result


def prepare(project,root,config):
    parent=project/PARENT
    old_config=json.loads((parent/'config.json').read_text())
    if old_config['specs']!=masked.specs() or old_config['discovery_parameters']!=config['discovery_parameters']:
        raise ValueError('Source weighting or discovery parameters changed')
    cumulative.prepare(project,root,config,task_list=tasks())
    with np.load(root/'native/metadata.npz',allow_pickle=False) as z:meta=dict(z)
    (root/'fits').mkdir(exist_ok=False)
    reused=[]
    for task in tasks():
        indices,_,weights=masked.validate_task(meta,task,config['discovery_parameters']['seed'])
        if task['task'] in NEW_TASKS:continue
        source_task=old_config['tasks'][task['task']]
        if source_task!=task:raise ValueError('Attempt to reuse changed cohort')
        source=cumulative.directory(parent,source_task)
        done=json.loads((source/'complete.json').read_text())
        report=json.loads((source/'report_complete.json').read_text())
        for receipt in (done,report):
            if receipt['status']!='complete' or receipt['task']!=task:raise ValueError('Incomplete reuse source')
            for name,sha in receipt['files'].items():
                if digest(source/name)!=sha:raise ValueError('Changed reuse file '+name)
        with np.load(source/'examples.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['indices'],indices)
            np.testing.assert_array_equal(z['ids'],meta['ids'][indices])
            np.testing.assert_array_equal(z['weights'],weights)
        selection=json.loads((source/'selection.json').read_text())
        if selection['spec']!=masked.specs()[task['target']]:raise ValueError('Changed source readout')
        # Read-only links to frozen results; never annotate or write into them.
        cumulative.directory(root,task).symlink_to(source,target_is_directory=True)
        reused.append(dict(task=task,source=str(source),complete_sha256=digest(source/'complete.json'),
                           report_complete_sha256=digest(source/'report_complete.json')))
    write_json(root/'reuse.json',dict(reused=reused,new_tasks=list(NEW_TASKS),
        policy='Verified identical cohorts/order/IDs/weights, raw-fit and report checksums. Sources unchanged.'))
    event('masked_cumulative_prepared',reused=len(reused),new_tasks=list(NEW_TASKS))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','discover','finalize'))
    p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',type=int,default=0);args=p.parse_args();require_allocation('cpu')
    if importlib.metadata.version('modisco')!='2.5.2':raise ValueError('Pinned TF-MoDISco required')
    config=json.loads((args.root/'config.json').read_text())
    if config['tasks']!=tasks() or config['specs']!=masked.specs():raise ValueError('Wrong cumulative family contract')
    if args.stage=='prepare':prepare(args.project,args.root,config)
    elif args.stage=='discover':
        if args.task not in NEW_TASKS:raise ValueError('Unchanged fits must be reused, not recomputed')
        task=tasks()[args.task]
        native.discover(args.root,config,args.task,task=task,spec=masked.specs()[task['target']])
        masked.annotate(args.project,args.root,task)
    else:
        reuse=json.loads((args.root/'reuse.json').read_text())
        if len(reuse['reused'])!=14:raise ValueError('Missing verified reused analyses')
        masked.finalize(args.root,task_list=tasks(),cumulative_breadth=True,
                        run_summary=dict(new_discoveries=2,reused_fits=14))
