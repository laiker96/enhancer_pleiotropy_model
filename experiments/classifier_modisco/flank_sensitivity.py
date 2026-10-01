"""Reprocess unchanged non-overlap fits with 0.2-bit flanks, on CECAR CPUs."""
import argparse
import json

from pathlib import Path

import numpy as np

from .common import digest, event, require_allocation, write_json
from .dual_motif_pipeline import TARGETS, core_rule, report, verify_sources
from .nonoverlap_motifs import GROUPS, compare_groups, verify_receipt
from .nonoverlap_motif_figure import render
from .tomtom_atlas import DATABASES


def setup(project, root, config):
    verify_sources(project, config)
    parent = project/config['parent']
    ready = verify_receipt(parent, 'setup_complete.json')
    if config['flank_threshold'] != .2 or config['report_suffix'] != '_flanks02':
        raise ValueError('Unexpected sensitivity protocol')
    for name in ('metadata.npz', 'native_actual.npy', 'native_hypothetical.npy'):
        (root/name).symlink_to(parent/name)
    for target in TARGETS:
        verify_receipt(parent/target, 'report_complete.json')
        verify_receipt(parent/target, 'comparison_complete.json')
        directory = root/target/'groups'; directory.mkdir(parents=True)
        for group, _, _ in GROUPS:
            source = parent/target/'groups'/group
            verify_receipt(source)
            (directory/group).symlink_to(source, target_is_directory=True)
    write_json(root/'setup_complete.json', dict(status='complete', files=ready['files'],
        counts=ready['counts'], parent_audit_sha256=ready['parent_audit_sha256'],
        config_sha256=digest(root/'config.json'), parent_setup_sha256=digest(parent/'setup_complete.json'),
        reuse='All six discovery fits and all attributions unchanged; flank threshold only.',
        rule=core_rule(.2)))
    event('flank_setup_complete', rule=core_rule(.2))


def boundary_comparison(old, new):
    """Account for every raw pattern, including motifs that fail either filter."""
    def index(audit):
        accepted = {r['id']:r for g in audit['groups'] for r in g['rows']}
        rejected = {r['id']:r for r in audit['exclusions']}
        if accepted.keys() & rejected.keys():
            raise ValueError('Motif both retained and excluded')
        return accepted, rejected
    before, old_failed = index(old); after, new_failed = index(new)
    if before.keys() | old_failed.keys() != after.keys() | new_failed.keys():
        raise ValueError('Sensitivity analysis must use exactly the same raw motifs')
    rows = []
    for ident in sorted(before.keys() | old_failed.keys()):
        first, second = before.get(ident), after.get(ident)
        if first and second:
            if first['supporting_discovery_enhancers'] != second['supporting_discovery_enhancers']:
                raise ValueError('Native cluster support changed')
            np.testing.assert_array_equal(first['full_pwm'], second['full_pwm'])
            a,b = first['quality'], second['quality']
            if b['start'] > a['start'] or b['end'] < a['end']:
                raise ValueError('Gentler flank filter unexpectedly shortened a retained motif')
            state = 'extended' if (a['start'],a['end']) != (b['start'],b['end']) else 'unchanged'
        else:
            state = 'added' if second else 'removed' if first else 'excluded_both'
        rows.append(dict(id=ident, change=state,
            old_quality=first['quality'] if first else old_failed[ident],
            new_quality=second['quality'] if second else new_failed[ident]))
    return dict(raw=len(rows), old_retained=len(before), new_retained=len(after),
        changes={key:sum(r['change']==key for r in rows)
                 for key in ('unchanged','extended','added','removed','excluded_both')}, rows=rows)


def sensitivity(project, root, config, target):
    result = root/TARGETS[target]; previous = project/config['parent']/TARGETS[target]
    old = json.loads((previous/'report_audit.json').read_text())
    new = json.loads((result/'report_audit.json').read_text())
    comparison = boundary_comparison(old, new)
    metrics = [{r['id']:r for r in json.loads((p/'cross_group_summary.json').read_text())['rows']}
               for p in (previous,result)]
    matches = [{db:json.loads((p/'annotation'/db/'matches.json').read_text())['best']
                for db in DATABASES} for p in (previous,result)]
    for row in comparison['rows']:
        ident = row['id']; query = ident.replace('/','__')
        for number,key in enumerate(('old','new')):
            row[key+'_metrics'] = metrics[number].get(ident)
            row[key+'_matches'] = {}
            for db in DATABASES:
                match = matches[number][db].get(query)
                row[key+'_matches'][db] = (dict(target_id=match['target_id'],
                    name=match['reference']['name'], p=match['p'], q=match['q']) if match else None)
    comparison.update(target=TARGETS[target], old_rule=old['rules']['trim'],new_rule=new['rules']['trim'],
        note='Exploratory flank sensitivity, not selection for a preferred TF match. Fresh FIMO/Tomtom for every retained PWM.',
        old_report_sha256=digest(previous/'report_audit.json'),new_report_sha256=digest(result/'report_audit.json'))
    write_json(result/'flank_sensitivity.json',comparison)
    write_json(result/'sensitivity_complete.json',dict(status='complete',files={
        'flank_sensitivity.json':digest(result/'flank_sensitivity.json')}))
    event('flank_sensitivity_complete',target=TARGETS[target],
        old_retained=comparison['old_retained'],new_retained=comparison['new_retained'],changes=comparison['changes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('setup','report'))
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',type=int,default=0)
    args=parser.parse_args();require_allocation('cpu')
    config=json.loads((args.root/'config.json').read_text())
    if args.stage=='setup':
        setup(args.project,args.root,config)
    else:
        if args.task not in (0,1):raise ValueError('Only two targets')
        ready=json.loads((args.root/'setup_complete.json').read_text())
        if ready['config_sha256'] != digest(args.root/'config.json'):
            raise ValueError('Configuration changed after setup')
        report(args.project,args.root,config,args.task,groups=GROUPS)
        compare_groups(args.project,args.root,config,args.task)
        sensitivity(args.project,args.root,config,args.task)
        render(args.project,args.root,config,args.task)


if __name__=='__main__':main()
