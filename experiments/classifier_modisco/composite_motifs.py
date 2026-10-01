"""Exploratory two-block PWM diagnostics; never merge or alter discovered motifs."""
import html
import json
from pathlib import Path

import numpy as np

from .common import digest, write_json
from .dual_motif_pipeline import informative_core
from .calibrated_tomtom import alignment_logos, label, match_text, page_start
from .report_breadth import sequence_logo
from .tomtom_atlas import query_id, run as tomtom_run


def split_candidates(row, cwm):
    """All informative disjoint left/right cores at every cut; no preferred TF."""
    pwm=np.asarray(row['trimmed_pwm']);offset=row['quality']['start']
    cwm=np.asarray(cwm)
    if cwm.shape!=(len(row['full_pwm']),4):raise ValueError('Unaligned raw contribution matrix')
    sign=1 if row['sign']=='positive' else -1
    parts={};pairs=set()
    for cut in range(5,len(pwm)-4):
        pair=[]
        for start,end in ((0,cut),(cut,len(pwm))):
            q=informative_core(pwm[start:end],flank_threshold=.2)
            if not q['passed']:break
            a,b=start+q['start'],start+q['end']
            if cwm[offset+a:offset+b].sum()*sign<=0:break
            key=(a,b)
            if key not in parts:
                matrix=pwm[a:b].tolist()
                parts[key]=dict(id=row['id']+f'__part_{a}_{b}',seqlets=row['seqlets'],
                    trimmed_pwm=matrix,quality=dict(passed=True),
                    consensus=''.join('ACGT'[i] for i in np.argmax(matrix,axis=1)),
                    start=a,end=b,source_pattern=row['id'])
            pair.append(key)
        if len(pair)==2:pairs.add(tuple(pair))
    used={key for pair in pairs for key in pair}
    return [parts[key] for key in sorted(used)],sorted(pairs)


def score_aligned(sequence,pwm):
    """Same max-logodds convention as motif_occurrence.pssm, uniform background."""
    p=np.asarray(pwm,float);seq=np.asarray(sequence)
    if seq.shape[1:]!=p.shape or not np.isin(seq,[0,1]).all() or not (seq.sum(2)==1).all():
        raise ValueError('Expected aligned one-hot native seqlets')
    weights=np.log2(((p+.001*.25)/(1+.001))/.25)
    maximum=weights.max(1).sum()
    if maximum<=0:raise ValueError('Uninformative subcore')
    return np.einsum('nlb,lb->n',seq,weights)/maximum


def joint_support(sequence,contribution,examples,row,left,right):
    offset=row['quality']['start'];parts=(left,right)
    if (sequence.shape!=contribution.shape or len(sequence)!=len(examples)
            or sequence.shape[1:]!=(len(row['full_pwm']),4)):
        raise ValueError('Seqlets are not aligned to the original PWM')
    scores=[];sign=1 if row['sign']=='positive' else -1;signed=[]
    for part in parts:
        a,b=offset+part['start'],offset+part['end']
        scores.append(score_aligned(sequence[:,a:b],part['trimmed_pwm']))
        signed.append(contribution[:,a:b].sum((1,2))*sign>0)
    support=[]
    for threshold in (.7,.8,.9):
        a,b=[scores[i]>=threshold for i in range(2)];both=a&b
        support.append(dict(threshold=threshold,seqlets=len(sequence),enhancers=len(np.unique(examples)),
            left_seqlets=int(a.sum()),right_seqlets=int(b.sum()),both_seqlets=int(both.sum()),
            both_enhancers=len(np.unique(examples[both])),
            both_source_sign_seqlets=int((both&signed[0]&signed[1]).sum()),
            both_source_sign_enhancers=len(np.unique(examples[both&signed[0]&signed[1]]))))
    return support


def analyze(folder,group,h5_path,binary,references):
    import h5py
    folder.mkdir(exist_ok=False)
    candidates={};queries=[];rows={r['id']:r for r in group['rows']}
    with h5py.File(h5_path,'r') as h5:
        for row in group['rows']:
            parts,pairs=split_candidates(row,h5[row['pattern']+'/contrib_scores'][:])
            if pairs:
                candidates[row['id']]=dict(parts=parts,pairs=pairs);queries.extend(parts)
    write_json(folder/'candidates.json',dict(patterns=candidates,queries=len(queries)))
    if queries:
        write_json(folder/'queries.json',dict(groups=[dict(rows=queries)]))
        tomtom_run(folder,folder/'queries.json',binary,references=references,database_keys=('jaspar',),
            query_rule='Exploratory two-block subqueries; all informative cuts; no reclustering or TF-directed selection.')
        matches=json.loads((folder/'jaspar/matches.json').read_text())['best']
    else:matches={}
    selected=[]
    with h5py.File(h5_path,'r') as h5:
        for ident,candidate in candidates.items():
            row=rows[ident];parts={(p['start'],p['end']):p for p in candidate['parts']}
            def pair_key(pair):
                m=[matches[query_id(parts[k])] for k in pair]
                return (max(x['q'] for x in m),max(x['p'] for x in m),sum(x['p'] for x in m),pair)
            best=min(candidate['pairs'],key=pair_key);left,right=[parts[k] for k in best]
            node=h5[row['pattern']+'/seqlets']
            support=joint_support(node['sequence'][:],node['contrib_scores'][:],node['example_idx'][:],row,left,right)
            lm,rm=[matches[query_id(p)] for p in (left,right)]
            selected.append(dict(id=ident,pattern=row['pattern'],sign=row['sign'],row=row,
                left=left,right=right,left_match=lm,right_match=rm,
                gap_bp=right['start']-left['end'],candidate_pairs=len(candidate['pairs']),support=support,
                both_nominal_q_le_005=max(lm['q'],rm['q'])<=.05,
                same_reference_profile=lm['target_id']==rm['target_id']))
    result=dict(task=group['task'],elements=group['elements'],patterns=selected,
        selected_by='Minimum worse-part q, then worse-part p, then sum p, then coordinates.',
        caveat='Exploratory selected submotif matches: q-values are not corrected across cuts/pairs. '
            'Co-support is in the discovery seqlets, not independent validation or proof of cooperativity.',
        score='Aligned subcore log-odds / theoretical maximum; uniform background; pseudocount 0.001; 0.7/0.8/0.9 sensitivity.',
        h5_sha256=digest(h5_path))
    write_json(folder/'diagnostics.json',result)
    return result


def render(groups):
    page=page_start('Candidate paired motifs — cumulative JASPAR analysis',('jaspar',))
    page.append('<p>Exploratory diagnostic, not a test of TF cooperativity. Every informative disjoint '
        'split was compared with JASPAR; the best two-block explanation is shown. Split-selected '
        'q-values are nominal: no additional correction across cuts/pairs. A repeat can match the '
        'same profile twice without representing two independent sites.</p><p>Co-support requires '
        'both subcores in the SAME original aligned seqlet. Scores use the discovered subcore PWMs, '
        'not a new JASPAR sequence scan. Thresholds are fractions of maximum log-odds, not p-values. '
        'Cumulative groups overlap; recurrence across them is not independent replication.</p>')
    page.append('<h2>Recurrence of candidate profile pairs</h2><p>Cells count discovered patterns '
        'with both selected part matches at nominal q ≤0.05 and at least one enhancer contributing '
        'a seqlet with both subcores at score ≥0.8 and the source contribution sign. Pairs are '
        'unordered here; this is not an orientation-specific grammar or an enrichment statistic. '
        'All candidates, including weak ones, remain in the detailed sections below.</p>')
    for target in dict.fromkeys(g['task']['target'] for g in groups):
        selected=[g for g in groups if g['task']['target']==target];recurrence={}
        for column,g in enumerate(selected):
            for item in g['patterns']:
                support=next(s for s in item['support'] if s['threshold']==.8)
                if not item['both_nominal_q_le_005'] or not support['both_source_sign_enhancers']:continue
                refs=sorted((item['left_match']['reference'],item['right_match']['reference']),key=lambda r:r['id'])
                key=(item['sign'],refs[0]['id'],refs[1]['id'])
                if key not in recurrence:
                    recurrence[key]=dict(names=[r['name'] for r in refs],counts=[0]*len(selected))
                recurrence[key]['counts'][column]+=1
        page.append('<h3>'+html.escape(label(target))+'</h3><div class="table-wrap"><table><tr><th>Sign / profile pair</th>'
            +''.join('<th>'+html.escape(label(g['task']['group']))+'</th>' for g in selected)+'</tr>')
        for key,value in sorted(recurrence.items()):
            name=' + '.join(f'{name} ({ident})' for name,ident in zip(value['names'],key[1:]))
            page.append('<tr><td>'+html.escape(key[0]+': '+name)+'</td>'
                +''.join(f'<td>{count}</td>' for count in value['counts'])+'</tr>')
        if not recurrence:page.append(f'<tr><td colspan="{len(selected)+1}">No candidates meet this diagnostic rule.</td></tr>')
        page.append('</table></div>')
    for g in groups:
        t=g['task'];page.append(f'<section><h2>{html.escape(label(t["target"]))} / {html.escape(label(t["group"]))}</h2>')
        if not g['patterns']:page.append('<p>No retained PWM admits two informative disjoint subcores.</p>')
        for item in g['patterns']:
            row=item['row'];lm,rm=item['left_match'],item['right_match']
            page.append(f'<article class="{row["sign"]}"><h3>{row["sign"]}: <code>{row["consensus"]}</code></h3>'
                f'<p>{html.escape(row["pattern"])}; {item["candidate_pairs"]} possible splits. '
                f'Gap between selected cores: {item["gap_bp"]} bp. '
                +('Same reference profile in both parts.' if item['same_reference_profile'] else 'Different reference profiles; not proof of distinct TFs.')+'</p>')
            page.append(sequence_logo(row['trimmed_pwm'],row['consensus']))
            for part,m in ((item['left'],lm),(item['right'],rm)):
                page.append(f'<p>Core [{part["start"]}, {part["end"]}): {match_text(m)}; nominal q={m["q"]:.3g}</p>')
                page.append(alignment_logos(part,m))
            page.append('<table><tr><th>Score cutoff</th><th>Both cores / seqlets</th><th>Both cores / cluster enhancers</th><th>Both also have source contribution sign</th></tr>')
            for s in item['support']:
                page.append(f'<tr><td>{s["threshold"]:.1f}</td><td>{s["both_seqlets"]}/{s["seqlets"]}</td>'
                    f'<td>{s["both_enhancers"]}/{s["enhancers"]}</td><td>{s["both_source_sign_enhancers"]} enhancers</td></tr>')
            page.append('</table></article>')
        page.append('</section>')
    return ''.join(page)+'</body></html>'
