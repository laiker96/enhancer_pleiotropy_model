"""Transparent, motif-diverse high-contrast illustrations from saved IG sites.

This selects clear examples, not a random or population-representative sample.
Background here means surrounding enhancer attribution, not genomic negatives.
"""
import itertools

import numpy as np


DIVERSE_STRATA = [('exact_1', 1, 'pattern_10'), ('ge_2', 2, 'Trl'),
                 ('exact_1', 1, 'pattern_13'), ('ge_4', 4, 'Trl'),
                 ('exact_1', 1, 'pattern_0'), ('ge_8', 8, 'Trl+cg')]
MIN_MEAN = .04
MIN_NATIVE_CONTRAST = 3.
MIN_LOCAL_CONTRAST = 2.
MIN_POSITIVE_FRACTION = .7
HALO = 2


def site_contrast(actual, selected, zoom):
    """Compare each signed site mean to absolute attribution outside site halos."""
    actual = np.asarray(actual, dtype=float)
    mask = np.ones(len(actual), dtype=bool)
    for site in selected:
        mask[max(0, site['start'] - HALO):min(len(actual), site['end'] + HALO)] = False
    left, right = zoom
    local_mask = mask[left:right]
    if mask.sum() < 20 or local_mask.sum() < 20:
        return None
    native_bg = float(np.abs(actual[mask]).mean())
    local_bg = float(np.abs(actual[left:right][local_mask]).mean())
    sites = []
    for site in selected:
        values = actual[site['start']:site['end']]
        mean = float(values.mean())
        positive_fraction = float((values > 0).mean())
        sites.append(dict(motif_id=site['motif_id'], mean_ig=mean,
            positive_fraction=positive_fraction,
            native_ratio=mean / max(native_bg, .001),
            local_ratio=mean / max(local_bg, .001)))
    minimum_ratio = min(min(s['native_ratio'], s['local_ratio']) for s in sites)
    minimum_mean = min(s['mean_ig'] for s in sites)
    return dict(sites=sites, native_background_abs_mean=native_bg,
        local_background_abs_mean=local_bg, native_background_bases=int(mask.sum()),
        local_background_bases=int(local_mask.sum()), minimum_ratio=minimum_ratio,
        score=minimum_mean * min(minimum_ratio, 10.),
        passes=(minimum_mean >= MIN_MEAN
                and min(s['native_ratio'] for s in sites) >= MIN_NATIVE_CONTRAST
                and min(s['local_ratio'] for s in sites) >= MIN_LOCAL_CONTRAST
                and min(s['positive_fraction'] for s in sites) >= MIN_POSITIVE_FRACTION))


def choose_configuration(actual, annotations, kind, zoom_interval):
    """Select one clear native motif or two distinct proximal Trl/cg sites."""
    trl = [s for s in annotations if s['best_match'] == 'Trl' and s['tomtom_q'] < .05]
    cg = [s for s in annotations if s['best_match'] == 'cg' and s['tomtom_q'] < .05]
    if kind == 'Trl+cg':
        configurations = itertools.product(trl, cg)
    elif kind == 'Trl':
        configurations = ([s] for s in trl)
    else:
        configurations = ([s] for s in annotations if s['motif_id'].split('/')[-1] == kind)
    candidates = []
    for config in configurations:
        sites = sorted(config, key=lambda s: (s['start'], s['end'], s['motif_id']))
        start, end = min(s['start'] for s in sites), max(s['end'] for s in sites)
        gap = None
        if kind == 'Trl+cg':
            gap = sites[1]['start'] - sites[0]['end']
            if not 3 <= gap <= 25 or end - start > 48:
                continue
        zoom = zoom_interval(start, end, len(actual))
        if kind == 'Trl' and any(s['start'] < zoom[1] and s['end'] > zoom[0] for s in cg):
            continue  # Trl-only zoom, not a claim of no cg elsewhere in enhancer.
        contrast = site_contrast(actual, sites, zoom)
        if contrast is None or not contrast['passes']:
            continue
        candidates.append(dict(anchor=sites[0], plot_sites=sites, zoom=list(zoom),
            contrast=contrast, pair_gap_bp=gap, example_kind=kind))
    if not candidates:
        return None
    return min(candidates, key=lambda c: (-c['contrast']['score'],
        tuple((s['start'], s['end'], s['motif_id']) for s in c['plot_sites'])))


def select_clear_example(candidates):
    if not candidates:
        raise ValueError('No motif configuration passes the declared contrast criteria')
    target = float(np.quantile([c['contrast']['score'] for c in candidates], .9))
    chosen = min(candidates, key=lambda c: (abs(c['contrast']['score'] - target), c['id']))
    shortlist = [dict(id=c['id'], contexts=c['active_contexts'],
        score=c['contrast']['score'], minimum_ratio=c['contrast']['minimum_ratio'])
        for c in sorted(candidates, key=lambda c: (-c['contrast']['score'], c['id']))[:20]]
    return dict(chosen, eligible_candidates=len(candidates), selection_shortlist=shortlist,
                selection_score_quantile=.9, selection_score_target=target)


def selection_protocol():
    return dict(purpose='Motif-diverse high-contrast training illustrations, not population representativeness',
        context_specific='Three different native degree-1 motifs: ewg-like pattern_10, su(Hw)-like pattern_13, and pattern_0 (C-rich; TF match nonsignificant). Different sole-active contexts.',
        pleiotropic='Exact degrees 2/4: Trl-like site with no significant cg-like scan match in its displayed zoom. Exact degree 8: nonoverlapping Trl/cg sites, gap 3-25 bp, joint span <=48 bp.',
        motif_pool='All retained positive native TF-MoDISco patterns; not restricted to panel B top-four subset',
        site_p_max=1e-4, trl_cg_tomtom_q_max=.05, minimum_mean_site_ig=MIN_MEAN,
        minimum_positive_base_fraction=MIN_POSITIVE_FRACTION,
        minimum_site_to_native_background_ratio=MIN_NATIVE_CONTRAST,
        minimum_site_to_local_background_ratio=MIN_LOCAL_CONTRAST,
        background='Mean absolute actual IG outside selected sites plus 2-bp halos, computed separately over the 60-bp zoom and entire native enhancer; minimum 20 background bases each; denominator floor 0.001 logit/base',
        ranking='For each enhancer, maximize min(site mean IG) * min(weakest local/native site-to-background ratio, 10). Within each requested motif/degree configuration, choose the enhancer closest to the 90th percentile of that score among quality-passing enhancers, rather than the maximum; ties by ID. Both paired sites must pass.',
        inference='Positive contributions and proximity do not establish TF occupancy, enhancer context specificity of the motif, or cooperative binding')
