"""Figure 3 with a fixed, information-filtered de novo PWM panel D."""
import json

import numpy as np

from . import paper_figure3_frequency as previous
from .figure3_denovo_data import NAME

base=previous.base
ROOT=base.PROJECT/'results'/NAME
DATA=ROOT/'cecar_results'
OUTPUT=base.PROJECT/'output/pdf/figure_3_context_positive_denovo_importance_20260930.pdf'


def importance_panel(fig,summary,top):
    from matplotlib.lines import Line2D
    if summary['profiles']!=66 or len(summary['exact'])!=66:
        raise ValueError('All 66 fixed de novo PWMs required')
    ids=[r['id'] for r in summary['exact']]
    if len(set(ids))!=66 or set(ids)!={p['id'] for p in summary['catalogue']}:
        raise ValueError('Catalogue and importance IDs differ')
    expected={p['id'] for p in summary['catalogue'] if p['name']=='Trl' and p['tomtom_q']<=.05}
    if set(summary['gaf_ids'])!=expected:raise ValueError('Changed GAF annotation policy')
    base.text(fig,16,top,'d',25,weight='bold')
    ax=base.draw.make_axes(fig,(138,top-242,1120,214))
    visible=[];highlighted=[]
    for row in sorted(summary['exact'],key=lambda r:r['id'] in expected):
        values=np.asarray([np.nan if v is None else v for v in row['means']])*1000
        if not np.isfinite(values).any():continue
        gaf=row['id'] in expected;visible.append(row['id'])
        if gaf:highlighted.append(row['id'])
        ax.plot(np.arange(1,9),values,color=base.GAF_COLOR if gaf else '#B3B6BA',
                alpha=.8 if gaf else .65,lw=1.5 if gaf else .8,
                marker='o',markersize=4 if gaf else 2.2,markeredgewidth=0,
                zorder=5 if gaf else 1,solid_capstyle='butt',clip_on=True)
    ax.plot([1,8],[0,0],color='#666666',lw=.5,zorder=0,solid_capstyle='butt',clip_on=True)
    ax.set(xlim=(.95,8.05),xticks=np.arange(1,9),xlabel='Degree of pleiotropy',
           ylabel='Mean motif importance\n(IG / bp × 10³)')
    base.draw.clean(ax);ax.spines['bottom'].set_bounds(1,8)
    ax.tick_params(labelsize=12,length=3,pad=5)
    ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
    ax.xaxis.labelpad=8;ax.yaxis.labelpad=12
    if highlighted:
        ax.legend(handles=[Line2D([],[],color=base.GAF_COLOR,lw=1.5,marker='o',markersize=4,
                                 markeredgewidth=0,label='GAF')],loc='upper left',fontsize=12,frameon=False)
    return dict(database=summary['database'],total_profiles=66,visible_profiles=len(visible),
                grouping='exact',highlighted=highlighted,gaf_catalogue=sorted(expected),
                no_hit_profiles=[ident for ident in ids if ident not in visible],
                frequency_shown=False,connected_points=True,line_domain=[1,8],
                selection='All best-match Trl PWMs with Tomtom q<=0.05; separate curves, no reclustering')


def comparison(summary,reference):
    """Export carrier counts and all exact-degree ranks, not just a favorable curve."""
    means=np.array([[np.nan if v is None else v for v in r['means']] for r in summary['exact']])
    catalogue={p['id']:p for p in summary['catalogue']};rows=[]
    for i,row in enumerate(summary['exact']):
        p=catalogue[row['id']]
        if p['name']!='Trl':continue
        values=means[i]
        rows.append(dict(**row,discovery_group=p['discovery_group'],tomtom_q=p['tomtom_q'],
            highlighted=p['gaf_highlight'],width=p['width'],
            ranks=[int(1+(means[:,k]>v).sum()) if np.isfinite(v) else None for k,v in enumerate(values)],
            degree8_over_degree1=float(values[7]/values[0])
                if np.isfinite(values[[0,7]]).all() and values[0]!=0 else None))
    ffs=next(r for r in reference['exact'] if r['id']==previous.previous.previous.GAF_ID)
    return dict(denovo_gaf=rows,flyfactorsurvey_gaf=ffs,
                no_hit_profiles=summary['no_hit_profiles'],caveat=summary['caveat'])


if __name__=='__main__':
    previous.render(denovo=True)
