"""Separate de novo motif discovery at each observed pleiotropy threshold."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .common import digest,event,load_inputs,require_allocation,write_json
from .discover import collect,discover,make_report,summarize_seqlets


def threshold_indices(labels,split,quality,group,seed):
    labels=np.asarray(labels); split=np.asarray(split); quality=np.asarray(quality)
    if labels.ndim!=2 or labels.shape[1]!=8 or not np.isin(labels,[0,1]).all() or not labels.any(1).all():
        raise ValueError("Eight observed binary activity labels required")
    if split.shape!=(len(labels),) or quality.shape!=(len(labels),) or quality.dtype!=np.bool_:
        raise ValueError("Aligned split labels and boolean quality mask required")
    low,high=group["minimum_breadth"],group["maximum_breadth"]
    if not 1<=low<=high<=8: raise ValueError("Invalid breadth bounds")
    breadth=labels.sum(1)
    indices=np.flatnonzero((split=="train") & quality & (breadth>=low) & (breadth<=high))
    if not len(indices): raise ValueError("No eligible training enhancers for "+group["name"])
    return np.random.default_rng(seed).permutation(indices)


def importance_summary(example_idx,contributions,eligible_count):
    """Give each supporting enhancer one vote, regardless of its seqlet count."""
    example_idx=np.asarray(example_idx); contributions=np.asarray(contributions)
    if (contributions.ndim!=3 or contributions.shape[2]!=4 or contributions.shape[1]<1
            or len(example_idx)!=len(contributions) or not np.isfinite(contributions).all()
            or not len(example_idx) or eligible_count<1):
        raise ValueError("Aligned finite seqlet contributions required")
    unique,inverse=np.unique(example_idx,return_inverse=True)
    if len(unique)>eligible_count: raise ValueError("Support exceeds the input cohort")
    site=contributions.sum((1,2),dtype=np.float64)
    counts=np.bincount(inverse)
    per_enhancer=np.bincount(inverse,weights=site)/counts
    per_base=per_enhancer/contributions.shape[1]
    return dict(supporting_enhancers=len(unique),assigned_enhancer_fraction=len(unique)/eligible_count,
        mean_site_contribution=float(per_enhancer.mean()),median_site_contribution=float(np.median(per_enhancer)),
        mean_contribution_per_base=float(per_base.mean()),median_contribution_per_base=float(np.median(per_base)),
        supporting_enhancer_positive_fraction=float((per_enhancer>0).mean()),
        seqlets=len(example_idx),aligned_seqlet_length=int(contributions.shape[1]))


def load_contract(root,project):
    config=json.loads((root/"config.json").read_text())
    parent=project/config["parent_experiment"]
    if digest(parent/"MANIFEST.sha256")!=config["parent_package_sha256"]:
        raise ValueError("Wrong parent attribution package")
    parameters,provenance,data=load_inputs(parent)
    ready=json.loads((parent/"runtime_ready.json").read_text())
    if ready["status"]!="passed" or ready["modisco"]!=parameters["modisco_version"]:
        raise ValueError("Successful TF-MoDISco runtime smoke required")
    return config,parent,parameters,provenance,data


def summarize_importance(output,data,selected,group_name):
    import h5py
    support=summarize_seqlets(None,output,data,selected)
    rows=[]
    with h5py.File(output/"motifs.h5","r") as f:
        for row in support:
            pattern=row["pattern"]; s=f[pattern]["seqlets"]
            stats=importance_summary(s["example_idx"][:],s["contrib_scores"][:],len(selected))
            if row["seqlets"]!=stats.pop("seqlets"):
                raise ValueError("Inconsistent seqlet counts: "+pattern)
            rows.append(dict(group=group_name,group_specific_pattern_id=group_name+"/"+pattern,
                **row,**stats,ranking_direction="positive" if pattern.startswith("pos_") else "negative"))
    # Each sign is ranked independently; rank is within a fitted group only.
    for sign in ("positive","negative"):
        chosen=[r for r in rows if r["ranking_direction"]==sign]
        chosen.sort(key=lambda r:(-abs(r["mean_contribution_per_base"]),r["pattern"]))
        for rank,row in enumerate(chosen,1): row["within_group_importance_rank"]=rank
    write_json(output/"motif_importance.json",rows)
    return rows


def finish_group(root,parent,parameters,config,data,selected,group,counts,recovered_inputs=None):
    output=root/"groups"/group["name"]
    rows=summarize_importance(output,data,selected,group["name"])
    if rows: make_report(output/"motifs.h5",output/"report")
    if recovered_inputs is not None:
        for name,sha in recovered_inputs.items():
            if digest(output/name)!=sha: raise ValueError("Recovery changed discovery input: "+name)
    write_json(output/"complete.json",dict(status="complete" if rows else "complete_no_patterns",group=group,
        patterns=counts,discovery_elements=len(selected),de_novo_discovery=True,
        analysis_config_sha256=digest(root/"config.json"),parent_attribution_sha256=digest(parent/"attribution_complete.json"),
        job_id=os.environ["SLURM_JOB_ID"],array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        reporting_code_sha256=digest(Path(__file__)),source_package_sha256=digest(root/"MANIFEST.sha256"),
        report_only_recovery=recovered_inputs is not None,recovered_input_sha256=recovered_inputs,
        parameters={k:parameters[k] for k in ("seed","modisco_version","max_seqlets_per_metacluster","target_seqlet_fdr")},
        outputs={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
        limitations=config["limitations"]))
    event("breadth_modisco_complete",group=group["name"],patterns=counts,report_only_recovery=recovered_inputs is not None)


def recover(root,project,group_index):
    """Regenerate reports from saved full fits; never run attribution or discovery."""
    require_allocation("cpu")
    config,parent,parameters,provenance,data=load_contract(root,project)
    if not 0<=group_index<len(config["groups"]): raise ValueError("Unknown group index")
    group=config["groups"][group_index]; output=root/"groups"/group["name"]
    if (output/"complete.json").exists(): raise FileExistsError("Group already complete: "+group["name"])
    parameters=dict(parameters,seed=parameters["seed"]+group_index)
    names=("motifs.h5","pilot.h5","discovery_examples.npz","discovery_sequences.npz",
           "discovery_hypothetical.npz","selection.json")
    recovered_inputs={name:digest(output/name) for name in names}
    selection=json.loads((output/"selection.json").read_text())
    if (selection["group"]!=group or selection["seed"]!=parameters["seed"]
            or selection["window"]!=512 or not selection["train_only"]):
        raise ValueError("Saved discovery selection differs from the group contract")
    with np.load(output/"discovery_examples.npz",allow_pickle=False) as saved:
        selected=saved["indices"]
        if (selected.ndim!=1 or selected.dtype.kind not in "iu" or not len(selected)
                or len(selected)!=selection["elements"] or len(np.unique(selected))!=len(selected)
                or (selected<0).any() or (selected>=len(data["labels"])).any()):
            raise ValueError("Invalid saved discovery indices")
        for key in ("ids","labels","chrom","summit"):
            if not np.array_equal(saved[key],data[key][selected]):
                raise ValueError("Saved discovery examples differ from cohort: "+key)
    breadth=data["labels"][selected].sum(1)
    if (not (data["split"][selected]=="train").all()
            or not ((breadth>=group["minimum_breadth"])&(breadth<=group["maximum_breadth"])).all()):
        raise ValueError("Saved discovery examples violate training/breadth selection")
    import h5py
    with h5py.File(output/"motifs.h5","r") as f:
        counts={sign:len(f.get(name,{})) for sign,name in (("positive","pos_patterns"),("negative","neg_patterns"))}
    event("breadth_report_recovery_start",group=group["name"],patterns=counts,discovery_rerun=False)
    finish_group(root,parent,parameters,config,data,selected,group,counts,recovered_inputs)


def run(root,project,group_index):
    require_allocation("cpu")
    config,parent,parameters,provenance,data=load_contract(root,project)
    if not 0<=group_index<len(config["groups"]): raise ValueError("Unknown group index")
    group=config["groups"][group_index]
    output=root/"groups"/group["name"]; output.mkdir(parents=True,exist_ok=True)
    if (output/"complete.json").exists(): raise FileExistsError("Group already complete: "+group["name"])
    hypothetical,quality=collect(parent,data,provenance)
    parameters=dict(parameters,seed=parameters["seed"]+group_index)
    selected=threshold_indices(data["labels"],data["split"],quality,group,parameters["seed"])
    breadth=data["labels"].sum(1)
    original=(data["split"]=="train") & (breadth>=group["minimum_breadth"]) & (breadth<=group["maximum_breadth"])
    sequences=np.eye(4,dtype=np.float32)[data["sequence"][selected,768:1280]]
    hyp=hypothetical[selected].transpose(0,2,1).copy()
    np.savez_compressed(output/"discovery_examples.npz",indices=selected,ids=data["ids"][selected],
        labels=data["labels"][selected],chrom=data["chrom"][selected],summit=data["summit"][selected])
    np.savez_compressed(output/"discovery_sequences.npz",sequences.transpose(0,2,1))
    np.savez_compressed(output/"discovery_hypothetical.npz",hyp.transpose(0,2,1))
    selection=dict(group=group,elements=len(selected),eligible_before_qc=int(original.sum()),
        quality_excluded=int((original&~quality).sum()),train_only=True,no_enhancer_downsampling=True,
        seed=parameters["seed"],window=512,
        exact_breadth_counts={str(k):int((breadth[selected]==k).sum()) for k in range(1,9)},
        prior_motif_database_used=False)
    write_json(output/"selection.json",selection)
    event("breadth_modisco_start",group=group["name"],elements=len(selected),window=512)
    pilot=discover(sequences,hyp,parameters,output/"pilot.h5",parameters["pilot_max_seqlets"])
    event("breadth_modisco_pilot",group=group["name"],patterns=pilot)
    counts=discover(sequences,hyp,parameters,output/"motifs.h5",parameters["max_seqlets_per_metacluster"])
    finish_group(root,parent,parameters,config,data,selected,group,counts)


def summarize(root):
    require_allocation("cpu")
    config=json.loads((root/"config.json").read_text())
    groups=[]; rows=[]
    for group in config["groups"]:
        output=root/"groups"/group["name"]
        completion=json.loads((output/"complete.json").read_text())
        if completion["analysis_config_sha256"]!=digest(root/"config.json"): raise ValueError("Group config differs")
        for name,sha in completion["outputs"].items():
            if digest(output/name)!=sha: raise ValueError("Group output changed")
        groups.append(dict(group=group["name"],status=completion["status"],patterns=completion["patterns"],
                           discovery_elements=completion["discovery_elements"]))
        rows.extend(json.loads((output/"motif_importance.json").read_text()))
    write_json(root/"breadth_motif_importance.json",dict(groups=groups,motifs=rows,limitations=config["limitations"]))
    write_json(root/"complete.json",dict(status="complete",groups=groups,independent_fits=len(groups),
        cross_group_motif_matching=False,heldout_enrichment=False,
        output_sha256=digest(root/"breadth_motif_importance.json")))
    event("all_breadth_discoveries_complete",groups=groups)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("discover","recover","summarize")); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--project",type=Path,required=True); p.add_argument("--group-index",type=int)
    a=p.parse_args()
    if a.stage in ("discover","recover"):
        if a.group_index is None: p.error("--group-index required for discovery/recovery")
        (run if a.stage=="discover" else recover)(a.root,a.project,a.group_index)
    else: summarize(a.root)
