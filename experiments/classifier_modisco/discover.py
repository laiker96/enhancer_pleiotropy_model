"""De novo TF-MoDISco; no motif database participates in discovery."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .common import (balanced_indices,digest,event,load_inputs,projected_contributions,
                     require_allocation,write_json)


def discover(sequences,hypothetical,config,output,cap):
    import modiscolite
    if importlib.metadata.version("modisco") != config["modisco_version"]:
        raise ValueError("Wrong TF-MoDISco release")
    if sequences.shape != hypothetical.shape or sequences.ndim !=3 or sequences.shape[2] !=4:
        raise ValueError("API requires matching [examples,length,ACGT] arrays")
    if not np.isfinite(hypothetical).all() or not np.isin(sequences,[0,1]).all() or not (sequences.sum(2)==1).all():
        raise ValueError("Invalid attribution or one-hot input")
    random.seed(config["seed"]); np.random.seed(config["seed"])
    keys=("n_leiden_runs","sliding_window_size","flank_size","trim_to_window_size",
          "initial_flank_to_add","final_flank_to_add","target_seqlet_fdr")
    positive,negative = modiscolite.tfmodisco.TFMoDISco(
        one_hot=sequences.astype(np.float32),hypothetical_contribs=hypothetical.astype(np.float32),
        max_seqlets_per_metacluster=cap,verbose=True,**{k:config[k] for k in keys})
    temporary=output.with_suffix(".partial.h5")
    modiscolite.io.save_hdf5(temporary,positive,negative,window_size=sequences.shape[1])
    temporary.replace(output)
    return {"positive":len(positive or []),"negative":len(negative or [])}


def make_report(h5,output):
    from modiscolite.report import report_motifs
    output.mkdir(exist_ok=True)
    # Relative paths keep the report usable after it is copied back locally.
    report_motifs(h5,output,img_path_suffix="",meme_motif_db=None,
                  is_writing_tomtom_matrix=False)


def synthetic_smoke(root):
    require_allocation("cpu")
    config=json.loads((root/"config.json").read_text())
    rng=np.random.default_rng(config["seed"])
    codes=rng.integers(0,4,(256,128),dtype=np.uint8)
    motif=np.asarray([0,1,2,0,3,1,1,2,3,0,2,2])
    hyp=rng.normal(0,.005,(256,128,4)).astype(np.float32)
    for i in range(len(codes)):
        start=int(rng.integers(40,76)); codes[i,start:start+len(motif)]=motif
        hyp[i,np.arange(start,start+len(motif)),motif]+=1
    sequence=np.eye(4,dtype=np.float32)[codes]
    output=root/"runtime_smoke"; output.mkdir(exist_ok=True)
    started=time.monotonic()
    counts=discover(sequence,hyp,config,output/"synthetic.h5",300)
    if counts["positive"] <1: raise RuntimeError("Synthetic planted motif was not recovered")
    make_report(output/"synthetic.h5",output/"report")
    write_json(root/"runtime_ready.json",dict(status="passed",modisco=config["modisco_version"],
        synthetic_patterns=counts,seconds=time.monotonic()-started,job_id=os.environ["SLURM_JOB_ID"]))
    event("modisco_runtime_ready",patterns=counts)


def collect(root,data,provenance):
    completion=json.loads((root/"attribution_complete.json").read_text())
    if completion["input_provenance_sha256"] != digest(root/"input_provenance.json"):
        raise ValueError("Attribution input contract differs")
    n=len(data["ids"])
    hyp=np.empty((n,4,512),np.float32); quality=np.zeros(n,bool); seen=np.zeros(n,bool)
    for name,sha in completion["chunks"].items():
        path=root/"chunks"/name
        if digest(path) !=sha: raise ValueError("Changed attribution chunk")
        with np.load(path,allow_pickle=False) as f:
            idx=f["indices"]
            if str(f["signature"]) != completion["signature"] or seen[idx].any(): raise ValueError("Repeated/wrong attribution chunk")
            hyp[idx]=f["hypothetical_central512"]
            actual=projected_contributions(data["sequence"][idx,768:1280],hyp[idx])
            np.testing.assert_allclose(actual,f["actual"][:,768:1280],atol=1e-7,rtol=1e-5)
            quality[idx]=f["quality_pass"]; seen[idx]=True
    if not seen.all(): raise ValueError("Incomplete cohort attribution")
    return hyp,quality


def summarize_seqlets(root,output,data,selected):
    import h5py
    rows=[]; instances=[]
    with h5py.File(output/"motifs.h5","r") as f:
        for group in ("pos_patterns","neg_patterns"):
            if group not in f: continue
            for name,p in f[group].items():
                s=p["seqlets"]
                local=s["example_idx"][:].astype(int)
                if (local<0).any() or (local>=len(selected)).any(): raise ValueError("Unknown seqlet example")
                idx=selected[local]
                widths=s["end"][:]-s["start"][:]
                if (widths<=0).any() or (s["start"][:]<0).any() or (s["end"][:]>512).any(): raise ValueError("Seqlet outside central512")
                unique=np.unique(idx); breadth=data["labels"][unique].sum(1)
                rows.append(dict(pattern=group+"/"+name,seqlets=len(idx),unique_enhancers=len(unique),
                    exact_breadth_counts={str(k):int((breadth==k).sum()) for k in range(1,9)},
                    nested_breadth_counts={"ge_"+str(k):int((breadth>=k).sum()) for k in range(2,9)},
                    interpretation="Training discovery support only, not population enrichment or held-out validation"))
                for row,i in enumerate(idx):
                    start=int(s["start"][row]); end=int(s["end"][row])
                    instances.append(dict(pattern=group+"/"+name,id=str(data["ids"][i]),
                        cohort_index=int(i),breadth=int(data["labels"][i].sum()),labels=data["labels"][i].tolist(),
                        chrom=str(data["chrom"][i]),start=int(data["summit"][i])-256+start,
                        end=int(data["summit"][i])-256+end,reverse=bool(s["is_revcomp"][row])))
    write_json(output/"motif_breadth_support.json",rows)
    write_json(output/"seqlet_instances.json",instances)
    return rows


def run(root):
    require_allocation("cpu")
    config,provenance,data=load_inputs(root)
    ready=json.loads((root/"runtime_ready.json").read_text())
    if ready["status"]!="passed" or ready["modisco"]!=config["modisco_version"]: raise ValueError("Successful runtime smoke required")
    hypothetical,quality=collect(root,data,provenance)
    selected=balanced_indices(data["labels"],data["split"],quality,config["discovery_breadth_bins"],
                              config["maximum_elements_per_bin"],config["seed"])
    output=root/"denovo"; output.mkdir(exist_ok=True)
    dna=data["sequence"][selected,768:1280]
    sequences=np.eye(4,dtype=np.float32)[dna]
    hyp=hypothetical[selected].transpose(0,2,1).copy()
    np.savez_compressed(output/"discovery_examples.npz",indices=selected,ids=data["ids"][selected],
                        labels=data["labels"][selected],chrom=data["chrom"][selected],summit=data["summit"][selected])
    np.savez_compressed(output/"discovery_sequences.npz",sequences.transpose(0,2,1))
    np.savez_compressed(output/"discovery_hypothetical.npz",hyp.transpose(0,2,1))
    write_json(output/"selection.json",dict(elements=len(selected),quality_excluded=int((~quality).sum()),
        train_only=True,seed=config["seed"],breadth_bins=config["discovery_breadth_bins"],
        elements_per_bin=len(selected)//len(config["discovery_breadth_bins"]),window=512,
        exact_breadth_counts={str(k):int((data["labels"][selected].sum(1)==k).sum()) for k in range(1,9)},
        prior_motif_database_used=False))
    event("modisco_discovery_start",elements=len(selected),window=512,max_seqlets=config["max_seqlets_per_metacluster"])
    # A smaller real-data pass verifies extraction/clustering before the full cap.
    pilot=discover(sequences,hyp,config,output/"pilot.h5",config["pilot_max_seqlets"])
    event("modisco_real_data_pilot",patterns=pilot)
    counts=discover(sequences,hyp,config,output/"motifs.h5",config["max_seqlets_per_metacluster"])
    rows=summarize_seqlets(root,output,data,selected)
    if rows: make_report(output/"motifs.h5",output/"report")
    write_json(root/"complete.json",dict(status="complete" if rows else "complete_no_patterns",patterns=counts,
        discovery_elements=len(selected),attributed_elements=len(data["ids"]),de_novo_discovery=True,
        known_motif_screen_run=False,heldout_enrichment_run=False,perturbation_run=False,
        model_id=config["model_id"],job_id=os.environ["SLURM_JOB_ID"],modisco=config["modisco_version"],
        outputs={str(p.relative_to(root)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
        limitations=config["limitations"]))
    event("modisco_complete",patterns=counts)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("runtime_smoke","discover")); p.add_argument("--root",type=Path,required=True)
    a=p.parse_args(); (synthetic_smoke if a.stage=="runtime_smoke" else run)(a.root)
