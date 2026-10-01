"""Prepare split-contained background pools without changing enhancer examples."""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from .background_matching import gc_features, sequence_key, match_background
from .data import CONTEXTS, SPLITS, assign_split, digest, load_split, write_json


def event(name, **values):
    print(json.dumps(dict(event=name,**values)),flush=True)


def prepare(enhancer_root, windows, locked_test_root, output):
    enhancer_root,windows,locked_test_root,output = map(Path,(enhancer_root,windows,locked_test_root,output))
    if output.exists(): raise FileExistsError(output)
    event("background_preparation_started")
    audit = json.loads((enhancer_root/"audit.json").read_text())
    meta = json.loads(windows.with_name("windows.metadata.json").read_text())
    if digest(windows) != meta["output"]["sha256"]: raise ValueError("Window hash mismatch")
    canonical = lambda obj: {k:{s:sorted(obj[k][s]) for s in SPLITS}
                             for k in ("chromosome_splits","region_splits")}
    if canonical(audit["regression_splits"]) != canonical(meta["inputs"]):
        raise ValueError("Background/enhancer split definitions differ")
    enhancers = {s:load_split(enhancer_root,s) for s in SPLITS}
    locked = json.loads((locked_test_root/"matching.json").read_text())
    test_path = locked_test_root/"matched_background.npz"
    if digest(test_path) != locked["background_sha256"]: raise ValueError("Locked test backgrounds changed")
    if digest(enhancer_root/"test.npz") not in locked["input_sha256"].values():
        raise ValueError("Locked test enhancer dataset mismatch")
    with np.load(test_path,allow_pickle=False) as f: test_bg = dict(f)
    if not np.array_equal(test_bg["enhancer_ids"],enhancers["test"]["ids"][test_bg["enhancer_index"]]):
        raise ValueError("Locked test pair alignment mismatch")
    forbidden = {sequence_key(row) for s in SPLITS for row in enhancers[s]["sequence"]}
    forbidden.update(sequence_key(row) for row in test_bg["sequence"])
    pools = {s:dict(ids=[],chrom=[],start=[],end=[],sequence=[],gc=[],keep=[]) for s in ("train","validation")}
    seen, coordinates = {}, set()
    counts = collections.Counter()
    lookup = np.full(256,255,dtype=np.uint8)
    lookup[np.frombuffer(b"ACGT",dtype=np.uint8)] = np.arange(4)
    with gzip.open(windows,"rt") as handle:
        for row in csv.DictReader(handle,delimiter="\t"):
            split = row["split"]
            if row["source"] != "genomic_background" or split not in pools: continue
            counts[split+"_candidates"] += 1
            chrom,start,end = row["chrom"],int(row["start"]),int(row["end"])
            if (end-start != 2048 or int(row["target_start"]) != start+768 or int(row["target_end"]) != start+1280
                    or assign_split(chrom,start,end,audit["regression_splits"]) != split):
                raise ValueError("Candidate crosses split or has wrong geometry")
            coordinate=(chrom,start,end)
            if coordinate in coordinates: raise ValueError("Duplicate candidate coordinate")
            coordinates.add(coordinate)
            codes=lookup[np.frombuffer(row["sequence"].encode("ascii"),dtype=np.uint8)]
            if len(codes)!=2048 or (codes>3).any(): raise ValueError("Invalid background DNA")
            key=sequence_key(codes)
            if key in forbidden:
                counts[split+"_catalog_or_locked_test_duplicate"] += 1
                continue
            if key in seen:
                previous_split,index=seen[key]
                pools[previous_split]["keep"][index]=False
                counts[split+"_background_duplicate"] += 1
                continue
            pool=pools[split];seen[key]=(split,len(pool["ids"]))
            for field,value in (("ids",row["record_id"]),("chrom",chrom),("start",start),("end",end),("sequence",codes),("keep",True)):
                pool[field].append(value)
            gc=(codes==1)|(codes==2)
            pool["gc"].append([gc.mean(),gc[768:1280].mean()])
            if sum(len(v["ids"]) for v in pools.values())%50000==0:
                event("background_candidates_read",counts=dict(counts))
    output.mkdir(parents=True)
    bg_root=output/"background";bg_root.mkdir()
    for filename in ("audit.json", "train.npz", "validation.npz", "test.npz"):
        shutil.copy2(enhancer_root/filename,output/filename)
    summaries={}
    for split,pool in pools.items():
        if counts[split+"_candidates"] != meta["candidate_and_sampling_counts"][split]["selected_background"]:
            raise ValueError("Background candidate count mismatch")
        features=np.asarray(pool["gc"]); chrom=np.asarray(pool["chrom"]);keep=np.asarray(pool["keep"])
        enh=enhancers[split];enh_gc=gc_features(enh["sequence"])
        matched=np.full(len(enh["ids"]),-1,dtype=int)
        for chromosome in sorted(set(enh["chrom"])):
            positive_idx=np.flatnonzero(enh["chrom"]==chromosome)
            candidates=np.flatnonzero((chrom==chromosome)&keep)
            if not len(candidates):continue
            local=match_background(enh_gc[positive_idx],features[candidates],seed=20260916)
            ok=local>=0;matched[positive_idx[ok]]=candidates[local[ok]]
        idx=np.flatnonzero(matched>=0);paired=matched[idx]
        if split=="train" and len(paired)<len(enh["ids"])//3:raise ValueError("Insufficient training matches")
        if not len(paired):raise ValueError("No background matches")
        delta=features[paired]-enh_gc[idx]
        if len(np.unique(paired))!=len(paired) or np.max(np.abs(delta))>.02+1e-12:raise ValueError("Invalid GC pairs")
        values={k:np.asarray(pool[k])[paired] for k in ("ids","chrom","start","end")}
        values.update(sequence=np.stack([pool["sequence"][i] for i in paired]),
                      labels=np.zeros((len(paired),8),dtype=np.uint8),enhancer_index=idx,
                      enhancer_ids=enh["ids"][idx],enhancer_gc=enh_gc[idx],background_gc=features[paired])
        np.savez_compressed(bg_root/(split+".npz"),**values)
        summaries[split]=dict(enhancers=len(enh["ids"]),background_pool=len(paired),
            unmatched_enhancer_ids=enh["ids"][matched<0].tolist(),
            gc_mean_absolute_difference=np.abs(delta).mean(0).tolist(),gc_max_absolute_difference=np.abs(delta).max(0).tolist())
        event("background_split_prepared",split=split,**{k:v for k,v in summaries[split].items() if k!='unmatched_enhancer_ids'})
    test_bg["labels"]=np.zeros((len(test_bg["ids"]),8),dtype=np.uint8)
    np.savez_compressed(bg_root/"test.npz",**test_bg)
    summaries["test"]=dict(enhancers=len(enhancers["test"]["ids"]),background_pool=len(test_bg["ids"]),locked_original_file_sha256=digest(test_path))
    write_json(bg_root/"audit.json",dict(status="complete",contexts=CONTEXTS,seed=20260916,splits=summaries,
        background_definition=locked["background_definition"],matching=locked["matching"],
        matching_seed_note="Train/validation use seed 20260916; test pairs are reused unchanged from seed 20260915 diagnostic",
        background_batch_contract="floor(N_enhancers/3) unique backgrounds per epoch, fresh deterministic permutation; all enhancer exposures and optimizer-step counts preserved",
        candidate_counts=dict(counts),regression_splits=audit["regression_splits"],
        outputs={s+".npz":digest(bg_root/(s+".npz")) for s in SPLITS},
        inputs={str(p.resolve()):digest(p) for p in (windows,windows.with_name("windows.metadata.json"),enhancer_root/"audit.json",test_path,locked_test_root/"matching.json")},
        source_sha256={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name("background_matching.py"))}))
    event("background_dataset_complete",output=str(output),splits=summaries)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("enhancer-root","windows","locked-test-root","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();prepare(a.enhancer_root,a.windows,a.locked_test_root,a.output)
