"""Raw enhancer-length TF-MoDISco -> informative cores -> frequency -> Tomtom.

No post-hoc merging, reclustering, CWM-relative trimming, or attribution rerun.
All real-data computation is restricted to a CECAR CPU allocation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, event, require_allocation, write_json
from .original_intervals import GROUPS
from .original_sequence_report import frequency, parse_fimo_line
from .tomtom_atlas import DATABASES, heights, meme_queries, query_id, read_meme, run as tomtom

TOP_N = 5
SITE_P = 1e-4


def informative_core(pwm, threshold=.2):
    """One contiguous native-pattern slice, with explicit pass/failure reasons."""
    pwm = np.asarray(pwm, dtype=float)
    if (pwm.ndim != 2 or pwm.shape[1] != 4 or not len(pwm)
            or not np.isfinite(pwm).all() or (pwm < 0).any()
            or not np.allclose(pwm.sum(1), 1, atol=1e-6)):
        raise ValueError("Invalid native probability matrix")
    pwm = pwm/pwm.sum(1, keepdims=True)
    information = np.array(heights(pwm.tolist())).sum(1)
    if not math.isfinite(threshold) or not 0 <= threshold < 2:
        raise ValueError("Invalid information threshold")
    starts = [i for i in range(len(pwm)-4) if (information[i:i+5] > threshold).all()]
    if not starts:
        return dict(passed=False, reason=f"no_five_consecutive_columns_above_{threshold:g}_bits")
    start, end = starts[0], starts[-1]+5
    info = information[start:end]
    result = dict(start=start, end=end, width=end-start,
                  mean_bits=float(info.mean()), total_bits=float(info.sum()))
    reason = ("core_longer_than_30_bp" if end-start > 30 else
              "mean_information_below_0.5_bits" if info.mean() < .5 else
              "total_information_below_5_bits" if info.sum() < 5 else "passed")
    return dict(result, passed=reason == "passed", reason=reason)


def filter_native(path, name, counts, threshold=.2, *, core_filter=None):
    """Apply the identical reporting filter to either native discovery window."""
    import h5py
    rows, exclusions = [], []
    with h5py.File(path, "r") as h5:
        for sign, prefix in (("positive", "pos_patterns"), ("negative", "neg_patterns")):
            if len(h5.get(prefix, {})) != counts[sign]:
                raise ValueError("Raw motif count differs from discovery audit")
            for key in sorted(h5.get(prefix, {})):
                node = h5[prefix][key]
                pwm, cwm = np.asarray(node["sequence"]), np.asarray(node["contrib_scores"])
                qc = informative_core(pwm, threshold) if core_filter is None else core_filter(pwm)
                ident, pattern = name+"/"+prefix+"/"+key, prefix+"/"+key
                if not qc["passed"]:
                    exclusions.append(dict(id=ident, sign=sign, **qc)); continue
                start, end = qc["start"], qc["end"]
                signed = float(cwm[start:end].sum())
                if not math.isfinite(signed) or signed*(1 if sign == "positive" else -1) <= 0:
                    exclusions.append(dict(id=ident, sign=sign, **dict(qc, passed=False, reason="trimmed_core_sign_disagrees"))); continue
                core = pwm[start:end].astype(float)
                core /= core.sum(1, keepdims=True)
                rows.append(dict(id=ident, pattern=pattern, sign=sign, quality=qc,
                    trimmed_pwm=core.tolist(), full_pwm=pwm.tolist(),
                    seqlets=len(node["seqlets/example_idx"]),
                    supporting_discovery_enhancers=len(np.unique(node["seqlets/example_idx"][:])),
                    signed_core_contribution=signed))
    return rows, exclusions


def rank_rows(group):
    """Keep discovery support and sequence prevalence as distinct quantities."""
    support = group.get("rank_by") == "native_support"
    for row in group["rows"]:
        row.pop("rank", None)
        if support:
            n, hits = group["discovery_elements"], row["supporting_discovery_enhancers"]
            if not 0 <= hits <= n or n <= 0:
                raise ValueError("Invalid discovery support denominator/count")
            row["attribution_support"] = dict(hits=hits, n=n, fraction=hits/n)
    for sign in ("positive", "negative"):
        selected = sorted((r for r in group["rows"] if r["sign"] == sign
                           and (support or r["frequency_quantifiable"])),
            key=lambda r: (-(r["attribution_support"]["fraction"] if support
                             else r["occurrence"]["train"]["fraction"]), r["id"]))
        for rank, row in enumerate(selected, 1):
            row["rank"] = rank


def original_support_scope():
    """Reuse the exact support-report policy with explicit original-interval provenance."""
    from .central512_report import SUPPORT_SCOPE
    return dict(SUPPORT_SCOPE,
        title="Enhancer-length motifs: attribution-supported prevalence",
        filename="enhancer50ref_support_top5",
        window_label="Original enhancers; 50 references",
        scan_region="original enhancer sequences",
        intro="Reuse the completed native TF-MoDISco fits from attribution tensors restricted to the ORIGINAL enhancer boundaries, not a central-512 crop. The frozen CNN epoch38 saw its full 2048-bp genomic input; the target was the mean observed-active-context forward/RC logit. These attributions consistently used 50 shuffled references. No old two-reference scores, additional motif clustering, merging or new attribution are used. The discovery and sequence-scan intervals are the original enhancers.",
        query_rule="Native original-enhancer 50-reference motifs trimmed to five-column runs >0.5 bits; 5-30bp; no extra clustering",
        native_discovery="reused exact original-enhancer fits; 50 shuffled references",
        missing_description=SUPPORT_SCOPE["missing_description"].replace("central 512-bp windows", "original enhancer sequences"))


def prepare(project, root):
    from .report_breadth import COLORS, GLYPHS
    config = json.loads((root/"config.json").read_text())
    mode = config.get("report_mode", "sequence_frequency")
    if mode not in ("sequence_frequency", "native_support_strict"):
        raise ValueError("Unknown original-enhancer report mode")
    support = mode == "native_support_strict"
    for name, expected in config["inputs"].items():
        if digest(project/name) != expected:
            raise ValueError("Changed frozen input: "+name)
    source, previous = project/config["source"], project/config["previous_report"]
    if support:
        choice = json.loads((source/"reference_choice.json").read_text())
        if choice["references"] != 50 or choice["reuse_old_two_reference_scores"]:
            raise ValueError("Expected original-enhancer attribution with 50 references")
    # The previous files contain original intervals, not the central 512 crop.
    prior = json.loads((previous/"motif_audit.json").read_text())
    for name in ("enhancers.fa", "cohort_metadata.npz", "training_background.txt"):
        if digest(previous/name) != prior["prepared_files"][name]:
            raise ValueError("Changed prepared sequence input: "+name)
    with np.load(previous/"cohort_metadata.npz", allow_pickle=False) as f:
        metadata = dict(f)
    groups, exclusions = [], []
    for name, low, high in GROUPS:
        directory = source/"groups"/name
        selection = json.loads((directory/"selection.json").read_text())
        discovery = json.loads((directory/"motifs.audit.json").read_text())
        if (selection["interval"] != "original_catalog_bounds" or not selection["train_only"]
                or discovery["padding_in_threshold_distribution"]
                or discovery["discovery_interval"] != "original_enhancer"):
            raise ValueError("Not a valid enhancer-length discovery")
        n = int(np.sum((metadata["split"] == "train") & (metadata["breadth"] >= low) & (metadata["breadth"] <= high)))
        if n != selection["elements"]:
            raise ValueError("Discovery/cohort disagreement")
        if support and selection["attribution_sha256"] != digest(source/"attribution_complete.json"):
            raise ValueError("Discovery/50-reference attribution hash mismatch")
        rows, excluded = filter_native(directory/"motifs.h5", name,
            {sign:discovery[sign]["patterns"] for sign in ("positive", "negative")}, threshold=.5 if support else .2)
        exclusions.extend(excluded)
        groups.append(dict(name=name, minimum_breadth=low, maximum_breadth=high, n=n,
            rows=rows, native_discovery=discovery))
        if support:
            groups[-1].update(rank_by="native_support", discovery_elements=selection["elements"])
    audit = dict(groups=groups, exclusions=exclusions, colors=COLORS, glyphs=GLYPHS,
        inputs=config["inputs"], previous_prepared_files=prior["prepared_files"],
        background=prior["background"], top_n=TOP_N,
        rules=dict(source="raw groups/*/motifs.h5 from completed enhancer-length TF-MoDISco",
            attribution="same frozen epoch38 CNN, full2048 input, 50 references; unchanged",
            reclustering=False, posthoc_merging=False,
            trim="first through last run of 5 consecutive columns >0.2 bits; retain internal positions",
            filter="core 5-30bp; mean information >=0.5 bits; total information >=5 bits; sign concordant",
            rank="training enhancer sequence-scan frequency; separate per group/sign; ID breaks ties",
            frequency="unique enhancers with >=1 FIMO hit / all enhancers in group and split",
            site_p=SITE_P, background="fixed train-only RC-symmetric zero-order background",
            selected="top5 frequency-quantifiable motifs per group/sign; fewer when insufficient; all filtered motifs annotated"))
    if support:
        audit["report_scope"] = original_support_scope()
        audit["rules"].update(attribution_references=50, model_input_bp=2048,
            discovery_interval="original_enhancer", scan_interval="original_enhancer",
            trim="first through last run of 5 consecutive columns >0.5 bits; preserve internal gaps; uniform IC background",
            rank="distinct native cluster-supporting enhancers / discovery enhancers; separately per group/sign; ID breaks ties",
            support="original seqlet cluster memberships, unchanged after trimming; not attribution at rescanned sites",
            selected="top5 informative motifs by native support, including scan-unquantifiable cores; scan prevalence shown separately")
    (root/"scans").mkdir()
    write_json(root/"cores.json", audit)
    event("raw_cores_prepared", retained=sum(len(g["rows"]) for g in groups), excluded=len(exclusions),
          counts={g["name"]:{sign:sum(r["sign"]==sign for r in g["rows"]) for sign in ("positive","negative")} for g in groups})
    return audit, metadata, previous


def scan(root, previous, group, metadata, background):
    rows = group["rows"]
    if not rows:
        return
    folder = root/"scans"/group["name"]; folder.mkdir()
    query = folder/"cores.meme"; query.write_text(meme_queries(dict(groups=[group])))
    lookup = {query_id(r):i for i,r in enumerate(rows)}
    widths = [len(r["trimmed_pwm"]) for r in rows]
    binary = root/"bin/fimo"
    command = [str(binary), "--text", "--thresh", str(SITE_P), "--motif-pseudo", ".1",
        "--bgfile", str(previous/"training_background.txt"), str(query), str(previous/"enhancers.fa")]
    best = np.ones((len(rows), len(metadata["ids"])))
    event("simple_scan_start", group=group["name"], motifs=len(rows))
    with (folder/"stderr.log").open("x") as stderr, gzip.open(folder/"hits.tsv.gz", "wt") as archive:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, text=True)
        try:
            header = next(process.stdout)
            if not header.startswith("motif_id\tmotif_alt_id\tsequence_name"):
                raise ValueError("Unexpected FIMO header")
            archive.write(header)
            for line in process.stdout:
                archive.write(line)
                if line.strip() and not line.startswith("#"):
                    parse_fimo_line(line, lookup, best, metadata["length"], widths)
            if process.wait(): raise RuntimeError("FIMO failed")
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
    # Prevent short cores with an unattainable site-p cutoff from being called
    # biologically absent. This is a small FIMO probe, not an extra motif scan.
    probe = folder/"best_possible.fa"
    with probe.open("x") as handle:
        for i,row in enumerate(rows):
            codes=np.argmax(np.asarray(row["trimmed_pwm"])/background,axis=1)
            handle.write(f">e{i}\n"+"".join("ACGT"[b] for b in codes)+"\n")
    probe_command = command[:]; probe_command[3] = "1"; probe_command[-1] = str(probe)
    result=subprocess.run(probe_command,text=True,capture_output=True,check=True)
    (folder/"best_possible.tsv").write_text(result.stdout)
    possible=np.ones((len(rows),len(rows)))
    for line in result.stdout.splitlines()[1:]:
        if line and not line.startswith("#"):
            parse_fimo_line(line,lookup,possible,widths,widths)
    for i,row in enumerate(rows):
        row["minimum_possible_p"]=float(possible[i,i])
        row["frequency_quantifiable"]=bool(possible[i,i] <= SITE_P)
        row["occurrence"]={}
        for split in ("train","validation","test"):
            mask=((metadata["split"]==split) & (metadata["breadth"]>=group["minimum_breadth"])
                  & (metadata["breadth"]<=group["maximum_breadth"]))
            row["occurrence"][split]=frequency(best[i],mask,metadata["length"],widths[i],SITE_P)
    rank_rows(group)
    np.savez_compressed(folder/"minimum_p.npz", query_ids=np.array(list(lookup)), minimum_p=best)
    write_json(folder/"complete.json",dict(command=command,probe_command=probe_command,
        motifs_sha256=digest(query),hits_sha256=digest(folder/"hits.tsv.gz")))
    event("simple_scan_done",group=group["name"])


def combgap(root, audit):
    reference=read_meme(root/"references"/DATABASES["jaspar"]["file"])["MA2107.1"]
    if reference["name"] != "cg": raise ValueError("Unexpected Combgap profile")
    path=root/"annotation/jaspar/tomtom.tsv"
    with path.open() as stream:
        reader=csv.DictReader((line for line in stream if line.strip() and not line.startswith("#")),delimiter="\t")
        by_query={}
        for match in reader:by_query.setdefault(match["Query_ID"],[]).append(match)
    records=[]
    for group in audit["groups"]:
        for row in group["rows"]:
            matches=sorted(by_query[query_id(row)],key=lambda x:(float(x["p-value"]),float(x["q-value"]),x["Target_ID"]))
            i,match=next((i,m) for i,m in enumerate(matches,1) if m["Target_ID"]=="MA2107.1")
            row["combgap"]=dict(rank=i,q=float(match["q-value"]),p=float(match["p-value"]),
                offset=int(match["Optimal_offset"]),orientation=match["Orientation"],overlap=int(match["Overlap"]))
            records.append(dict(id=row["id"],sign=row["sign"],**row["combgap"]))
    write_json(root/"combgap.json",dict(reference=reference,matches=records,
        present_in=dict(jaspar=True,flyreg=False,flyfactorsurvey=False),
        source="https://jaspar.elixir.no/matrix/MA2107.1/",identity="https://flybase.org/reports/FBgn0000289"))
    with (root/"combgap_matches.tsv").open("x") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]),delimiter="\t");writer.writeheader();writer.writerows(records)


def main(project, root):
    require_allocation("cpu")
    for binary,flag in (("fimo","--version"),("tomtom","-version")):
        if subprocess.check_output([str(root/"bin"/binary),flag],text=True).strip()!="5.5.9":
            raise ValueError("Expected MEME5.5.9")
    interval=json.loads((root/"config.json").read_text()).get("interval", "original_enhancer")
    if interval == "central512":
        from .central512_report import prepare as prepare_central
        audit,metadata,previous=prepare_central(project,root)
    elif interval == "original_enhancer":
        audit,metadata,previous=prepare(project,root)
    else:
        raise ValueError("Unknown discovery/report interval")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda g:scan(root,previous,g,metadata,np.array(audit["background"]["probabilities"])),audit["groups"]))
    write_json(root/"frequency_audit.json",audit)
    if audit.get("report_scope", {}).get("rank_by") == "native_support":
        with (root/"attribution_support.tsv").open("x") as handle:
            writer=csv.writer(handle, delimiter="\t")
            writer.writerow(["motif", "sign", "rank", "native_support_enhancers", "discovery_enhancers", "support_fraction"])
            for g in audit["groups"]:
                for r in g["rows"]:
                    s=r["attribution_support"]
                    writer.writerow([r["id"], r["sign"], r["rank"], s["hits"], s["n"], s["fraction"]])
    with (root/"frequency.tsv").open("x") as handle:
        writer=csv.writer(handle,delimiter="\t")
        writer.writerow(["motif","sign","width","training_rank","split","hits","n","fraction","quantifiable"])
        for g in audit["groups"]:
            for r in g["rows"]:
                for split,v in r["occurrence"].items():
                    writer.writerow([r["id"],r["sign"],len(r["trimmed_pwm"]),r.get("rank",""),split,v["hits"],v["n"],v["fraction"],r["frequency_quantifiable"]])
    annotation=root/"annotation";annotation.mkdir()
    (annotation/"references").symlink_to(root/"references",target_is_directory=True)
    tomtom(annotation,root/"frequency_audit.json",root/"bin/tomtom",
           query_rule=audit.get("report_scope", {}).get("query_rule", "Information-trimmed raw enhancer-length TF-MoDISco motifs; 5-30bp; no extra clustering"))
    combgap(root,audit)
    write_json(root/"report_audit.json",audit)
    from .simple_report_pdf import render
    render(root,audit)
    outputs=[root/"cores.json",root/"frequency_audit.json",root/"frequency.tsv",root/"report_audit.json",root/"combgap.json",root/"combgap_matches.tsv"]
    if (root/"attribution_support.tsv").exists():
        outputs.append(root/"attribution_support.tsv")
    outputs+=list((root/"output/pdf").glob("*"))
    write_json(root/"complete.json",dict(status="complete",job=os.environ["SLURM_JOB_ID"],
        reclustered=False,attribution_rerun=False,native_discovery=audit.get("report_scope", {}).get("native_discovery", "reused exact original-enhancer fits"),
        retained=sum(len(g["rows"]) for g in audit["groups"]),top_n=TOP_N,
        files={str(p.relative_to(root)):digest(p) for p in outputs},visual_qa="pending"))
    event("simple_motif_report_complete")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",required=True,type=Path)
    parser.add_argument("--root",required=True,type=Path)
    args=parser.parse_args();main(args.project.resolve(),args.root.resolve())
