"""Prepare saved central-512 native motifs for the unchanged simple report."""
import json
import shutil

import numpy as np

from .common import digest, event, write_json
from .original_intervals import GROUPS
from .simple_report import filter_native, SITE_P, TOP_N


SCOPE = dict(
    title="Central-512-bp TF-MoDISco motifs",
    filename="central512_native_top5",
    scan_region="central 512-bp windows centered on the saved DHS summit",
    intro="The completed native TF-MoDISco fits used only the central 512-bp attribution tensors (input positions 768:1280). The same frozen CNN epoch38 saw its full 2048-bp genomic input. Its target was the mean observed-active-context forward/RC logit. These older attributions used TWO shuffled references, whereas the enhancer-length report used FIFTY: this is not a window-only controlled comparison. Native motifs are reused directly; no new attribution, clustering or merging is performed. The 512-bp window can include enhancer flanks.",
    missing="A missing negative panel means no native pattern survives the reporting and scan filters, not absence of negative regulation.",
    query_rule="Information-trimmed raw central512 TF-MoDISco motifs; 5-30bp; no extra clustering",
    native_discovery="reused exact central512 fits; 2 shuffled references; not 50-reference matched")

SUPPORT_SCOPE = dict(SCOPE,
    title="Central-512 motifs: attribution-supported prevalence",
    filename="central512_support_top5", rank_by="native_support",
    query_rule="Native central512 motifs trimmed to five-column runs >0.5 bits; 5-30bp; no extra clustering",
    trim_description="Exploratory stricter trimming: retain the span from the first to the last run of five consecutive columns with information >0.5 bits (uniform background). Apply identically to every motif, including CA and GA repeats. Keep cores 5-30 bp, mean information >=0.5 bits/base, total >=5 bits, and concordant contribution sign. Preserve internal gaps. Display, FIMO and Tomtom use exactly the same trimmed core; raw PWMs are unchanged.",
    rank_description="Rank separately by group/sign using distinct enhancers with a seqlet in the ORIGINAL native cluster divided by the number of discovery enhancers. Count each enhancer once per cluster. Membership is not reassigned after trimming: this measures original cluster support, not attribution at every occurrence of the trimmed PWM. Groups and clusters may overlap. Primary bars use a shared 0-100% axis; this is not enrichment or mean attribution strength.",
    missing_description="Show fewer than five when too few native patterns pass the core filters; missing negative motifs do not imply absence of negative regulation. Secondary sequence percentages scan BOTH strands of central 512-bp windows using FIMO5.5.9, site p<=1e-4 and the fixed train-only background: distinct hit-containing enhancers / ALL enhancers in that group and split. Train and validation are separate; test counts are saved. An unattainable cutoff is N/A, not absence, and does not exclude a support-ranked motif. Site p is not enhancer-level FDR.")


def central_codes(data):
    sequence = np.asarray(data["sequence"])
    n = len(data["ids"])
    if sequence.shape != (n, 2048) or sequence.dtype.kind not in "iu":
        raise ValueError("Expected aligned 2048-bp integer-coded native inputs")
    codes = sequence[:, 768:1280]
    if not np.isin(codes, range(4)).all():
        raise ValueError("Non-ACGT central sequence")
    return codes


def prepare(project, root):
    from .report_breadth import COLORS, GLYPHS
    config = json.loads((root/"config.json").read_text())
    mode = config.get("report_mode", "sequence_frequency")
    if mode not in ("sequence_frequency", "native_support_strict"):
        raise ValueError("Unknown central512 report mode")
    support = mode == "native_support_strict"
    for name, expected in config["inputs"].items():
        if digest(project/name) != expected:
            raise ValueError("Changed frozen input: "+name)
    source, parent = project/config["source"], project/config["parent"]
    previous = project/config["previous_report"]
    attribution = json.loads((parent/"attribution_config.json").read_text())
    if attribution["references"] != 2 or attribution["checkpoint_epoch"] != 38:
        raise ValueError("Saved central512 attribution contract changed")
    with np.load(parent/"cohort.npz", allow_pickle=False) as f:
        data = dict(f)
    codes = central_codes(data)
    labels = data["labels"]
    if labels.shape != (len(codes), 8) or not np.isin(labels, [0, 1]).all() or not labels.any(1).all():
        raise ValueError("Invalid observed context labels")
    if len(np.unique(data["ids"])) != len(codes):
        raise ValueError("Duplicate enhancer IDs")
    prior = json.loads((previous/"motif_audit.json").read_text())
    for name in ("cohort_metadata.npz", "training_background.txt"):
        if digest(previous/name) != prior["prepared_files"][name]:
            raise ValueError("Changed common sequence-scan input: "+name)
    with np.load(previous/"cohort_metadata.npz", allow_pickle=False) as f:
        if any(not np.array_equal(f[k], data[k]) for k in ("ids", "split", "chrom")) or not np.array_equal(f["breadth"], labels.sum(1)):
            raise ValueError("Central/original cohort ordering or labels differ")
    metadata = dict(ids=data["ids"], breadth=labels.sum(1), split=data["split"],
        chrom=data["chrom"], start=data["summit"].astype(np.int64)-256,
        end=data["summit"].astype(np.int64)+256, length=np.full(len(codes), 512))
    groups, exclusions = [], []
    for name, low, high in GROUPS:
        directory = source/"groups"/name
        selection = json.loads((directory/"selection.json").read_text())
        complete = json.loads((directory/"complete.json").read_text())
        definition = dict(name=name, minimum_breadth=low, maximum_breadth=high)
        if (selection["group"] != definition or selection["window"] != 512
                or not selection["train_only"] or not selection["no_enhancer_downsampling"]
                or complete["status"] != "complete" or complete["group"] != definition):
            raise ValueError("Not the completed train-only central512 discovery")
        if complete["parent_attribution_sha256"] != digest(parent/"attribution_complete.json"):
            raise ValueError("Wrong parent attribution completion")
        with np.load(directory/"discovery_examples.npz", allow_pickle=False) as saved:
            indices = saved["indices"]
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                    or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(codes)).any()):
                raise ValueError("Invalid discovery example indices")
            for key in ("ids", "labels", "chrom", "summit"):
                if not np.array_equal(saved[key], data[key][indices]):
                    raise ValueError("Discovery/cohort mismatch: "+key)
        eligible = ((metadata["split"] == "train") & (metadata["breadth"] >= low) & (metadata["breadth"] <= high))
        if (not eligible[indices].all() or len(indices) != selection["elements"]
                or len(indices) != complete["discovery_elements"]
                or int(eligible.sum()) != selection["eligible_before_qc"]
                or int(eligible.sum())-len(indices) != selection["quality_excluded"]):
            raise ValueError("Training/breadth/quality cohort disagreement")
        rows, excluded = filter_native(directory/"motifs.h5", name, complete["patterns"], threshold=.5 if support else .2)
        exclusions.extend(excluded)
        groups.append(dict(**definition, n=int(eligible.sum()), rows=rows,
            native_discovery=complete, discovery_elements=len(indices), quality_excluded=selection["quality_excluded"]))
        if support:
            groups[-1]["rank_by"] = "native_support"
    with (root/"enhancers.fa").open("x") as handle:
        for i, sequence in enumerate(codes):
            handle.write(f">e{i}\n"+"".join("ACGT"[b] for b in sequence)+"\n")
    np.savez_compressed(root/"cohort_metadata.npz", **metadata)
    shutil.copy2(previous/"training_background.txt", root/"training_background.txt")
    audit = dict(groups=groups, exclusions=exclusions, colors=COLORS, glyphs=GLYPHS,
        inputs=config["inputs"], background=prior["background"], top_n=TOP_N, report_scope=SUPPORT_SCOPE if support else SCOPE,
        prepared_files={n:digest(root/n) for n in ("enhancers.fa", "cohort_metadata.npz", "training_background.txt")},
        rules=dict(source="raw central512 groups/*/motifs.h5", attribution_references=2,
            comparability="enhancer-length report used 50 references; not a window-only comparison",
            model_input_bp=2048, attribution_and_scan_input_slice=[768,1280],
            reclustering=False, posthoc_merging=False,
            trim="first through last run of 5 consecutive columns >0.2 bits; retain internal positions",
            filter="core 5-30bp; mean information >=0.5 bits; total information >=5 bits; sign concordant",
            rank="training enhancer sequence-scan frequency; separate per group/sign; ID breaks ties",
            frequency="unique enhancers with >=1 central512 FIMO hit / all enhancers in group and split",
            site_p=SITE_P, background="same fixed training enhancer-derived RC-symmetric zero-order background as original-interval report"))
    if support:
        audit["rules"].update(
            trim="first through last run of 5 consecutive columns >0.5 bits; preserve internal gaps; uniform IC background",
            rank="distinct native cluster-supporting enhancers / discovery enhancers; separately per group/sign; ID breaks ties",
            support="original seqlet cluster memberships, unchanged after trimming; not attribution at rescanned sites",
            selected="top5 informative motifs by native support, including scan-unquantifiable cores; scan prevalence shown separately")
    (root/"scans").mkdir()
    write_json(root/"cores.json", audit)
    event("central512_cores_prepared", retained=sum(len(g["rows"]) for g in groups), excluded=len(exclusions),
        references=2, model_input_bp=2048, scan_bp=512,
        counts={g["name"]:{s:sum(r["sign"]==s for r in g["rows"]) for s in ("positive","negative")} for g in groups})
    return audit, metadata, root
