"""Native-length signed TF-MoDISco; filter representative PWMs, never recluster."""
from functools import partial
import json
from pathlib import Path
import subprocess

import numpy as np

from classifier_modisco.original_intervals import GROUPS, discover_original
from classifier_modisco.dual_motif_pipeline import informative_core, core_rule
from classifier_modisco.simple_report import filter_native, rank_rows
from classifier_modisco.tomtom_atlas import meme_queries, read_meme, parse_matches
from .common import digest, write_json


def groups(cfg):
    grouping = cfg["discovery"]["grouping"]
    if grouping == "nonoverlap":
        return [("specific", 1, 1), ("intermediate", 2, 5), ("high", 6, 8)]
    if grouping == "cumulative":
        return GROUPS
    raise ValueError("grouping must be nonoverlap or cumulative")


def target_map(values, name):
    from classifier_transfer.data import CONTEXTS
    targets = tuple('calibrated_probability_'+c for c in CONTEXTS) + ('mean_active_logit',)
    names = list(values["targets"])
    if tuple(names) != targets:
        raise ValueError("Attribution target order changed")
    if name == "calibrated_breadth":
        return values["hypothetical"][:, :8].sum(1), values["breadth_quality_pass"] & values["quality_pass"][:, :8].all(1)
    if name not in names:
        raise ValueError("Unknown attribution target")
    index = names.index(name)
    return values["hypothetical"][:, index], values["quality_pass"][:, index]


def run(cfg, task):
    definitions = groups(cfg)
    if not 0 <= task < len(definitions):
        raise ValueError("Invalid discovery task")
    name, low, high = definitions[task]
    work = Path(cfg["paths"]["work"])
    root = work / "attribution"
    signature = json.loads((root / "manifest.json").read_text())["signature"]
    files = {}
    for shard in range(cfg["attribution"]["shards"]):
        receipt = json.loads((root / f"shard_{shard}.json").read_text())
        if receipt["status"] != "complete" or receipt["signature"] != signature or set(files) & set(receipt["chunks"]):
            raise ValueError("Incomplete, incompatible or duplicate attribution shards")
        files.update(receipt["chunks"])
    with np.load(work / "catalog/cohort.npz", allow_pickle=False) as f:
        cohort = dict(f)
    with np.load(work / "catalog/intervals.npz", allow_pickle=False) as f:
        intervals = dict(f)
    selected, tracks, sequences, lengths = [], [], [], []
    seen = np.zeros(len(cohort["ids"]), bool)
    target = cfg["discovery"]["target"]
    eligible_n = 0
    for filename in sorted(files):
        path = root / "chunks" / filename
        if digest(path) != files[filename]:
            raise ValueError("Attribution chunk changed")
        with np.load(path, allow_pickle=False) as f:
            idx = f["indices"]
            if (seen[idx].any() or str(f["signature"]) != signature
                    or not np.array_equal(f["ids"], cohort["ids"][idx])):
                raise ValueError("Duplicate/misaligned attribution chunk")
            seen[idx] = True
            hyp, quality = target_map(f, target)
            for row, i in enumerate(idx):
                breadth = int(cohort["labels"][i].sum())
                if cohort["split"][i] != "train" or not low <= breadth <= high:
                    continue
                eligible_n += 1
                if not quality[row]:
                    continue
                start, length = int(intervals["offset"][i]), int(intervals["length"][i])
                selected.append(int(i)); lengths.append(length)
                sequences.append(np.eye(4, dtype=np.float32)[cohort["sequence"][i, start:start+length]])
                tracks.append(hyp[row, :, start:start+length].T)
    if not seen.all() or not selected:
        raise ValueError("Missing attributions or empty discovery group")
    width = max(lengths)
    seq = np.zeros((len(selected), width, 4), np.float32)
    hyp = np.zeros_like(seq)
    for row, length in enumerate(lengths):
        seq[row, :length] = sequences[row]; hyp[row, :length] = tracks[row]
    out = work / "motifs" / target / name
    out.mkdir(parents=True, exist_ok=False)
    parameters = {k: cfg["discovery"][k] for k in ("seed", "sliding_window_size", "flank_size",
        "trim_to_window_size", "initial_flank_to_add", "final_flank_to_add", "n_leiden_runs", "target_seqlet_fdr")}
    parameters["seed"] += task
    # The seqlet cap must not preferentially sample early catalog chromosomes.
    order = np.random.default_rng(parameters["seed"]).permutation(len(selected))
    seq, hyp = seq[order], hyp[order]
    selected = np.asarray(selected)[order].tolist()
    lengths = np.asarray(lengths)[order]
    audit = discover_original(seq, hyp, np.asarray(lengths), parameters, out / "motifs.h5",
                              cfg["discovery"]["max_seqlets_per_sign"])
    counts = {sign: audit[sign]["patterns"] for sign in ("positive", "negative")}
    flank = cfg["discovery"]["flank_threshold"]
    rows, excluded = filter_native(out / "motifs.h5", name, counts,
                                   core_filter=partial(informative_core, flank_threshold=flank))
    group = dict(name=name, minimum_breadth=low, maximum_breadth=high,
                 discovery_elements=len(selected), rank_by="native_support", rows=rows)
    rank_rows(group)
    report = dict(groups=[group], exclusions=excluded, rule=core_rule(flank),
        target=target, attribution_signature=signature, selection=selected, eligible_train_enhancers=eligible_n,
        quality_excluded=eligible_n-len(selected), bar_semantics="Distinct discovery enhancers assigned a seqlet / discovery N; not sequence-scan prevalence",
        reclustered=False, attribution_input_bp=2048, discovery_interval="native enhancer")
    write_json(out / "report.json", report)
    (out / "queries.meme").write_text(meme_queries(report))
    expected = {r["id"].replace("/", "__") for r in rows}
    base = Path(cfg["config_path"]).parent / cfg.get("project_root", "..")
    for db, filename in cfg.get("motif_databases", {}).items():
        if not db or Path(db).name != db:
            raise ValueError("Database names must be simple filenames")
        path = (base / filename).resolve()
        targets = read_meme(path)
        if not rows:
            write_json(out / (db + ".matches.json"), dict(best={}, reason="No informative motifs"))
            continue
        version = subprocess.check_output(["tomtom", "-version"], text=True).strip()
        if version != "5.5.9":
            raise ValueError("Tomtom 5.5.9 required")
        command = ["tomtom", "-text", "-dist", "pearson", "-min-overlap", "5", "-motif-pseudo", ".1",
                   "-thresh", "1", "-verbosity", "2", str(out / "queries.meme"), str(path)]
        with (out / (db + ".tomtom.tsv")).open("x") as stream:
            subprocess.run(command, stdout=stream, check=True)
        best, counts = parse_matches((out / (db + ".tomtom.tsv")).read_text(), expected, targets)
        write_json(out / (db + ".matches.json"), dict(best=best, returned_matches_per_query=counts,
            database_sha256=digest(path), version=version, command=command))
    html_report(out, report, cfg.get("motif_databases", {}), base)
    write_json(out / "complete.json", dict(status="complete", files={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}))


def html_report(out, report, databases, base):
    """Filtered native patterns, both signs, with clearly labelled seqlet support."""
    from html import escape
    from classifier_modisco.report_breadth import sequence_logo
    annotations = {}
    for name, filename in databases.items():
        annotations[name] = (json.loads((out / (name+".matches.json")).read_text())["best"],
                             read_meme((base / filename).resolve()))
    group = report["groups"][0]
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Native enhancer motifs</title>',
        '<style>body{font:15px sans-serif;max-width:1200px;margin:35px auto}table{width:100%;border-collapse:collapse}'
        'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}svg{max-width:300px;width:100%}'
        '.positive{color:#2466c4}.negative{color:#ce3646}</style>',
        '<h1>'+escape(group["name"])+': '+escape(report["target"])+'</h1>',
        '<p>'+escape(report["bar_semantics"])+'. No post-discovery reclustering. All passing motifs are shown, sorted within sign.</p>',
        '<table><tr><th>Sign / rank</th><th>Native motif</th><th>Enhancer support</th><th>Database matches (q)</th></tr>']
    for row in sorted(group["rows"], key=lambda r: (r["sign"] != "positive", r["rank"])):
        support = row["attribution_support"]
        hits = []
        for name, (matches, motifs) in annotations.items():
            match = matches.get(row["id"].replace("/", "__"))
            if match is not None:
                motif = motifs[match["target_id"]]
                hits.append(escape(name+": "+motif["name"])+f' (q={match["q"]:.3g})'+
                            sequence_logo(np.asarray(motif["pwm"]), motif["name"]))
        parts.append(f'<tr><td class="{row["sign"]}">{row["sign"]} / {row["rank"]}</td><td>'+
            sequence_logo(np.asarray(row["trimmed_pwm"]), row["id"])+
            f'</td><td>{support["hits"]}/{support["n"]} ({100*support["fraction"]:.1f}%)</td>'+
            '<td>'+('<br>'.join(hits) or 'No database supplied')+'</td></tr>')
    parts.append('</table></html>')
    (out / "report.html").write_text("\n".join(parts))
