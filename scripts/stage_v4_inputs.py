"""Verify and stage the frozen v4 model inputs without modifying exported files."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import tarfile


ARCHIVE_SHA256 = "0202c8a4fdb1105da658e7895dc2916670f634ec1b4aecb4cd942914e9e19052"
ARCHIVE_ROOT = "drosophila_ccre_regulatory_analysis_bundle_v4"
CONTEXTS = ("ab", "e13", "e5", "ead", "hid", "lb", "o", "wid")
H3_ALIASES = {
    **{f"{c}_h3k27ac_rep{r}": f"{c}_h3k27ac_rep{r}"
       for c, reps in (("ab", 2), ("e13", 2), ("e5", 2), ("ead", 1), ("hid", 2), ("lb", 2))
       for r in range(1, reps + 1)},
    "o_h3k27ac_rep1": "o_h3k27ac_cutrun_rep1",
    "wid_h3k27ac_rep1": "wid_h3k27ac_gfp2",
    "wid_h3k27ac_rep2": "wid_h3k27ac_gfp3",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_compute_node():
    if socket.gethostname().split(".")[0] == "neocranex" or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("V4 processing requires a Slurm compute-node allocation")


def parse_manifest(text):
    result = {}
    for line in text.splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.removeprefix("./")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name or name in result:
            raise ValueError(f"Unsafe or duplicate manifest path: {name}")
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid SHA256 in archive manifest")
        result[name] = checksum
    return result


def required_files():
    files = {
        "README.md", "reference/dm6.fa", "reference/dm6.fa.fai",
        "reference/dm6.chrom.sizes", "reference/dm6.blacklist.bed",
        "manifests/samples.tsv", "manifests/atlas-atac.library-review.tsv",
        "manifests/atlas-h3k27ac.library-review.tsv", "provenance/atlas/selection.json",
        "regulatory/quantification/normalization_factors.tsv", "regulatory/signal_tracks.tsv",
    }
    files.update(f"regulatory/master_dhs/{name}" for name in (
        "master_dhs.bed", "master_dhs_summits.bed", "master_dhs.json",
        "master_dhs_context_matrix.tsv", "master_dhs_membership.tsv"))
    files.update(f"regulatory/h3k27ac/peaks/{lib}_peaks.broadPeak" for lib in H3_ALIASES.values())
    files.update(f"regulatory/normalized_mean_bigwig/{c}.{assay}.mean.background_tmm.{ext}"
                 for c in CONTEXTS for assay in ("atac", "h3k27ac") for ext in ("bw", "json"))
    return sorted(files)


def copy_verified_member(tar, name, target, checksum):
    member = tar.getmember(f"{ARCHIVE_ROOT}/{name}")
    if not member.isfile():
        raise ValueError(f"Expected a regular archive member: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tar.extractfile(member) as source, target.open("xb") as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != checksum:
        raise ValueError(f"Archive payload checksum mismatch: {name}")


def stage_archive(archive, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing existing v4 destination: {destination}")
    print("Verifying complete v4 archive SHA256", flush=True)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("V4 archive checksum mismatch")
    with tarfile.open(archive, "r:") as tar:
        names = [m.name for m in tar.getmembers()]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive member names")
        manifest_text = tar.extractfile(f"{ARCHIVE_ROOT}/FILE_MANIFEST.sha256").read().decode()
        manifest = parse_manifest(manifest_text)
        files = required_files()
        for name in files:
            if name not in manifest or not tar.getmember(f"{ARCHIVE_ROOT}/{name}").isfile():
                raise ValueError(f"Missing regular checksummed input: {name}")
        destination.mkdir(parents=True, exist_ok=False)
        for i, name in enumerate(files, 1):
            copy_verified_member(tar, name, destination / name, manifest[name])
            print(f"Staged {i}/{len(files)}: {name}", flush=True)
    selection = json.loads((destination / "provenance/atlas/selection.json").read_text())
    master = json.loads((destination / "regulatory/master_dhs/master_dhs.json").read_text())
    if tuple(selection["contexts"]) != CONTEXTS or tuple(master["contexts"]) != CONTEXTS or master["master_dhs_count"] != 81035:
        raise ValueError("Unexpected v4 master/context registry")
    with (destination / "manifests/samples.tsv").open() as handle:
        samples = list(csv.DictReader(handle, delimiter="\t"))
    if len(samples) != 35 or len({x["library_id"] for x in samples}) != 34 or len({x["accession"] for x in samples}) != 35:
        raise ValueError("Unexpected v4 accession/library counts")
    if {x["context"] for x in samples} != set(CONTEXTS):
        raise ValueError("Unexpected sample contexts")
    if set(selection["h3k27ac_libraries"]) != set(H3_ALIASES.values()):
        raise ValueError("H3K27ac aliases do not cover exactly the selected libraries")
    aliases = destination / "model_h3k27ac_peaks"
    aliases.mkdir()
    mapping = []
    for alias, library in sorted(H3_ALIASES.items()):
        source = destination / f"regulatory/h3k27ac/peaks/{library}_peaks.broadPeak"
        target = aliases / f"{alias}_peaks.broadPeak"
        shutil.copyfile(source, target)
        mapping.append(dict(model_alias=alias, source_library=library,
                            protocol="CUT&RUN" if "cutrun" in library else "ChIP-seq",
                            source=str(source.relative_to(destination)),
                            target=str(target.relative_to(destination)), sha256=sha256(target)))
    (destination / "h3k27ac_aliases.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (destination / "staging.json").write_text(json.dumps(dict(
        archive=str(archive), archive_sha256=ARCHIVE_SHA256,
        archive_manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),
        inputs={name: manifest[name] for name in files}, h3k27ac_aliases=mapping,
        master_contexts=CONTEXTS, master_elements=81035,
        normalization="background-TMM; no renormalization; v4 library-CPM tracks excluded",
        caveat="Ovary is CUT&RUN; generic source JSON unit_semantics does not change the protocol"
    ), indent=2) + "\n")
    print("V4 staging verified; all 14 peak libraries mapped without changing source files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    require_compute_node()
    stage_archive(args.archive, args.destination)


if __name__ == "__main__":
    main()
