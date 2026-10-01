#!/usr/bin/env python3
"""Export only plotted Figure 3 values and logo style; no checkpoints/DNA tensors."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    source = json.loads(args.source.read_text())
    audit = json.loads(args.audit.read_text())
    if not all(k in source for k in ("panel_b", "panel_c", "panel_d", "panel_e", "pdf_sha256")):
        raise ValueError("Expected latest b-e figure source, not an earlier b-d source")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {"source.json": source, "style.json": dict(groups=[], colors=audit["colors"], glyphs=audit["glyphs"])}
    hashes = {}
    for name, values in files.items():
        path = args.output / name
        path.write_text(json.dumps(values, indent=2, allow_nan=False)+"\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(status="complete", files=hashes, historical_pdf_sha256=source["pdf_sha256"],
        source_file_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        audit_file_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        role="Render-only historical figure replay. These values are not outputs of a newly trained model.")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__": main()
