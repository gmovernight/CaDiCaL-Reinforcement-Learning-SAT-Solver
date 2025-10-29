#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
import sys
import tarfile
import zipfile
import shutil
import gzip
from pathlib import Path
from typing import List, Tuple, Optional


CNF_HEADER_RE = re.compile(r"^\s*p\s+cnf\s+(\d+)\s+(\d+)\s*$", re.IGNORECASE)
CNF_CLAUSE_RE = re.compile(r"^\s*(?:-?\d+\s+)+0\s*$")


def extract_archives(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_dir():
            continue
        name = p.name.lower()
        try:
            if name.endswith('.zip'):
                with zipfile.ZipFile(p, 'r') as zf:
                    zf.extractall(dest)
            elif name.endswith(('.tar.gz', '.tgz', '.tar.xz', '.tar.bz2', '.tar')):
                with tarfile.open(p, 'r:*') as tf:
                    tf.extractall(dest)
            elif name.endswith('.cnf.gz'):
                # Expand single compressed CNF keeping filename
                out = dest / p.with_suffix('').name
                with gzip.open(p, 'rb') as fin, open(out, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
            else:
                # Copy other files as-is (might already be .cnf)
                if name.endswith('.cnf'):
                    shutil.copy2(p, dest / p.name)
        except Exception as e:
            print(f"WARN: failed to extract {p}: {e}")


def find_cnf_files(root: Path) -> List[Path]:
    return [p for p in root.rglob('*.cnf') if p.is_file()]


def read_text_any(path: Path) -> Optional[str]:
    for enc in ('utf-8', 'latin-1'):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return path.read_text(errors='ignore')
    except Exception:
        return None


def normalize_cnf_text(raw: str) -> Optional[str]:
    lines = raw.splitlines()
    # Find header line index
    hdr_idx = -1
    vars_clauses: Tuple[int, int] | None = None
    for i, ln in enumerate(lines):
        m = CNF_HEADER_RE.match(ln)
        if m:
            hdr_idx = i
            vars_clauses = (int(m.group(1)), int(m.group(2)))
            break
    if hdr_idx < 0:
        return None
    out: List[str] = []
    out.append('c normalized by prepare_sat_instances.py')
    out.append(lines[hdr_idx].strip())
    # Append only comment lines and well-formed clause lines after header
    for ln in lines[hdr_idx+1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith('c'):
            out.append(s)
        elif CNF_CLAUSE_RE.match(s):
            out.append(s)
        # ignore everything else
    return '\n'.join(out) + '\n'


def classify_family(path: Path) -> str:
    name = path.name.lower()
    # Try to find family tokens in the name
    for fam in (
        'uf50-218', 'uuf50-218', 'uf75-325', 'uuf75-325', 'uf100-430', 'uuf100-430',
        'uf125-538', 'uuf125-538', 'uf150-645', 'uuf150-645', 'flat50-115', 'flat75-180', 'flat100-239'
    ):
        if fam in name:
            return fam
    # Fallback to uf/uuf with numbers
    m = re.search(r"u?uf\d+-\d+", name)
    if m:
        return m.group(0)
    if 'flat' in name:
        m = re.search(r"flat\d+-\d+", name)
        if m:
            return m.group(0)
    return 'misc'


def split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    train = idx[:t]
    val = idx[t:t+v]
    test = idx[t+v:]
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare and split SAT instances')
    ap.add_argument('--src', default='SAT_Instances', help='source folder with archives')
    ap.add_argument('--work', default='SAT_Instances/extracted', help='working extraction dir')
    ap.add_argument('--out', default='data/instances', help='output base dir for splits')
    ap.add_argument('--train', type=float, default=0.8)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.1)
    args = ap.parse_args()

    src = Path(args.src)
    work = Path(args.work)
    out = Path(args.out)
    if not src.exists():
        print(f"ERROR: missing src folder: {src}")
        sys.exit(1)

    print(f"Extracting archives from {src} -> {work}")
    extract_archives(src, work)

    print("Scanning for CNF files...")
    cnfs = find_cnf_files(work)
    print(f"Found {len(cnfs)} .cnf files")

    # Normalize and group by family
    families: dict[str, List[Path]] = {}
    normalized_root = work / 'normalized'
    normalized_root.mkdir(parents=True, exist_ok=True)

    for p in sorted(cnfs):
        raw = read_text_any(p)
        if raw is None:
            print(f"WARN: failed to read {p}")
            continue
        norm = normalize_cnf_text(raw)
        if norm is None:
            print(f"WARN: skipping {p} (no p cnf header found)")
            continue
        fam = classify_family(p)
        fam_dir = normalized_root / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        out_path = fam_dir / p.name
        out_path.write_text(norm)
        families.setdefault(fam, []).append(out_path)

    # Split per family
    print("Splitting per family...")
    for fam, files in families.items():
        files = sorted(files)
        n = len(files)
        if n == 0:
            continue
        train_idx, val_idx, test_idx = split_indices(n, args.train, args.val)
        splits = [('train', train_idx), ('val', val_idx), ('test', test_idx)]
        for split_name, idxs in splits:
            dest_dir = out / split_name / fam
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i in idxs:
                src_file = files[i]
                shutil.copy2(src_file, dest_dir / src_file.name)
        print(f"family {fam}: {n} -> train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    print(f"Done. Output in {out}/<train|val|test>/<family>/")


if __name__ == '__main__':
    main()

