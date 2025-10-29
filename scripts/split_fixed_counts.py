#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
import sys
import tarfile
import zipfile
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
                out = dest / p.with_suffix('').name
                with gzip.open(p, 'rb') as fin, open(out, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
            elif name.endswith('.cnf'):
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
    # Find the header line
    hdr_idx = -1
    for i, ln in enumerate(lines):
        if CNF_HEADER_RE.match(ln):
            hdr_idx = i
            break
    if hdr_idx < 0:
        return None
    out: List[str] = []
    out.append('c normalized by split_fixed_counts.py')
    out.append(lines[hdr_idx].strip())
    for ln in lines[hdr_idx+1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith('c'):
            out.append(s)
        elif CNF_CLAUSE_RE.match(s):
            out.append(s)
    return '\n'.join(out) + '\n'


def classify_family(path: Path) -> str:
    """Infer family name from filename or parent directories."""
    candidates = [path.name.lower()] + [part.lower() for part in path.parts]
    for fam in (
        'uuf50-218', 'uf50-218', 'uuf100-430', 'uf100-430', 'flat50-115'
    ):
        if any(fam in s for s in candidates):
            return fam
    # generic fallbacks if patterns exist
    for s in candidates:
        m = re.search(r"u?uf\d+-\d+", s)
        if m:
            return m.group(0)
        m = re.search(r"flat\d+-\d+", s)
        if m:
            return m.group(0)
    return 'misc'


def fixed_split(files: List[Path], counts: Tuple[int, int, int]) -> Tuple[List[Path], List[Path], List[Path]]:
    files = sorted(files)
    n_train, n_val, n_test = counts
    train = files[:n_train]
    val = files[n_train:n_train+n_val]
    test = files[n_train+n_val:n_train+n_val+n_test]
    return train, val, test


def write_lists(base_out: Path, train: List[Path], val: List[Path], test: List[Path]) -> None:
    base_out.mkdir(parents=True, exist_ok=True)
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(Path.cwd()))
        except Exception:
            return str(p)
    (base_out / 'train.lst').write_text('\n'.join(rel(p) for p in train) + '\n')
    (base_out / 'val.lst').write_text('\n'.join(rel(p) for p in val) + '\n')
    (base_out / 'test.lst').write_text('\n'.join(rel(p) for p in test) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description='Split SAT instances with fixed per-family counts and normalize CNFs')
    ap.add_argument('--src', default='SAT_Instances', help='folder containing archives or CNF files')
    ap.add_argument('--work', default='SAT_Instances/extracted', help='working dir for extraction')
    ap.add_argument('--out', default='data/instances', help='output base dir')
    args = ap.parse_args()

    src = Path(args.src)
    work = Path(args.work)
    out = Path(args.out)
    if not src.exists():
        print(f"ERROR: missing src folder: {src}")
        sys.exit(1)

    print(f"[1/4] Extracting archives from {src} -> {work}")
    extract_archives(src, work)

    print(f"[2/4] Collecting and normalizing CNFs")
    cnfs = find_cnf_files(work)
    fam_map: Dict[str, List[Path]] = {}
    norm_root = work / 'normalized'
    norm_root.mkdir(parents=True, exist_ok=True)
    for p in sorted(cnfs):
        raw = read_text_any(p)
        if raw is None:
            print(f"WARN: cannot read {p}")
            continue
        norm = normalize_cnf_text(raw)
        if norm is None:
            print(f"WARN: skipping {p} (missing DIMACS header)")
            continue
        fam = classify_family(p)
        fam_dir = norm_root / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        outp = fam_dir / p.name
        outp.write_text(norm)
        fam_map.setdefault(fam, []).append(outp)

    print(f"[3/4] Fixed per-family splits")
    # Desired counts per family
    desired: Dict[str, Tuple[int,int,int]] = {
        'uf50-218': (600, 100, 100),
        'uuf50-218': (600, 100, 100),
        'uf100-430': (600, 100, 100),
        'uuf100-430': (600, 100, 100),
        'flat50-115': (200, 25, 25),
    }

    # Aggregate for .lst writing
    all_train: List[Path] = []
    all_val: List[Path] = []
    all_test: List[Path] = []

    for fam, counts in desired.items():
        files = sorted(fam_map.get(fam, []))
        if not files:
            print(f"WARN: no files found for family {fam}")
            continue
        n_avail = len(files)
        n_req = sum(counts)
        if n_avail < n_req:
            print(f"WARN: {fam}: only {n_avail} files available, requested {n_req}; using available range")
        tr, va, te = fixed_split(files, counts)
        # Copy to output structure
        for split_name, lst in [('train', tr), ('val', va), ('test', te)]:
            dest_dir = out / split_name / fam
            dest_dir.mkdir(parents=True, exist_ok=True)
            for p in lst:
                shutil.copy2(p, dest_dir / p.name)
        print(f"{fam}: {len(files)} avail -> train={len(tr)} val={len(va)} test={len(te)}")
        all_train.extend([out / 'train' / fam / p.name for p in tr])
        all_val.extend([out / 'val' / fam / p.name for p in va])
        all_test.extend([out / 'test' / fam / p.name for p in te])

    print(f"[4/4] Writing split lists (train.lst/val.lst/test.lst)")
    write_lists(out, all_train, all_val, all_test)
    print(f"Done. Output in {out}/<train|val|test>/<family>/ and {out}/*.lst")


if __name__ == '__main__':
    main()
