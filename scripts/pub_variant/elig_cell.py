
# Runs before the pipeline (the stage functions are defined in the next cell and called in cell 12):
#  1. writes null_engine.py next to the pipeline modules (SHA-256 pinned). The patched frozen cells 39 and 45 import it
#     and give every reader of Engine 2's lists (ENG, eng_lists.json) Engine 1's own lists in Engine 2's schema.
#  2. records the mounts: every dataset of kernel-metadata.json present, none of the detached ones.
#  3. commit run only: hashes every mounted file of those datasets (and the competition files) that has a row in the
#     embedded copy of casmi26-pubm's v43_mounted_input_hashes.json, with frozen cell 54's digest (full SHA-256 up to
#     32 MiB, else size + SHA-256 of the first and last 4 MiB), matched by its path inside the dataset (the v4g
#     container held each upstream dataset in a folder of the same name), else by file name + size. Prints MATCH /
#     MISMATCH counts and the mismatching names; log only, it never stops the run. MATCH of a file over 32 MiB means
#     size + first and last 4 MiB only (the v43 rows hold no more), so:
#  4. commit run only: the larger files of the datasets that replace container folders (ELIG_FULL_DATASETS; v4b is
#     checked in full by frozen cell 26) are hashed in full and compared with the container MANIFEST's sha256
#     (ELIG_FULL_REFS); the others (fpnet_full1.pt, glacier.pt, the metric wheels) are recorded in elig0_report.json.
def _elig_write_null_engine():
    path = CODE / 'null_engine.py'
    path.write_text(NULL_ENGINE_SRC, encoding='utf-8')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == NULL_ENGINE_SHA256, 'null_engine.py differs from its pin'
    return path


def _elig_root(ref):
    owner, slug = ref.split('/', 1)
    for p in (INPUT / 'datasets' / owner / slug, INPUT / slug):
        if p.is_dir():
            return p
    return None


def _elig_detached_present():
    found = []
    for ref in ELIG_DETACHED:
        slug = ref.split('/', 1)[1]
        found += [str(p) for p in [INPUT / slug] + sorted((INPUT / 'datasets').glob(f'*/{slug}')) if p.is_dir()]
    return found


def _elig_digest(path, limit=32 << 20, edge=4 << 20):
    """Frozen cell 54's quick_digest: (bytes, digest, full)."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open('rb') as source:
        if size <= limit:
            for block in iter(lambda: source.read(8 << 20), b''):
                h.update(block)
        else:
            h.update(source.read(edge))
            source.seek(size - edge)
            h.update(source.read(edge))
    return size, h.hexdigest(), size <= limit


def _elig_reference(rows):
    """v43 rows keyed by (dataset folder, path inside it) and by (file name, bytes)."""
    container = 'datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets/'
    by_rel, by_name = {}, {}
    for r in rows:
        p = r['path'].split('/kaggle/input/', 1)[-1]
        parts = p.split('/')
        if p.startswith(container) and len(parts) > 4:     # a folder of the container = one upstream dataset
            slug, rel = parts[3], '/'.join(parts[4:])
        elif parts[0] == 'datasets' and len(parts) > 3:
            slug, rel = parts[2], '/'.join(parts[3:])
        elif parts[0] == 'competitions' and len(parts) > 2:
            slug, rel = parts[1], '/'.join(parts[2:])
        else:
            slug, rel = '', p
        entry = dict(path=r['path'], bytes=int(r['bytes']), full=bool(r.get('full')),
                     digest=r['sha256'] if r.get('full') else r['sha256_head_tail'])
        by_rel[(slug, rel)] = entry
        by_name.setdefault((parts[-1], entry['bytes']), []).append(entry)
    return by_rel, by_name


def _elig_full_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def elig_hash_check(datasets, comp, rows, full_refs=None, full_datasets=(), limit=32 << 20, edge=4 << 20, show=40):
    """Compare the mounted files with the v43 rows (and the larger files of full_datasets in full with full_refs);
    returns the report (never raises on a mismatch)."""
    t0 = time.time()
    full_refs = full_refs or {}
    by_rel, by_name = _elig_reference(rows)
    report = dict(match=0, match_full=0, match_head_tail=0, mismatch=0, by_path=0, by_name_size=0, no_row=0,
                  missing_rows=0, bytes_read=0, datasets={}, mismatches=[], missing=[], no_row_examples=[],
                  full=dict(files=0, bytes=0, match=0, mismatch=0, recorded=0, seconds=0.0, mismatches=[], recorded_files=[],
                            references_unchecked=[]))
    full_seen = set()
    targets = [(ref, ref.split('/', 1)[1], root) for ref, root in datasets] + [('competition', comp.name, comp)]
    for ref, slug, root in targets:
        d = dict(root=str(root) if root else None, files=0, match=0, mismatch=0, no_row=0)
        report['datasets'][ref] = d
        if root is None:
            continue
        seen = set()
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            rel = path.relative_to(root).as_posix()
            d['files'] += 1
            if ref in full_datasets and path.stat().st_size > limit:     # 4. full SHA-256 of the larger files
                t1, name, size = time.time(), f'{ref}/{rel}', path.stat().st_size
                sha, fr = _elig_full_sha256(path), report['full']
                fr['files'] += 1; fr['bytes'] += size; fr['seconds'] += time.time() - t1; report['bytes_read'] += size
                full_seen.add(name)
                if name in full_refs:
                    ok_full = (size, sha) == tuple(full_refs[name])
                    fr['match' if ok_full else 'mismatch'] += 1
                    if not ok_full:
                        fr['mismatches'].append(dict(file=name, bytes=size, sha256=sha, reference_bytes=full_refs[name][0],
                                                     reference_sha256=full_refs[name][1]))
                else:
                    fr['recorded'] += 1
                    fr['recorded_files'].append(dict(file=name, bytes=size, sha256=sha))
            row, how = by_rel.get((slug, rel)), 'by_path'
            if row is None:
                same = by_name.get((path.name, path.stat().st_size)) or []
                if same and len({r['digest'] for r in same}) == 1:
                    row, how = same[0], 'by_name_size'
            if row is None:
                d['no_row'] += 1; report['no_row'] += 1
                if len(report['no_row_examples']) < show:
                    report['no_row_examples'].append(f'{ref}/{rel}')
                continue
            seen.add(row['path'])
            size, digest, full = _elig_digest(path, limit, edge) if path.stat().st_size == row['bytes'] else \
                (path.stat().st_size, None, None)
            report['bytes_read'] += (size if full else 2 * edge) if digest else 0
            ok = digest is not None and digest == row['digest'] and full == row['full']
            report[how] += 1
            key = 'match' if ok else 'mismatch'
            report[key] += 1; d[key] += 1
            if ok:
                report['match_full' if full else 'match_head_tail'] += 1
            if not ok:
                report['mismatches'].append(dict(file=f'{ref}/{rel}', reference=row['path'], bytes=size,
                                                 reference_bytes=row['bytes'], digest=digest, reference_digest=row['digest']))
        for (s, rel), row in by_rel.items():            # reference rows of this dataset that are not on disk
            if s == slug and row['path'] not in seen and not (root / rel).is_file():
                report['missing_rows'] += 1
                report['missing'].append(f'{ref}/{rel}')
    report['full']['references_unchecked'] = sorted(set(full_refs) - full_seen)
    report['full']['seconds'] = round(report['full']['seconds'], 1)
    report['seconds'] = round(time.time() - t0, 1)
    return report


def elig0_setup():
    t0 = time.time()
    path = _elig_write_null_engine()
    roots = [(ref, _elig_root(ref)) for ref in ELIG_DATASETS]
    absent = [ref for ref, root in roots if root is None]
    detached = _elig_detached_present()
    spec_w = globals().get('SPEC_W')
    rep = dict(elig=ELIG, null_engine=str(path), null_engine_sha256=NULL_ENGINE_SHA256, frozen=ELIG_FROZEN,
               frozen_sha256_in_use=FROZEN_SHA256, datasets={ref: str(root) if root else None for ref, root in roots},
               absent=absent, detached_present=detached, spec_w=spec_w, checked_versions=ELIG_CHECKED_VERSIONS)
    print(f"ELIG {ELIG} | F1 datasets mounted {len(roots) - len(absent)} / {len(roots)}"
          f"{' (ABSENT: ' + ', '.join(absent) + ')' if absent else ''} | detached ones present: "
          f"{', '.join(detached) if detached else 'none'} | F2a null Engine 2 (null_engine.py {NULL_ENGINE_SHA256[:12]}) | "
          f"F3 SRC_BITS 0 (frozen cell 26), SPEC_W {spec_w} | F4 external reference spectra off (frozen cell 36) | "
          f"frozen source {FROZEN_SHA256[:12]} (patched from {ELIG_FROZEN['original_sha256'][:12]}; cells "
          f"{ELIG_FROZEN['patched_cells']})", flush=True)
    if spec_w != 0 or FROZEN_SHA256 != ELIG_FROZEN['patched_sha256'] or detached:
        print('ELIG WARNING: this run is not the elig0 configuration (SPEC_W, frozen pin or detached mounts)', flush=True)
    if IS_RERUN:
        print('ELIG HASH CHECK skipped in the scoring rerun (the commit run checks the same mounted versions)', flush=True)
    else:
        try:
            rows = json.loads(V43_HASHES_JSON)
            assert hashlib.sha256(V43_HASHES_JSON.encode()).hexdigest() == V43_HASHES_SHA256, 'embedded table differs'
            chk = elig_hash_check([(ref, root) for ref, root in roots if ref in ELIG_UPSTREAM], COMP, rows,
                                  full_refs=ELIG_FULL_REFS, full_datasets=ELIG_FULL_DATASETS)
            rep['hash_check'] = chk
            print(f"ELIG HASH CHECK | MATCH {chk['match']} (full SHA-256 {chk['match_full']}; over 32 MiB by size + first / "
                  f"last 4 MiB only {chk['match_head_tail']}) | MISMATCH {chk['mismatch']} | matched by path "
                  f"{chk['by_path']}, by name + size {chk['by_name_size']} | files without a v43 row {chk['no_row']} "
                  f"(not hashed) | v43 rows of these datasets missing on disk {chk['missing_rows']} | "
                  f"{chk['bytes_read'] / 1e9:.2f} GB read | {chk['seconds']} s", flush=True)
            fr = chk['full']
            print(f"ELIG FULL HASH | files over 32 MiB of {len(ELIG_FULL_DATASETS)} re-mounted datasets (v4b: frozen cell 26) "
                  f"| hashed {fr['files']} ({fr['bytes'] / 1e9:.2f} GB, {fr['seconds']} s) | vs container MANIFEST sha256: "
                  f"MATCH {fr['match']} | MISMATCH {fr['mismatch']} | references unchecked {len(fr['references_unchecked'])} "
                  f"| recorded without a reference {fr['recorded']} (elig0_report.json)", flush=True)
            for m in fr['mismatches']:
                print(f"  ELIG FULL MISMATCH {m['file']} | bytes {m['bytes']} vs {m['reference_bytes']} | "
                      f"{m['sha256'][:12]} vs {m['reference_sha256'][:12]}", flush=True)
            for name in fr['references_unchecked']:
                print(f'  ELIG FULL UNCHECKED {name} (not on disk, or not over 32 MiB)', flush=True)
            for m in fr['recorded_files']:
                print(f"  ELIG FULL SHA256 {m['file']} {m['bytes']} {m['sha256']}", flush=True)
            for ref, d in chk['datasets'].items():
                print(f"  ELIG HASH {ref}: files {d['files']} | match {d['match']} | mismatch {d['mismatch']} | "
                      f"no row {d['no_row']}", flush=True)
            for m in chk['mismatches'][:40]:
                print(f"  ELIG HASH MISMATCH {m['file']} | bytes {m['bytes']} vs {m['reference_bytes']} | "
                      f"{str(m['digest'])[:12]} vs {m['reference_digest'][:12]}", flush=True)
            for name in chk['missing'][:40]:
                print(f'  ELIG HASH MISSING {name}', flush=True)
        except Exception as exc:                          # log only: the check never stops the run
            rep['hash_check'] = dict(error=repr(exc)[:400])
            print('ELIG HASH CHECK FAILED (log only):', repr(exc)[:400], flush=True)
    rep['seconds'] = round(time.time() - t0, 1)
    (ROOT / 'elig0_report.json').write_text(json.dumps(rep, indent=2, default=str))
    return rep


ELIG_REPORT = elig0_setup()
