#!/usr/bin/env python3
"""Can the sweeps be found without doing any population genetics?

Checks the shipped data for anything that identifies a selected site by a
property of the FILE rather than of the biology: a distinct allele alphabet,
a non-integer or suspiciously round coordinate, an extra INFO field, a
differently shaped record, metadata naming selection. Also checks that
nothing inside the environment image carries the truth, and that the
selected mutations really are absent from the genotypes.

Every check prints PASS or FAIL. A FAIL means an agent could score without
analysing anything.
"""
import gzip, json, os, re, sys
from collections import Counter

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(name)


def read_vcf(path):
    op = gzip.open if path.endswith(".gz") else open
    meta, header, rows = [], None, []
    with op(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                meta.append(line)
            elif line.startswith("#CHROM"):
                header = line.split("\t")
            else:
                rows.append(line)
    return meta, header, rows


def main():
    if len(sys.argv) < 3:
        print("usage: leak_audit.py DATA_DIR TRUTH_JSON"); return 2
    data_dir, truth_path = sys.argv[1], sys.argv[2]
    truth = json.load(open(truth_path))
    vcf = os.path.join(data_dir, "variants.vcf.gz")

    print("=== shipped files ===")
    for f in sorted(os.listdir(data_dir)):
        print(f"  {f}  ({os.path.getsize(os.path.join(data_dir, f))} bytes)")
    check("no tree sequence shipped",
          not any(f.endswith((".trees", ".ts", ".tsz")) for f in os.listdir(data_dir)))

    meta, header, rows = read_vcf(vcf)
    print(f"\n=== VCF: {len(rows)} records, {len(header) - 9} samples ===")

    print("\n=== record-level uniformity ===")
    chrom = []; pos = []; ref = []; alt = []; info = []; fmt = []; ncol = Counter()
    for r in rows:
        f = r.split("\t")
        ncol[len(f)] += 1
        chrom.append(f[0]); pos.append(int(f[1])); ref.append(f[3]); alt.append(f[4])
        info.append(f[7]); fmt.append(f[8])
    pos = np.array(pos)

    check("every record has the same number of columns", len(ncol) == 1,
          f"column counts: {dict(ncol)}")
    check("ID column is always '.'", all(r.split("\t")[2] == "." for r in rows))
    check("QUAL column is always '.'", all(r.split("\t")[5] == "." for r in rows))
    check("FILTER column is always PASS", all(r.split("\t")[6] == "PASS" for r in rows))
    check("FORMAT column is always GT", set(fmt) == {"GT"})
    info_keys = {k.split("=")[0] for i in info for k in i.split(";")}
    check("INFO carries nothing but AA", info_keys == {"AA"}, f"keys: {sorted(info_keys)}")
    check("AA always equals REF",
          all(i == f"AA={r}" for i, r in zip(info, ref)))

    print("\n=== allele alphabet ===")
    ra = Counter(ref); aa = Counter(alt)
    check("REF alleles are single ACGT", set(ra) <= set("ACGT"), f"{dict(ra)}")
    check("ALT alleles are single ACGT", set(aa) <= set("ACGT"), f"{dict(aa)}")

    print("\n=== coordinates ===")
    check("all positions are positive integers", bool(np.all(pos >= 1)))
    nround = int(np.sum(pos % 1000 == 0))
    check("no excess of round coordinates", nround <= max(5, len(pos) // 500),
          f"{nround} positions divisible by 1000 out of {len(pos)}")
    check("positions ascend within each chromosome",
          all(np.all(np.diff(pos[np.array(chrom) == c]) > 0)
              for c in sorted(set(chrom))))

    print("\n=== metadata text ===")
    banned = re.compile(r"sweep|select|benefic|fitness|s=|coeff|sw0\d|truth|origin",
                        re.IGNORECASE)
    bad_meta = [m for m in meta if banned.search(m)]
    check("no VCF header line mentions selection", not bad_meta, str(bad_meta[:2]))
    bad_names = [h for h in header[9:] if banned.search(h)]
    check("no sample name mentions selection", not bad_names, str(bad_names[:3]))

    print("\n=== the selected sites themselves ===")
    sw = truth["sweeps"]
    present = []
    for s in sw:
        c = f"chr{s['chrom']}"
        p = s["position_local"]
        hit = [r for r in rows if r.startswith(f"{c}\t{p}\t")]
        if hit:
            present.append((c, p))
    check("no selected site appears as a variant", not present, str(present))

    # Positions should not be identifiable by local record density alone being
    # exactly zero over the sweep: diversity is reduced, not absent.
    print("\n=== local SNP density at sweeps vs elsewhere ===")
    dens_sw, dens_bg = [], []
    for c in sorted(set(chrom)):
        cp = pos[np.array(chrom) == c]
        cnum = int(c[3:])
        sw_here = [s["position_local"] for s in sw if s["chrom"] == cnum]
        for p in sw_here:
            dens_sw.append(int(np.sum(np.abs(cp - p) <= 25_000)))
        for p in range(1_000_000, 11_000_000, 500_000):
            if all(abs(p - q) > 1_500_000 for q in sw_here):
                dens_bg.append(int(np.sum(np.abs(cp - p) <= 25_000)))
    if dens_sw and dens_bg:
        print(f"  median SNPs per 50kb: at sweeps {np.median(dens_sw):.0f}, "
              f"background {np.median(dens_bg):.0f}")
        check("sweep windows are not empty of variants", min(dens_sw) > 0,
              f"minimum {min(dens_sw)}")

    print("\n=== nothing in the environment carries the truth ===")
    env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "environment"))
    leaked = []
    sw_positions = {str(s["position_local"]) for s in sw} | \
                   {str(s["position_global"]) for s in sw}
    for root, _, files in os.walk(env):
        for f in files:
            fp = os.path.join(root, f)
            if fp.endswith(".vcf.gz"):
                continue
            try:
                txt = open(fp, "r", errors="ignore").read()
            except Exception:
                continue
            for p in sw_positions:
                if len(p) >= 6 and p in txt:
                    leaked.append((os.path.relpath(fp, env), p))
            if re.search(r"\bsw0\d\b|selection_coefficient|completion_gens_ago", txt):
                leaked.append((os.path.relpath(fp, env), "truth-shaped text"))
    check("no environment file contains a sweep position or truth field",
          not leaked, str(leaked[:4]))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} LEAK CHECK(S) FAILED: {FAILURES}")
        return 1
    print("all leak checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
