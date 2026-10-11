#!/usr/bin/env python3
"""Print the calibration configurations the shipped zoo requires.

run_training.sh reads these rather than carrying its own copy. The copy
drifted: sw04's selection coefficient moved from 0.0130 to 0.0200 and five
origins changed, and the calibration set kept the old ones -- so
fit_estimators was fitting on configurations the shipped replicate no longer
contains, and the shipped-mix weighting in fit_classifier died looking for a
configuration that had never been simulated.

One row per shipped sweep, in run_training.sh's format:

    label  s  origin  n_origins  cold_factor  tag

cold_factor is the rate factor of the cold spot a sweep sits in, or "-" at
the base rate. A sweep inside a cold spot gets a _cs_ tag so its replicates
are distinguishable from a base-rate sweep at the same coefficient.
"""
import sys

import config as C


def rows():
    cold = set(C.SWEEPS_IN_COLDSPOT)
    out = []
    for sid, chrom, frac, s, origin, n_origins, klass in C.SWEEP_SPEC:
        pos = C.genome_config().chrom_offset(chrom) + frac * C.CHROM_LEN
        factor = "-"
        if sid in cold:
            local = pos - C.genome_config().chrom_offset(chrom)
            for cs in C.genome_config().cold_spots:
                if cs.chrom == chrom and cs.start <= local <= cs.end:
                    factor = f"{cs.factor:g}"
                    break
        tag = ("_cs_s" if sid in cold else "_s") + f"{round(s * 10000):04d}"
        out.append((klass, s, origin, n_origins, factor, tag, sid))
    return out


def main() -> int:
    want_cold = "--cold" in sys.argv
    for klass, s, origin, n_origins, factor, tag, sid in rows():
        if want_cold != (factor != "-"):
            continue
        print(f'  "{klass:<14} {s:<8.4f} {origin:<5} {n_origins:<4} '
              f'{factor:<5} {tag}"   # {sid}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
