# confounded-sweep-scan

A whole-genome scan where three different things produce a sweep-like signal
and only one of them is selection. The agent is given phased genotypes for 50
diploids across four 12 Mb chromosomes and a non-uniform recombination map,
and has to report the population's demographic history together with every
positively selected site: where it is, which of four modes of sweep it was,
how strong the selection was, and when the sweep finished.

## Difficulty

The data are synthetic. Real genomes do not come with a list of selected
sites annotated with their selection coefficients, ages and modes of origin,
and that list is what makes the task gradable. They are produced by a forward
simulation in fwdpy11, recapitated with msprime under a bottleneck-and-
expansion history and overlaid with neutral mutations. Generator, seeds and
all calibration measurements are under `authoring/`.

Three sources of difficulty, each measured rather than asserted.

**The demography mimics selection genome-wide.** The population contracts
from 12,000 to 800 for 400 generations starting 3,400 generations ago, then
expands to 9,000. That depresses diversity and skews the site frequency
spectrum everywhere, which is locally what a sweep does. A scan calibrated
against a constant-size null reports the bottleneck.

**Recombination cold spots mimic selection locally.** Three windows of 300 to
600 kb carry rates 8 to 12 times below the genome average, on three of the
four chromosomes, so an agent cannot assume one per chromosome. Fewer
independent genealogies per window means heavier tails for every statistic.
Over ten neutral replicates with no selection anywhere, a cut at the genome's
own 99.9th percentile fires this much more often inside the cold spots than
outside:

| statistic | enrichment inside cold spots |
| --- | --- |
| diversity | 20.6x |
| Tajima's D | 9.6x |
| nSL | 9.6x |
| H12 | 8.8x |
| H2/H1 | 7.4x |

The map is given to the agent, so this is a rule to infer, not a secret. But
a scan run at its defaults will report cold spots as sweeps.

**No single statistic finds all four sweep classes.** Measured at the sweep
against each replicate's own background, over independent single-sweep
replicates:

| class | diversity ratio | Tajima's D shift | H12 ratio | H2/H1 ratio |
| --- | --- | --- | --- | --- |
| hard, recent, complete | 0.033 | -3.03 | 11.7 | 0.02 |
| soft, recent, complete | ~0.05 | ~-3.0 | ~11 | 0.78 - 1.05 |
| hard, old, complete | 0.098 | -2.82 | 3.54 | 0.24 |
| partial, still rising | 0.233 | -2.36 | 9.99 | 0.04 |

H12 collapses for old sweeps. nSL reads near zero at any completed sweep,
since the sweep is over and no high-frequency young haplotype remains to
carry a long tract, so it finds partial sweeps and little else. Diversity
locates a partial sweep to a median 360 kb against 20 kb for H12. Hard and
soft sweeps are separated by H2/H1 and essentially nothing else, their
diversity, Tajima's D and H12 signatures agreeing to within noise.

Expert time: around 20 hours, most of it on null calibration and on deciding
which statistic to trust for which class.

## Reference solution

`solution/solve.sh` runs `solution/pipeline/`, in this order, because the
order is the point: the null a sweep must stand out from depends on the
history, so the history comes first.

1. **Demography** by simulated method of moments. Propose a history, simulate
   it with msprime, compare the binned unfolded spectrum and mean diversity
   against the observed ones computed over windows remaining after the
   lowest-diversity decile is dropped, so sweeps are not absorbed into the
   demography. Every proposal reuses one seed, which makes the objective
   deterministic instead of noisy and lets coarse random search plus local
   refinement and a coordinate polish converge.
2. **Null** of ten neutral replicates under the fitted history using the
   provided recombination map, scanned identically to the observed genome.
3. **Detection**: each observed window becomes a z-score against null windows
   in the same local-rate bin, across diversity, Tajima's D, H12 and extreme
   nSL fraction. H2/H1 is excluded here on purpose, because it sits near its
   neutral value at a soft sweep and scoring on it would penalise the hardest
   class to find. The threshold is set so the expected number of null windows
   per genome is a fraction of one; contiguous passing windows cluster into
   one call.
4. **Characterisation** from the measured separation above, plus two
   calibrated estimators: trough width times local rate equals 0.56 times s
   (measured 0.536, 0.571, 0.564, 0.561 across the four classes), and
   completion time from `2 * N_current * pi_min / pi_background` divided by a
   class-specific bias of 1.17, 2.02 or 2.46.

Validated against three independent neutral replicates with known truth: all
five demography gates pass with margin, N_ancestral 1.02 to 1.16x, N_current
1.03 to 1.13x, bottleneck strength 0.95 to 1.06x, expansion time within 2 to
20%.

## Verification

`tests/test_sweep_scan.py` grades `/app/results.json` against a sealed static
truth file. Pure python, no simulation at grading time, no network.

Ground truth is the generator's configuration plus the realized trajectory
the forward simulation recorded: scheduled position, selection coefficient,
the generation the allele passed 98% frequency, final frequency, and the
number of independent origins surviving to the present. Nothing in it comes
from a solver.

Calls are deduplicated, then matched to true sweeps greedily by ascending
distance within a window taken from measured localisation spread. Gates are
conjunctive: ancestral and current size within a factor; the bottleneck's
duration-over-depth within a factor; expansion time within a fraction; a
structural check that the history contracts and recovers; per-class recall;
a cap on calls matching no true sweep, with cold-spot calls counting against
it; class accuracy over matched calls; selection coefficients within a
factor; and completion time within tolerance, graded separately inside each
completed class.

Two decisions are worth naming.

**Bottleneck depth is graded only in combination.** Depth and duration are
not separately identifiable from the spectrum, since the genealogy integrates
duration over size. The reference overestimates depth 2.4 to 2.6 times on
every seed tried while recovering `duration / N_bottleneck` to within 6%, so
grading depth alone would have failed a correct method. The gate is on the
combination.

**Timing is graded per class.** Graded in aggregate it was satisfiable by a
constant: the recent and soft sweeps finish within a few hundred generations
of each other, so a flat 300 covered six of the eight gradable sweeps and
cleared any aggregate threshold. Per class no constant works, because the old
sweeps finish roughly five times further back while the tolerance is about
half.

Eighteen submissions were run through the real verifier
(`authoring/evidence/verifier_selftest.py`). The perfect answer passes, and so
does the perfect answer plus one false positive, confirming the false-positive
cap is not merely strict. Fifteen degenerate submissions fail: carpet bombing
every 100 kb, one constant class, a constant selection coefficient, two
different constant timings, all timings doubled, every call at a chromosome
centre, all positions shifted 400 kb, a flat constant-size history,
submissions missing an entire class, and NaN injection.

## Leak controls

The forward-phase selected mutations are dropped from the exported genotypes
rather than relabelled. fwdpy11 writes them with a `0`/`1` allele alphabet
against ACGT for everything msprime produces, at non-integer positions, and
with metadata stating the selection coefficient outright, so any one of the
three identifies every sweep with a single grep. Nothing about the signal is
lost: a sweep is a property of the genealogy, not of the beneficial mutation,
so dropping the causal site only means sweeps must be found from linked
variation, as in any real scan. Selected positions are also jittered off
round coordinates, fwdpy11 individual metadata (which carries each
individual's fitness and so labels carriers) is scrubbed, only present-day
samples are exported, and no tree sequence is shipped.

## What is not here

`instruction.md` and the author-facts block in `task.toml`
(`author_name`, `author_email`, `author_organization`, `author_profile`,
`conflicts_of_interest`, `relevant_experience`) are deliberately absent or
blank. The instruction has to be the author's own words, and the pipeline
screens it for generated text; the author facts are claims about a real
person. Both are the author's to write.

The instruction must state, at minimum: the absolute output path
`/app/results.json`; the exact JSON schema, including that `time_gens_ago` is
the completion time of a sweep and null for one still segregating; that
selection is additive with per-allele advantage s; the mutation rate; and the
four class names exactly as the verifier expects them
(`hard_recent_complete`, `hard_old_complete`, `soft_standing`, `partial`).
All of this is also stated in `environment/data/README_DATA.txt`, which the
agent can read, but the brief should not rely on that alone. It must end with
the required closing sentence using the integer agent timeout from
`task.toml`.
