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

**No single statistic finds all four sweep classes.** Measured in the window
containing the selected site, against each replicate's own background,
grouped by what each sweep actually turned out to be:

| realized class | diversity | Tajima's D | H12 | H2/H1 | nSL | n |
| --- | --- | --- | --- | --- | --- | --- |
| hard, recent, complete | 0.073 | -2.65 | 8.46 | 0.071 | 0.044 | 11 |
| hard, old, complete | 0.154 | -2.37 | 1.35 | 0.797 | 0.030 | 8 |
| soft, complete | 0.743 | **+0.89** | 4.52 | 0.436 | 0.016 | 4 |
| partial, still rising | 0.528 | -1.44 | 7.46 | 0.111 | 0.389 | 8 |

Diversity and H12 as ratios to background, Tajima's D and nSL as
differences.

H12 collapses for old sweeps, from about 8.5 down to 1.3, because
recombination erodes the swept haplotype block. nSL reads near zero at any
completed sweep, since no high-frequency young haplotype remains to carry a
long tract, so it responds to partial sweeps and little else. Diversity
locates a partial sweep to a median 380 kb against 25 kb for H12.

The soft sweeps are the subtlest: their diversity sits at 0.74 of
background, barely depressed, so a diversity scan does not see them at all
and only H12 does. What identifies them is that Tajima's D at a soft sweep
is **positive**, +0.21 to +1.15, where every other class is negative with a
maximum of -0.25: several frequent haplotypes leave an excess of
intermediate-frequency variants.

Worth noting what does NOT work, since it is the obvious choice: H2/H1,
normally the hard-versus-soft discriminator, fails here. At an old hard
sweep the block has decayed far enough that H2/H1 recovers to 0.46-0.99,
overlapping the 0.27-0.62 measured at soft sweeps.

## Why shortcuts fail, gate by gate

Measured on the shipped replicate. Each strategy differs from the reference in
one respect and is handed the reference's own classification and estimation
step, so the differences are differences in detection and calibration.
`authoring/evidence/baselines.py` reproduces the table.

| strategy | calls | false pos | fails on |
| --- | --- | --- | --- |
| one statistic, genome-wide cut (diversity) | 2 | 0 | demography, recall |
| one statistic (Tajima's D) | 2 | 0 | demography, recall |
| one statistic (H12) | 2 | 0 | demography, recall |
| one statistic (nSL) | 1 | 0 | demography, recall, timing |
| all statistics, constant-size null | 14 | 4 | demography, **false positives** |
| all statistics, correct history, rate-blind threshold | 8 | 1 | **recall** |
| all statistics, correct null, per-window cut | 7 | 1 | **recall** |
| the reference | 9 | 0 | **passes** |

Every required piece of work is load-bearing, and each one is enforced by a
different gate.

A scan run at its defaults on any single statistic finds one or two of the
ten sweeps. Skipping the demographic inference is the most instructive
failure: a constant-size null finds **every** sweep, 10 of 10, and invents
four more, so it fails on false positives alone. That is the bottleneck
confound in one line. Ignoring the recombination map while otherwise doing
everything right costs both old sweeps, because a threshold pooled across
rates is set by the heavy tail the cold spots contribute. And thresholding
single windows instead of clusters costs all three soft sweeps.

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
3. **Detection**: each observed window gets a rank-based p-value against null
   windows in the same local-rate bin, for six statistics, combined as
   -2 sum log p. Rank-based rather than a z-score because these statistics
   are strongly skewed, so a z is not on a calibrated scale: thresholding a
   max-z for a fixed false-positive budget put the cut at z = 10 and missed
   both old sweeps despite their diversity sitting at 0.21 of background.

   The six are diversity (low), Tajima's D in **both** directions, H12
   (high), H2/H1 (low) and extreme-nSL fraction (high). Including both
   directions of Tajima's D is not double-counting: in a Fisher combination
   a window extreme in one tail has a p near one in the other and
   contributes nothing from it, so the pair acts as a two-sided test. It is
   there because soft sweeps push Tajima's D the opposite way from hard
   ones. With only the one-sided set, soft sweeps contributed to a single
   statistic and two of three were missed.

   Calls are made at the level of CLUSTERS, not single windows. A footprint
   here spans one to two megabases, a hundred to two hundred windows at a
   10 kb step, each moderately elevated. Requiring any one window to clear a
   cut set for 0.3 expected false windows per genome puts that cut at the
   99.993rd percentile of the null, and measured on the shipped replicate
   the three soft sweeps peaked at 49.3, 55.4 and 55.9 against a threshold of
   56.44, with a genome-wide median of 12.7: real signal, two of them short
   by a single point. So clusters are formed at the null's 99th percentile
   and judged by total mass, the summed excess over that threshold. A broad,
   consistently elevated footprint accumulates far more mass than the short
   runs noise produces, and the mass cut is calibrated by applying the
   identical procedure to every null replicate, which keeps the
   false-positive budget exact and makes it a budget on clusters, which is
   what a call is.
4. **Characterisation** by a nearest-centroid rule on four features read at
   the peak: diversity and H2/H1 as ratios to background, Tajima's D and the
   extreme-nSL fraction as differences. Calibrated on 32 independent
   single-sweep replicates, 90.6% leave-one-out accuracy against 69% for the
   best single feature. Plus two estimators: the selection coefficient from
   the trough width times the local rate, with the width measured as the
   contiguous span recovering halfway from the trough's own floor rather
   than crossing half of background (which the shallow classes never do),
   meeting a factor-of-three tolerance on 91% of replicates and on all of
   both hard classes; and the completion time from diversity recovery with a
   class-specific bias removed, for single-origin completed sweeps only.

Validated against four independent neutral replicates with known truth:
N_ancestral 1.00 to 1.16x, N_current 0.88 to 1.12x, bottleneck severity 1.14
to 1.48x, all inside their gates.

The false-positive control is the sharper test. Run end to end on a genome
simulated under the same history and the same map but with no selection
anywhere, the pipeline reports **zero** sweeps. That is the counterpart to
the cold-spot enrichment above: a genome-wide threshold fires 7 to 21 times
more often inside cold spots than outside, while a null simulated under the
fitted history and compared rate-bin by rate-bin fires not at all.

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

**Bottleneck depth is graded only in combination, and its timing not at
all.** Depth and duration are not separately identifiable from the spectrum,
since the genealogy integrates duration over size, so the gate is on
`duration / N_bottleneck`, recovered at 1.14 to 1.48 times truth across
seeds where depth alone was off by 2.4 to 2.6. Timing is not graded: the
same four fits placed the expansion at 0.64, 0.68, 1.31 and 2.63 times its
true time, so one in four fails a 40% tolerance while any tolerance admitting
2.63 is vacuous. The objective has a broad flat minimum spanning roughly 1500
to 4000 generations for a truth of 3000, which is the information content of
a spectrum from 100 haplotypes rather than a failure of the search.

**Completion time is graded only for single-origin sweeps.** Diversity
recovery dates a sweep by how far below background its centre still sits,
which presumes the sweep drove diversity to zero. A soft sweep leaves it at
0.74 of background, and the same estimator reads such sweeps as roughly
eighteen times older than they are.

**Timing is graded per class.** Graded in aggregate it was satisfiable by a
constant: the recent and soft sweeps finish within a few hundred generations
of each other, so a flat 300 covered six of the eight gradable sweeps and
cleared any aggregate threshold. Per class no constant works, because the old
sweeps finish roughly five times further back while the tolerance is about
half.

## Validation performed

Run with `authoring/assemble.sh` and `authoring/validate.sh`.

**The oracle reaches reward 1.** Nine calls on the shipped replicate, zero
false positives, recall 3/3 recent hard, 1/2 old, 3/3 soft, 2/2 partial,
class accuracy 7 of 9, selection coefficients within a factor of three on 8
of 9, and all four graded completion times inside tolerance.

**It is deterministic.** Two independent runs produce byte-identical
submissions and identical calibrated thresholds, a cluster-forming cut of
31.56 and a mass cut of 97.3.

**The verifier container behaves.** Built from `tests/Dockerfile` and run
against the real images: the oracle's own output scores `reward=1` with a
CTRF report written, a perfect submission scores 1, an empty `/app` scores 0,
a malformed `results.json` scores 0, and the container has no network.

**The data carries no shortcut.** `authoring/evidence/leak_audit.py` passes
every check, including that no selected site appears as a variant at all.

**`task.toml` validates** against harbor's own task schema.

Not done here: a full `harbor run` of the oracle and nop inside this sandbox
needs the environment image, whose apt layer cannot be built because the
egress policy reaches PyPI but not Debian mirrors, and pip inside a docker
build needs the proxy CA injected. Neither applies in the real pipeline. The
pip layer was built and its imports exercised; the apt packages are standard
bookworm names and are convenience tooling the reference does not use.

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
agent can read, but the brief should not rely on that alone.

The file must be plain prose with no Markdown headings, and must end with a
blank line followed by exactly this sentence, matching the integer
`[agent].timeout_sec` of 21600 in `task.toml`:

```
You have 21600 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
```

The schema the verifier expects is:

```json
{
  "demography": {
    "N_ancestral": 12000,
    "N_bottleneck": 800,
    "N_current": 9000,
    "T_bottleneck_start_gens_ago": 3400,
    "T_bottleneck_end_gens_ago": 3000
  },
  "sweeps": [
    {
      "chrom": 1,
      "position": 7447962,
      "class": "hard_recent_complete",
      "selection_coefficient": 0.035,
      "time_gens_ago": 300
    }
  ]
}
```

`position` is 1-based within its chromosome. `time_gens_ago` is the
generation before present at which the sweep completed, and null for a sweep
still segregating. At most 200 calls are accepted.
