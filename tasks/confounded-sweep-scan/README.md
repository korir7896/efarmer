# confounded-sweep-scan

A whole-genome scan where three different things produce a sweep-like signal
and only one of them is selection. The agent is given phased genotypes for 50
diploids across four 12 Mb chromosomes and a non-uniform recombination map,
and has to report the population's demographic history together with every
positively selected site: where it is, which of four modes of sweep it was,
how strong the selection was, and, for the completed single-origin ones, how
long ago it finished.

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

**No single statistic finds all four sweep classes, and neither axis of the
class scheme has a statistic that carries it.** The four classes are the two
binary properties a scan can be asked to recover, crossed: complete or
partial (did the beneficial allele pass 0.98), hard or soft (is one
independent origin of it still present, or several). Measured in the window
containing the selected site, against each replicate's own background, over
51 independent single-sweep replicates grouped by what each sweep actually
turned out to be:

| realized class | diversity | Tajima's D | H12 | H2/H1 | nSL | n |
| --- | --- | --- | --- | --- | --- | --- |
| hard_complete | 0.119 | -2.44 | 7.409 | 0.084 | +0.01 | 16 |
| hard_partial | 0.801 | -0.63 | 6.596 | 0.104 | +0.41 | 9 |
| soft_complete | 0.794 | +0.53 | 4.298 | 0.673 | -0.02 | 16 |
| soft_partial | 0.811 | -0.03 | 4.949 | 0.549 | +0.20 | 10 |

Diversity, H12 and H2/H1 as ratios to background, Tajima's D and the
extreme-nSL fraction as differences; median over replicates.

The medians separate. The distributions do not, and that is the difficulty.

H2/H1 is the textbook hard-versus-soft discriminator and it does not work
here, because it is confounded by age: across the hard_complete replicates
it runs from 0.009 to 1.012 of background, which swallows the 0.025 to 0.861
the soft completed ones occupy. At a recent hard sweep one haplotype
dominates and H2/H1 is near zero; at an old one recombination has broken up
the swept block and H2/H1 has recovered to background. Diversity is the
obvious complete-versus-partial discriminator and it is confounded by
softness: 0.011 to 0.719 across hard_complete against 0.383 to 1.167 across
hard_partial, because a soft completed sweep leaves diversity near
background exactly as a partial one does.

What does separate the classes is different features on each axis, and for
softness a different feature depending on completeness. Completeness is
carried by diversity together with the number of segregating sites in the
window. Softness among completed sweeps is carried by Tajima's D, which
several frequent haplotypes push positive where one pushes it negative, with
the extent of the H12 elevation controlling for the sweep's strength.
Softness among partial sweeps is carried by H2/H1 after all -- usable there
precisely because age is not in play, since every partial sweep is still
rising. Measured leave-one-configuration-out, that gets 82.4% against 72.9%
for a single nearest-centroid rule over all four classes at once, and 90.6%
on the softness axis among completed sweeps against 73.7% among partial
ones.

H12 collapses for old sweeps, because recombination erodes the swept
haplotype block, so a scan that leans on it misses them. nSL reads near zero
at any completed sweep, since no high-frequency young haplotype remains to
carry a long tract: it is the only statistic that responds strongly to
partial sweeps, +0.41 and +0.22 against +0.01 and -0.02, and it responds to
almost nothing else.

**And the two quantities asked about each sweep are relations that have to be
derived and calibrated.** The selection coefficients in the shipped
replicate span a factor of 9.4 and the ages of the datable sweeps a factor
of eighteen, so no single plausible number satisfies either requirement. The
textbook relation for strength, that the footprint extends to about s over
the local recombination rate, is necessary but not sufficient: fitted as
trough width times local rate over a per-class constant it put 74% of
replicates inside a factor of 2.5 and ranked them at only 0.32, because a
soft sweep leaves no trough whose width can be measured. Taking the swept
extent from the H12 elevation instead, where diversity is flat, brings that
to 98% and 0.83.

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
4. **Characterisation** of each called region, calibrated on 48 independent
   single-sweep replicates simulated under the same history at several
   strengths per class (51 of them) and grouped by what each sweep actually turned out to
   be. Every score quoted here is leave-one-CONFIGURATION-out, not
   leave-one-replicate-out: the shipped sweeps sit at the very (s, age)
   configurations the replicates were simulated at, so holding out one
   replicate at a time would reward a rule that had memorised those few
   pairs.

   **Class** by one rule per axis rather than one rule over four classes:
   82.4% against 72.9% for a nearest-centroid rule on the same features.
   Completeness comes from diversity with the number of segregating sites in
   the window; softness from Tajima's D with the extent of the H12
   elevation among completed sweeps, and from H2/H1 among partial ones,
   fitted separately because the axis is much easier among completed sweeps
   (90.6%) than among those still rising (73.7%).

   **Selection coefficient** from a log-linear fit in two measured features
   per class: 98% within a factor of 2.5 with a rank correlation of 0.83
   pooled across classes. The textbook relation, footprint extent times the
   local rate, is the backbone of it, but on its own, as trough width over a
   per-class constant, it reached only 74% and ranked at 0.32 -- a soft sweep
   leaves no trough whose width can be measured. The soft classes take their
   extent from the H12 elevation instead. All width features are scaled by
   the local recombination rate, because what strength sets is the genetic
   length of the footprint, not its length in base pairs.

   **Completion time** from H12 decay with the footprint extent controlling
   for strength: 15 of 16 inside the verifier's tolerance, rank correlation
   0.88. Dating by residual diversity at the trough's floor instead, which
   is the obvious choice, managed 9 of 16. Reported for every sweep called
   complete, because the schema requires a number there, and graded only
   where it is estimable.

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
truth file. Pure python, no simulation at grading time, no network, and no
numpy or scipy: the verifier image carries pytest and nothing else.

Ground truth is the generator's configuration plus the realized trajectory
the forward simulation recorded: scheduled position, selection coefficient,
the generation the allele passed 0.98, final frequency, and the number of
independent origins surviving to the present. Nothing in it comes from a
solver.

Calls are deduplicated at 250 kb, then matched to true sweeps greedily by
ascending distance, same chromosome, within 250 kb. The gates are
conjunctive:

| gate | threshold |
| --- | --- |
| `N_ancestral`, `N_current` | each within a factor of 1.6 |
| bottleneck `duration / N_bottleneck` | within a factor of 2.5 |
| history is a real contraction and recovery | structural |
| recall, `hard_complete` | at least 3 of 4 |
| recall, each other class | at least 1 of 2 |
| calls matching no true sweep | at most 1, cold spots included |
| class, over matched calls | at least 70% correct |
| `s` magnitude, over matched calls | at least 70% within a factor of 2.5 |
| `s` ordering, over matched calls | rank correlation at least 0.50 |
| completion time, true `hard_complete` sweeps | at least 3 inside max(50%, 150 generations) |

Four decisions are worth naming.

**Bottleneck depth is graded only in combination, and its timing not at
all.** Depth and duration are not separately identifiable from the spectrum,
since the genealogy integrates duration over size, so the gate is on
`duration / N_bottleneck`, recovered at 1.14 to 1.48 times truth across
seeds where depth alone was off by 2.4 to 2.6. Timing is not graded: the
same four fits placed the expansion at 0.64, 0.68, 1.31 and 2.63 times its
true time, so one in four fails a 40% tolerance while any tolerance admitting
2.63 is vacuous. The objective has a broad flat minimum spanning roughly 1500
to 4000 generations for a truth of 3000, which is the information content of
a spectrum from 100 haplotypes rather than a failure of the search. There is
no gate on expansion time, and nothing in the instruction asks for one to be
graded.

**A sweep's age is not part of its class.** An earlier version of this task
split the completed hard sweeps into `hard_recent_complete` and
`hard_old_complete` at a fixed boundary. That put age inside the class
label, and the consequence was that one constant per class satisfied the
timing gate without dating anything: the review of that version said so, and
it was right. The classes are now the two binary properties a scan can be
asked to recover, crossed, and age is graded only as a continuous quantity.

**Completion time is graded only where it is estimable.** Diversity recovery
dates a sweep by how far below background its centre still sits, which
presumes the sweep drove diversity to zero. A soft sweep never did, so the
clock has no zero to start from. Timing is therefore graded over the sweeps
whose TRUE class is `hard_complete` -- the true class, not the reported one,
so a misclassification is not punished twice -- and the four of them in the
shipped replicate completed far enough apart that no single number lands
inside more than two of their tolerance windows.

**Selection coefficients are graded twice, on magnitude and on order.** The
magnitude test alone is not enough and the bundle does not pretend
otherwise: `authoring/evidence/gate_feasibility.py` searches for the single
coefficient that satisfies the most magnitude tests and reports it. The
ordering test is what cannot be satisfied without measuring strength. A
constant produces nothing but ties, which score zero by the convention in
`grading.spearman`, so no constant passes whatever its value, and neither
does an answer whose coefficients are real numbers unrelated to the data.

### Where the thresholds come from

Worth separating, because the three bounds rest on different evidence and
only one of them involves the reference solution.

**That the gates are passable** is measured on independent single-sweep
replicates, one chromosome and one sweep each, simulated under the same
history at several strengths per class -- data the shipped genome had no
part in. `gate_feasibility.py` reports how often the estimators land inside
each gate on those replicates, and every pass fraction is set below what was
measured there. The demography gates rest on four independent neutral
replicates with known truth, again not the shipped one.

**That the gates are not trivially passable** is established without
reference to the reference at all. `gate_feasibility.py` searches for the
best possible constant selection coefficient and the best possible constant
completion time against the shipped truth and reports what each achieves;
`verifier_selftest.py` runs those and twenty-odd other degenerate
submissions through the real verifier.

**The ablation table above is neither of those, and should not be read as
justifying a threshold.** Every row in it is handed the reference's own
classification and estimation step, deliberately, so that the rows differ
only in detection and calibration. It shows which pieces of the detection
pipeline are load-bearing. It says nothing about whether a different
classifier or a different estimator would clear the characterisation gates,
and no threshold was set from it.

What none of this establishes is that a method unlike the reference would
pass. The claim is narrower: every gate is met by at least one honest
method, measured out of sample, and none is met by an answer that estimates
nothing.

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

**Both harbor gates pass.** `harbor run -a oracle -e docker` returns reward
**1.0** and `harbor run -a nop -e docker` returns reward **0.0**, one trial
each, zero errors, through the real harness: environment image built,
`solve.sh` run inside it, `/app/results.json` copied out as the declared
artifact, and the separate verifier container grading it.

Those two runs used a locally patched copy of the bundle, because this
sandbox's egress policy reaches PyPI but not Debian mirrors and pip inside a
docker build needs the proxy CA injected. The patch drops the apt layer and
adds the CA; nothing else differs, and neither change applies in the real
pipeline, which has open network. The apt packages are standard bookworm
names and are convenience tooling the reference does not use.

Not run here: `harbor check`, which needs a model API key this session does
not have.

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
