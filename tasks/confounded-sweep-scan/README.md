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
111 independent single-sweep replicates at 37 configurations, grouped by
what each sweep actually turned out to be:

| realized class | diversity | Tajima's D | H12 | H2/H1 | nSL | n |
| --- | --- | --- | --- | --- | --- | --- |
| hard_complete | 0.119 | -2.43 | 5.142 | 0.169 | +0.01 | 40 |
| hard_partial | 0.540 | -0.83 | 6.596 | 0.150 | +0.45 | 21 |
| soft_complete | 0.700 | +0.23 | 4.427 | 0.642 | -0.01 | 26 |
| soft_partial | 0.787 | +0.00 | 5.418 | 0.488 | +0.20 | 24 |

Diversity, H12 and H2/H1 as ratios to background, Tajima's D and the
extreme-nSL fraction as differences; median over replicates.

The medians separate. The distributions do not, and that is the difficulty.

H2/H1 is the textbook hard-versus-soft discriminator and it does not work
here, because it is confounded by age: across the hard_complete replicates
it runs from 0.007 to 1.012 of background, which swallows the 0.013 to 0.861
the soft completed ones occupy. At a recent hard sweep one haplotype
dominates and H2/H1 is near zero; at an old one recombination has broken up
the swept block and H2/H1 has recovered to background. Diversity is the
obvious complete-versus-partial discriminator and it is confounded by
softness: 0.011 to 0.719 across hard_complete against 0.158 to 1.431 across
hard_partial, because a soft completed sweep leaves diversity near
background exactly as a partial one does.

What does separate the classes is different features on each axis, and for
softness a different feature set depending on completeness. Completeness is
carried by the floor of the diversity trough with the H12 elevation and the
standardized-nSL fraction: a completed sweep removed variation and left one
or a few long haplotypes, while one still rising carries a long YOUNG
high-frequency haplotype against a background that is otherwise intact,
which is what nSL responds to. Softness among completed sweeps is carried by
the trough floor with Tajima's D and H2/H1 -- one frequent haplotype drives
diversity to nothing and Tajima's D far negative, several leave both milder
and H2/H1 high. Softness among partial sweeps is carried by the extent of
the H12 elevation with H2/H1 after all -- usable there precisely because age
is not in play, since every partial sweep is still rising. Measured
leave-one-configuration-out, that gets 80.2% four-way against 75.7% for the
best nearest-centroid rule over all four classes at once, and 92.4% on the
softness axis among completed sweeps against 80.0% among partial ones.

H12 collapses for the oldest sweeps, because recombination erodes the swept
haplotype block, so a scan that leans on it misses them. nSL reads near zero
at any completed sweep, since no high-frequency young haplotype remains to
carry a long tract: it is the only statistic that responds strongly to
partial sweeps, +0.45 and +0.20 against +0.01 and -0.01, and it responds to
almost nothing else.

**And the two quantities asked about each sweep are relations that have to be
derived and calibrated.** The selection coefficients in the shipped
replicate span a factor of 9.4 and the ages of the datable sweeps a factor
of 15.8, so no single plausible number satisfies either requirement. The
textbook relation for strength, that the footprint extends to about s over
the local recombination rate, is necessary but not sufficient: fitted as
trough width times local rate over a per-class constant it put 74% of
replicates inside a factor of 2.5 and ranked them at only 0.32, because a
soft sweep leaves no trough whose width can be measured. Taking the swept
extent from the H12 elevation instead, wherever diversity is flat, brings
that to 109 of 109 inside a factor of 2.5 and a rank correlation of +0.82.

## Why shortcuts fail, gate by gate

Measured on the shipped replicate. Each strategy differs from the reference in
one respect and is handed the reference's own classification and estimation
step, so the differences are differences in detection and calibration.
`authoring/evidence/baselines.py` reproduces the table.

| strategy | calls | false pos | fails on |
| --- | --- | --- | --- |
| one statistic, genome-wide cut (diversity) | 1 | 0 | demography, recall, class, s, s ordering, timing |
| one statistic (Tajima's D) | 2 | 0 | demography, recall, s ordering, timing |
| one statistic (H12) | 1 | 0 | demography, recall, class, s, s ordering, timing |
| one statistic (nSL) | 2 | 0 | demography, recall, s ordering, timing |
| all statistics, constant-size null | 12 | 4 | demography, **false positives**, class, s ordering |
| all statistics, correct history, rate-blind threshold | 9 | 1 | **recall**, timing |
| all statistics, correct null, per-window cut | 6 | 1 | **recall**, class, s ordering, timing |
| the reference | 8 | 0 | **passes** |

Every required piece of work is load-bearing, and each one is enforced by a
different gate.

A scan run at its defaults on any single statistic finds one or two of the
ten sweeps, and which ones depends on the statistic: diversity and Tajima's
D find only completed hard sweeps, nSL only a partial one and a soft partial
one, and none of them finds enough of any class to clear recall.

Skipping the demographic inference is the most instructive failure. A
constant-size null finds **more** sweeps than the reference does, all four
hard_complete sweeps among them, and invents four more, so it fails on false
positives. That is the bottleneck confound in one line: the contraction
depresses diversity and skews the spectrum genome-wide, so a scan calibrated
against a constant-size population sees signal everywhere.

Ignoring the recombination map while otherwise doing everything right finds
nine regions and still loses two of the four hard_complete sweeps, because a
threshold pooled across rates is set by the heavy tail the cold spots
contribute. And thresholding single windows instead of clusters costs both
soft completed sweeps outright.

The pipeline's one free parameter is its false-positive budget, the expected
number of spurious CLUSTERS per genome, which sets the cluster-mass cut. It
is set to 0.5, half the verifier's cap of one unmatched call. This is what
every value does on the shipped replicate, with detection the only thing
varying and the characterisation step fixed:

| budget | mass cut | calls | matched | FP | class | `s` spread | `s` rho | timed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 132.5 | 8 | 8 | 0 | 0.75 | 2.76x | 0.786 | 3/3 |
| 0.3 | 130.0 | 8 | 8 | 0 | 0.75 | 2.76x | 0.786 | 3/3 |
| **0.5** | 73.9 | 9 | 9 | 0 | 0.78 | 5.20x | 0.850 | 3/3 |
| 0.8 | 49.5 | 10 | 9 | 1 | 0.67 | 2.74x | 0.717 | 3/3 |
| 1.0 | 47.3 | 10 | 9 | 1 | 0.67 | 2.74x | 0.717 | 3/3 |
| 1.5 | 39.1 | 11 | 9 | 2 | 0.67 | 2.80x | 0.717 | 3/3 |
| 2.0 | 34.7 | 11 | 9 | 2 | 0.67 | 2.80x | 0.717 | 3/3 |
| 3.0 | 27.7 | 12 | 10 | 2 | 0.60 | 5.35x | 0.794 | 4/4 |

Everything from 0.2 to 1.0 clears every gate; 1.5 and above breach the
false-positive cap. 0.5 is the largest budget that still makes no false call
at all, and it is where the margins are widest. Note the last row: a budget
of 3.0 finds the fourth hard_complete sweep and dates it, at the cost of two
false positives and a class accuracy below the gate -- the recall the
reference leaves on the table is reachable, just not for free.

This number was chosen with the answer visible, which is why the rule it
follows matters more than the value: sit at half the verifier's stated cap.

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
   windows in the same local-rate bin, for six one-sided tests over five
   statistics, combined as -2 sum log p. Rank-based rather than a z-score because these statistics
   are strongly skewed, so a z is not on a calibrated scale: thresholding a
   max-z for a fixed false-positive budget put the cut at z = 10 and missed
   the two oldest sweeps despite their diversity sitting at 0.21 of
   background. That was measured on an earlier replicate of this task, built
   to the same history and the same map.

   The six are diversity (low), Tajima's D in **both** directions, H12
   (high), H2/H1 (low) and extreme-nSL fraction (high). Including both
   directions of Tajima's D is not double-counting: in a Fisher combination
   a window extreme in one tail has a p near one in the other and
   contributes nothing from it, so the pair acts as a two-sided test. It is
   there because soft sweeps push Tajima's D the opposite way from hard
   ones, a median of +0.23 against -2.43 at the sweep. With only the one-sided set, soft
   sweeps contributed to a single statistic and two of the three in an
   earlier replicate were missed.

   Calls are made at the level of CLUSTERS, not single windows. A footprint
   here spans one to two megabases, a hundred to two hundred windows at a
   10 kb step, each moderately elevated. Requiring any one window to clear a
   cut set for 0.3 expected false windows per genome puts that cut at the
   99.993rd percentile of the null. On an earlier replicate the three soft
   sweeps peaked at 49.3, 55.4 and 55.9 against such a threshold of 56.44,
   with a genome-wide median of 12.7: real signal, two of them short by a
   single point. On the shipped replicate that substitution, everything else
   held fixed, loses both soft completed sweeps and still reports a false
   positive. So clusters are formed at the null's 99th percentile
   and judged by total mass, the summed excess over that threshold. A broad,
   consistently elevated footprint accumulates far more mass than the short
   runs noise produces, and the mass cut is calibrated by applying the
   identical procedure to every null replicate, which keeps the
   false-positive budget exact and makes it a budget on clusters, which is
   what a call is.
4. **Characterisation** of each called region by fixed constants, fitted
   offline on 111 independent single-sweep replicates at 37 configurations,
   simulated at several strengths per class and grouped by what each sweep
   actually turned out to be. Those replicates use the TRUE demography, not a
   fitted one, and ten of the 37 configurations are the shipped sweeps' own;
   the Verification section below says what that does and does not establish.
   Every score quoted here is leave-one-CONFIGURATION-out, not
   leave-one-replicate-out: holding out one replicate at a time would reward
   a rule that had memorised those few (s, age) pairs.

   **Class** by one rule per axis rather than one rule over four classes:
   80.2% four-way against 75.7% for the best nearest-centroid rule on three
   features. Completeness comes from the floor of the diversity trough, the
   H12 elevation and the standardized-nSL fraction (91.0%). Softness comes
   from the trough floor with Tajima's D and H2/H1 among completed sweeps
   (92.4%), and from the extent of the H12 elevation with H2/H1 among partial
   ones (80.0%), fitted separately because the axis is easier among completed
   sweeps than among those still rising. The three feature sets were chosen
   by a joint search over the top six candidates per axis, 216 triples,
   scored four-way: picking each axis on its own accuracy is the wrong
   objective, because the softness rules are fitted inside a completeness
   group and applied to the PREDICTED group, so the axes interact.

   Selecting over the ten shipped configurations alone picked a different
   triple and scored it at 0.824 there. On all 37 configurations that triple
   falls well below this one, which is what selecting on the graded instances
   looks like from the inside; `authoring/evidence/fit_classifier.py
   --shipped-only` reproduces both numbers.

   **Selection coefficient** from a log-linear fit in the level of one
   statistic at the peak and one footprint width per class: 109 of 109 within
   a factor of 2.5 with a pooled rank correlation of +0.82. The textbook
   relation, footprint extent times the local rate, is the backbone of it,
   but on its own, as trough width over a per-class constant, it reached only
   74% and ranked at 0.32 -- a soft sweep leaves no trough whose width can be
   measured. Only `hard_complete` takes its width from the diversity trough;
   the other three take it from the H12 elevation. All width features are
   scaled by the local recombination rate, because what strength sets is the
   genetic length of the footprint, not its length in base pairs.

   **Completion time** from H12 decay with the footprint extent controlling
   for strength: 36 of 40 inside the verifier's tolerance, rank correlation
   +0.93. Dating by residual diversity at the trough's floor instead, which
   is the obvious choice, managed 9 of 16 on the configurations it was
   measured against. Reported for every sweep called complete, because the
   schema requires a number there, and graded only where it is estimable.

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
| `s` spread, over matched calls | largest over smallest at least 2.0 |
| `s` ordering, over matched calls | rank correlation at least 0.30 |
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

**Selection coefficients are graded three ways, on magnitude, spread and
order.** The magnitude test alone is not enough and the bundle does not
pretend otherwise: `authoring/evidence/gate_feasibility.py` searches for the
single coefficient that satisfies the most magnitude tests and reports it.
The spread gate requires the reported coefficients to range over at least a
factor of 2, which a constant cannot do at all. The ordering gate requires a
rank correlation of 0.30; a constant produces nothing but ties, which score
zero by the convention in `grading.spearman`, so no constant passes whatever
its value.

The honest limit of that pair is measured rather than glossed: a constant
with noise added clears the ordering gate on 0.223 of tries at nine matched
calls and 0.290 at the smallest match set the recall gates permit, measured
over 20,000 random answers per match set in `gate_feasibility.py`. The
deterministic alternative -- dropping the magnitude tolerance to a factor of
1.5, where the best constant reaches only 0.667 and so fails outright --
was measured too and rejected, because it leaves the reference at 0.778 and
the independent solver at 0.286, excluding a method that orders every sweep
correctly. A one-in-five escape for an answer that also has to clear eleven
other gates is the lesser cost.

### Where the thresholds come from

Three claims, resting on different evidence. They are separated because the
first is much better supported than the third, and an earlier version of
this section overstated the third.

**That the gates are NOT trivially passable** is established without the
reference solution at all, and `authoring/evidence/gate_feasibility.py`
prints every number in this paragraph.

For the selection coefficients, a single constant fails the spread gate
(1.00 against 2.00) and the ordering gate (0.000 against 0.30) with
certainty, on any match set and whatever its value. The magnitude gate alone
does not exclude it: enumerating all 135 match sets the recall gates permit,
the best a constant achieves at a factor of 2.5 is 1.000, because the recall
gates allow a match set whose true coefficients span only 3.2 (0.025 to
0.080). The deterministic alternative was measured and rejected -- at a
factor of 1.5 the best constant reaches 0.667, below the 0.70 threshold, but
there the reference scores 0.778, one error from failing, and a second
independent implementation 0.286. A gate only one method can clear, by one
instance, is worse than one a noisy cheat escapes. **The honest limit, measured rather
than asserted: a constant with noise added passes magnitude and spread, and
the ordering gate catches it only most of the time.** Over 20,000 random
answers per match set, a random order clears a correlation of 0.30 on 0.223
of tries at nine matched calls, 0.190 at ten, and 0.290 at the smallest
match set the recall gates permit -- so the worst case is about one escape
in 3.5. Ten instances do not support a stronger claim than that.

For the completion times, the best single constant dates 2 of the 4
hard_complete sweeps where 3 are required, and one constant per class does
no better since there is only one dated class. That holds for every
replicate the generator would accept, not just this draw: over the
acceptance WINDOWS, the four tolerance ranges have no point in common three
at a time. **The honest limit: two constants, 160 and 1060, date all four.**
With four dateable sweeps no count-based gate can exclude that. The gate
excludes one constant, which is what it claims and nothing stronger.

Twenty-eight degenerate submissions, including the best possible constant of
each kind found by search rather than picked, go through the real verifier in
`authoring/evidence/verifier_selftest.py`.

**That the gates ARE passable** rests on three measurements, in increasing
order of how much they tell you.

1. The reference clears all twelve. On its own this is the weakest of the
   three, and the reason is worth stating plainly: its characterisation
   constants are fitted on single-sweep replicates simulated under the TRUE
   demography, not a fitted one, so it carries no demographic-inference
   error that a real analysis would; and ten of the 37 configurations those
   replicates cover ARE the shipped sweeps' own, so its feature sets and
   coefficients were selected with those ten in the pool.

2. The same pipeline, with every characterisation constant re-SELECTED and
   re-fitted on 60 replicates across 20 configurations the shipped replicate
   never uses, also clears every gate: 9 matched calls, no false positives,
   class right on 7 of 9, coefficients inside a factor of 2.5 on 8 of 9 with
   a spread of 5.4x and a rank correlation of 0.517, and all three localized
   hard_complete sweeps dated inside tolerance
   (`authoring/evidence/holdout_calibration.py`). Nothing in that version is
   fitted at a point it is graded on.

   Getting there required fixing the measurement twice, and both corrections
   matter. Refitting only the coefficients and keeping the shipped feature
   sets regraded at a rank correlation of 0.267 -- selecting features is
   itself fitting, and the pair chosen for `soft_complete` on the shipped
   configurations carries almost no information about s away from them. And
   the first re-selection ignored coverage, picking a predictor that is zero
   at the oldest sweeps, so the oldest sweep fell through to a fallback and
   the timing gate failed on an artefact.

   Removing the circularity changed the reference too, and not for the
   better on paper: re-selecting its feature sets over all 37 configurations
   rather than the ten shipped ones moved four-way class accuracy from the
   0.824 measured on those ten to 0.802 over all of them, and it is the
   wider number that means anything. The s gate was rebuilt during that
   work, from a single rank-correlation threshold to the magnitude, spread
   and ordering triple described above, because a threshold calibrated
   against an in-sample correlation is calibrated against an artefact.

3. `authoring/evidence/independent_solver.py` is a second solver sharing no
   method with the reference -- demography by Nelder-Mead on four moments
   instead of L2 on a binned spectrum, detection by a sliding 600 kb
   macro-window mean of per-rate-bin rank scores over three statistics
   instead of a Fisher combination of six judged by cluster mass, linear
   discriminant analysis for the class, s from the Tajima's D trough width,
   and the diversity-recovery clock. Its constants come only from the
   out-of-configuration replicates. It shares the windowed statistics and
   the VCF reader, which are measurement rather than method.

   Its submission goes through the **real verifier**, in the verifier
   container with `--network none`, not through any authoring grader:
   `authoring/evidence/indep_results.json` is the file, and the ctrf report
   from that run names which gate functions passed. It clears **10 of the 12
   gates**. It passes all four demography gates
   with a visibly different history, a shallow long bottleneck of 5425 for
   2326 generations against the truth's deep short 800 for 400 -- which is
   the non-identifiability the duration-over-depth gate was designed around,
   demonstrated rather than asserted. It passes every recall gate, class
   accuracy at 5 of 7, and all three coefficient gates, with a rank
   correlation of 0.847 on its seven matched calls -- level with the
   reference's 0.850 on nine, from a completely different estimator.

   It fails exactly two, and the verifier's own messages say which. The
   false-positive cap: `2 reported sweeps match no true sweep (limit 1):
   [(1, 3985000), (4, 7515000)]`, one of them at the edge of a cold spot its
   quantile rate bins do not fully absorb. And timing: `0 of 3 localized
   hard_complete sweeps dated within tolerance`, systematically -- 373
   against 133, 684 against 292, 3108 against 1214, each about 2.5 times too
   old. That is the
   diversity-recovery clock, which the reference measured at 9 of 16 against
   36 of 40 for H12 decay. **So the timing gate separates a good clock from
   a poor one, which is what a gate should do -- and no independent
   implementation has yet been shown to clear it.** Three good-faith
   improvement passes went into that solver; a fourth, tuned until it
   passed, would be the circularity this exercise exists to test for.

**The ablation table above is neither of those, and no threshold was set
from it.** Every row is handed the reference's own classification and
estimation step, deliberately, so the rows isolate detection and
calibration.

### Margins

| gate | reference | threshold | margin |
| --- | --- | --- | --- |
| `N_ancestral` | 1.03x | within 1.6x | wide |
| `N_current` | 0.95x | within 1.6x | wide |
| bottleneck `duration / N_bottleneck` | 1.56x | within 2.5x | wide |
| history contracts and recovers | yes | structural | — |
| recall, `hard_complete` | 3 of 4 | at least 3 | **none** |
| recall, `soft_complete` | 2 of 2 | at least 1 | 1 sweep |
| recall, `hard_partial` | 2 of 2 | at least 1 | 1 sweep |
| recall, `soft_partial` | 2 of 2 | at least 1 | 1 sweep |
| unmatched calls | 0 | at most 1 | 1 call |
| class accuracy | 7 of 9 | at least 70% | 1 call |
| `s` magnitude | 9 of 9 | at least 70% | 2 calls |
| `s` spread | 5.2x | at least 2.0x | wide |
| `s` ordering | 0.850 | at least 0.30 | 0.55 |
| completion time | 3 of 3 | at least 3 | **none** |

Two gates are cleared with no margin, and that is structural rather than
tuned. The timing gate needs three correct dates because a single constant
achieves two; three is the smallest requirement a constant cannot meet. And
the recall gate must supply three `hard_complete` sweeps for three dates to
be gradeable at all. Neither number can be lowered without making the
timing gate satisfiable by a constant, and the reference finds exactly three
of the four. The fourth, at s=0.0085 and completed 1214 generations ago, is
found by the independent solver, and by the reference itself at a looser
false-positive budget, so it is not beyond reach -- the reference declines it
to stay inside the cap, which is a judgement call rather than a limit.

**What none of this establishes** is that an arbitrary correct method clears
every gate. One structurally independent implementation was built, scored by
the real verifier, and cleared ten of twelve. Its two failures are traceable
to specific weaker components -- a detector whose rate bins leak at a cold
spot edge, and a clock the reference had already measured as the worse of
two -- rather than to the thresholds. That is informative, and it is one
implementation, not a proof either way.

Two gates in particular have been cleared by exactly one method each. Timing
is cleared by the reference and by its holdout-recalibrated variant, both of
which use the H12-decay clock, and by nothing else here. The false-positive
cap is cleared by every version of the reference and by neither of the
detection ablations nor the independent solver. If a reviewer wants one
number for how much this bundle has been stress-tested, it is that: twelve
gates, three methods, and no gate cleared by a method that estimates
nothing.

## Validation performed

Run with `authoring/assemble.sh` and `authoring/validate.sh`.

**The oracle reaches reward 1: all twelve gates.** Nine calls on the shipped
replicate, zero false positives against a cap of one. Recall 3 of 4
hard_complete, 2 of 2 soft_complete, 2 of 2 hard_partial, 2 of 2
soft_partial -- every call matched a true sweep. The right class on 7 of 9
matched calls against a 70% threshold; selection coefficients inside a
factor of 2.5 on 9 of 9, spanning a factor of 5.2 against a 2.0 threshold,
with a rank correlation of 0.850 against a 0.30 threshold; all three
localized hard_complete sweeps dated inside tolerance, which is the number
the gate requires.

Demography: ancestral size 1.03 times truth, current size 0.95, and
duration over depth 1.56. That last one is worth looking at next to what it
is made of: the fit put the bottleneck at 3889 individuals for 3026
generations where the truth is 800 for 400, so its DEPTH is off by a factor
of 4.9 while the combination the gate grades is off by 1.56. That is the
non-identifiability the gate exists for, in the oracle's own output.

The one sweep it misses is the one it should find hardest: the weakest
hard_complete sweep, at s=0.0085 and completed 1214 generations ago, whose
footprint is both narrow and half recovered. At a false-positive budget of
3.0 the pipeline does find and date it, at the cost of two false positives
and a class accuracy below the gate -- see the budget table above.

**It is deterministic.** Two independent runs produce byte-identical
submissions, md5 `be9f76f169bdd5fe20abf85c695117ef`, and identical
calibrated thresholds: a cluster-forming cut of 31.65 and a mass cut of
73.9.

**Twenty-eight submissions were run through the real verifier**
(`authoring/evidence/verifier_selftest.py`), graded against the shipped
`tests/truth.json` rather than a stand-in. Two pass: the perfect answer, and
the perfect answer plus one false positive, which confirms the cap is not
merely strict. Twenty-six fail, among them the four cases this redesign
exists for -- the best possible constant selection coefficient, found by
search rather than picked (0.0180, which satisfies the magnitude test on 8
of 10 and the ordering test on nothing), a constant coefficient with noise
added, the best possible constant completion time (142, which dates 2 of the
4 hard_complete sweeps where 3 are needed), and one constant time per class.
The rest: no file, an empty object, no sweeps, a carpet of calls every
100 kb, every call given one class in each of two ways, a constant
coefficient at 0.3 and at the true geometric mean, coefficients correct in
magnitude but reversed in rank, constant times at 300 and 1600, every time
doubled, a constant-size history, each of the four classes dropped entirely,
every position shifted 400 kb, a NaN coefficient, complete sweeps submitted
with no time at all, every call parked at a chromosome centre, and the
perfect answer plus three false positives.

**The gates are feasible and non-trivial, and the design guarantee holds.**
`authoring/evidence/gate_feasibility.py` reports four things. The shipped
estimators' accuracy on the calibration replicates, labelled in-sample
because that is what it is -- 109 of 109 inside a factor of 2.5 and 38 of 40
dated inside tolerance, which checks the constants were pasted in correctly
and nothing more; the honest held-out figures come from
`fit_estimators.py --emit`. The best a constant can do against the shipped
truth, over all 135 match sets the recall gates permit. How often a jittered
constant escapes the ordering gate, by match-set size. And -- over the
generator's acceptance WINDOWS rather than this one draw -- that no constant
can date three of the four hard_complete sweeps for any replicate the
generator would accept, which makes the timing gate a property of the design
rather than of this realization.

It also prints the limit of the timing gate: two well-chosen constants,
(160, 1060), date all four. With only four dateable sweeps no count-based
gate can exclude that, so the gate excludes ONE constant, which is what it
claims, and nothing stronger.

**The data carries no shortcut.** `authoring/evidence/leak_audit.py` passes
every check, including that no selected site appears as a variant at all.
Median SNP density is 104 per 50 kb at the sweeps against 189 in the
background, with no sweep window empty of variants.

**`task.toml` validates** against harbor's own task schema.

**The verifier container behaves.** Built from `tests/Dockerfile` and run
against the real image with `--network none`: the oracle's own output scores
`reward=1`, a perfect submission scores 1, an empty `/app` scores 0, a
malformed `results.json` scores 0, an empty JSON object scores 0, and
`/logs/verifier/ctrf.json` is written at the canonical path in every case,
naming all eleven gate functions -- the twelve pytest items collapse to
eleven names because the two demography-size parameters share one. A socket
call from inside the container fails with a resolver error, so it really has
no network, and `import numpy` fails, so the verifier really does run on
pytest and the standard library alone.

The same container scored the structurally independent solver's submission,
`authoring/evidence/indep_results.json`: `reward=0`, with the ctrf report
naming nine of the eleven gate functions passed and exactly two failed,
`test_false_positives` and `test_timing_of_completed_sweeps`. That is the
10-of-12 scorecard above, produced by the shipped verifier rather than by an
authoring script.

**Both harbor gates pass.** `harbor run -a oracle -e docker` returns reward
**1.0** and `harbor run -a nop -e docker` returns **0.0**, one trial each,
zero exceptions, through the real harness: environment image built,
`solve.sh` run inside it for eight minutes, `/app/results.json` copied out
as the declared artifact, and the separate verifier container grading it.

Those runs used a locally patched copy of the bundle. Two changes, neither
of which applies in the real pipeline and neither of which touches what the
gates exercise. First, the apt layer is dropped and the sandbox's proxy CA
injected, because this sandbox's egress policy reaches PyPI but not Debian
mirrors and pip inside a docker build does not trust the intercepting
proxy; the apt packages are standard bookworm names and are convenience
tooling the reference does not use. Second, each Dockerfile's base image is
replaced by one already in the local cache, because Docker Hub began
rate-limiting `python:3.12-slim-bookworm` (HTTP 429) partway through this
session. The cached images were built from these very Dockerfiles earlier
and carry the same pinned versions, numpy 2.5.3 through pandas 2.3.3 for
the environment and pytest 8.4.1 with pytest-json-ctrf 0.3.5 for the
verifier. The shipped bundle is untouched by either.

Not run here: `harbor check`, which needs a model API key this session does
not have.

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
