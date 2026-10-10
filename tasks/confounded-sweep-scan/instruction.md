The data in /app/data come from one diploid population: a phased VCF of
resequenced individuals with the ancestral allele annotated at every site,
and a piecewise-constant recombination map. /app/data/README_DATA.txt
documents the file formats and lists the parameters that may be treated as
known, among them the mutation rate and the fitness convention that fixes
what a selection coefficient means.

Two facts about these data are not recoverable from the files alone. The
population passed through a bottleneck and later grew. And a number of sites
in the genome have been targets of recent positive selection, but the
selected sites themselves are not in the VCF: a variant that reached
fixation is monomorphic in the sample and has been filtered out, as have
multiallelic sites.

Recover the demographic history and the selected sites, and write both to
/app/results.json.

## Definitions

These are the definitions the answer is graded against.

* A sweep is **complete** if the beneficial allele reached a frequency of
  0.98 or more in the population, and **partial** if it is still segregating
  below that frequency today.
* A sweep is **soft** if two or more independent origins of the beneficial
  allele -- separate mutations at the same site, each arising on its own
  haplotype background -- are still present in the population today, and
  **hard** if exactly one is.

Crossing the two gives the four classes `hard_complete`, `hard_partial`,
`soft_complete` and `soft_partial`. Nothing in these labels refers to a
sweep's age.

* The **selection coefficient** `s` is per-copy and additive, under the
  fitness convention given in README_DATA.txt.
* A complete sweep's **completion time** is the number of generations before
  the present at which the beneficial allele first reached frequency 0.98.
* `T_bottleneck_start_gens_ago` and `T_bottleneck_end_gens_ago` are the
  generations before the present at which the population size dropped to
  `N_bottleneck` and at which it rose from it.

## Output

/app/results.json must be a single JSON object with two keys.

`"demography"`: an object with the numeric fields `"N_ancestral"`,
`"N_bottleneck"`, `"N_current"`, `"T_bottleneck_start_gens_ago"` and
`"T_bottleneck_end_gens_ago"`. All five must be positive numbers.

`"sweeps"`: a list with one object per selected site reported, each with

* `"chrom"`: integer chromosome number, 1-based;
* `"position"`: 1-based position in base pairs within that chromosome;
* `"class"`: exactly one of the four class names above;
* `"selection_coefficient"`: a number strictly between 0 and 1;
* `"time_gens_ago"`: the completion time in generations. Required for every
  sweep reported as `hard_complete` or `soft_complete`; ignored, and may be
  omitted or null, for the two partial classes.

For example:

```json
{"demography": {"N_ancestral": 10000, "N_bottleneck": 1200, "N_current": 9000,
                "T_bottleneck_start_gens_ago": 3400,
                "T_bottleneck_end_gens_ago": 3000},
 "sweeps": [{"chrom": 2, "position": 4531000, "class": "soft_complete",
             "selection_coefficient": 0.03, "time_gens_ago": 180}]}
```

## How the answer is graded

Every one of the following must hold. All are measured against the
simulation's recorded truth.

**Demography.** `N_ancestral` and `N_current` must each be within a factor of
1.6 of the truth. The bottleneck is graded through
`(T_bottleneck_start_gens_ago - T_bottleneck_end_gens_ago) / N_bottleneck`,
which must be within a factor of 2.5 of the truth; its depth and its
duration are only jointly identifiable from a single present-day sample, and
its absolute timing is not recoverable from these data at all, so neither is
graded on its own. The reported history must also be a real contraction and
recovery: `N_bottleneck` below both other sizes, and
`T_bottleneck_start_gens_ago` greater than `T_bottleneck_end_gens_ago`.

**Localisation.** Reported sweeps closer together than 250 kb are collapsed
to one before grading. A reported sweep is counted as finding a true one if
it is on the same chromosome and within 250 kb of it; each true sweep can be
found by at most one report and vice versa. At least three of the true
`hard_complete` sweeps and at least one true sweep of each of the other three
classes must be found. At most one reported sweep may fail to match a true
sweep; a second unmatched report fails the task, whether it sits in a
recombination cold spot or anywhere else.

**Class.** Of the reported sweeps that matched a true sweep, at least 70% must
carry the right class.

**Selection coefficient.** Over the same matched sweeps, at least 70% of the
reported coefficients must be within a factor of 2.5 of the truth, and the
rank correlation between the reported and the true coefficients must be at
least 0.50. The true coefficients span a factor of more than nine, so one
number repeated cannot satisfy both.

**Completion time.** Graded over the matched sweeps whose true class is
`hard_complete`, using the true class rather than the reported one. At least
three of them must be dated to within 50% of the true completion time, or
within 150 generations, whichever is larger.

The network is open; use whatever population-genetics software you prefer.

You have 21600 seconds to complete this task. Do not cheat by using online
solutions or hints specific to this task.
