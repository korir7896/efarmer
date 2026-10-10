# Tuning the sweep origins in context

`tune_trajectories.py` measures a sweep's trajectory on a one-megabase genome
carrying that sweep alone, because a trajectory depends on N and s rather
than on genome length and that makes replication affordable. For the
completed sweeps that holds. For the partial ones it does not, and the
shipped replicate is a forty-eight-megabase genome carrying ten sweeps, four
of them stronger than the one being measured.

`generate.py --forward-only` runs the forward phase of the real
configuration, prints every sweep's realized outcome and stops, skipping
recapitation and the VCF. The acceptance windows in `config.ACCEPT` are set
from these runs.

## Three independent draws of the configuration as first written

`--seed-offset 0`, `2` and (the pre-fix) `1`, with sw09 at twelve origins:

| sweep | s | origin | offset 0 | offset 2 |
| --- | --- | --- | --- | --- |
| sw01 hard_complete | 0.0800 | 280 | 90 gens ago | 109 |
| sw02 hard_complete | 0.0300 | 710 | 330 | 397 |
| sw03 hard_complete | 0.0085 | 2300 | 1214 | 1179 |
| sw04 hard_complete | 0.0250 | 2500 | 2097 | 1946 |
| sw05 soft_complete | 0.0180 | 650 | 174, 4 origins | 129, 6 origins |
| sw06 soft_complete | 0.0450 | 380 | 165, 6 origins | 295, 4 origins |
| sw07 hard_partial | 0.0140 | 530 | freq 0.063 | freq 0.702 |
| sw08 hard_partial | 0.0350 | 230 | freq 0.369 | freq 0.479 |
| sw09 soft_partial | 0.0110 | 620 | freq 0.000, 1 origin | freq 0.861, 1 origin |
| sw10 soft_partial | 0.0600 | 130 | freq 0.723, 3 origins | freq 0.754, 3 origins |

Every completion time is inside its window in both draws, and the one-sweep
tuning predicted them closely. The partial sweeps are another matter:
sw07 ranged from 0.063 to 0.702 where eight isolated replicates at the same
origin spanned 0.112-0.833 with a median of 0.599.

sw09 failed in both draws for the same reason, and it is arithmetic rather
than luck. A founding lineage escapes drift with probability about 2s, so
the introductions a soft sweep needs scale as 1/(2s). At s=0.011 twelve
origins leave about 0.26 expected survivors, and both draws left exactly
one, which makes the sweep hard by the definition the verifier grades
against. Twelve is enough for sw10 only because 2s is five times larger
there.

## After sw09 was given 150 origins

`--seed-offset 3`, with sw09 at 150 origins and its origin at 560:

| sweep | realized | |
| --- | --- | --- |
| sw01 | hard_complete | 116 gens ago |
| sw02 | hard_complete | 362 |
| sw03 | hard_complete | 1108 |
| sw04 | hard_complete | 2105 |
| sw05 | soft_complete | 121, 2 origins |
| sw06 | **hard_complete** | 294, **1 origin** |
| sw07 | hard_partial | freq 0.823 |
| sw08 | hard_partial | freq 0.875 |
| sw09 | soft_partial | freq 0.801, 3 origins |
| sw10 | soft_partial | freq 0.664, 3 origins |

sw09 realizes as scheduled. The single failure in this draw is sw06, whose
surviving origins came out at 6, 4 and 1 across the three draws: a soft sweep
stops being soft once it has been complete for a few hundred generations,
because the surviving beneficial haplotypes drift and one takes over.

## What this means for how long generation takes

Per-sweep acceptance is around 1.0 for the four completed hard sweeps and
for sw05 and sw10, and around two thirds for sw06, sw07 and sw08, so a whole
pass is accepted roughly three times in ten. The forward driver backtracks
to the earliest failing sweep rather than discarding the replicate, so this
converges in a handful of restarts rather than needing a three-in-ten draw
of ten independent sweeps at once.

The windows are deliberately no tighter than the timing gate's
non-triviality requires, since rejecting an early sweep replays every sweep
after it: `gate_feasibility.design_guarantee()` is what checks that they are
still tight enough.
