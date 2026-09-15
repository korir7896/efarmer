# CrossPhase gate report

`S*` = **59.209**

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| Public-tuning ceiling | argmax on proxy scores <= S* | 58.783 vs 59.209 (IRM|lam=100000,warm=0.65); per-family proxy selection reaches 58.783 | PASS |
| Reconstruction ceiling | argmax on rebuilt worlds <= S* | 51.109 vs 59.209 (IRM|lam=100,warm=0.3) | PASS |
| Proxy correlation | Spearman in [0.5, 0.8] | 0.502 over 106 configurations | PASS |
| Transition separation | A-to-C differs by >= 20% of budget | 14%; all three differ A 0.65, B 0.50, C 0.51 of budget (+/- 0.013 probe resolution) | FAIL |
| Warm-up anti-transfer | A/B-optimal warm-up suboptimal on C | max Q_C cost 0.0967; families differing: IRM, VREx, GroupDRO, SD, EQRM | PASS |
| Rank reversal | best baseline differs across >= 2 settings | A:IRM, B:IRM, C:VREx | PASS |
| Penalty anti-transfer | coefficient argmax differs A/B vs C | IRM, SD, EQRM | PASS |
| Loss-scale routes | mid-run scale change must not beat S* | uniform 0.050/decade; best mid-run probe 43.32 (+7.53 over ERM, 15.89 below S*) | PASS |
| Binding-world audit | no world binds > 90% of cells | (0.1, 0.1, 1.0, 1.0) 86%, (0.9, 0.9, 1.0, 0.45) 4%, diverged 4% | PASS |
| Fingerprint policy | declared before trials | legal; settings separable at 88%, branch probe S=45.212 | PASS |
| Headroom | reference solution beats S* | 64.518 (+5.309); nuisance-free ceiling 73.896 | PASS |

## Cue recovery from public data

| Setting | z1 | z2 | median-split cap |
|---|---|---|---|
| A | 1.000 | 0.994 | 1.000 |
| B | 0.998 | 0.995 | 0.996 |
| C | 0.996 | 1.000 | 0.992 |

Both nuisance cues are recoverable from raw signals essentially perfectly, so the reconstruction route is fully live and is assumed available to any competent agent.  It is gated, not prevented.

## Binding-world audit

The cue-flipped world binds the minimum in 86% of measured cells, against a 90% limit, so this gate **passes**.  Stated plainly either way, because the margin is not large: `Q` is in practice close to `sqrt(I * T_flip)`.

The remaining worlds are not decorative -- they bind in the other cells, and they bind for the strongest methods, which is where a worst-case minimum has to bite.  Diverged runs are excluded from the denominator: counting a run that produced no measurement as evidence about which world binds once made this audit read 64% when 32% of its cells were failures.

The known remedy if this drifts back over the limit is to tighten the core-attenuation world from 0.45 to 0.30, which is measured to bind far more often against IRM in setting C.  It costs a full re-sweep and gate re-run, and is recorded here rather than applied.

## Combination rule

Across the 28 competent configurations, the geometric mean spreads scores over 6.147 points (sd 1.913); an outer minimum over settings spreads them over 9.285 (sd 3.287).  On raw spread the **outer minimum is the wider of the two**, and an earlier draft of this report claimed the opposite while printing these same numbers.

Raw spread is not decisive either way -- it is not normalised by the seed-level noise within a configuration, so a wider spread can be scale rather than signal.  The geometric mean is kept on a different ground, which does not depend on the comparison above: an outer minimum over settings reports only the worst setting and discards the other two entirely, so an objective that is excellent on A and B and mediocre on C scores identically to one that is mediocre everywhere.  That is precisely the distinction this task exists to make -- the reference solution's whole advantage is holding A and B while improving C -- and an outer minimum would be blind to it.  It would also compound with the minimum already taken inside each `Q` over target worlds, applying a worst-case twice.

