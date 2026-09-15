# CrossPhase gate report

`S*` = **57.706**

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| Public-tuning ceiling | argmax on proxy scores <= S* | 45.562 vs 57.706 (VREx|lam=1000,warm=0.5); per-family proxy selection reaches 57.140 | PASS |
| Reconstruction ceiling | argmax on rebuilt worlds <= S* | 51.109 vs 57.706 (IRM|lam=100,warm=0.3) | PASS |
| Proxy correlation | Spearman in [0.5, 0.8] | 0.575 over 77 configurations | PASS |
| Transition separation | A-to-C differs by >= 20% of budget | 20% (A=520, B=110, C=360 of 800) | PASS |
| Warm-up anti-transfer | A/B-optimal warm-up suboptimal on C | max Q_C cost 0.1540; families differing: VREx, EQRM | PASS |
| Rank reversal | best baseline differs across >= 2 settings | A:IRM, B:IRM, C:VREx | PASS |
| Penalty anti-transfer | coefficient argmax differs A/B vs C | VREx, SD | PASS |
| Loss-scale routes | mid-run scale change must not beat S* | uniform 0.050/decade; best mid-run probe 43.32 (+5.93 over ERM, 14.38 below S*) | PASS |
| Binding-world audit | no world binds > 90% of cells | (0.1, 0.1, 1.0, 1.0) 91%, (0.1, 0.9, 1.0, 1.0) 3%, (0.9, 0.9, 1.0, 0.45) 3% | FAIL |
| Fingerprint policy | declared before trials | legal; settings separable at 95%, branch probe S=37.512 | PASS |
| Headroom | reference solution beats S* | 60.544 (+2.838) | PASS |

## Cue recovery from public data

| Setting | z1 | z2 | median-split cap |
|---|---|---|---|
| A | 1.000 | 0.994 | 1.000 |
| B | 0.998 | 0.995 | 0.996 |
| C | 0.996 | 1.000 | 0.992 |

Both nuisance cues are recoverable from raw signals essentially perfectly, so the reconstruction route is fully live and is assumed available to any competent agent.  It is gated, not prevented.

## Binding-world audit

The cue-flipped world binds the minimum in 91% of cells, one point over the 90% limit, so **this gate fails and the threshold has not been moved to suit it**.  Stated plainly, as the design requires when one world dominates: `Q` is in practice `sqrt(I * T_flip)`.

The remaining worlds are kept rather than dropped because they are not decorative -- they bind in the other cells, and they bind for IRM and V-REx, the two strongest methods, which is exactly where a worst-case minimum has to bite.  The alternative fix, tightening the core-attenuation world from 0.45 to 0.30, is measured to bind far more often against IRM in setting C and would likely clear the gate; it needs a full re-sweep and gate re-run to move one point on a heuristic threshold, and is recorded here as the known remedy rather than applied.

## Combination rule

Across the 17 competent configurations, the geometric mean spreads scores over 11.275 points (sd 3.574); an outer minimum over settings spreads them over 17.893 (sd 4.514).  The geometric mean is kept because it discriminates between methods better, not because it scores lower.

