# CrossPhase gate report

`S*` = **57.706**

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| Public-tuning ceiling | argmax on proxy scores <= S* | 51.109 vs 57.706 (IRM|lam=100,warm=0.3) | PASS |
| Reconstruction ceiling | argmax on rebuilt worlds <= S* | 51.109 vs 57.706 (IRM|lam=100,warm=0.3) | PASS |
| Proxy correlation | Spearman in [0.5, 0.8] | 0.251 over 77 configurations | FAIL |
| Transition separation | A-to-C differs by >= 20% of budget | 20% (A=520, B=110, C=360 of 800) | PASS |
| Warm-up anti-transfer | A/B-optimal warm-up suboptimal on C | max Q_C cost 0.1540; families differing: VREx, EQRM | PASS |
| Rank reversal | best baseline differs across >= 2 settings | A:IRM, B:IRM, C:VREx | PASS |
| Penalty anti-transfer | coefficient argmax differs A/B vs C | VREx, SD | PASS |
| Loss-scale routes | mid-run scale change must not beat S* | uniform 0.050/decade; best mid-run probe 43.32 (+5.93 over ERM, 14.38 below S*) | PASS |
| Binding-world audit | no world binds > 90% of cells | (0.1, 0.1, 1.0, 1.0) 91%, (0.1, 0.9, 1.0, 1.0) 3%, (0.9, 0.9, 1.0, 0.45) 3% | FAIL |
| Fingerprint policy | declared before trials | legal; settings separable at 95%, branch probe S=43.087 | PASS |
| Headroom | reference solution beats S* | 60.544 (+2.838) | PASS |

## Cue recovery from public data

| Setting | z1 | z2 | median-split cap |
|---|---|---|---|
| A | 1.000 | 0.994 | 1.000 |
| B | 0.998 | 0.995 | 0.996 |
| C | 0.996 | 1.000 | 0.992 |

Both nuisance cues are recoverable from raw signals essentially perfectly, so the reconstruction route is fully live and is assumed available to any competent agent.  It is gated, not prevented.

## Combination rule

Across the 17 competent configurations, the geometric mean spreads scores over 11.275 points (sd 3.574); an outer minimum over settings spreads them over 17.893 (sd 4.514).  The geometric mean is kept because it discriminates between methods better, not because it scores lower.

