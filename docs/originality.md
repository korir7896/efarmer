# Originality memo

CrossPhase is original work.  Nothing in it is a copy, a rename, or a repackaging
of an existing public benchmark task.  This memo records what is new, what is
borrowed, and where the boundary sits.

## What is new

**The research question.** Existing invariant-learning benchmarks ask which
objective is most robust to a spurious correlation of a *given strength*.
CrossPhase asks a different question: when the shortcut-to-stable transition
happens at a *different point in training* in each setting, and when the noise
floor differs between them, can a single fixed objective be optimal everywhere?
The task is built so that the answer is no, and the gates in `gates/` measure
that rather than assuming it.

**The generator.** The stable feature is the sign of the cross-channel phase lag
of a core oscillation — invariant to global phase, to per-example gain, and to
anything added equally to both channels, but only recoverable by comparing the
two channels.  The two nuisance cues (a level cue and a band cue) are easier to
read than the stable one, which is what creates the shortcut.  We are not aware
of a published benchmark built on cross-channel phase; the closest relatives
(coloured MNIST, waterbirds, spurious-feature synthetic suites) all place the
shortcut in an additive or colour channel of a static image.

**The hidden shift.** Setting C changes the *mechanism* of the nuisance rather
than its strength: the level cue becomes a multiplicative gain and the band cue
becomes a monotone drift envelope, at a longer window, a weaker core and a higher
noise floor.  Gain preserves the core's shape while rescaling the logits, which
is what makes logit-magnitude and risk-variance methods behave differently there.
A shift of mechanism rather than of degree is the point; a setting that only
turned the same knob further would not test anything new.

**The scoring rule.** `Q = sqrt(I * min_j T_j)` over a world set that deliberately
mixes two kinds of stress — cue-flipped worlds that punish shortcut reliance and
core-attenuated / noise-raised worlds that punish over-regularisation — so that
the binding world is contested rather than fixed.  The binding-world audit in
`reports/` is the evidence, not an assumption.

**The public/proxy split.** The agent gets the training generators for A and B
and a set of *proxy* validation worlds at different coordinates with an
independent pool seed.  Proxy-to-official rank correlation is a measured,
reported, tunable quantity rather than an accident.

## What is borrowed, and cited

The six reproduced objectives are published methods, implemented from their
papers and cited in `agent/method_references.md`: ERM, IRMv1, V-REx, GroupDRO,
Spectral Decoupling and EQRM.  Using published baselines is the point of a
reproduction; the scaffold, the data, the shift and the evaluation around them
are ours.

Two grids are shifted from their papers' headline coefficient ranges.  The reason
is recorded in `crossphase/core/methods.py` and in `agent/method_references.md`,
and in both cases the shifted range still contains a published-scale endpoint.

## Boundaries we chose, and why

The objective contract exposes per-environment logits and targets only.  That
excludes Fishr, IGA, CORAL, MMD-alignment and IB-IRM by construction, because
each needs features, representations or per-parameter gradients.  That is a
boundary, not an oversight, and it is stated in `agent/method_references.md` so
that a reviewer can see it was a decision.
