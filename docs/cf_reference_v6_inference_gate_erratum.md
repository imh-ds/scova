# SCOVA-CF v6 simultaneous-inference gate erratum

The first v6 aggregation applied a two-sided equality check to familywise
error in every focused cell. That criterion was invalid in two cases:

- a focused family with no true null contrast has structural familywise error
  zero, not the nominal alpha; and
- a simultaneous procedure that controls familywise error conservatively is
  valid, provided its error does not exceed the calibrated upper bound.

The r4 aggregation checks the upper error bound only for families containing
at least one true null. It retains the existing simultaneous-coverage and
null-omnibus checks. This is a gate-semantics correction, not a change to the
estimator, simulations, target population, support policy, or thresholds.

The completed v6 shard records are retained unchanged and may be reaggregated
under r4. The resulting evidence artifact records the r4 commit and preserves
the source-shard commit for audit.
