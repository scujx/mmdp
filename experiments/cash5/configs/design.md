# Fixed route-mask comparison

The route-mask comparison evaluates Flat, True (MMDP), U1–U5 and C1–C5 under the same raw PPO,
continuous route/amount decoder, and executor. Each setting uses training seeds
42–51. All restricted masks contain six stage-zero routes; stage one retains the
same eight persistent routes.

## Mask construction

Generation uses only the public route catalogue and account roles, never episode
states or returns. The catalogue has fourteen ordered stage-zero routes. True
is the complement of the persistent subset. Route order always follows the
public catalogue.

C1–C5 preserve all three master-to-operating funding routes. Enumerate the 164
six-route alternatives satisfying that coverage, excluding True. For Ck, use
NumPy 2.0.1 PCG64 seed 982200+k and reject only previously accepted C masks.

U1–U5 sample from the 2997 six-route subsets remaining after excluding True and
the five accepted C masks. For Uk, use PCG64 seed 982100+k and reject only
previously accepted U masks. Topological diagnostics do not trigger redrawing.
`mask_inventory.json` records the draws and resulting route sets. Regeneration
checks that the resulting mask memberships match this configuration.

## Training and interpretation

Training uses the shared PPO settings and final 500,224-transition policies,
without validation-based endpoint selection. The test bank is
980000–980199. Statistics resample paired training seeds after averaging test
instances; alternative-mask groups are averaged within each training seed.

Equal route count holds selector bucket sizes and the STOP decoding interval
fixed, but does not fix learned action probabilities. C preserves direct funding,
not investment opportunities, reverse flow, optimal value, or all feasible flows.
The controls change route identities under a shared decoder and executor; they
are not a fixed-global-logit ablation or an isolated causal test of expiration.
