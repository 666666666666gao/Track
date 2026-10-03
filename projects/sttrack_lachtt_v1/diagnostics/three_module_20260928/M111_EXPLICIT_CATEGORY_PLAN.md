# M111: explicitly supervised initial-category evidence

M110 completed both fixed-state arms, but reviewed descriptions had no net
selection benefit over the same final's Empty condition. Module A's existing
three-channel evidence MLP has never received labels for those channels. Only
indirect localization gradients have trained it. This is a missing task from
the existing Stage1 A+B plan, not evidence that larger language weights help.

The 24 candidate-review category labels refer to the old generated category.
All24 strings differ from the corrected full slot0 phrase used in M107-M110;
some differences change meaning, others are refinements. Do not inherit those
labels onto corrected phrases, nor interpret low IoU as a physical identity
negative. No candidate support labels are used in M111.

Bind each initialization-review status to the exact original `generated_category`
string it rates. Restrict supervision to `first_frame_category_observability=clear`.
The authorized model review yields98 fitting queries (80 supported,17 conflicting,
1 uncertain) and19 held-out queries (15,3,1). Other35 initialization reviews are
excluded from this category task. These are model weak labels; no additional
human-confirmed file has been received. Review notes, later frames and
video-only attributes never enter the model. Future/multiframe review evidence
can make these weak labels imperfect even with the clear filter.

Reuse the M101 visual parent, M108 bank, and M110 frozen parameter list. Both
arms receive the same reviewed descriptions for candidate selection, the same
seed2027,2544 fitting states,12 epochs/480 updates,batch64,AdamW3e-4 and fixed
final checkpoint. The two arms differ only in whether the new unweighted
three-class category CE participates in optimization (weight0 versus weight1).
Original localization BCE plus native candidate preservation remain unchanged;
no A/B rank term or quality loss is added. Both arms compute the same extra
category forward for logging. Weight0 returns the original localization plus
preservation loss directly, without a zero-weight extra autograd branch.

Each update additionally draws32 queries uniformly with replacement from the98
fitting records using an isolated seeded CPU generator. It reads only the cached
immutable initial RoI/context/masks/depth validity, using that same initial
region as its current observation. Single original category occupies slot0;
other slots are masked. It supervises the existing absolute evidence logits,
not their Empty difference. The only prototype change returns an already
computed tensor; parameters and old candidate calculations stay unchanged.
Both arms have identical query draws and no development-label gradients.

This task is initialization category verification. It is not a label for later
candidate identity, and one uncertain fitting example cannot establish a
general visibility/abstention capability. The corrected descriptions remain
different inputs from original category queries; whether the task transfers to
candidate selection is precisely what the experiment must measure.

GPU0 runs weight0 and GPU1 weight1, first separate3-update sanity jobs, then
the two finals. Reuse Python3.8/Torch1.13.1+cu116/RTX3090; install nothing.
Assert frozen parameters/buffers exact and Empty score/quality exact on all3039
cached states. Sanity reports gradients and integrity only, no development
category accuracy or checkpoint. Original M107-M110 artifacts remain untouched.

At the fixed final, report category CE and class-wise confusion/recall on98 fit
and19 development sequences, candidate selection on495 held-out states, all
Empty/generic/reviewed content conditions, paired changes versus own Empty,
rescues, harms and old four capacity checks. Category accuracy alone cannot
authorize promotion. Semantic selection must improve over own Empty without
new healthy failures before considering recursion. There is no automatic
promotion, recursive training, CDTB/VOT fitting, public benchmark evaluation,
seed scan, coefficient search or best-epoch selection in this run.

Also report the constant majority-class reference (80/98,15/19) and the
same-query/different-label subsets (21 fit,3 development). These diagnostics
test whether aggregate accuracy hides category-prior collapse; their small
size and model-label uncertainty preclude a general instance-identity claim.

Raw CSV, private manifest and text banks stay outside public Git. Publish only
source, input hashes/counts and aggregate results. Source review and GPU sanity
must pass before final jobs. Store small new outputs on `/root/autodl-tmp`;
the almost-full root overlay precludes creating another large cache. Runtime
estimate2-3minutes including encoding/sanity, based on M110's49-second pair.
First completion observation after180seconds; no repeated GPU polling.
