# M95 initial-feature origin diagnostic

M90 froze its initial RoI from `NativeReferenceBank.before_decision` in the
first noninitial tracking forward. The pixels of that template come from
the first frame, but the post-TSG representation is computed jointly with
frame1 search tokens. It is not a t0-only encoding. M91 compares that template
representation with current search representations. Its pure cosine result
(8/495 correct versus native268) does not establish a lack of initial visual
identity; it may also depend on differing feature origins. This is an observed
implementation distinction, not a proven cause of M91's weak result.

Without changing the M90 current candidate cache or native trajectory, collect
two references for all152 legal Train initializations in an auxiliary t0
forward: the initial-template RoI and the t0-search RoI. Both use the same
initial RGB-D image and legal initialization box, zero initial query, frozen
native STTrack, and existing 4×4 bilinear RoI sampling. Do not read subsequent
GT or semantic labels during collection. Do not commit the auxiliary query.
Verify frame0 state is unchanged and the first real tracking frame reproduces
the saved native box and score. Use two GPU shards, one-sequence smoke first.

Then, on the identical M90 Top-10 candidates and Train labels, report pure
mean-RoI RGB-D cosine choice for cached first-use template, t0 template, and
t0 search references. Give fit130/development22 counts, native/Top10 coverage,
rescues and breaks, and all strata. This is a fixed-state feature-origin
diagnostic, not a language effect, learned identity model, recursive tracker,
or official nine-metric result. It cannot replace human phrase review.

If neither t0 reference improves candidate choices, do not launch another
visual-selector training merely because references are cleaner. If a t0
origin gives a measured improvement, compare it in a same-budget visual
control before letting it enter A+B. No public test images, new search range,
memory writer, geometric head, or text-bank change enters M95.
