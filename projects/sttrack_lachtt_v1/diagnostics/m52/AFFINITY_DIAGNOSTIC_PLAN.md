# M52 affinity readout diagnostic

Exploratory static analysis prepared after both M52 heads were trained, while
their frozen recursive evaluation was still running. It does not change the
tracker, its checkpoints, the training data, or the advancement gates.

Use the exact existing native and fixed-M45 policy caches and both trained
heads. Reuse the sealed-data audit. Run the head on CPU and require its
classifier choices to equal the already recorded GPU choices for every event.
Compare the classifier with the column of the learned affinity matrix indexed
by the actual previous-choice input. This causal readout receives no GT. An
unmatched result maps to candidate zero solely to match existing output
semantics; it does not implement a new lost-target state.

Separately, report a privileged readout from the previous GT candidate column,
restricted to events with such a previous candidate. Compare classifier,
causal readout, and oracle on the identical restricted subset. Label this as
oracle capacity, never as deployable performance. All event IoU metrics keep
the existing static denominator and unavailable-GT convention.

The purpose is to check whether the already learned auxiliary matching output
has useful target-selection information before designing online identity
propagation. There is no threshold fitting, no new training, and no public
evaluation. The fixed-M45 policy cache is not a newly generated trajectory
from either M52 head. This analysis is neither a KeepTrack reproduction nor
evidence that a persistent identity mechanism will work.
