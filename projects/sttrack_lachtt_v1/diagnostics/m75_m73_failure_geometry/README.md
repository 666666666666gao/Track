# M75 post-result CPU search-geometry census

All 71 M73 Category H10 onsets on reused Train development22 are included. Search geometry is reconstructed from the prior public bbox using the actual native ceil/round crop convention. No new tracking calls, optimizer steps or seeds.

20/71 target centers and 17/71 complete GT boxes are inside the Category crop; 58 onsets follow invalid GT. These annotation-based H10 segments need not be independent first failures. Invalid GT is not an absence label. This differs from the old VOT multi-start failure census.

The M74 strict content harms form 8 unique physical intervals: 4 center-inside and 4 center-outside. Each content trajectory has its own history and crop. This comparison is not a same-state prediction-head intervention, and geometric coverage does not prove a correct candidate exists.

No public evaluation or independent model-review PASS is authorized. See the full event table in result.json and the project master section 5.142 for the next diagnostic and training direction.
