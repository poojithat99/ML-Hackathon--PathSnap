# PathSnap experiment audit

- The original system checks raw inputs, projects coordinates to local metres,
  reranks 48 midpoint neighbours by point-to-segment distance, and scores the
  nearest eight directed arcs with four features. Five classifiers share
  trip-grouped folds; the selected logistic scorer is followed by Viterbi.
- Reference arcs are an ordered route, not point labels. Nearest-reference
  geometry ignores that order and can mislabel loops, intersections, and
  overlapping directions. Distance also influences the reference proxy itself.
  Freeze this proxy for paired comparisons; ask organizers about official
  scoring and investigate ordered alignment separately, not as a metric change.
- On training, 189,554 of 208,417 observations had their proxy arc in the
  baseline shortlist (90.95% point-weighted coverage). There were 1,667,336
  candidate rows, 11.37% positive before fit subsampling. Trip-local proxy
  classes ranged from 56 to 2,095 (median 247.5); arc IDs are not global.
- A midpoint search can miss long arcs. The instantaneous heading is noisy
  when stopped or sampling is irregular. GPS speed is shared by all candidates
  at a point and cannot change plain logistic top-1 ranking by itself.
- The original tuned classifier's student-trip agreement ranged from 56.8%
  (00000077) to 88.9% (00000007). Student-test macro F1 and CV macro F1 are
  distinct. Baseline CV weights folds equally; student metrics weight trips
  equally. Continuity excludes time steps above a fixed 15 s.
- Viterbi uses log candidate probabilities and a flat gap penalty for arcs
  that neither stay nor join directly. It does not insert intermediate arcs.
  Continuity can increase at the expense of fitting the observations.
- Original grouped folds isolate trips and fit scaling inside each fold.
  Candidate features and inference do not read reference routes. Keep reserved
  evaluator data untouched and select every new configuration on training CV.
- A local training-split pass took ~2.0 s for file I/O and candidate features
  together, ~2.6 s for proxy alignment. Candidate windows scale with shortlist
  size, and dense decoder transitions scale quadratically in candidate width;
  cache datasets and out-of-fold probabilities within each experiment run.
- XGBoost is installed but cannot import here because its native library
  requires unavailable libomp. Keep the experiment runnable without it.
