export const INITIAL_ROUND = 'initial';

export function answerStorageKey(dataset, reviewer, id, round) {
  const base = `rgbd-review-v1:${dataset}:${reviewer}:${id}`;
  return round === INITIAL_ROUND ? base : `${base}:${round}`;
}

export function newestAnswer(local, published) {
  if (!local.updated_at) return published;
  if (!published.updated_at) return local;
  return Date.parse(local.updated_at) >= Date.parse(published.updated_at) ? local : published;
}

export function savedAnswer(item, manifest, reviewer, storage) {
  const local = JSON.parse(storage.getItem(answerStorageKey(manifest.dataset, reviewer, item.id, manifest.human_review_round)) || '{}');
  const published = item.human_reviews[reviewer] || {};
  const answer = newestAnswer(local, published);
  return {...answer, _published:answer === published && Boolean(answer.status)};
}
