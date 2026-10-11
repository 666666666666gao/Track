"""User-defined two-F-score selection and consecutive dual-decline early stop."""


def next_decision(previous, current, prior_streak, best_score):
    assert len(previous) == len(current) == 2 and 0 <= prior_streak < 2
    dual_decline = all(now < before for now, before in zip(current, previous))
    streak = prior_streak + 1 if dual_decline else 0
    score = sum(current) / 2
    return dict(dual_decline=dual_decline, consecutive_dual_declines=streak,
                stop=streak == 2, score=score, replace_best=best_score is None or score > best_score)
