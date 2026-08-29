"""Baseline-first counterfactual veto for SUTrack template updates.

The direct identity-only tracker immediately commits a safe-v1 template
candidate.  This tracker therefore keeps that exact new-template branch as
the public/protected path while rolling the old-template state forward only
as a counterfactual.  The controller may promote the counterfactual branch
after two consecutive future-frame advantages, which semantically vetoes a
harmful template update.  Every hold, timeout, conflict, or error stays on the
direct-baseline branch.
"""

from lib.test.tracker.sutrack_transaction import (
    SUTRACKProtectedTransaction,
)


class SUTRACKBaselineFirstTemplateVeto(SUTRACKProtectedTransaction):
    """Preserve direct-baseline behavior unless an old-template veto wins."""

    @staticmethod
    def _orient_template_transaction_snapshots(
            old_template_snapshot, new_template_snapshot):
        return new_template_snapshot, old_template_snapshot

    def _write_transaction_trace(self, payload):
        enriched = dict(payload)
        enriched['branch_semantics'] = {
            'protected': 'direct_baseline_new_template',
            'tentative': 'counterfactual_old_template',
            'promote': 'veto_template_update',
            'rollback': 'keep_direct_baseline_update',
            'hold_public_output': 'direct_baseline_new_template',
        }
        super()._write_transaction_trace(enriched)


def get_tracker_class():
    return SUTRACKBaselineFirstTemplateVeto


__all__ = [
    'SUTRACKBaselineFirstTemplateVeto',
    'get_tracker_class',
]
