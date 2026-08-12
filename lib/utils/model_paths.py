"""Resolve local pretrained assets without coupling configs to one project copy."""

import os


def resolve_model_reference(reference, env_var='', search_dir='', local_only=True, label='model'):
    """Resolve an environment override, absolute path, or project-relative asset."""
    override = os.environ.get(env_var, '').strip() if env_var else ''
    candidate = override or str(reference or '').strip()
    if not candidate:
        raise FileNotFoundError(
            '{} is not configured{}'.format(label, ' (set {})'.format(env_var) if env_var else ''))

    candidate = os.path.expanduser(os.path.expandvars(candidate))
    if not os.path.isabs(candidate) and search_dir:
        local_candidate = os.path.abspath(os.path.join(search_dir, candidate))
        if os.path.exists(local_candidate) or local_only:
            candidate = local_candidate

    if local_only and not os.path.exists(candidate):
        hint = ' Set {} to the existing local path.'.format(env_var) if env_var else ''
        raise FileNotFoundError('{} does not exist: {}.{}'.format(label, candidate, hint))
    return candidate
