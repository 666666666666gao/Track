"""Pure-Python policy boundaries; no model, transport or dataset is involved."""
import json
from cad_template_profiles import PRIMARY_PROFILES, get_profile, profile_dict


def main():
    checks = 0
    def verify(value):
        nonlocal checks
        assert value
        checks += 1
    verify(get_profile().name == 'U0')
    verify(PRIMARY_PROFILES == ('U0', 'U1', 'U2'))
    for frame in range(1, 101):
        for native in [.0, .65, .7, .75, .8, 1.]:
            for final in [.2, .65, .8]:
                for C in [.45, .5, .55]:
                    old = frame % 50 == 0 and native > .75 and C > .5
                    profile = get_profile('U0')
                    new = profile.qualifies(frame, native, final) and profile.accepts(C)
                    verify(new == old)
    verify(not get_profile('U0').qualifies(5, .8, .8))
    verify(get_profile('U1').qualifies(5, .8, .2))
    verify(not get_profile('U1').qualifies(5, .7, .99))
    verify(get_profile('U2').qualifies(5, .7, .2))
    verify(not get_profile('U2').qualifies(5, .65, .99))
    verify(not get_profile('U2').qualifies(6, .9, .99))
    verify(not get_profile('U2').accepts(.5))
    verify(get_profile('U2').accepts(.55))
    verify(not get_profile('U2').qualifies(5, .2, .99))
    verify(get_profile('U3').qualifies(5, .2, .7))
    verify(not get_profile('U3').qualifies(5, .9, .65))
    verify(not get_profile('U3').accepts(.45))
    verify(get_profile('U4').accepts())
    verify(not profile_dict('U4')['use_C'])
    result = dict(status='complete_pure_python_template_profile_boundaries', checks=checks,
        scope='policy logic only; U0 equality over 5400 synthetic scalar cases, plus strict boundaries',
        profiles={name:profile_dict(name) for name in ['U0','U1','U2','U3','U4']},
        real_tracking_calls=0, model_instances=0, NN_calls=0, GPU_queries=0,
        original_R0_progress_queries=0, evaluation_type='synthetic_policy_only')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
