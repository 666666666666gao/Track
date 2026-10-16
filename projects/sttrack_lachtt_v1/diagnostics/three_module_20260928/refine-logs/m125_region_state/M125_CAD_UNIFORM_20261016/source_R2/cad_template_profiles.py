"""One dataset-independent template profile per complete three-dataset run."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TemplateProfile:
    name: str
    interval: int
    response_threshold: float
    response_source: str
    use_C: bool
    C_threshold: float = .5

    def qualifies(self, frame, native_same_position_response, final_selected_response):
        response = native_same_position_response if self.response_source == 'native_selected' else final_selected_response
        return frame % self.interval == 0 and response > self.response_threshold

    def accepts(self, prediction=None):
        if not self.use_C:
            return True
        assert prediction is not None
        return prediction > self.C_threshold


PROFILES = {
    'U0': TemplateProfile('U0', 50, .75, 'native_selected', True),
    'U1': TemplateProfile('U1', 5, .75, 'native_selected', True),
    'U2': TemplateProfile('U2', 5, .65, 'native_selected', True),
    'U3': TemplateProfile('U3', 5, .65, 'final_selected', True),
    'U4': TemplateProfile('U4', 5, .65, 'final_selected', False),
}
PRIMARY_PROFILES = ('U0', 'U1', 'U2')


def get_profile(name='U0'):
    return PROFILES[name]


def profile_dict(name):
    return asdict(get_profile(name))
