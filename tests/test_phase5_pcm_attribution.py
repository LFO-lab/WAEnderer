"""Coverage follows PCM generations, including transition and underrun edges."""
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pytest
from eval_scripts.benchmark_dual_inference import ObservedPlayer


@pytest.mark.parametrize('after_generation,transition,underrun,credited', [
    (7,'idle',False,True), (8,'idle',False,False),
    (7,'crossfading',False,False), (7,'idle',True,False),
])
def test_callback_credits_only_stable_audible_generation(after_generation,transition,underrun,credited):
    before = dict(generation=7,transition_status='idle')
    after = dict(generation=after_generation,transition_status=transition)
    states = iter([before,after])
    player = ObservedPlayer.__new__(ObservedPlayer)
    player.__dict__.update(dict(underruns=0,buffer_underruns=0,device_underruns=0,
        get_state=lambda:next(states),generation_scenarios={7:'manual:T8',8:'manual:T32'},
        callback_times=[],underrun_events=[],nonfinite=False,blocks=0,rendered_samples=0,
        scenario_samples=defaultdict(int),excluded_samples=0,peak=0.,sum_square=0.))
    def render(self,output,frames,time_info,status):
        output.fill(.1)
        if underrun:
            self.underruns += 1
            self.buffer_underruns += 1
            # The event snapshot does not consume the final attribution state.
            self.get_state = lambda:after
    with patch('eval_scripts.benchmark_dual_inference.DecoderPlayer._callback',render):
        ObservedPlayer._callback(player,np.empty((16,2),np.float32),16,None,None)
    assert player.scenario_samples.get('manual:T8',0) == (16 if credited else 0)
    assert player.scenario_samples.get('manual:T32',0) == 0
    assert player.excluded_samples == (0 if credited else 16)
