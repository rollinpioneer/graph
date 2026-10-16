from typing import Any
import jax
from agents.gfp import GFPAgent
from bfc import BFCAgent, target

class FrozenTD(BFCAgent):
    frozen_vabc: Any = None

    def td_parts(self, batch, rng):
        _, key = jax.random.split(rng)
        obs = batch['next_observations']
        vobs = obs if self.config['target_use_next'] else batch['observations']
        current = self.sample_actions(obs, key)
        p = dict(self.network.params)
        p['modules_actor'] = self.reference
        p['modules_vabc'] = self.frozen_vabc
        fixed = GFPAgent(self.rng, self.network.replace(params=p), self.config)
        reference = fixed.sample_actions(obs, key)
        flow = fixed.sample_flow_actions(vobs, key)
        qc, qr, qf = [self.critic_eval(obs, a) for a in (current, reference, flow)]
        y, yr, d = target(batch['rewards'], batch['masks'], self.config['discount'], qc, qr, qf, self.epsilon)
        return y, yr, d, qc, qr, qf
