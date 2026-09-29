# T_P_SR V1 geometry / mechanism review (saved data only)

No new environment, provider, reset, skill, physical attempt, representation forward, RL or optimizer was used.

## What the bins are

Pool configs: 64; physically tested scenes: 24. Configs with nonzero camera projected overlap: pool 7, physical 3.

| bin | pool | physical | xy distance min..max | clearance proxy min..max | overlap min..max | direct | relocation |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 6 | 0.1519..0.2048 | 0.0919..0.1448 | 0.0..0.0 | 12/12 | 4/12 |
| 1 | 16 | 6 | 0.1054..0.1498 | 0.0454..0.0898 | 0.0..0.0 | 10/12 | 6/12 |
| 2 | 16 | 6 | 0.0770..0.1046 | 0.0170..0.0446 | 0.0..0.08562676479739051 | 12/12 | 12/12 |
| 3 | 16 | 6 | 0.0509..0.0743 | -0.0091..0.0143 | 0.0..0.3249358045666608 | 12/12 | 4/12 |

The field `occlusion_bin` is a distance bin. It is kept unchanged in old CSVs; the derived column `distance_bin_originally_named_occlusion` is used here.

## Three notions that must not be merged

1. Centre distance near (recorded). 2. Camera occlusion of the target (only a projection proxy recorded). 3. Real gripper approach/descend/close/lift interference (not recorded and not measurable from saved fields). `approach_clearance_proxy = centre_distance - 0.06` is arithmetic on the centre distance and is not a gripper collision distance. Zero projected overlap therefore does not imply that mechanical interference is absent.

Helpful scenes: ['T_P_SR_pool_33'].

## Read-only look at SkillExecutor._pick (hypotheses, not findings)

Source sha256 06d3741723c5236afc9e1234a3229164b33e1609114fcfa82d023a672fd37ff6. Sequence: hover above object xy at hover z, descend to grasp z, press 8 mm lower, close hold, lift to the fixed show pose (0.02, -0.08, table_top+0.16), refresh perception.

- Hover (APPROACH): the vertical hover above the target xy may be affected by an interferer only if it is tall enough to reach hover z; hypothesis.
- Descend/press (INTERACT): finger opening axes near the target xy may collide with a nearby interferer; depends on the finger axis versus the target-interferer axis.
- Close hold: closing fingers may push an interferer within finger travel; hypothesis.
- Lift to show pose: a held target could drag or hit a neighbour; hypothesis.
- Which phase a given `axis` (x or y) affects is unverified: no sub-phase trace or contact evidence was saved.

```python
    def _pick(self, trace, obj, timeout, sim_start):
        xyz = self._object_xyz(obj)
        hover = np.array([xyz[0], xyz[1], self._hover_z()])
        grasp = np.array([xyz[0], xyz[1], self._grasp_z(float(xyz[2]), OBJECT_HALF[2])])
        press = np.array([grasp[0], grasp[1], grasp[2] - 0.008])
        if not self._move_to(trace, hover, GRIP_OPEN, timeout, sim_start, tag="APPROACH", tol=0.02, max_steps=180):
            return
        if not self._move_to(trace, grasp, GRIP_OPEN, timeout, sim_start, tag="INTERACT", tol=0.008, max_steps=260):
            return
        if not self._move_to(trace, press, GRIP_OPEN, timeout, sim_start, tag="INTERACT", tol=0.01, max_steps=80):
            pass
        if not self._hold(trace, GRIP_CLOSE, 36, timeout, sim_start):
            return
        # Present the held object to agentview so RGB-D postcondition is observable.
        show = np.array([0.02, -0.08, float(self.env.table_top_z + 0.16)])
        if not self._move_to(trace, show, GRIP_CLOSE, timeout, sim_start, tag="RETREAT", tol=0.03, max_steps=200):
            if not self._move_to(trace, hover, GRIP_CLOSE, timeout, sim_start, tag="RETREAT", tol=0.04, max_steps=120):
                return
        self._refresh_perception()
        trace.exit_reason = "NORMAL_TERMINATION"

```
