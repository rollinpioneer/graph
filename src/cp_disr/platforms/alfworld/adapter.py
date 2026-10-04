"""ALFWorld TextWorld adapter under the public-information boundary.

The underlying TextWorld env is started WITHOUT the `facts` info request, so the full hidden
world state is never materialised in this process. Everything the agent may use is derived
from the feedback text and admissible commands of real raw commands.
Hidden placement exists only in the separate catalog file (oracle partition) which this
module never imports.
"""
import warnings
from dataclasses import dataclass, field

from ...common import DataIntegrityError
from .observation import parse_arrival, parse_examine, parse_initial, parse_open, type_of

MAX_STEPS = 50


class ProtocolError(DataIntegrityError):
    pass


@dataclass(frozen=True)
class MacroResult:
    kind: str
    target: str
    commands: tuple
    duration: int
    ok: bool
    status: str  # OK | FAILED | WON | TRUNCATED
    feedback: tuple = ()


@dataclass(frozen=True)
class PublicState:
    """Everything a policy/prior/script may see. Holds no reference to the environment."""
    goal_otype: str
    goal_rtype: str
    receptacles: tuple
    feasible: tuple
    unchecked: tuple
    goal_instance: object   # canonical goal-type receptacle (or None)
    raw_steps: int
    max_steps: int
    checked: tuple          # ((receptacle, items), ...)
    target_in: tuple        # ((receptacle, True/False), ...) for checked receptacles
    target_found: bool
    holding: object
    placed: bool
    legal: tuple
    done: bool


class AlfEpisode:
    def __init__(self, gamefile, tables, max_steps=MAX_STEPS):
        warnings.filterwarnings("ignore")
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self.gamefile = gamefile
        self.tables = tables
        self.max_steps = max_steps
        infos = textworld.EnvInfos(won=True, admissible_commands=True, feedback=True)
        self._env = textworld.start(gamefile, infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        state = self._env.reset()
        self._admissible = tuple(state["admissible_commands"])
        public = parse_initial(state["feedback"])
        self.initial_feedback = state["feedback"]
        self.receptacles = tuple(public["receptacles"])
        self.goal_otype = public["goal_otype"]
        self.goal_rtype = public["goal_rtype"]
        self.feasible = tuple(r for r in self.receptacles if tables.can_contain(type_of(r), self.goal_otype))
        self.goal_instances = tuple(r for r in self.receptacles if type_of(r) == self.goal_rtype)
        self.raw_steps = 0
        self.commands = []
        self.checked = {}       # receptacle -> tuple(items seen)
        self.target_seen = {}   # receptacle -> tuple(target-class instances seen there)
        self.holding = None
        self.placed = False
        self.done = False
        self.won = False
        self.truncated = False
        self.success_step = None

    # ------------------------------------------------------------------ raw layer
    def _raw(self, command):
        if self.done:
            raise ProtocolError("episode already finished")
        if command not in self._admissible:
            raise ProtocolError("command not admissible: " + command)
        state, _reward, _done = self._env.step(command)
        self.raw_steps += 1
        self.commands.append(command)
        self._admissible = tuple(state["admissible_commands"])
        if state["won"]:
            self.done = self.won = True
            self.success_step = self.raw_steps
        elif self.raw_steps >= self.max_steps:
            self.done = self.truncated = True
        return state["feedback"]

    # ------------------------------------------------------------------ public status
    @property
    def target_found(self):
        return any(self.target_seen.get(r) for r in self.target_seen)

    def legal(self):
        """Public legality only (no hidden fact is consulted)."""
        if self.done:
            return ()
        out = []
        if self.holding is None and not self.target_found:
            out += [("CHECK", r) for r in self.feasible if r not in self.checked]
        if self.holding is None:
            out += [("TAKE", r) for r in self.feasible if self.target_seen.get(r)]
        if self.holding is not None and not self.placed and self.goal_instances:
            out.append(("PUT", self.goal_instances[0]))
        return tuple(out)

    def public(self):
        return PublicState(
            goal_otype=self.goal_otype,
            goal_rtype=self.goal_rtype,
            receptacles=self.receptacles,
            feasible=self.feasible,
            unchecked=tuple(r for r in self.feasible if r not in self.checked),
            goal_instance=self.goal_instances[0] if self.goal_instances else None,
            raw_steps=self.raw_steps,
            max_steps=self.max_steps,
            checked=tuple(sorted(self.checked.items())),
            target_in=tuple(sorted((r, bool(self.target_seen.get(r))) for r in self.checked)),
            target_found=self.target_found or self.holding is not None or self.placed,
            holding=self.holding,
            placed=self.placed,
            legal=self.legal(),
            done=self.done,
        )

    # ------------------------------------------------------------------ macro layer
    def _reveal(self, r, feedback_log, commands):
        """Make receptacle r accessible and return its visible items (None if cut off by the step cap).

        Several receptacles can share one location; `go to r` is then not admissible while the agent
        stands there and `examine r` is used instead. The choice is read off the public admissible list.
        """
        adm = self._admissible
        if "go to " + r in adm:
            cmd = "go to " + r
            fb = self._raw(cmd)
            commands.append(cmd)
            feedback_log.append(fb)
            arrival = parse_arrival(fb)
            if arrival is None or arrival["at"] != r:
                raise ProtocolError("arrival not confirmed for " + r)
            closed, items = arrival["closed"], arrival["items"]
        elif "examine " + r in adm:
            cmd = "examine " + r
            fb = self._raw(cmd)
            commands.append(cmd)
            feedback_log.append(fb)
            seen = parse_examine(fb, r)
            if seen is None:
                raise ProtocolError("examine not understood for " + r)
            closed, items = seen["closed"], seen["items"]
        else:
            raise ProtocolError("receptacle not reachable: " + r)
        if closed:
            if self.done:
                return None
            fb = self._raw("open " + r)
            commands.append("open " + r)
            feedback_log.append(fb)
            opened = parse_open(fb)
            if opened is None or opened["opened"] != r:
                raise ProtocolError("open not confirmed for " + r)
            items = opened["items"]
        return items
    def _status(self, ok):
        return "WON" if self.won else "TRUNCATED" if self.truncated else "OK" if ok else "FAILED"

    def check(self, r):
        if ("CHECK", r) not in self.legal():
            raise ProtocolError("CHECK not legal: " + r)
        start, commands, log = self.raw_steps, [], []
        items = self._reveal(r, log, commands)
        if items is not None:
            self.checked[r] = tuple(items)
            self.target_seen[r] = tuple(sorted(i for i in items if type_of(i) == self.goal_otype))
        return MacroResult("CHECK", r, tuple(commands), self.raw_steps - start, items is not None, self._status(items is not None), tuple(log))

    def take(self, r):
        if ("TAKE", r) not in self.legal():
            raise ProtocolError("TAKE not legal: " + r)
        start, commands, log = self.raw_steps, [], []
        obj = self.target_seen[r][0]
        if "take %s from %s" % (obj, r) not in self._admissible:
            self._reveal(r, log, commands)
        ok = False
        if not self.done:
            fb = self._raw("take %s from %s" % (obj, r))
            commands.append("take %s from %s" % (obj, r))
            log.append(fb)
            ok = fb.startswith("You pick up")
        if ok:
            self.holding = obj
            self.target_seen[r] = tuple(x for x in self.target_seen[r] if x != obj)
        return MacroResult("TAKE", r, tuple(commands), self.raw_steps - start, ok, self._status(ok), tuple(log))

    def put(self, g):
        if ("PUT", g) not in self.legal():
            raise ProtocolError("PUT not legal: " + g)
        start, commands, log = self.raw_steps, [], []
        if "move %s to %s" % (self.holding, g) not in self._admissible:
            self._reveal(g, log, commands)
        ok = False
        if not self.done:
            cmd = "move %s to %s" % (self.holding, g)
            fb = self._raw(cmd)
            commands.append(cmd)
            log.append(fb)
            ok = fb.startswith("You move")
            if ok:
                self.placed = True
                self.holding = None
        return MacroResult("PUT", g, tuple(commands), self.raw_steps - start, ok, self._status(ok), tuple(log))

    def execute(self, kind, target):
        return {"CHECK": self.check, "TAKE": self.take, "PUT": self.put}[kind](target)

    # ------------------------------------------------------------------ replay
    @classmethod
    def replay(cls, gamefile, tables, macros, max_steps=MAX_STEPS):
        ep = cls(gamefile, tables, max_steps)
        for kind, target in macros:
            ep.execute(kind, target)
        return ep

    def close(self):
        self._env.close()
