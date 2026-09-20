"""External contracts. Protocols have no pretend-success implementation."""
from dataclasses import dataclass
from typing import Protocol,runtime_checkable
from .facts import FactRecord

@dataclass(frozen=True)
class Observation:
    frame_id:str
    rgb_ref:str
    depth_ref:str
    proprioception:tuple[float,...]
    capture_seconds:float
    available_seconds:float

@dataclass(frozen=True)
class Perception:
    observation:Observation
    object_measurements:tuple
    checkpoint_hash:str
    calibration_hash:str

@dataclass(frozen=True)
class ExecutionResult:
    execution_id:str
    controller_exit:str
    start_seconds:float
    end_seconds:float
    evidence_ids:tuple[str,...]

@dataclass(frozen=True)
class EvaluationInput:
    task_id:str
    env_id:str
    episode_id:str
    evidence_refs:tuple[str,...]
    elapsed_seconds:float
    interval_start_seconds:float
    interval_end_seconds:float
    # No prior, graph embedding or nominal facts allowed in this schema.

@dataclass(frozen=True)
class TaskResult:
    success:bool
    terminated:bool
    truncated:bool
    reason:str
    reward_events:tuple[tuple[float,float],...]

@runtime_checkable
class EnvironmentAdapter(Protocol):
    def reset(self,task_id:str,case_id:str,seed:int)->None: ...
    def close(self)->None: ...

@runtime_checkable
class SkillExecutor(Protocol):
    def execute(self,candidate_id:str,timeout_seconds:float)->ExecutionResult: ...

@runtime_checkable
class ObservationProvider(Protocol):
    def observe(self)->Observation: ...

@runtime_checkable
class PerceptionAdapter(Protocol):
    def infer(self,observation:Observation)->Perception: ...

@runtime_checkable
class FactVerifier(Protocol):
    def verify(self,measurement:Perception,execution:ExecutionResult|None)->tuple[FactRecord,...]: ...

@runtime_checkable
class TaskEvaluator(Protocol):
    def evaluate(self,value:EvaluationInput)->TaskResult: ...

@runtime_checkable
class SafetyManager(Protocol):
    def can_execute(self,candidate_id:str,observation:Observation)->bool: ...
    def end_no_candidates(self,env_id:str,episode_id:str)->str: ...

@runtime_checkable
class DurationProvider(Protocol):
    def now_seconds(self)->float: ...
    def duration_seconds(self,start:float,end:float)->float: ...

@runtime_checkable
class RuntimeFactory(Protocol):
    def create(self,manifest:dict): ...

@runtime_checkable
class SnapshotBuilder(Protocol):
    def build(self,previous_snapshot,records:tuple[FactRecord,...],observation:Observation,execution:ExecutionResult,clock_seconds:float): ...
