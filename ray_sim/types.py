"""
Ray Simulation Engine - Type Definitions

Faithfully models the data structures from the Ray paper:
- Global Control Store (GCS) tables
- Task/Object metadata
- Step events for visualization
- Dynamic task graph (data edges, control edges, stateful edges)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json


# === ID Types ===
ObjectID = str
TaskID = str
NodeID = str
ActorID = str
FunctionID = str


# === Enums ===

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    SCHEDULED = "scheduled"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"

class StepPhase(Enum):
    """Major phases matching the paper's end-to-end flow (Fig 5)"""
    INIT = "init"
    TASK_SUBMIT = "task_submit"
    LOCAL_SCHEDULE = "local_schedule"
    GLOBAL_SCHEDULE = "global_schedule"
    DATA_FETCH = "data_fetch"
    TASK_EXECUTE = "task_execute"
    RESULT_REGISTER = "result_register"
    RESULT_GET = "result_get"
    ACTOR_CREATE = "actor_create"
    ACTOR_METHOD = "actor_method"

class EdgeType(Enum):
    """Three edge types from the paper's dynamic task graph model"""
    DATA = "data"           # Data object ↔ Task dependency
    CONTROL = "control"     # Nested remote function invocation
    STATEFUL = "stateful"   # Consecutive actor method calls


# === Core Data Structures ===

@dataclass
class TaskSpec:
    """Specification of a task (remote function invocation or actor method)"""
    task_id: TaskID
    function_name: str
    args: List[ObjectID]
    num_returns: int = 1
    resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1})
    is_actor_method: bool = False
    actor_id: Optional[ActorID] = None
    calling_task: Optional[TaskID] = None  # For control edges
    node_constraint: Optional[NodeID] = None  # For actor methods (must run on actor's node)

@dataclass
class TaskInfo:
    """Runtime info about a task"""
    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[NodeID] = None
    result_objects: List[ObjectID] = field(default_factory=list)
    worker_id: Optional[str] = None

@dataclass
class ObjectInfo:
    """Metadata for an object in the distributed object store"""
    object_id: ObjectID
    location: NodeID
    size: int = 100  # bytes (simplified)
    value: Any = None
    created_by_task: Optional[TaskID] = None
    is_actor_handle: bool = False

@dataclass
class ActorInfo:
    """Information about an actor instance"""
    actor_id: ActorID
    class_name: str
    node_id: NodeID
    methods: List[str] = field(default_factory=list)
    last_method_task: Optional[TaskID] = None  # For stateful edges

@dataclass
class FunctionInfo:
    """Information about a registered remote function"""
    function_name: str
    num_returns: int = 1
    resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1})
    is_actor_class: bool = False
    actor_methods: List[str] = field(default_factory=list)


# === Visual Hint Types ===

@dataclass
class ArrowHint:
    """Describes an animated arrow to show data/control flow"""
    from_id: str  # Source component ID (e.g., "N1_driver", "GCS", "global_scheduler")
    to_id: str    # Target component ID
    label: str = ""
    arrow_type: str = "data"  # "data" | "control" | "notification"
    style: str = "solid"      # "solid" | "dashed"

@dataclass
class HighlightHint:
    """Describes a component to highlight"""
    component_id: str  # e.g., "N1_object_store", "GCS_object_table"
    highlight_type: str = "active"  # "active" | "new" | "modified"

@dataclass
class TaskGraphNode:
    """A node in the dynamic task graph"""
    node_id: str       # Task ID or Object ID
    node_type: str     # "task" | "data" | "actor_method"
    label: str
    status: str = "pending"  # "pending" | "running" | "completed"

@dataclass
class TaskGraphEdge:
    """An edge in the dynamic task graph"""
    from_id: str
    to_id: str
    edge_type: EdgeType


# === Step Event ===

@dataclass
class StepEvent:
    """A single step in the execution, corresponding to one interaction between components.
    
    This directly models the numbered steps in the paper's Figure 5.
    """
    step_number: int = 0
    phase: StepPhase = StepPhase.INIT
    description: str = ""
    detail: str = ""  # More detailed description for the event log
    
    # Which components are involved
    source: str = ""     # Component initiating the action
    target: str = ""     # Component receiving the action
    
    # Visual hints for the frontend
    arrows: List[ArrowHint] = field(default_factory=list)
    highlights: List[HighlightHint] = field(default_factory=list)
    
    # Task graph updates at this step
    new_graph_nodes: List[TaskGraphNode] = field(default_factory=list)
    new_graph_edges: List[TaskGraphEdge] = field(default_factory=list)
    
    # Data changes at this step
    data_changes: Dict[str, Any] = field(default_factory=dict)


# === System State Snapshot ===

@dataclass
class NodeState:
    """State of a single node"""
    node_id: NodeID
    object_store: Dict[ObjectID, Any] = field(default_factory=dict)  # object_id -> value
    local_queue: List[TaskID] = field(default_factory=list)
    workers: List[str] = field(default_factory=list)
    actors: List[ActorID] = field(default_factory=list)
    is_driver: bool = False

@dataclass
class GCSState:
    """State of the Global Control Store"""
    object_table: Dict[ObjectID, Dict] = field(default_factory=dict)  # object_id -> {location, size}
    task_table: Dict[TaskID, Dict] = field(default_factory=dict)      # task_id -> {status, node, ...}
    function_table: Dict[str, Dict] = field(default_factory=dict)     # function_name -> info
    actor_table: Dict[ActorID, Dict] = field(default_factory=dict)    # actor_id -> info

@dataclass
class SystemSnapshot:
    """Complete state of the system at a given step"""
    step_number: int
    nodes: Dict[NodeID, NodeState] = field(default_factory=dict)
    gcs: GCSState = field(default_factory=GCSState)
    global_scheduler_queue: List[TaskID] = field(default_factory=list)
    event: Optional[StepEvent] = None
    task_graph_nodes: Dict[str, TaskGraphNode] = field(default_factory=dict)
    task_graph_edges: List[TaskGraphEdge] = field(default_factory=list)


# === Program Definition ===

@dataclass
class ProgramOp:
    """Base class for program operations"""
    op_type: str = ""

@dataclass
class RegisterFunction(ProgramOp):
    """Register a remote function"""
    op_type: str = "register_function"
    function_name: str = ""
    num_returns: int = 1
    resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1})

@dataclass
class RegisterActorClass(ProgramOp):
    """Register an actor class"""
    op_type: str = "register_actor_class"
    class_name: str = ""
    methods: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=lambda: {"CPU": 1})

@dataclass
class PutOp(ProgramOp):
    """Put a value into the object store"""
    op_type: str = "put"
    object_id: ObjectID = ""
    value: Any = None
    node: NodeID = "N1"

@dataclass
class RemoteCallOp(ProgramOp):
    """Call a remote function"""
    op_type: str = "remote_call"
    function_name: str = ""
    args: List[ObjectID] = field(default_factory=list)
    calling_node: NodeID = "N1"  # Which node's driver is making the call
    calling_task: Optional[TaskID] = None  # For nested calls (control edges)

@dataclass
class ActorCreateOp(ProgramOp):
    """Create an actor instance"""
    op_type: str = "actor_create"
    class_name: str = ""
    actor_id: ActorID = ""
    node: NodeID = ""  # If empty, scheduler decides
    calling_node: NodeID = "N1"

@dataclass
class ActorMethodCallOp(ProgramOp):
    """Call an actor method"""
    op_type: str = "actor_method_call"
    actor_id: ActorID = ""
    method_name: str = ""
    args: List[ObjectID] = field(default_factory=list)
    calling_node: NodeID = "N1"
    calling_task: Optional[TaskID] = None

@dataclass
class GetOp(ProgramOp):
    """Get the value of a future (ray.get)"""
    op_type: str = "get"
    object_id: ObjectID = ""
    calling_node: NodeID = "N1"


@dataclass
class RayProgram:
    """A complete Ray program definition"""
    name: str
    description: str
    num_nodes: int = 2
    operations: List[ProgramOp] = field(default_factory=list)
    node_labels: Dict[NodeID, str] = field(default_factory=dict)
