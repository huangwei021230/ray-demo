"""
Ray Simulation Engine - Core Execution Engine

Faithfully implements the Ray paper's architecture:
- Global Control Store (GCS): key-value store with pub-sub (Section 4.2)
- Bottom-Up Distributed Scheduler: local + global (Section 4.3)
- In-Memory Distributed Object Store: shared memory per node (Section 4.4)
- Step-by-step execution matching the paper's Figure 5 flow

The engine breaks down each Ray operation (remote call, actor creation, etc.)
into discrete micro-steps that can be individually stepped through and visualized.
"""

from __future__ import annotations
import copy
import json
from typing import Dict, List, Optional, Any, Callable
from .types import *


class GlobalControlStore:
    """Global Control Store (GCS) - Section 4.2 of the paper.
    
    A key-value store with pub-sub functionality.
    Uses sharding for scale and chain replication for fault tolerance.
    Stores: Object Table, Task Table, Function Table, Actor Table.
    
    Key design: enables every component in the system to be stateless.
    """
    
    def __init__(self):
        self.object_table: Dict[ObjectID, Dict] = {}  # object_id -> {location, size, created_by}
        self.task_table: Dict[TaskID, Dict] = {}       # task_id -> {spec, status, node}
        self.function_table: Dict[str, Dict] = {}       # function_name -> info
        self.actor_table: Dict[ActorID, Dict] = {}      # actor_id -> {class_name, node, methods, last_method}
        
        # Pub-sub callbacks
        self._subscribers: Dict[str, List[Callable]] = {
            "object_table": [],
            "task_table": [],
            "function_table": [],
            "actor_table": [],
        }
    
    def register_function(self, name: str, info: Dict):
        self.function_table[name] = info
        self._publish("function_table", name, info)
    
    def register_object(self, object_id: ObjectID, location: NodeID, size: int = 100,
                        created_by: Optional[TaskID] = None, is_actor_handle: bool = False):
        entry = {"location": location, "size": size, "created_by": created_by,
                 "is_actor_handle": is_actor_handle}
        self.object_table[object_id] = entry
        self._publish("object_table", object_id, entry)
    
    def update_object_location(self, object_id: ObjectID, location: NodeID):
        if object_id in self.object_table:
            self.object_table[object_id]["location"] = location
            self._publish("object_table", object_id, self.object_table[object_id])
    
    def get_object_location(self, object_id: ObjectID) -> Optional[Dict]:
        return self.object_table.get(object_id)
    
    def register_task(self, task_id: TaskID, info: Dict):
        self.task_table[task_id] = info
        self._publish("task_table", task_id, info)
    
    def update_task_status(self, task_id: TaskID, status: str, node: Optional[NodeID] = None):
        if task_id in self.task_table:
            self.task_table[task_id]["status"] = status
            if node:
                self.task_table[task_id]["node"] = node
            self._publish("task_table", task_id, self.task_table[task_id])
    
    def register_actor(self, actor_id: ActorID, info: Dict):
        self.actor_table[actor_id] = info
        self._publish("actor_table", actor_id, info)
    
    def update_actor_last_method(self, actor_id: ActorID, task_id: TaskID):
        if actor_id in self.actor_table:
            self.actor_table[actor_id]["last_method"] = task_id
    
    def subscribe(self, table: str, callback: Callable):
        self._subscribers[table].append(callback)
    
    def _publish(self, table: str, key: str, value: Any):
        for cb in self._subscribers.get(table, []):
            cb(key, value)
    
    def get_state(self) -> GCSState:
        return GCSState(
            object_table=copy.deepcopy(self.object_table),
            task_table=copy.deepcopy(self.task_table),
            function_table=copy.deepcopy(self.function_table),
            actor_table=copy.deepcopy(self.actor_table),
        )


class LocalObjectStore:
    """Per-node in-memory distributed object store - Section 4.4.
    
    - Shared memory for zero-copy data sharing between tasks on same node
    - Objects replicated from remote nodes before task execution
    - Immutable data only
    - LRU eviction to disk (simplified in simulation)
    """
    
    def __init__(self, node_id: NodeID):
        self.node_id = node_id
        self.objects: Dict[ObjectID, Any] = {}  # object_id -> value
    
    def put(self, object_id: ObjectID, value: Any):
        self.objects[object_id] = value
    
    def get(self, object_id: ObjectID) -> Optional[Any]:
        return self.objects.get(object_id)
    
    def has(self, object_id: ObjectID) -> bool:
        return object_id in self.objects
    
    def get_all_ids(self) -> List[ObjectID]:
        return list(self.objects.keys())


class LocalScheduler:
    """Per-node local scheduler - Section 4.3.
    
    Bottom-up scheduling: tries to schedule tasks locally first.
    Forwards to global scheduler only if:
    - Node is overloaded (queue exceeds threshold)
    - Cannot satisfy task's resource requirements (e.g., lacks GPU)
    """
    
    def __init__(self, node_id: NodeID, threshold: int = 5):
        self.node_id = node_id
        self.threshold = threshold
        self.task_queue: List[TaskID] = []
        self.available_resources: Dict[str, float] = {"CPU": 4, "GPU": 0}
        # Synchronous engine — by the time the next op arrives the queue has
        # drained, so len(task_queue) never grows. To still demonstrate the
        # "overload → forward" branch of bottom-up scheduling, we also track a
        # cumulative admission count that survives op boundaries. The global
        # scheduler still sees realistic per-tick queue depths via heartbeats.
        self.admitted_recent: int = 0

    def is_overloaded(self) -> bool:
        return (len(self.task_queue) >= self.threshold
                or self.admitted_recent >= self.threshold)
    
    def can_satisfy(self, resources: Dict[str, float]) -> bool:
        for res, amount in resources.items():
            if self.available_resources.get(res, 0) < amount:
                return False
        return True
    
    def should_forward(self, task: TaskSpec) -> bool:
        """Decide whether to forward task to global scheduler"""
        if self.is_overloaded():
            return True
        if not self.can_satisfy(task.resources):
            return True
        # Actor methods must run on the actor's node
        if task.is_actor_method and task.node_constraint and task.node_constraint != self.node_id:
            return True
        return False
    
    def enqueue(self, task_id: TaskID):
        self.task_queue.append(task_id)
        self.admitted_recent += 1

    def dequeue(self) -> Optional[TaskID]:
        if self.task_queue:
            return self.task_queue.pop(0)
        return None


class GlobalScheduler:
    """Global scheduler - Section 4.3.
    
    Receives tasks forwarded from local schedulers.
    Selects best node based on:
    - Estimated waiting time = queue_time + transfer_time
    - queue_time = task_queue_size * avg_task_execution_time
    - transfer_time = total_remote_input_size / avg_bandwidth
    
    Can be replicated for scalability (shares state via GCS).
    """
    
    def __init__(self):
        self.pending_tasks: List[TaskID] = []
        self.node_loads: Dict[NodeID, Dict] = {}  # from heartbeats
        self.avg_task_duration: float = 5.0  # ms (exponential averaging)
        self.avg_bandwidth: float = 1000.0   # MB/s (exponential averaging)
        # Populated by select_node so callers can include scoring details in
        # event descriptions (paper §4.3 "estimated waiting time").
        self.last_scoring: List[Dict] = []
    
    def update_node_load(self, node_id: NodeID, queue_size: int, resources: Dict[str, float]):
        self.node_loads[node_id] = {
            "queue_size": queue_size,
            "resources": resources,
        }
    
    def select_node(self, task: TaskSpec, gcs: GlobalControlStore,
                    available_nodes: List[NodeID]) -> NodeID:
        """Select the best node for a task using the paper's algorithm:
        minimize estimated_waiting_time = estimated_queue_time + estimated_transfer_time
        """
        best_node = None
        best_time = float('inf')
        scoring: List[Dict] = []

        for node_id in available_nodes:
            # Check if node has required resources
            load = self.node_loads.get(node_id, {"queue_size": 0, "resources": {"CPU": 4, "GPU": 0}})
            can_run = True
            for res, amount in task.resources.items():
                if load["resources"].get(res, 0) < amount:
                    can_run = False
                    break
            if not can_run:
                scoring.append({
                    "node": node_id, "eligible": False,
                    "reason": f"missing resources (need {task.resources}, has {load['resources']})",
                })
                continue

            # Estimated queue time
            queue_time = load["queue_size"] * self.avg_task_duration

            # Estimated transfer time for remote inputs
            remote_input_size = 0
            remote_args: List[ObjectID] = []
            for arg_id in task.args:
                loc_info = gcs.get_object_location(arg_id)
                if loc_info and loc_info["location"] != node_id:
                    remote_input_size += loc_info["size"]
                    remote_args.append(arg_id)

            transfer_time = (remote_input_size / (1024 * 1024)) / self.avg_bandwidth * 1000  # ms

            total_time = queue_time + transfer_time
            scoring.append({
                "node": node_id, "eligible": True,
                "queue_size": load["queue_size"],
                "queue_time_ms": round(queue_time, 3),
                "remote_bytes": remote_input_size,
                "remote_args": remote_args,
                "transfer_time_ms": round(transfer_time, 3),
                "total_ms": round(total_time, 3),
            })

            if total_time < best_time:
                best_time = total_time
                best_node = node_id

        # Fallback: pick first available node with resources
        if best_node is None and available_nodes:
            best_node = available_nodes[0]

        self.last_scoring = scoring
        return best_node

    def format_scoring(self) -> str:
        """Render last_scoring as a one-line breakdown for event detail."""
        parts = []
        for s in self.last_scoring:
            if not s.get("eligible", True):
                parts.append(f"{s['node']}: ineligible ({s['reason']})")
            else:
                parts.append(
                    f"{s['node']}: queue={s['queue_size']}×{self.avg_task_duration}ms"
                    f"+transfer={s['remote_bytes']}B/{self.avg_bandwidth}MB·s"
                    f" → {s['total_ms']}ms"
                )
        return "; ".join(parts)
    
    def enqueue(self, task_id: TaskID):
        self.pending_tasks.append(task_id)
    
    def dequeue(self, task_id: TaskID):
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)


class Node:
    """A node in the Ray cluster.
    
    Contains: Local Scheduler, Object Store, Workers, Actors.
    May contain a Driver (user program).
    """
    
    def __init__(self, node_id: NodeID, is_driver: bool = False,
                 resources: Optional[Dict[str, float]] = None):
        self.node_id = node_id
        self.is_driver = is_driver
        self.local_scheduler = LocalScheduler(node_id)
        self.object_store = LocalObjectStore(node_id)
        self.workers: List[str] = [f"{node_id}_worker_{i}" for i in range(2)]
        self.worker_tasks: Dict[str, Optional[TaskID]] = {
            w: None for w in self.workers
        }
        self.actors: List[ActorID] = []
        self.resources = resources or {"CPU": 4, "GPU": 0}
        self.local_scheduler.available_resources = dict(self.resources)
        self.is_dead: bool = False

    def get_state(self) -> NodeState:
        return NodeState(
            node_id=self.node_id,
            object_store=dict(self.object_store.objects),
            local_queue=list(self.local_scheduler.task_queue),
            workers=list(self.workers),
            worker_tasks=dict(self.worker_tasks),
            actors=list(self.actors),
            is_driver=self.is_driver,
            is_dead=self.is_dead,
            resources=dict(self.resources),
        )


class ExecutionEngine:
    """Step-by-step execution engine.
    
    Simulates the complete Ray execution flow from the paper,
    breaking each operation into discrete micro-steps that can be
    individually stepped through and visualized.
    
    Matches the paper's Figure 5 step-by-step flow.
    """
    
    def __init__(self):
        self.nodes: Dict[NodeID, Node] = {}
        self.gcs = GlobalControlStore()
        self.global_scheduler = GlobalScheduler()
        self.tasks: Dict[TaskID, TaskInfo] = {}
        self.actors_info: Dict[ActorID, ActorInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.object_values: Dict[ObjectID, Any] = {}
        
        # Step tracking
        self.steps: List[SystemSnapshot] = []
        self.current_step = -1
        self._step_counter = 0
        
        # Task graph
        self._graph_nodes: Dict[str, TaskGraphNode] = {}
        self._graph_edges: List[TaskGraphEdge] = []
        
        # ID counters
        self._task_counter = 0
        self._object_counter = 0
        self._actor_counter = 0

        # Burst-submission window state (see BurstStart / BurstEnd ops).
        self._burst_mode: bool = False
        self._burst_pending: List[Dict] = []
    
    def _next_task_id(self) -> TaskID:
        tid = f"task_{self._task_counter}"
        self._task_counter += 1
        return tid
    
    def _next_object_id(self) -> ObjectID:
        oid = f"obj_{self._object_counter}"
        self._object_counter += 1
        return oid
    
    def _next_actor_id(self) -> ActorID:
        aid = f"actor_{self._actor_counter}"
        self._actor_counter += 1
        return aid
    
    def initialize_cluster(self, num_nodes: int, node_labels: Optional[Dict[NodeID, str]] = None,
                           node_resources: Optional[Dict[NodeID, Dict[str, float]]] = None,
                           driver_node: Optional[NodeID] = None):
        """Initialize the Ray cluster with the given number of nodes."""
        self.nodes.clear()
        for i in range(num_nodes):
            nid = f"N{i+1}"
            is_driver = (nid == (driver_node or "N1"))
            # Use per-node resources if specified, otherwise defaults
            if node_resources and nid in node_resources:
                resources = dict(node_resources[nid])
            else:
                resources = {"CPU": 4, "GPU": 0}
                if i >= 1:
                    resources["GPU"] = 1
            self.nodes[nid] = Node(nid, is_driver=is_driver, resources=resources)
            self.global_scheduler.update_node_load(nid, 0, resources)
        
        self._record_initial_state()
    
    def _record_initial_state(self):
        """Record the initial empty state."""
        event = StepEvent(
            step_number=self._step_counter,
            phase=StepPhase.INIT,
            description="Ray cluster initialized",
            detail=f"Cluster started with {len(self.nodes)} nodes. GCS, Global Scheduler, and Local Schedulers are ready.",
            source="system",
            target="system",
            highlights=[HighlightHint("GCS", "active"), HighlightHint("global_scheduler", "active")],
        )
        # Add all node highlights
        for nid in self.nodes:
            event.highlights.append(HighlightHint(f"{nid}_local_scheduler", "active"))
            event.highlights.append(HighlightHint(f"{nid}_object_store", "active"))
        
        self._record_step(event)
    
    def _record_step(self, event: StepEvent):
        """Record a step with the current system state."""
        event.step_number = self._step_counter
        
        snapshot = SystemSnapshot(
            step_number=self._step_counter,
            nodes={nid: node.get_state() for nid, node in self.nodes.items()},
            gcs=self.gcs.get_state(),
            global_scheduler_queue=list(self.global_scheduler.pending_tasks),
            event=event,
            task_graph_nodes=copy.deepcopy(self._graph_nodes),
            task_graph_edges=copy.deepcopy(self._graph_edges),
        )
        
        self.steps.append(snapshot)
        self._step_counter += 1
    
    def _add_graph_node(self, node_id: str, node_type: str, label: str, status: str = "pending"):
        gn = TaskGraphNode(node_id=node_id, node_type=node_type, label=label, status=status)
        self._graph_nodes[node_id] = gn
    
    def _add_graph_edge(self, from_id: str, to_id: str, edge_type: EdgeType):
        ge = TaskGraphEdge(from_id=from_id, to_id=to_id, edge_type=edge_type)
        self._graph_edges.append(ge)
    
    # === High-level operations ===
    
    def execute_register_function(self, op: RegisterFunction):
        """Register a remote function with GCS and distribute to all workers.
        
        Paper: "The remote function add() is automatically registered with the GCS
        upon initialization and distributed to every worker in the system (step 0)."
        """
        func_info = FunctionInfo(
            function_name=op.function_name,
            num_returns=op.num_returns,
            resources=op.resources,
        )
        self.functions[op.function_name] = func_info
        
        # Register with GCS and distribute to all workers — single step
        self.gcs.register_function(op.function_name, {
            "num_returns": op.num_returns,
            "resources": op.resources,
        })
        
        node_list = list(self.nodes.keys())
        event = StepEvent(
            phase=StepPhase.INIT,
            description=f"Function '{op.function_name}()' registered & distributed to {', '.join(node_list)}",
            detail=f"Function '{op.function_name}()' is registered in the GCS Function Table "
                   f"and distributed to all workers on {', '.join(node_list)}. "
                   f"It requires {op.resources} and returns {op.num_returns} object(s).",
            source="driver",
            target="GCS",
            highlights=[HighlightHint("GCS_function_table", "new")] +
                      [HighlightHint(f"{nid}_worker", "active") for nid in node_list],
            arrows=[ArrowHint("N1_driver", "GCS", f"register {op.function_name}()", "control", "dashed")] +
                   [ArrowHint("GCS", f"{nid}_worker", f"push {op.function_name}()", "control", "dashed") for nid in node_list],
        )
        self._record_step(event)
    
    def execute_register_actor_class(self, op: RegisterActorClass):
        """Register an actor class with GCS."""
        func_info = FunctionInfo(
            function_name=op.class_name,
            num_returns=1,
            resources=op.resources,
            is_actor_class=True,
            actor_methods=op.methods,
        )
        self.functions[op.class_name] = func_info
        
        self.gcs.register_function(op.class_name, {
            "is_actor_class": True,
            "methods": op.methods,
            "resources": op.resources,
        })
        
        event = StepEvent(
            phase=StepPhase.INIT,
            description=f"Actor class '{op.class_name}' registered with GCS",
            detail=f"Actor class '{op.class_name}' is registered in the GCS. "
                   f"It exposes methods: {', '.join(op.methods)}. Requires {op.resources}.",
            source="driver",
            target="GCS",
            highlights=[HighlightHint("GCS_function_table", "new")],
            arrows=[ArrowHint("N1_driver", "GCS", f"register {op.class_name}", "control", "dashed")],
        )
        self._record_step(event)
    
    def execute_put(self, op: PutOp):
        """Put a value into a node's object store and register with GCS.
        
        Paper: ray.put() stores the value in the local object store and
        registers it in the GCS Object Table.
        """
        node = self.nodes[op.node]
        
        # Store value and register with GCS — single step
        node.object_store.put(op.object_id, op.value)
        self.object_values[op.object_id] = op.value
        self.gcs.register_object(op.object_id, op.node, size=100)
        
        # Add to task graph as a data node
        # Use object_id as label (e.g., "a", "b") for clarity
        self._add_graph_node(op.object_id, "data", op.object_id, "completed")
        
        event = StepEvent(
            phase=StepPhase.TASK_SUBMIT,
            description=f"ray.put({op.value}) → {op.object_id} stored on {op.node} & registered in GCS",
            detail=f"Driver on {op.node} stores value {op.value} in the local object store "
                   f"and registers its location in the GCS Object Table. Object ID: {op.object_id}",
            source=f"{op.node}_driver",
            target=f"{op.node}_object_store",
            highlights=[HighlightHint(f"{op.node}_object_store", "new"), HighlightHint("GCS_object_table", "new")],
            arrows=[ArrowHint(f"{op.node}_driver", f"{op.node}_object_store", f"put({op.value})", "data"),
                    ArrowHint(f"{op.node}_object_store", "GCS", f"register {op.object_id}@{op.node}", "control", "dashed")],
            new_graph_nodes=[TaskGraphNode(op.object_id, "data", op.object_id, "completed")],
            data_changes={"gcs_object_table_add": {op.object_id: {"location": op.node, "size": 100}}},
        )
        self._record_step(event)
    
    def execute_remote_call(self, op: RemoteCallOp) -> List[ObjectID]:
        """Execute a remote function call, following the paper's Figure 5 flow.
        
        Simplified step breakdown:
        1. Driver submits task to local scheduler
        2. Scheduling decision: local or global (1-2 steps)
        3. Fetch missing data (single step for all args)
        4. Worker executes task, stores results & registers in GCS (single step)
        """
        task_id = self._next_task_id()
        if op.create_result:
            if op.result_id:
                # Use custom result ID (e.g., "c" for add(a,b)=c)
                result_ids = [op.result_id]
            else:
                result_ids = [self._next_object_id() for _ in range(
                    self.functions.get(op.function_name, FunctionInfo(op.function_name)).num_returns
                )]
        else:
            result_ids = []
        
        task_spec = TaskSpec(
            task_id=task_id,
            function_name=op.function_name,
            args=op.args,
            num_returns=len(result_ids),
            resources=self.functions.get(op.function_name, FunctionInfo(op.function_name)).resources,
            calling_task=op.calling_task,
        )
        
        task_info = TaskInfo(spec=task_spec, status=TaskStatus.PENDING, result_objects=result_ids)
        self.tasks[task_id] = task_info
        
        # Add task node to graph
        task_label = op.label if op.label else f"{op.function_name}()"
        self._add_graph_node(task_id, "task", task_label)
        
        # Add data edges: args → task
        for arg_id in op.args:
            self._add_graph_edge(arg_id, task_id, EdgeType.DATA)
        
        # Add control edge if nested call
        if op.calling_task:
            self._add_graph_edge(op.calling_task, task_id, EdgeType.CONTROL)
        
        # Add data edges: task → results (only if create_result is True)
        if op.create_result:
            for rid in result_ids:
                result_label = op.result_label if op.result_label else f"result_{rid}"
                self._add_graph_node(rid, "data", result_label)
                self._add_graph_edge(task_id, rid, EdgeType.DATA)
        
        calling_node = self.nodes[op.calling_node]
        
        # ---- Step 1: Driver/parent task submits task to local scheduler ----
        task_info.status = TaskStatus.READY
        
        # Register task in GCS
        self.gcs.register_task(task_id, {
            "function": op.function_name,
            "args": op.args,
            "status": "ready",
            "node": None,
        })
        
        # Determine the source of the call (driver or parent task)
        source_label = f"{op.calling_node}_driver"
        if op.parent_task and op.parent_task in self.tasks:
            source_label = f"{op.calling_node}_worker"  # Parent task is running on a worker
        
        submit_description = f"Driver calls {op.function_name}.remote({', '.join(op.args)}) → creates {task_id}"
        if op.parent_task:
            submit_description = f"Nested call: {op.function_name}.remote({', '.join(op.args)}) → creates {task_id}"
        
        event = StepEvent(
            phase=StepPhase.TASK_SUBMIT,
            description=submit_description,
            detail=f"{'Nested within ' + op.parent_task + '. ' if op.parent_task else ''}"
                   f"{op.function_name}.remote() invoked on {op.calling_node} with args {op.args}. "
                   f"A TaskSpec is created (task_id={task_id}) and submitted to {op.calling_node}'s local scheduler. "
                   f"The call immediately returns futures: {result_ids}.",
            source=source_label,
            target=f"{op.calling_node}_local_scheduler",
            highlights=[HighlightHint(f"{op.calling_node}_local_scheduler", "active")],
            arrows=[ArrowHint(source_label, f"{op.calling_node}_local_scheduler",
                              f"submit {task_id}", "control")],
            new_graph_nodes=[TaskGraphNode(task_id, "task", f"{op.function_name}()", "pending")],
            new_graph_edges=[TaskGraphEdge(arg, task_id, EdgeType.DATA) for arg in op.args] +
                           ([TaskGraphEdge(op.calling_task, task_id, EdgeType.CONTROL)] if op.calling_task else []),
        )
        self._record_step(event)
        
        # ---- Step 2: Scheduling decision ----
        should_forward = calling_node.local_scheduler.should_forward(task_spec)
        
        if should_forward:
            reason = "node overloaded" if calling_node.local_scheduler.is_overloaded() else \
                     f"cannot satisfy resource requirements ({task_spec.resources})" if not calling_node.local_scheduler.can_satisfy(task_spec.resources) else \
                     "task requires different node"
            
            # Local scheduler forwards to global scheduler → global selects node — single step
            self.global_scheduler.enqueue(task_id)
            
            # Query GCS for arg locations
            arg_locations = {}
            for arg_id in op.args:
                loc = self.gcs.get_object_location(arg_id)
                if loc:
                    arg_locations[arg_id] = loc["location"]
            
            # Select best node — only living nodes are candidates
            available_nodes = [nid for nid, n in self.nodes.items() if not n.is_dead]
            selected_node = self.global_scheduler.select_node(task_spec, self.gcs, available_nodes)

            if selected_node is None:
                selected_node = available_nodes[0]

            task_info.assigned_node = selected_node
            task_info.status = TaskStatus.SCHEDULED
            self.gcs.update_task_status(task_id, "scheduled", selected_node)

            scoring_breakdown = self.global_scheduler.format_scoring()

            event = StepEvent(
                phase=StepPhase.GLOBAL_SCHEDULE,
                description=f"{op.calling_node} forwards {task_id} → global scheduler assigns to {selected_node}",
                detail=f"Local scheduler on {op.calling_node} cannot schedule {task_id} ({reason}). "
                       f"Global scheduler scores candidates by estimated_waiting_time = queue_time + transfer_time "
                       f"(paper §4.3). Scores: {scoring_breakdown}. "
                       f"Arg locations from GCS: {', '.join(f'{k}@{v}' for k, v in arg_locations.items()) or 'none'}.",
                source=f"{op.calling_node}_local_scheduler",
                target=f"{selected_node}_local_scheduler",
                highlights=[HighlightHint("global_scheduler", "active"), HighlightHint(f"{selected_node}_local_scheduler", "new")],
                arrows=[ArrowHint(f"{op.calling_node}_local_scheduler", "global_scheduler",
                                  f"forward {task_id}", "control"),
                        ArrowHint("global_scheduler", f"{selected_node}_local_scheduler",
                                  f"assign {task_id}", "control")],
            )
            self._record_step(event)
            
            # Task has been assigned: dequeue from global, enqueue into target node's local scheduler
            self.global_scheduler.dequeue(task_id)
            target_node = self.nodes[selected_node]
            target_node.local_scheduler.enqueue(task_id)
            self.global_scheduler.update_node_load(
                selected_node, max(len(target_node.local_scheduler.task_queue), target_node.local_scheduler.admitted_recent),
                target_node.local_scheduler.available_resources)
            
        else:
            # Schedule locally
            task_info.assigned_node = op.calling_node
            task_info.status = TaskStatus.SCHEDULED
            self.gcs.update_task_status(task_id, "scheduled", op.calling_node)
            calling_node.local_scheduler.enqueue(task_id)
            self.global_scheduler.update_node_load(
                op.calling_node, max(len(calling_node.local_scheduler.task_queue), calling_node.local_scheduler.admitted_recent),
                calling_node.local_scheduler.available_resources)
            
            event = StepEvent(
                phase=StepPhase.LOCAL_SCHEDULE,
                description=f"{op.calling_node}'s local scheduler schedules {task_id} locally",
                detail=f"The local scheduler on {op.calling_node} decides to schedule {task_id} locally. "
                       f"The node has available resources and is not overloaded. "
                       f"This is the common case (bottom-up scheduling: most tasks are local).",
                source=f"{op.calling_node}_local_scheduler",
                target=f"{op.calling_node}_local_scheduler",
                highlights=[HighlightHint(f"{op.calling_node}_local_scheduler", "active")],
            )
            self._record_step(event)
            
            selected_node = op.calling_node
        
        target_node = self.nodes[selected_node]

        if self._burst_mode:
            # Defer Step 3+4. Task stays queued on target_node so successive
            # submissions can see the queue depth grow and trigger overload.
            self._burst_pending.append({
                "task_id": task_id, "selected_node": selected_node,
                "op": op, "result_ids": result_ids,
            })
            return result_ids

        self._finish_remote_call(task_id, selected_node, op, result_ids)
        return result_ids

    def _finish_remote_call(self, task_id: TaskID, selected_node: NodeID,
                             op: RemoteCallOp, result_ids: List[ObjectID]):
        """Steps 3+4 of execute_remote_call (data fetch + dispatch + execute).

        Extracted so burst mode can defer them — the burst handler invokes this
        for each pending task once the submission window closes.
        """
        task_info = self.tasks[task_id]
        target_node = self.nodes[selected_node]

        # Before fetching, replay lineage for any arg that lives on a dead node
        # (paper §4.2.1) — otherwise the worker would read empty bytes.
        for arg_id in list(op.args):
            loc_info = self.gcs.get_object_location(arg_id)
            if loc_info:
                src = self.nodes.get(loc_info["location"])
                if src and src.is_dead and not target_node.object_store.has(arg_id):
                    self._ensure_object_alive(arg_id, op.calling_node)

        # ---- Step 3: Fetch missing data (single step for all args) ----
        missing_args = [arg_id for arg_id in op.args if not target_node.object_store.has(arg_id)]
        local_args = [arg_id for arg_id in op.args if target_node.object_store.has(arg_id)]

        if missing_args:
            # Replicate all missing args
            replication_info = []
            source_nodes_for_args = {}
            for arg_id in missing_args:
                loc_info = self.gcs.get_object_location(arg_id)
                source_node_id = loc_info["location"] if loc_info else op.calling_node
                value = self.nodes[source_node_id].object_store.get(arg_id)
                target_node.object_store.put(arg_id, value)
                replication_info.append(f"{arg_id} from {source_node_id}")
                source_nodes_for_args[arg_id] = source_node_id
            
            # Build arrows: GCS lookup + data replication (grouped by source node)
            fetch_arrows = [ArrowHint(f"{selected_node}_object_store", "GCS",
                                       f"lookup {missing_args}", "control", "dashed")]
            # Group args by source node for deduplication
            src_groups = {}
            for arg_id, src_node in source_nodes_for_args.items():
                src_groups.setdefault(src_node, []).append(arg_id)
            for src_node, arg_ids in src_groups.items():
                fetch_arrows.append(ArrowHint(f"{src_node}_object_store", f"{selected_node}_object_store",
                                               f"replicate {', '.join(arg_ids)}", "data"))
            
            zero_copy_note = (
                f" Local args {local_args} read shared-memory zero-copy with no transfer."
                if local_args else ""
            )
            event = StepEvent(
                phase=StepPhase.DATA_FETCH,
                description=f"{selected_node} replicates missing args: {', '.join(replication_info)}"
                            + (f" (+ {len(local_args)} zero-copy)" if local_args else ""),
                detail=f"Plasma replicate-on-read (paper §4.2.3): missing args are pulled point-to-point "
                       f"from the recorded GCS location into {selected_node}'s object store so the worker "
                       f"can mmap them shared-memory.{zero_copy_note} "
                       f"Replications: {', '.join(replication_info)}.",
                source=f"{selected_node}_object_store",
                target=f"{selected_node}_object_store",
                highlights=[HighlightHint(f"{selected_node}_object_store", "new"),
                           HighlightHint("GCS_object_table", "active")],
                arrows=fetch_arrows,
            )
            self._record_step(event)
        elif op.args:
            # All args already local → demonstrate Plasma same-node zero-copy path
            event = StepEvent(
                phase=StepPhase.DATA_FETCH,
                description=f"{selected_node}: all {len(op.args)} arg(s) already local → zero-copy mmap",
                detail=f"Args {op.args} are already in {selected_node}'s object store. "
                       f"The worker maps them shared-memory directly — no network transfer "
                       f"(paper §4.2.3 'same-node zero-copy reads via shared memory').",
                source=f"{selected_node}_object_store",
                target=f"{selected_node}_worker",
                highlights=[HighlightHint(f"{selected_node}_object_store", "active")],
                arrows=[ArrowHint(f"{selected_node}_object_store", f"{selected_node}_worker",
                                  f"mmap {op.args}", "data", "dashed")],
            )
            self._record_step(event)

        # ---- Step 4a: Dispatch task to worker (worker becomes busy) ----
        target_node.local_scheduler.dequeue()
        self.global_scheduler.update_node_load(
            selected_node, max(len(target_node.local_scheduler.task_queue), target_node.local_scheduler.admitted_recent),
            target_node.local_scheduler.available_resources)
        
        worker_id = target_node.workers[0] if target_node.workers else f"{selected_node}_worker_0"
        target_node.worker_tasks[worker_id] = task_id
        
        task_info.status = TaskStatus.RUNNING
        task_info.worker_id = worker_id
        self.gcs.update_task_status(task_id, "running", selected_node)
        self._graph_nodes[task_id].status = "running"
        
        dispatch_event = StepEvent(
            phase=StepPhase.TASK_EXECUTE,
            description=f"Local scheduler on {selected_node} dispatches {task_id} → {worker_id}",
            detail=f"The local scheduler on {selected_node} pops {task_id} from its queue "
                   f"and dispatches it to {worker_id}. Worker is now busy. "
                   f"Local scheduler is free to accept new tasks (non-blocking).",
            source=f"{selected_node}_local_scheduler",
            target=f"{selected_node}_worker",
            highlights=[HighlightHint(f"{selected_node}_worker", "active"),
                       HighlightHint(f"{selected_node}_local_scheduler", "active")],
            arrows=[ArrowHint(f"{selected_node}_local_scheduler", f"{selected_node}_worker",
                              f"dispatch {task_id}", "control")],
        )
        self._record_step(dispatch_event)
        
        # ---- Step 4b: Worker executes, stores results, registers in GCS ----
        result_values = self._simulate_function_execution(op.function_name, op.args)
        
        target_node.worker_tasks[worker_id] = None
        
        for rid, rval in zip(result_ids, result_values):
            target_node.object_store.put(rid, rval)
            self.object_values[rid] = rval
            self._graph_nodes[rid].status = "completed"
            self.gcs.register_object(rid, selected_node, size=100, created_by=task_id)
        
        task_info.status = TaskStatus.COMPLETED
        self.gcs.update_task_status(task_id, "completed", selected_node)
        self._graph_nodes[task_id].status = "completed"
        
        complete_event = StepEvent(
            phase=StepPhase.TASK_EXECUTE,
            description=f"Worker {worker_id} completes {op.function_name}()" + 
                       (f" → {result_ids} registered in GCS" if result_ids else ""),
            detail=f"Worker {worker_id} on {selected_node} finishes executing {op.function_name}(). " +
                   (f"Stores results {result_ids} in local object store and registers them in GCS. "
                    f"Results: {dict(zip(result_ids, result_values))}" if result_ids else "No return values.") +
                   " Worker is now idle.",
            source=f"{selected_node}_worker",
            target=f"{selected_node}_object_store" if result_ids else f"{selected_node}_worker",
            highlights=[HighlightHint(f"{selected_node}_worker", "active")] +
                      ([HighlightHint(f"{selected_node}_object_store", "new"),
                        HighlightHint("GCS_object_table", "new")] if result_ids else []),
            arrows=([ArrowHint(f"{selected_node}_worker", f"{selected_node}_object_store",
                              f"write {result_ids}", "data"),
                     ArrowHint(f"{selected_node}_object_store", "GCS",
                              f"register {result_ids}", "control", "dashed")] if result_ids else []),
            new_graph_nodes=[TaskGraphNode(rid, "data", f"result", "completed") for rid in result_ids],
            new_graph_edges=[TaskGraphEdge(task_id, rid, EdgeType.DATA) for rid in result_ids],
            data_changes={"gcs_object_table_add": {rid: {"location": selected_node, "size": 100} for rid in result_ids}},
        )
        self._record_step(complete_event)

    def execute_actor_create(self, op: ActorCreateOp) -> ActorID:
        """Create an actor instance.
        
        Paper: Actor creation follows the same scheduling flow as tasks:
        1. Driver submits to local scheduler
        2. Local scheduler forwards to global scheduler
        3. Global scheduler selects best node (or uses specified node)
        4. Target node creates the actor
        """
        actor_id = op.actor_id if op.actor_id else self._next_actor_id()
        
        func_info = self.functions.get(op.class_name, FunctionInfo(op.class_name))
        
        # ---- Step 1: Driver submits to local scheduler ----
        submit_event = StepEvent(
            phase=StepPhase.TASK_SUBMIT,
            description=f"Driver submits {op.class_name}.remote() → create {actor_id}",
            detail=f"Driver on {op.calling_node} wants to create an instance of {op.class_name}. "
                   f"Submitted to {op.calling_node}'s local scheduler.",
            source=f"{op.calling_node}_driver",
            target=f"{op.calling_node}_local_scheduler",
            highlights=[HighlightHint(f"{op.calling_node}_local_scheduler", "active")],
            arrows=[ArrowHint(f"{op.calling_node}_driver", f"{op.calling_node}_local_scheduler",
                              f"submit create {actor_id}", "control")],
        )
        self._record_step(submit_event)
        
        # ---- Step 2: Local scheduler forwards to global scheduler ----
        # Determine which node the actor will be on
        if op.node:
            target_node_id = op.node
        else:
            available = list(self.nodes.keys())
            target_node_id = self.global_scheduler.select_node(
                TaskSpec(task_id="actor_create", function_name=op.class_name,
                        args=[], resources=func_info.resources),
                self.gcs, available
            ) or available[0]
        
        forward_event = StepEvent(
            phase=StepPhase.GLOBAL_SCHEDULE,
            description=f"{op.calling_node} forwards → global scheduler selects {target_node_id} for {actor_id}",
            detail=f"Local scheduler forwards actor creation to global scheduler. "
                   f"Global scheduler selects {target_node_id} "
                   f"(has GPU resources for {op.class_name}).",
            source=f"{op.calling_node}_local_scheduler",
            target=f"{target_node_id}_local_scheduler",
            highlights=[HighlightHint("global_scheduler", "active"), HighlightHint(f"{target_node_id}_local_scheduler", "new")],
            arrows=[ArrowHint(f"{op.calling_node}_local_scheduler", "global_scheduler",
                              f"forward create {actor_id}", "control"),
                    ArrowHint("global_scheduler", f"{target_node_id}_local_scheduler",
                              f"assign {actor_id}", "control")],
        )
        self._record_step(forward_event)
        
        # ---- Step 3: Target node creates the actor ----
        target_node = self.nodes[target_node_id]
        target_node.actors.append(actor_id)
        
        # Register actor with GCS
        actor_info = ActorInfo(
            actor_id=actor_id,
            class_name=op.class_name,
            node_id=target_node_id,
            methods=func_info.actor_methods,
        )
        self.actors_info[actor_id] = actor_info
        self.gcs.register_actor(actor_id, {
            "class_name": op.class_name,
            "node": target_node_id,
            "methods": func_info.actor_methods,
        })
        
        # Create actor handle
        handle_id = f"handle_{actor_id}"
        target_node.object_store.put(handle_id, f"<ActorHandle:{actor_id}>")
        self.object_values[handle_id] = f"<ActorHandle:{actor_id}>"
        self.gcs.register_object(handle_id, target_node_id, is_actor_handle=True)
        
        # Add to task graph
        self._add_graph_node(actor_id, "actor_method", f"{op.class_name}", "completed")
        
        # Add control edge if called from a parent task
        if op.calling_task:
            self._add_graph_edge(op.calling_task, actor_id, EdgeType.CONTROL)
        
        create_event = StepEvent(
            phase=StepPhase.ACTOR_CREATE,
            description=f"{target_node_id} creates actor {actor_id} ({op.class_name})",
            detail=f"Actor {actor_id} is created on {target_node_id} and registered in GCS Actor Table. "
                   f"It exposes methods: {', '.join(func_info.actor_methods)}.",
            source=f"{target_node_id}_local_scheduler",
            target=f"{target_node_id}_actor",
            highlights=[HighlightHint(f"{target_node_id}_actor", "new"), HighlightHint("GCS_actor_table", "new")],
            arrows=[ArrowHint(f"{target_node_id}_local_scheduler", f"{target_node_id}_actor",
                              f"create {actor_id}", "control"),
                    ArrowHint(f"{target_node_id}_actor", "GCS",
                              f"register {actor_id}", "control", "dashed")],
            new_graph_nodes=[TaskGraphNode(actor_id, "actor_method", f"{op.class_name}", "completed")],
            new_graph_edges=[TaskGraphEdge(op.calling_task, actor_id, EdgeType.CONTROL)] if op.calling_task else [],
        )
        self._record_step(create_event)
        
        return actor_id
    
    def execute_actor_method(self, op: ActorMethodCallOp) -> List[ObjectID]:
        """Execute an actor method call.
        
        Paper: Actor method calls follow the same scheduling flow as tasks:
        1. Driver submits to local scheduler
        2. Local scheduler forwards to global scheduler (because actor has node constraint)
        3. Global scheduler assigns to the actor's node
        4. Target node's local scheduler dispatches to actor worker
        """
        task_id = self._next_task_id()
        result_ids = [self._next_object_id()]
        
        actor_info = self.actors_info.get(op.actor_id)
        if not actor_info:
            return result_ids
        
        # Actor methods MUST run on the actor's node
        target_node_id = actor_info.node_id
        
        task_spec = TaskSpec(
            task_id=task_id,
            function_name=f"{op.actor_id}.{op.method_name}",
            args=op.args,
            num_returns=1,
            resources={"CPU": 1},
            is_actor_method=True,
            actor_id=op.actor_id,
            calling_task=op.calling_task,
            node_constraint=target_node_id,
        )
        
        task_info = TaskInfo(spec=task_spec, status=TaskStatus.PENDING,
                            result_objects=result_ids, assigned_node=target_node_id)
        self.tasks[task_id] = task_info
        
        # Register task in GCS
        self.gcs.register_task(task_id, {
            "function": f"{op.actor_id}.{op.method_name}",
            "args": op.args,
            "status": "pending",
            "node": None,
        })
        
        # Add to task graph
        task_label = op.label if op.label else f"{op.method_name}()"
        self._add_graph_node(task_id, "actor_method", task_label)
        
        # Data edges: args → task
        for arg_id in op.args:
            self._add_graph_edge(arg_id, task_id, EdgeType.DATA)
        
        # Stateful edge: from previous method (or actor init if first method) to current method
        stateful_source = actor_info.last_method_task if actor_info.last_method_task else actor_info.actor_id
        self._add_graph_edge(stateful_source, task_id, EdgeType.STATEFUL)
        
        # Control edge if nested
        if op.calling_task:
            self._add_graph_edge(op.calling_task, task_id, EdgeType.CONTROL)
        
        # Data edges: task → results
        for rid in result_ids:
            result_label = op.result_label if op.result_label else f"result"
            self._add_graph_node(rid, "data", result_label)
            self._add_graph_edge(task_id, rid, EdgeType.DATA)
        
        calling_node = self.nodes[op.calling_node]
        
        # ---- Step 1: Driver submits to local scheduler ----
        stateful_info = f" Stateful edge: {stateful_source} → {task_id}."
        
        submit_event = StepEvent(
            phase=StepPhase.TASK_SUBMIT,
            description=f"Call {op.actor_id}.{op.method_name}() → {task_id}",
            detail=f"Driver on {op.calling_node} invokes {op.actor_id}.{op.method_name}(). "
                   f"The call is submitted to {op.calling_node}'s local scheduler.{stateful_info}",
            source=f"{op.calling_node}_driver",
            target=f"{op.calling_node}_local_scheduler",
            highlights=[HighlightHint(f"{op.calling_node}_local_scheduler", "active")],
            arrows=[ArrowHint(f"{op.calling_node}_driver", f"{op.calling_node}_local_scheduler",
                              f"submit {task_id}", "control")],
        )
        self._record_step(submit_event)
        
        # ---- Step 2: Local scheduler forwards to global scheduler ----
        # Actor methods always need forwarding because they have node constraints
        self.global_scheduler.enqueue(task_id)
        
        # Query GCS for arg locations
        arg_locations = {}
        for arg_id in op.args:
            loc = self.gcs.get_object_location(arg_id)
            if loc:
                arg_locations[arg_id] = loc["location"]
        
        forward_event = StepEvent(
            phase=StepPhase.GLOBAL_SCHEDULE,
            description=f"{op.calling_node} forwards {task_id} → global scheduler assigns to {target_node_id}",
            detail=f"Local scheduler on {op.calling_node} forwards actor method {task_id} to global scheduler "
                   f"(actor {op.actor_id} is bound to {target_node_id}). "
                   f"Global scheduler assigns to {target_node_id}.",
            source=f"{op.calling_node}_local_scheduler",
            target=f"{target_node_id}_local_scheduler",
            highlights=[HighlightHint("global_scheduler", "active"), HighlightHint(f"{target_node_id}_local_scheduler", "new")],
            arrows=[ArrowHint(f"{op.calling_node}_local_scheduler", "global_scheduler",
                              f"forward {task_id}", "control"),
                    ArrowHint("global_scheduler", f"{target_node_id}_local_scheduler",
                              f"assign {task_id}", "control")],
        )
        self._record_step(forward_event)
        
        # Dequeue from global, enqueue into target node
        self.global_scheduler.dequeue(task_id)
        target_node = self.nodes[target_node_id]
        target_node.local_scheduler.enqueue(task_id)
        self.global_scheduler.update_node_load(
            target_node_id, max(len(target_node.local_scheduler.task_queue), target_node.local_scheduler.admitted_recent),
            target_node.local_scheduler.available_resources)
        
        # ---- Step 3: Fetch missing args ----
        missing_args = [arg_id for arg_id in op.args if not target_node.object_store.has(arg_id)]
        fetch_events = []
        for arg_id in missing_args:
            loc_info = self.gcs.get_object_location(arg_id)
            source_node_id = loc_info["location"] if loc_info else op.calling_node
            value = self.nodes[source_node_id].object_store.get(arg_id)
            target_node.object_store.put(arg_id, value)
            
            fetch_events.append(StepEvent(
                phase=StepPhase.DATA_FETCH,
                description=f"{target_node_id} fetches missing arg: {arg_id} from {source_node_id}",
                detail=f"{target_node_id}'s object store doesn't have {arg_id}. "
                       f"Looks up GCS → found at {source_node_id}. Replicates locally.",
                source=f"{source_node_id}_object_store",
                target=f"{target_node_id}_object_store",
                highlights=[HighlightHint(f"{target_node_id}_object_store", "modified"),
                           HighlightHint("GCS_object_table", "active")],
                arrows=[ArrowHint(f"{target_node_id}_object_store", "GCS",
                                  f"lookup {arg_id}", "control", "dashed"),
                        ArrowHint(f"{source_node_id}_object_store", f"{target_node_id}_object_store",
                                  f"replicate {arg_id}", "data")],
            ))
        
        if fetch_events:
            self._record_step(fetch_events[0])
        
        # ---- Step 4: Local scheduler dispatches to actor worker ----
        target_node.local_scheduler.dequeue()
        self.global_scheduler.update_node_load(
            target_node_id, max(len(target_node.local_scheduler.task_queue), target_node.local_scheduler.admitted_recent),
            target_node.local_scheduler.available_resources)
        
        actor_worker = target_node.workers[0] if target_node.workers else f"{target_node_id}_worker_0"
        target_node.worker_tasks[actor_worker] = task_id
        task_info.status = TaskStatus.RUNNING
        self._graph_nodes[task_id].status = "running"
        
        dispatch_event = StepEvent(
            phase=StepPhase.ACTOR_METHOD,
            description=f"Local scheduler on {target_node_id} dispatches {task_id} → {actor_worker}",
            detail=f"The local scheduler dispatches actor method {op.method_name}() to {actor_worker}.",
            source=f"{target_node_id}_local_scheduler",
            target=f"{target_node_id}_actor",
            highlights=[HighlightHint(f"{target_node_id}_actor", "active"),
                       HighlightHint(f"{target_node_id}_local_scheduler", "active")],
            arrows=[ArrowHint(f"{target_node_id}_local_scheduler", f"{target_node_id}_actor",
                              f"dispatch {task_id}", "control")],
        )
        self._record_step(dispatch_event)
        
        # ---- Step 5: Worker executes and stores results ----
        target_node.worker_tasks[actor_worker] = None
        
        task_info.status = TaskStatus.COMPLETED
        self._graph_nodes[task_id].status = "completed"
        
        for rid, rval in zip(result_ids, [f"<rollout_result>"]):
            target_node.object_store.put(rid, rval)
            self.object_values[rid] = rval
            self.gcs.register_object(rid, target_node_id, created_by=task_id)
            self._graph_nodes[rid].status = "completed"
        
        actor_info.last_method_task = task_id
        self.gcs.update_actor_last_method(op.actor_id, task_id)
        
        complete_event = StepEvent(
            phase=StepPhase.ACTOR_METHOD,
            description=f"Actor {op.actor_id}.{op.method_name}() completes → {result_ids}",
            detail=f"Worker {actor_worker} finishes {op.method_name}(), stores result in local object store, "
                   f"registers in GCS. Actor state updated.",
            source=f"{target_node_id}_actor",
            target=f"{target_node_id}_object_store",
            highlights=[HighlightHint(f"{target_node_id}_actor", "active"),
                       HighlightHint(f"{target_node_id}_object_store", "new"),
                       HighlightHint("GCS_object_table", "new")],
            arrows=[ArrowHint(f"{target_node_id}_actor", f"{target_node_id}_object_store",
                              f"store {result_ids[0]}", "data"),
                    ArrowHint(f"{target_node_id}_object_store", "GCS",
                              f"register {result_ids[0]}", "control", "dashed")],
        )
        self._record_step(complete_event)
        
        return result_ids
        
        # Actor methods MUST run on the actor's node
        target_node_id = actor_info.node_id
        
        task_spec = TaskSpec(
            task_id=task_id,
            function_name=f"{op.actor_id}.{op.method_name}",
            args=op.args,
            num_returns=1,
            resources={"CPU": 1},
            is_actor_method=True,
            actor_id=op.actor_id,
            calling_task=op.calling_task,
            node_constraint=target_node_id,
        )
        
        task_info = TaskInfo(spec=task_spec, status=TaskStatus.PENDING,
                            result_objects=result_ids, assigned_node=target_node_id)
        self.tasks[task_id] = task_info
        
        # Add to task graph
        task_label = op.label if op.label else f"{op.method_name}()"
        self._add_graph_node(task_id, "actor_method", task_label)
        
        # Data edges: args → task
        for arg_id in op.args:
            self._add_graph_edge(arg_id, task_id, EdgeType.DATA)
        
        # Stateful edge: from previous method (or actor init if first method) to current method
        stateful_source = actor_info.last_method_task if actor_info.last_method_task else actor_info.actor_id
        self._add_graph_edge(stateful_source, task_id, EdgeType.STATEFUL)
        
        # Control edge if nested
        if op.calling_task:
            self._add_graph_edge(op.calling_task, task_id, EdgeType.CONTROL)
        
        # Data edges: task → results
        for rid in result_ids:
            result_label = op.result_label if op.result_label else f"result"
            self._add_graph_node(rid, "data", result_label)
            self._add_graph_edge(task_id, rid, EdgeType.DATA)
        
        target_node = self.nodes[target_node_id]
        
        # Enqueue into target node's local scheduler
        target_node.local_scheduler.enqueue(task_id)
        self.global_scheduler.update_node_load(
            target_node_id, max(len(target_node.local_scheduler.task_queue), target_node.local_scheduler.admitted_recent),
            target_node.local_scheduler.available_resources)
        
        # Fetch missing args
        missing_args = [arg_id for arg_id in op.args if not target_node.object_store.has(arg_id)]
        for arg_id in missing_args:
            loc_info = self.gcs.get_object_location(arg_id)
            source_node_id = loc_info["location"] if loc_info else op.calling_node
            value = self.nodes[source_node_id].object_store.get(arg_id)
            target_node.object_store.put(arg_id, value)
        
        # Step 1: Actor method call + fetch args
        stateful_info = f" Stateful edge: {stateful_source} → {task_id}."
        
        fetch_info = ""
        if missing_args:
            fetch_info = f" Fetched missing args: {', '.join(missing_args)}."
        
        event = StepEvent(
            phase=StepPhase.ACTOR_METHOD,
            description=f"Call {op.actor_id}.{op.method_name}() → {task_id} on {target_node_id}",
            detail=f"Actor method {op.method_name}() is invoked on {op.actor_id} (node {target_node_id}). "
                   f"Methods on the same actor execute serially.{stateful_info}{fetch_info}",
            source=f"{op.calling_node}_driver",
            target=f"{target_node_id}_actor",
            highlights=[HighlightHint(f"{target_node_id}_actor", "active")],
            arrows=[ArrowHint(f"{op.calling_node}_driver", f"{target_node_id}_actor",
                              f"call {op.method_name}()", "control")],
            new_graph_nodes=[TaskGraphNode(task_id, "actor_method", f"{op.method_name}()", "pending")],
            new_graph_edges=[TaskGraphEdge(stateful_source, task_id, EdgeType.STATEFUL)] +
                           ([TaskGraphEdge(op.calling_task, task_id, EdgeType.CONTROL)]
                           if op.calling_task else []),
        )
        self._record_step(event)
        
        # Step 2a: Local scheduler dispatches to actor worker (worker becomes busy)
        target_node.local_scheduler.dequeue()
        self.global_scheduler.update_node_load(
            target_node_id, max(len(target_node.local_scheduler.task_queue), target_node.local_scheduler.admitted_recent),
            target_node.local_scheduler.available_resources)
        
        actor_worker = target_node.workers[0] if target_node.workers else f"{target_node_id}_worker_0"
        target_node.worker_tasks[actor_worker] = task_id
        task_info.status = TaskStatus.RUNNING
        self.gcs.update_task_status(task_id, "running", target_node_id)
        self._graph_nodes[task_id].status = "running"
        
        dispatch_event = StepEvent(
            phase=StepPhase.ACTOR_METHOD,
            description=f"Local scheduler on {target_node_id} dispatches {task_id} → {actor_worker} (actor worker)",
            detail=f"The local scheduler dispatches actor method {op.method_name}() to {actor_worker}. "
                   f"Worker is now busy executing {task_id}.",
            source=f"{target_node_id}_local_scheduler",
            target=f"{target_node_id}_actor",
            highlights=[HighlightHint(f"{target_node_id}_actor", "active"),
                       HighlightHint(f"{target_node_id}_local_scheduler", "active")],
            arrows=[ArrowHint(f"{target_node_id}_local_scheduler", f"{target_node_id}_actor",
                              f"dispatch {task_id}", "control")],
        )
        self._record_step(dispatch_event)
        
        # Step 2b: Execute method, store results, register in GCS
        result_values = self._simulate_function_execution(
            f"{op.actor_id}.{op.method_name}", op.args
        )
        
        target_node.worker_tasks[actor_worker] = None
        
        task_info.status = TaskStatus.COMPLETED
        self.gcs.update_task_status(task_id, "completed", target_node_id)
        self._graph_nodes[task_id].status = "completed"
        
        for rid, rval in zip(result_ids, result_values):
            target_node.object_store.put(rid, rval)
            self.object_values[rid] = rval
            self.gcs.register_object(rid, target_node_id, created_by=task_id)
            self._graph_nodes[rid].status = "completed"
        
        actor_info.last_method_task = task_id
        self.gcs.update_actor_last_method(op.actor_id, task_id)
        
        complete_event = StepEvent(
            phase=StepPhase.ACTOR_METHOD,
            description=f"Actor {op.actor_id}.{op.method_name}() completes → {result_ids}",
            detail=f"{actor_worker} finishes executing {op.actor_id}.{op.method_name}(). "
                   f"Result {result_ids} stored in local object store and registered with GCS. "
                   f"Worker is now idle.",
            source=f"{target_node_id}_actor",
            target=f"{target_node_id}_object_store",
            highlights=[HighlightHint(f"{target_node_id}_actor", "active"),
                       HighlightHint(f"{target_node_id}_object_store", "new"),
                       HighlightHint("GCS_object_table", "new")],
            arrows=[ArrowHint(f"{target_node_id}_actor", f"{target_node_id}_object_store",
                              f"write {result_ids}", "data"),
                    ArrowHint(f"{target_node_id}_object_store", "GCS",
                              f"register {result_ids}@{target_node_id}", "control", "dashed")],
            new_graph_nodes=[TaskGraphNode(rid, "data", f"result", "completed") for rid in result_ids],
            new_graph_edges=[TaskGraphEdge(task_id, rid, EdgeType.DATA) for rid in result_ids],
        )
        self._record_step(complete_event)
        
        return result_ids
    
    def execute_burst_start(self, op: BurstStart):
        """Open a batch-submission window — tasks queue up without dispatching.

        Used by demos (e.g. load_balancing) that want the audience to see the
        local queue actually fill up and trigger overload-based forwarding.
        """
        self._burst_mode = True
        self._burst_pending = []
        event = StepEvent(
            phase=StepPhase.TASK_SUBMIT,
            description=f"⏸ Burst window opened — submitted tasks will queue but not dispatch",
            detail=(op.note or "")
                   + " Bottom-up scheduling (§4.3) admits tasks locally until the queue "
                     "hits the overload threshold; subsequent submissions are forwarded "
                     "to the global scheduler. Watch the local-queue depth grow.",
            source="driver",
            target="system",
        )
        self._record_step(event)

    def execute_burst_end(self, op: BurstEnd):
        """Drain all tasks queued since BurstStart in submission order."""
        self._burst_mode = False
        pending = self._burst_pending
        self._burst_pending = []
        event = StepEvent(
            phase=StepPhase.TASK_SUBMIT,
            description=f"▶ Burst window closes — draining {len(pending)} queued task(s)",
            detail=(op.note or "")
                   + " Each node's local scheduler now dispatches what it accepted. "
                     "Forwarded tasks dispatch on their assigned nodes — all three nodes "
                     "participate in execution.",
            source="system",
            target="system",
        )
        self._record_step(event)
        for entry in pending:
            self._finish_remote_call(
                entry["task_id"], entry["selected_node"],
                entry["op"], entry["result_ids"],
            )

    def execute_node_fail(self, op: NodeFailOp):
        """Simulate a node crash (paper §4.2.1 / Figure 11).

        The node's object store is wiped and the node is marked dead.
        Subsequent ray.get() calls on objects that lived only on this node
        will trigger lineage replay.
        """
        if op.node_id not in self.nodes:
            return
        node = self.nodes[op.node_id]
        lost_objects = list(node.object_store.objects.keys())
        lost_actors = list(node.actors)
        node.object_store.objects.clear()
        node.local_scheduler.task_queue.clear()
        for w in node.worker_tasks:
            node.worker_tasks[w] = None
        node.is_dead = True
        # Detach actors from the node so future actor methods would need replay.
        for aid in lost_actors:
            ainfo = self.actors_info.get(aid)
            if ainfo:
                ainfo.node_id = ""
        # Remove from global scheduler's load table so it's not selected again.
        self.global_scheduler.node_loads.pop(op.node_id, None)

        event = StepEvent(
            phase=StepPhase.NODE_FAIL,
            description=f"💥 Node {op.node_id} fails — local object store and worker state lost",
            detail=f"Node {op.node_id} crashes. Its object store loses {len(lost_objects)} object(s) "
                   f"({lost_objects}). Actors {lost_actors} are gone. The GCS still holds the lineage "
                   f"(task_table + object_table.created_by), which Ray uses to reconstruct objects "
                   f"on demand (paper §4.2.1 'fault tolerance via lineage').",
            source=op.node_id,
            target="system",
            highlights=[HighlightHint(f"{op.node_id}_object_store", "modified"),
                        HighlightHint(f"{op.node_id}_local_scheduler", "modified"),
                        HighlightHint("GCS_object_table", "active")],
        )
        self._record_step(event)

    def _ensure_object_alive(self, object_id: ObjectID,
                              calling_node: NodeID,
                              depth: int = 0) -> Optional[NodeID]:
        """Make sure object_id exists on some living node, replaying lineage if not.

        Returns the node id where the object now lives, or None if it cannot be
        reconstructed (e.g. came from ray.put on a dead node — no lineage).
        Used by execute_get to implement paper §4.2.1 lineage-based recovery.
        """
        # 1. Is it alive on a non-dead node? (GCS location may be stale.)
        for nid, n in self.nodes.items():
            if not n.is_dead and n.object_store.has(object_id):
                return nid

        # 2. Look up lineage in GCS object_table.
        obj_entry = self.gcs.get_object_location(object_id)
        if not obj_entry:
            return None
        producing_task = obj_entry.get("created_by")
        if not producing_task or producing_task not in self.tasks:
            # No lineage (e.g. ray.put on a now-dead node).
            event = StepEvent(
                phase=StepPhase.LINEAGE_REPLAY,
                description=f"⚠ Object {object_id} cannot be reconstructed — no lineage",
                detail=f"GCS has no producing task for {object_id} (it was created by ray.put "
                       f"on a now-dead node). The paper notes ray.put values are not lineage-recoverable.",
                source="GCS",
                target=f"{calling_node}_driver",
                highlights=[HighlightHint("GCS_object_table", "active")],
            )
            self._record_step(event)
            return None

        task_info = self.tasks[producing_task]
        spec = task_info.spec

        # 3. Announce the replay.
        event = StepEvent(
            phase=StepPhase.LINEAGE_REPLAY,
            description=f"🔁 Lineage replay: re-execute {producing_task} ({spec.function_name}) to recover {object_id}",
            detail=f"GCS object_table says {object_id} was created_by={producing_task}. "
                   f"The GCS task_table still has the TaskSpec → Ray re-executes the task on a surviving node "
                   f"(paper §4.2.1, Figure 11). Recursing into dependencies first if needed.",
            source="GCS",
            target="global_scheduler",
            highlights=[HighlightHint("GCS_task_table", "active"),
                        HighlightHint("GCS_object_table", "active")],
            arrows=[ArrowHint("GCS", "global_scheduler", f"replay {producing_task}", "control", "dashed")],
        )
        self._record_step(event)

        # 4. Recursively ensure each arg is alive.
        for arg in spec.args:
            self._ensure_object_alive(arg, calling_node, depth + 1)

        # 5. Pick a surviving node that satisfies the resources.
        survivors = [nid for nid, n in self.nodes.items() if not n.is_dead]
        if not survivors:
            return None
        new_node_id = self.global_scheduler.select_node(spec, self.gcs, survivors) or survivors[0]
        new_node = self.nodes[new_node_id]

        # 6. Make sure args are present on the new node (replicate from living copies).
        for arg in spec.args:
            if new_node.object_store.has(arg):
                continue
            for src_id, src in self.nodes.items():
                if not src.is_dead and src.object_store.has(arg):
                    new_node.object_store.put(arg, src.object_store.get(arg))
                    break

        # 7. Re-execute the function and re-register the outputs.
        new_values = self._simulate_function_execution(spec.function_name, spec.args)
        for rid, val in zip(task_info.result_objects, new_values):
            new_node.object_store.put(rid, val)
            self.object_values[rid] = val
            self.gcs.register_object(rid, new_node_id, created_by=producing_task)
        task_info.assigned_node = new_node_id

        event = StepEvent(
            phase=StepPhase.LINEAGE_REPLAY,
            description=f"✅ {producing_task} replayed on {new_node_id} → {task_info.result_objects} reconstructed",
            detail=f"Re-executed {spec.function_name}({spec.args}) on {new_node_id}. "
                   f"Result(s) {task_info.result_objects} re-registered in GCS. The downstream ray.get() "
                   f"can now proceed transparently.",
            source=f"{new_node_id}_worker",
            target=f"{new_node_id}_object_store",
            highlights=[HighlightHint(f"{new_node_id}_object_store", "new"),
                        HighlightHint("GCS_object_table", "new")],
            arrows=[ArrowHint(f"{new_node_id}_worker", f"{new_node_id}_object_store",
                              f"write {task_info.result_objects}", "data"),
                    ArrowHint(f"{new_node_id}_object_store", "GCS",
                              f"re-register {task_info.result_objects}@{new_node_id}",
                              "control", "dashed")],
        )
        self._record_step(event)
        return new_node_id

    def execute_get(self, op: GetOp):
        """Execute ray.get() - retrieve the value of a future.
        
        Simplified: 1 step (local hit or fetch+return), or 2 steps (wait+fetch for pending).
        """
        node = self.nodes[op.calling_node]

        # If GCS says the object lives on a dead node (or is missing entirely),
        # trigger lineage replay before proceeding.
        loc_info_pre = self.gcs.get_object_location(op.object_id)
        if loc_info_pre:
            loc_node = self.nodes.get(loc_info_pre["location"])
            if loc_node and loc_node.is_dead and not node.object_store.has(op.object_id):
                self._ensure_object_alive(op.object_id, op.calling_node)

        if node.object_store.has(op.object_id):
            # Object is local — single step
            value = node.object_store.get(op.object_id)
            
            event = StepEvent(
                phase=StepPhase.RESULT_GET,
                description=f"ray.get({op.object_id}) → returns {value}",
                detail=f"Object {op.object_id} found in local object store on {op.calling_node}. Value: {value}",
                source=f"{op.calling_node}_driver",
                target=f"{op.calling_node}_object_store",
                highlights=[HighlightHint(f"{op.calling_node}_object_store", "active")],
                arrows=[ArrowHint(f"{op.calling_node}_driver", f"{op.calling_node}_object_store",
                                  f"get({op.object_id})", "data")],
            )
            self._record_step(event)
        else:
            # Need to fetch from remote — single step
            loc_info = self.gcs.get_object_location(op.object_id)
            
            if loc_info:
                source_node_id = loc_info["location"]
                value = self.nodes[source_node_id].object_store.get(op.object_id)
                node.object_store.put(op.object_id, value)
                
                event = StepEvent(
                    phase=StepPhase.RESULT_GET,
                    description=f"ray.get({op.object_id}) → fetched from {source_node_id}, returns {value}",
                    detail=f"Object {op.object_id} not local. GCS lookup → {source_node_id}. "
                           f"Replicated and returned. Value: {value}",
                    source=f"{source_node_id}_object_store",
                    target=f"{op.calling_node}_object_store",
                    highlights=[HighlightHint(f"{op.calling_node}_object_store", "new"),
                               HighlightHint("GCS_object_table", "active")],
                    arrows=[ArrowHint(f"{op.calling_node}_object_store", "GCS",
                                      f"lookup {op.object_id}", "control", "dashed"),
                            ArrowHint(f"{source_node_id}_object_store", f"{op.calling_node}_object_store",
                                      f"replicate {op.object_id}", "data")],
                )
                self._record_step(event)
            else:
                # Object not yet available — wait then fetch
                event = StepEvent(
                    phase=StepPhase.RESULT_GET,
                    description=f"ray.get({op.object_id}) → waiting for result...",
                    detail=f"Object {op.object_id} has not been created yet. "
                           f"The driver registers a callback with GCS and waits.",
                    source=f"{op.calling_node}_object_store",
                    target="GCS",
                    highlights=[HighlightHint("GCS_object_table", "active")],
                    arrows=[ArrowHint(f"{op.calling_node}_object_store", "GCS",
                                      f"subscribe({op.object_id})", "control", "dashed")],
                )
                self._record_step(event)
                
                # Find the producing task and fetch
                producing_task = None
                for tid, tinfo in self.tasks.items():
                    if op.object_id in tinfo.result_objects:
                        producing_task = tid
                        break
                
                if producing_task:
                    source_node_id = self.tasks[producing_task].assigned_node
                    if source_node_id and self.nodes[source_node_id].object_store.has(op.object_id):
                        value = self.nodes[source_node_id].object_store.get(op.object_id)
                        node.object_store.put(op.object_id, value)
                        
                        event = StepEvent(
                            phase=StepPhase.RESULT_GET,
                            description=f"ray.get({op.object_id}) → returns {value}",
                            detail=f"Object created by {producing_task} on {source_node_id}. "
                                   f"Replicated to {op.calling_node}. Value: {value}",
                            source=f"{source_node_id}_object_store",
                            target=f"{op.calling_node}_object_store",
                            highlights=[HighlightHint(f"{op.calling_node}_object_store", "new")],
                        )
                        self._record_step(event)
    
    def _simulate_function_execution(self, function_name: str, args: List[ObjectID]) -> List[Any]:
        """Simulate the execution of a remote function.
        
        In a real system, this would execute the actual function code.
        In our simulation, we compute simple results for demonstration.
        """
        arg_values = [self.object_values.get(a, f"<{a}>") for a in args]
        
        if function_name == "add":
            try:
                result = sum(v for v in arg_values if isinstance(v, (int, float)))
            except:
                result = f"add({arg_values})"
            return [result]
        elif function_name == "create_policy":
            return ["<policy_v0>"]
        elif function_name == "update_policy":
            return ["<policy_v1>"]
        elif "." in function_name:
            # Actor method
            parts = function_name.split(".")
            method = parts[-1]
            if method == "rollout":
                return [f"<observations>"]
            elif method == "__init__":
                return [None]
            else:
                return [f"<{method}_result>"]
        else:
            return [f"<{function_name}_result>"]
    
    def run_program(self, program: RayProgram):
        """Execute a complete Ray program, recording all steps."""
        # Initialize cluster
        self.nodes.clear()
        self.gcs = GlobalControlStore()
        self.global_scheduler = GlobalScheduler()
        self.tasks.clear()
        self.actors_info.clear()
        self.functions.clear()
        self.object_values.clear()
        self.steps.clear()
        self.current_step = -1
        self._step_counter = 0
        self._task_counter = 0
        self._object_counter = 0
        self._actor_counter = 0
        self._graph_nodes.clear()
        self._graph_edges.clear()
        
        self.initialize_cluster(program.num_nodes, program.node_labels,
                               program.node_resources, program.driver_node)
        
        # Track parent tasks that are still running
        running_parent_tasks = set()
        
        # Execute each operation
        for op in program.operations:
            # If this operation has a parent_task, mark parent as running
            if hasattr(op, 'parent_task') and op.parent_task:
                if op.parent_task in self.tasks:
                    parent_task_info = self.tasks[op.parent_task]
                    if parent_task_info.status == TaskStatus.COMPLETED:
                        # Revert parent to running state for nested calls
                        parent_task_info.status = TaskStatus.RUNNING
                        if op.parent_task in self._graph_nodes:
                            self._graph_nodes[op.parent_task].status = "running"
                    running_parent_tasks.add(op.parent_task)
            
            if isinstance(op, RegisterFunction):
                self.execute_register_function(op)
            elif isinstance(op, RegisterActorClass):
                self.execute_register_actor_class(op)
            elif isinstance(op, PutOp):
                self.execute_put(op)
            elif isinstance(op, RemoteCallOp):
                result_ids = self.execute_remote_call(op)
                # Store the mapping for future reference
                for rid in result_ids:
                    self.object_values[rid] = self.object_values.get(rid, rid)
            elif isinstance(op, ActorCreateOp):
                self.execute_actor_create(op)
            elif isinstance(op, ActorMethodCallOp):
                result_ids = self.execute_actor_method(op)
                for rid in result_ids:
                    self.object_values[rid] = self.object_values.get(rid, rid)
            elif isinstance(op, GetOp):
                self.execute_get(op)
            elif isinstance(op, NodeFailOp):
                self.execute_node_fail(op)
            elif isinstance(op, BurstStart):
                self.execute_burst_start(op)
            elif isinstance(op, BurstEnd):
                self.execute_burst_end(op)
            
            # Check if this was the last nested call for a parent task
            if hasattr(op, 'parent_task') and op.parent_task:
                # Check if there are more operations with the same parent_task
                current_idx = program.operations.index(op)
                has_more_nested = False
                for next_op in program.operations[current_idx + 1:]:
                    if hasattr(next_op, 'parent_task') and next_op.parent_task == op.parent_task:
                        has_more_nested = True
                        break
                
                if not has_more_nested:
                    # No more nested calls, mark parent as completed
                    if op.parent_task in self.tasks:
                        parent_task_info = self.tasks[op.parent_task]
                        parent_task_info.status = TaskStatus.COMPLETED
                        if op.parent_task in self._graph_nodes:
                            self._graph_nodes[op.parent_task].status = "completed"
                    running_parent_tasks.discard(op.parent_task)
        
        # Add final step
        event = StepEvent(
            phase=StepPhase.INIT,
            description="Program execution complete",
            detail="All operations have been executed. The program is complete.",
            source="system",
            target="system",
        )
        self._record_step(event)
    
    def get_snapshot(self, step: int) -> Optional[SystemSnapshot]:
        """Get the system state at a given step."""
        if 0 <= step < len(self.steps):
            return self.steps[step]
        return None
    
    def get_total_steps(self) -> int:
        return len(self.steps)
    
    def to_json(self, step: int) -> Dict:
        """Serialize the system state at a given step to JSON for the frontend."""
        snapshot = self.get_snapshot(step)
        if not snapshot:
            return {"error": "invalid step"}
        
        result = {
            "step_number": step,
            "total_steps": len(self.steps),
            "event": {
                "step_number": snapshot.event.step_number,
                "phase": snapshot.event.phase.value,
                "description": snapshot.event.description,
                "detail": snapshot.event.detail,
                "source": snapshot.event.source,
                "target": snapshot.event.target,
                "arrows": [{"from": a.from_id, "to": a.to_id, "label": a.label,
                           "type": a.arrow_type, "style": a.style}
                          for a in snapshot.event.arrows],
                "highlights": [{"id": h.component_id, "type": h.highlight_type}
                              for h in snapshot.event.highlights],
                "data_changes": snapshot.event.data_changes,
            },
            "nodes": {},
            "gcs": {
                "object_table": snapshot.gcs.object_table,
                "task_table": snapshot.gcs.task_table,
                "function_table": snapshot.gcs.function_table,
                "actor_table": snapshot.gcs.actor_table,
            },
            "global_scheduler_queue": snapshot.global_scheduler_queue,
            "task_graph": {
                "nodes": {nid: {"id": n.node_id, "type": n.node_type, "label": n.label, "status": n.status}
                         for nid, n in snapshot.task_graph_nodes.items()},
                "edges": [{"from": e.from_id, "to": e.to_id, "type": e.edge_type.value}
                         for e in snapshot.task_graph_edges],
            },
        }
        
        for nid, nstate in snapshot.nodes.items():
            result["nodes"][nid] = {
                "node_id": nstate.node_id,
                "is_driver": nstate.is_driver,
                "is_dead": nstate.is_dead,
                "object_store": {k: str(v) for k, v in nstate.object_store.items()},
                "local_queue": nstate.local_queue,
                "workers": nstate.workers,
                "worker_tasks": nstate.worker_tasks,
                "actors": nstate.actors,
                "resources": nstate.resources,
            }
        
        return result
