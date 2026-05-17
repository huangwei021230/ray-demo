"""
Ray Demo - Example Programs

Two example programs matching the paper:
1. add(a, b) - Paper's Figure 5 end-to-end example
2. RL Training - Paper's Figure 2 dynamic task graph example
"""

from .types import *


def create_add_example() -> RayProgram:
    """Create the add(a, b) example from the paper's Figure 5.

    This is the paper's primary walkthrough example showing the
    complete end-to-end flow of a remote function call:
    - Function registration
    - Task submission from driver to local scheduler
    - Local scheduler decision (forward to global)
    - Global scheduler queries GCS and selects node
    - Data replication for missing arguments
    - Worker execution via shared memory
    - Result registration in GCS
    - ray.get() retrieves result via GCS pub-sub
    """
    return RayProgram(
        name="add(a, b) — Paper Figure 5",
        description=(
            "The paper's end-to-end example: executing add.remote(a, b) where "
            "a is on N1 and b is on N2. Shows the complete flow through "
            "local scheduler → global scheduler → GCS lookup → data replication → "
            "worker execution → result registration → ray.get(). "
            "Matches Figure 5 in the paper."
        ),
        num_nodes=2,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 0},
        },
        operations=[
            # Step 0: Register function
            RegisterFunction(
                function_name="add",
                num_returns=1,
                resources={"CPU": 1},
            ),
            # Put objects a and b on different nodes
            PutOp(object_id="a", value=1, node="N1"),
            PutOp(object_id="b", value=2, node="N2"),
            # Call add.remote(a, b) from N1
            RemoteCallOp(
                function_name="add",
                args=["a", "b"],
                calling_node="N1",
            ),
            # Get the result
            GetOp(object_id="obj_0", calling_node="N1"),
        ],
    )


def create_rl_example() -> RayProgram:
    """Create the RL training example from the paper's Figure 2.

    This matches the Python code in the paper's Figure 1:
    - @ray.remote def create_policy()
    - @ray.remote(num_gpus=1) class Simulator with rollout method
    - @ray.remote(num_gpus=2) def update_policy(policy, *rollouts)
    - @ray.remote def train_policy()

    Shows: actors, stateful edges, nested remote calls, dynamic task graph.
    We model two iterations of the training loop with 2 simulators.

    Task graph structure (matching paper Figure 2):
    - T0 (train_policy) is the top-level task that controls everything
    - T1 (create_policy) creates initial policy
    - A10, A20 are Simulator actors
    - A11, A21 are first rollout iterations
    - A12, A22 are second rollout iterations
    - T2, T3 are update_policy calls

    Edges:
    - Control edges: T0 → {A10, A20, T1, A11, A21}
    - Data edges: T1→policy1, policy1→{A11,A21}, A11→rollout11, etc.
    - Stateful edges: A10→{A11,A12}, A20→{A21,A22} (from actor init to methods)
    """
    return RayProgram(
        name="RL Training — Paper Figure 2",
        description=(
            "The paper's RL training example (Figure 1 & 2): train_policy() "
            "creates a policy and simulator actors, runs parallel rollouts, "
            "and updates the policy. Demonstrates: actors, stateful edges, "
            "nested remote calls, and the dynamic task graph computation model."
        ),
        num_nodes=3,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (GPU×2)", "N3": "N3 (GPU×1)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 2},
            "N3": {"CPU": 4, "GPU": 1},
        },
        operations=[
            # Register functions and actor classes
            RegisterFunction(
                function_name="create_policy",
                num_returns=1,
                resources={"CPU": 1},
            ),
            RegisterFunction(
                function_name="update_policy",
                num_returns=1,
                resources={"CPU": 1, "GPU": 2},
            ),
            RegisterActorClass(
                class_name="Simulator",
                methods=["rollout"],
                resources={"CPU": 1, "GPU": 1},
            ),
            RegisterFunction(
                function_name="train_policy",
                num_returns=1,
                resources={"CPU": 1},
            ),
            # === T0: train_policy() — top-level task ===
            # This creates task_0 which will be T0
            # create_result=False because T0 is a control task with no data output
            RemoteCallOp(
                function_name="train_policy",
                args=[],
                calling_node="N1",
                create_result=False,
                label="T0: train_policy",
            ),
            # === T0's body: all subsequent calls have calling_task="task_0" ===
            # 1. T1: create_policy() — creates initial policy (policy1 = obj_0)
            RemoteCallOp(
                function_name="create_policy",
                args=[],
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → T1
                label="T1: create_policy",
                result_label="policy1",
            ),
            # 2. A10, A20: Create simulator actors
            ActorCreateOp(
                class_name="Simulator",
                actor_id="A10",  # Actor 1
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A10
            ),
            ActorCreateOp(
                class_name="Simulator",
                actor_id="A20",  # Actor 2
                calling_node="N2",
                calling_task="task_0",  # Control edge: T0 → A20
            ),
            # 3. A11, A21: First rollout iteration
            # Stateful edges: A10→A11, A20→A21 (created automatically by engine)
            # Data edges: policy1(obj_0) → A11, A11 → rollout11(obj_1)
            ActorMethodCallOp(
                actor_id="A10",
                method_name="rollout",
                args=["obj_0"],  # policy1
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A11
                label="A11: rollout",
                result_label="rollout11",
            ),
            ActorMethodCallOp(
                actor_id="A20",
                method_name="rollout",
                args=["obj_0"],  # policy1
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A21
                label="A21: rollout",
                result_label="rollout21",
            ),
            # 4. T2: First update_policy
            # Data edges: policy1(obj_0), rollout11(obj_1), rollout21(obj_2) → T2
            # T2 → policy2(obj_3)
            RemoteCallOp(
                function_name="update_policy",
                args=["obj_0", "obj_1", "obj_2"],  # policy1, rollout11, rollout21
                calling_node="N1",
                label="T2: update_policy",
                result_label="policy2",
            ),
            # 5. A12, A22: Second rollout iteration
            # Stateful edges: A11→A12, A21→A22 (created automatically by engine)
            # Data edges: policy2(obj_3) → A12, A12 → rollout12(obj_4)
            ActorMethodCallOp(
                actor_id="A10",
                method_name="rollout",
                args=["obj_3"],  # policy2
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A12
                label="A12: rollout",
                result_label="rollout12",
            ),
            ActorMethodCallOp(
                actor_id="A20",
                method_name="rollout",
                args=["obj_3"],  # policy2
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A22
                label="A22: rollout",
                result_label="rollout22",
            ),
            # 6. T3: Second update_policy
            # Data edges: policy2(obj_3), rollout12(obj_4), rollout22(obj_5) → T3
            # T3 → policy3(obj_6)
            RemoteCallOp(
                function_name="update_policy",
                args=["obj_3", "obj_4", "obj_5"],  # policy2, rollout12, rollout22
                calling_node="N1",
                label="T3: update_policy",
                result_label="policy3",
            ),
            # 7. Get the final result
            GetOp(object_id="obj_6", calling_node="N1"),
        ],
    )


def create_simple_local_example() -> RayProgram:
    """Simple example where task is scheduled locally.

    Shows the bottom-up scheduler's common case: local scheduling.
    The driver puts data and calls a function, and the local scheduler
    handles it without involving the global scheduler.
    """
    return RayProgram(
        name="Local Scheduling — Bottom-Up",
        description=(
            "Demonstrates the common case in Ray's bottom-up scheduler: "
            "when the local node has available resources and the task's "
            "data is local, the local scheduler handles the task directly "
            "without forwarding to the global scheduler. This is why the "
            "paper calls it 'bottom-up' — most tasks are handled at the leaves."
        ),
        num_nodes=2,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 0},
        },
        operations=[
            RegisterFunction(
                function_name="double",
                num_returns=1,
                resources={"CPU": 1},
            ),
            PutOp(object_id="x", value=5, node="N1"),
            RemoteCallOp(
                function_name="double",
                args=["x"],
                calling_node="N1",
            ),
            GetOp(object_id="obj_0", calling_node="N1"),
        ],
    )


def create_chained_tasks_example() -> RayProgram:
    """Example with chained tasks showing data edges and control edges.

    Shows how futures flow through a pipeline of tasks, creating
    data edges (data dependencies) and control edges (nested calls).
    """
    return RayProgram(
        name="Chained Tasks — Data & Control Edges",
        description=(
            "Demonstrates chained remote function calls where the output of one "
            "task becomes the input to the next. Shows data edges between tasks "
            "and how Ray's dynamic task graph captures these dependencies. "
            "Tasks are: load_data → preprocess → train → evaluate."
        ),
        num_nodes=3,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker+)", "N3": "N3 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 1},
            "N3": {"CPU": 4, "GPU": 2},
        },
        operations=[
            RegisterFunction(
                function_name="load_data", num_returns=1, resources={"CPU": 1}
            ),
            RegisterFunction(
                function_name="preprocess", num_returns=1, resources={"CPU": 1}
            ),
            RegisterFunction(
                function_name="train", num_returns=1, resources={"CPU": 1, "GPU": 1}
            ),
            RegisterFunction(
                function_name="evaluate", num_returns=1, resources={"CPU": 1}
            ),
            # Chain: load_data → preprocess → train → evaluate
            RemoteCallOp(function_name="load_data", args=[], calling_node="N1"),
            # obj_0 = loaded data
            RemoteCallOp(function_name="preprocess", args=["obj_0"], calling_node="N1"),
            # obj_1 = preprocessed data
            RemoteCallOp(function_name="train", args=["obj_1"], calling_node="N1"),
            # obj_2 = trained model
            RemoteCallOp(function_name="evaluate", args=["obj_2"], calling_node="N1"),
            # obj_3 = evaluation result
            GetOp(object_id="obj_3", calling_node="N1"),
        ],
    )


ALL_PROGRAMS = {
    "add": create_add_example,
    "rl": create_rl_example,
    "local": create_simple_local_example,
    "chained": create_chained_tasks_example,
}
