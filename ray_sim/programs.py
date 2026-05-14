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
    We model one iteration of the training loop with 2 simulators.
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
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (GPU)", "N3": "N3 (GPU)"},
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
                resources={"CPU": 1, "GPU": 1},
            ),
            RegisterActorClass(
                class_name="Simulator",
                methods=["rollout"],
                resources={"CPU": 1},
            ),
            RegisterFunction(
                function_name="train_policy",
                num_returns=1,
                resources={"CPU": 1},
            ),
            # train_policy() body:
            # 1. Create policy
            RemoteCallOp(
                function_name="create_policy",
                args=[],
                calling_node="N1",
                calling_task=None,
            ),
            # policy_id is obj_0
            # 2. Create simulator actors
            ActorCreateOp(
                class_name="Simulator",
                actor_id="sim1",
                calling_node="N1",
            ),
            ActorCreateOp(
                class_name="Simulator",
                actor_id="sim2",
                calling_node="N1",
            ),
            # 3. Run rollouts on each actor (parallel) — iteration 1
            ActorMethodCallOp(
                actor_id="sim1",
                method_name="rollout",
                args=["obj_0"],  # policy
                calling_node="N1",
            ),
            ActorMethodCallOp(
                actor_id="sim2",
                method_name="rollout",
                args=["obj_0"],  # policy
                calling_node="N1",
            ),
            # 3b. Run rollouts on each actor — iteration 2
            # This creates STATEFUL EDGES: sim1.rollout[2] depends on sim1.rollout[1]
            ActorMethodCallOp(
                actor_id="sim1",
                method_name="rollout",
                args=["obj_0"],  # same policy
                calling_node="N1",
            ),
            ActorMethodCallOp(
                actor_id="sim2",
                method_name="rollout",
                args=["obj_0"],  # same policy
                calling_node="N1",
            ),
            # 4. Update policy with rollout results
            # obj_0 from create_policy, obj_2 from sim1.rollout[1], obj_3 from sim2.rollout[1]
            # obj_4 from sim1.rollout[2], obj_5 from sim2.rollout[2]
            RemoteCallOp(
                function_name="update_policy",
                args=["obj_0", "obj_2", "obj_3", "obj_4", "obj_5"],
                calling_node="N1",
            ),
            # 5. Get the final result
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
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)", "N3": "N3 (Worker)"},
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
