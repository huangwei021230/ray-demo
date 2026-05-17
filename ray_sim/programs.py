"""
Ray Demo - Example Programs

Each program isolates one concept from the Ray paper (Moritz et al., OSDI '18,
"Ray: A Distributed Framework for Emerging AI Applications"). The
`paper_mapping` field on RayProgram is rendered alongside the description so
the audience always sees which paper section/figure the scenario maps to.

Catalogue:
  add              — Figure 5 end-to-end walkthrough
  rl               — Figure 1/2 dynamic task graph (actors + nested calls)
  local            — bottom-up "common case" (§4.3)
  chained          — data + control edges, pipeline parallelism (§3 / Fig 4)
  load_balancing   — local-first, forward on overload (§4.3)
  hot_object       — Plasma replicate-on-read vs zero-copy mmap (§4.2.3)
  fault_tolerance  — lineage-based reconstruction after a node crash (§4.2.1, Fig 11)
"""

from .types import *


def create_add_example() -> RayProgram:
    """add(a, b) — paper's Figure 5 end-to-end walkthrough."""
    return RayProgram(
        name="add(a, b) — Paper Figure 5",
        description=(
            "论文的端到端示例：执行 add.remote(a, b)，其中 a 在 N1、b 在 N2。"
            "完整跑通：本地调度器 → 全局调度器 → 查询 GCS → 数据复制 → "
            "Worker 执行 → 结果注册 → ray.get()。对应论文 Figure 5。"
        ),
        paper_mapping=(
            "对应论文 Figure 5（§4.5 “Putting Everything Together”）。"
            "演示 Ray 每次 remote call 的完整消息流："
            "Driver 把 TaskSpec 提交给本地调度器；因为 b 在 N2，本地调度器无法满足资源 "
            "→ 转发给全局调度器（§4.3）；全局调度器查询 GCS Object Table 获取参数位置，"
            "按 estimated_waiting_time 最小的规则选节点；Plasma 跨节点拉取缺失参数（§4.2.3）；"
            "Worker 执行后把结果写回本地对象存储并注册到 GCS Object Table；"
            "Driver 的 ray.get() 再通过 GCS 查到位置取回结果。"
            "所有有状态组件只通过 GCS 通信 —— 这是让调度器和 Worker 保持无状态、可横向扩展的核心设计（§4.2）。"
        ),
        num_nodes=2,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 0},
        },
        operations=[
            RegisterFunction(function_name="add", num_returns=1, resources={"CPU": 1}),
            PutOp(object_id="a", value=1, node="N1"),
            PutOp(object_id="b", value=2, node="N2"),
            RemoteCallOp(function_name="add", args=["a", "b"], calling_node="N1"),
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
            "论文的 RL 训练示例（Figure 1 & 2）：train_policy() 创建一个 policy 和若干 simulator actor，"
            "并行跑 rollout，再用结果更新 policy。"
            "演示：actor、stateful edge（有状态边）、嵌套 remote 调用、以及动态任务图计算模型。"
        ),
        paper_mapping=(
            "对应论文 Figure 1（用户的 Python 代码）和 Figure 2（生成的动态任务图），"
            "属于 §2 “Motivation and Requirements” 与 §3 “Programming and Computation Model”。"
            "该示例同时展示 Ray 在 task 模型之上的两种互补原语："
            "(1) 无状态 **task**（create_policy / update_policy）产生 **data edge**（数据边）；"
            "(2) 有状态 **actor**（Simulator），同一 actor 上连续的方法调用产生 **stateful edge**（有状态边）——"
            "既保证串行执行，又给 Ray 一条可回放的链用于 actor 恢复（§4.2.1）。"
            "update_policy 的 GPU 约束触发全局调度器决策（§4.3），观众能看到打分过程。"
            "完整的动态任务图（task + actor 方法链 + 结果对象）正是论文中 Ray 区别于纯 BSP 或纯 actor 系统的关键。"
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
    """Local scheduling — the bottom-up common case."""
    return RayProgram(
        name="Local Scheduling — Bottom-Up",
        description=(
            "演示 Ray bottom-up 调度器的常见情形：当本地节点资源足够、参数也在本地时，"
            "本地调度器直接处理任务，不会上抛给全局调度器。"
            "这就是论文称之为 “bottom-up” 的原因 —— 绝大多数任务在叶子节点就消化掉了。"
        ),
        paper_mapping=(
            "对应论文 §4.3 “Bottom-Up Distributed Scheduler” 和 Figure 4（分层调度）。"
            "论文的核心调度论点：单一全局调度器无法支撑每秒百万级 task 的吞吐，"
            "所以每个节点的 **本地调度器** 默认就地接收任务，只有满足不了时才上抛。"
            "本示例隔离演示这条默认路径：数据在 N1、资源也够 —— 不转发、不走全局调度器、"
            "GCS 只发生一次 task_table 写入。"
            "与 `load_balancing` 和 `add` 对比，可以看到另外两条上抛路径（过载 / 资源不足）。"
        ),
        num_nodes=2,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 0},
        },
        operations=[
            RegisterFunction(function_name="double", num_returns=1, resources={"CPU": 1}),
            PutOp(object_id="x", value=5, node="N1"),
            RemoteCallOp(function_name="double", args=["x"], calling_node="N1"),
            GetOp(object_id="obj_0", calling_node="N1"),
        ],
    )


def create_chained_tasks_example() -> RayProgram:
    """Chained tasks — data and control edges through a pipeline."""
    return RayProgram(
        name="Chained Tasks — Data & Control Edges",
        description=(
            "演示链式 remote 调用：上一个 task 的输出作为下一个 task 的输入。"
            "展示 task 之间的 data edge（数据边）以及 Ray 的动态任务图如何捕获这些依赖。"
            "流水线：load_data → preprocess → train → evaluate。"
        ),
        paper_mapping=(
            "对应论文 §3 “Programming and Computation Model”，重点是 **动态任务图**："
            "data edge（一个 task 消费另一个 task 的 future）和 control edge（嵌套 remote 调用）。"
            "本示例展示 `f.remote(g.remote(...))` 这种调用是如何在运行时**懒构建**任务图的 —— "
            "图不是像 TensorFlow 静态图或 Spark DAG 那样事先声明的。"
            "图中的每条边都对应 GCS 里 object_table（data edge）或 task_table（control edge）的一行，"
            "这也是后续 lineage-based recovery（§4.2.1）能成立的基础 —— 参见 `fault_tolerance`。"
        ),
        num_nodes=3,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker+)", "N3": "N3 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 1},
            "N3": {"CPU": 4, "GPU": 2},
        },
        operations=[
            RegisterFunction(function_name="load_data", num_returns=1, resources={"CPU": 1}),
            RegisterFunction(function_name="preprocess", num_returns=1, resources={"CPU": 1}),
            RegisterFunction(function_name="train", num_returns=1, resources={"CPU": 1, "GPU": 1}),
            RegisterFunction(function_name="evaluate", num_returns=1, resources={"CPU": 1}),
            RemoteCallOp(function_name="load_data", args=[], calling_node="N1"),
            RemoteCallOp(function_name="preprocess", args=["obj_0"], calling_node="N1"),
            RemoteCallOp(function_name="train", args=["obj_1"], calling_node="N1"),
            RemoteCallOp(function_name="evaluate", args=["obj_2"], calling_node="N1"),
            GetOp(object_id="obj_3", calling_node="N1"),
        ],
    )


def create_load_balancing_example() -> RayProgram:
    """Load-based forwarding — show local scheduler escalation on overload."""
    return RayProgram(
        name="Load Balancing — Local Overload Forwarding",
        description=(
            "Driver 从 N1 一次性提交 7 个轻量 task。N1 的本地调度器按 bottom-up 原则就地准入，"
            "直到队列长度达到过载阈值（5）。前 5 个 task 留在 N1；之后的 task 被上抛给全局调度器，"
            "由后者依据心跳上报的节点负载分散到 N2 和 N3。"
            "与 `add` 的上抛触发条件对比 —— 那个是资源不足，这个是过载。"
        ),
        paper_mapping=(
            "对应论文 §4.3 “Bottom-Up Distributed Scheduler” 最后一段："
            "本地调度器在 “节点过载” **或** “无法满足资源需求” 时上抛。"
            "本示例隔离演示**过载**这一条路径（其他 demo 没覆盖）。"
            "全局调度器随后按论文公式打分：estimated_waiting_time = queue_time + transfer_time，"
            "事件 detail 里直接显示了每个候选节点的分数。"
            "心跳保持全局调度器的 `node_loads` 表新鲜 —— 一个节点队列变长，下次决策就会避开它。"
            "这正是论文 §6 评测中 Ray 能做到百万 task/s 吞吐而调度器不成瓶颈的原因。"
        ),
        num_nodes=3,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)", "N3": "N3 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 0},
            "N3": {"CPU": 4, "GPU": 0},
        },
        operations=[
            RegisterFunction(function_name="quick", num_returns=1, resources={"CPU": 1}),
            # Open burst window so the queue actually grows visibly — without
            # this the synchronous engine would drain the queue between every
            # submission and the audience would never see N1 'overloaded'.
            BurstStart(note="Driver dispatches a burst of 7 quick.remote() calls from N1."),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
            BurstEnd(note="All 7 tasks have been admitted somewhere — now the workers run."),
            GetOp(object_id="obj_6", calling_node="N1"),
        ],
    )


def create_hot_object_example() -> RayProgram:
    """Hot object — Plasma replicate-on-read + same-node zero-copy."""
    return RayProgram(
        name="Hot Object — Plasma Replicate-on-Read vs Zero-Copy",
        description=(
            "Driver 在 N1 上 put 一个大对象 X，随后三次 remote 读 X："
            "一次从 N1（同节点，zero-copy mmap），一次调度到 N2，一次到 N3"
            "（后两次都会触发 Plasma replicate-on-read）。"
            "观众能直观对比：共享内存读（无传输）与点对点复制（N1 到 N2、N3 的传输箭头）。"
        ),
        paper_mapping=(
            "对应论文 §4.2.3 “In-Memory Distributed Object Store”（即 Plasma）。"
            "展示两个关键特性："
            "(a) **同节点 zero-copy 读** —— 对象放在共享内存，同节点任何 worker 直接 mmap，无序列化、无拷贝；"
            "(b) **按需复制（replicate-on-read）** —— 当 task 被调度到没有输入对象的节点，"
            "对象存储会点对点拉一份并缓存本地，之后该节点上的 task 也都能 zero-copy 读取。"
            "论文认为正是这个机制让 Ray 能让多个 worker 共享 “热对象”（模型权重、数据集等），"
            "而不必把数据塞进 parameter server（§6 评测，Figure 8b）。"
        ),
        num_nodes=3,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)", "N3": "N3 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 0},
            "N3": {"CPU": 4, "GPU": 0},
        },
        operations=[
            RegisterFunction(function_name="read_x", num_returns=1, resources={"CPU": 1}),
            PutOp(object_id="X", value="<large_tensor>", node="N1"),
            # Same node as X → zero-copy mmap.
            RemoteCallOp(function_name="read_x", args=["X"], calling_node="N1"),
            # Different nodes → replicate-on-read fans out from N1.
            RemoteCallOp(function_name="read_x", args=["X"], calling_node="N2"),
            RemoteCallOp(function_name="read_x", args=["X"], calling_node="N3"),
            GetOp(object_id="obj_0", calling_node="N1"),
        ],
    )


def create_fault_tolerance_example() -> RayProgram:
    """Fault tolerance — lineage replay after a node crash."""
    return RayProgram(
        name="Fault Tolerance — Lineage-Based Recovery",
        description=(
            "跨节点跑一条 4 阶段流水线（load_data → preprocess → train → evaluate）。"
            "当 train 在 N2 上完成后，**N2 崩溃**：其对象存储被清空。"
            "Driver 随后对下游的 evaluate 结果发起 ray.get() —— "
            "Ray 沿 GCS 中记录的依赖链（object_table.created_by → task_table.spec）回溯，"
            "在幸存节点上重跑丢失的 task，ray.get() 透明地返回成功。"
        ),
        paper_mapping=(
            "对应论文 §4.2.1 “Fault Tolerance” 和 Figure 11，即基于 **lineage**（任务依赖谱系）的恢复机制。"
            "Ray 的关键设计：**所有**有状态组件都把更新发布到 GCS（object_table / task_table / actor_table），"
            "因此 worker 或整个节点宕掉时，集群丢的只是“字节”而不是“配方”。"
            "恢复一个对象时，系统在 GCS 里查 `created_by` 对应的 task，"
            "递归保证该 task 的参数也是活的（必要时按相同方式回放它们的依赖），"
            "然后在幸存节点上重跑那个确定性函数。"
            "论文也强调：这种 lineage 恢复只对 task 输出自动生效；"
            "若是 ray.put() 在已宕节点上塞的值，由于没有生产它的 task，无法重建。"
            "这一场景正是 §4.2 中 “GCS 作为单一事实源” 架构存在的意义。"
        ),
        num_nodes=3,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)", "N3": "N3 (Worker)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 1},
            "N3": {"CPU": 4, "GPU": 1},
        },
        operations=[
            RegisterFunction(function_name="load_data", num_returns=1, resources={"CPU": 1}),
            RegisterFunction(function_name="preprocess", num_returns=1, resources={"CPU": 1}),
            RegisterFunction(function_name="train", num_returns=1, resources={"CPU": 1, "GPU": 1}),
            RegisterFunction(function_name="evaluate", num_returns=1, resources={"CPU": 1}),
            RemoteCallOp(function_name="load_data", args=[], calling_node="N1"),
            RemoteCallOp(function_name="preprocess", args=["obj_0"], calling_node="N1"),
            RemoteCallOp(function_name="train", args=["obj_1"], calling_node="N1"),
            # train likely lands on N2 (GPU). Kill N2 before evaluate runs.
            NodeFailOp(node_id="N2"),
            RemoteCallOp(function_name="evaluate", args=["obj_2"], calling_node="N1"),
            GetOp(object_id="obj_3", calling_node="N1"),
        ],
    )


ALL_PROGRAMS = {
    "add": create_add_example,
    "rl": create_rl_example,
    "local": create_simple_local_example,
    "chained": create_chained_tasks_example,
    "load_balancing": create_load_balancing_example,
    "hot_object": create_hot_object_example,
    "fault_tolerance": create_fault_tolerance_example,
}
