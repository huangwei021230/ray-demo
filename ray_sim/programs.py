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
        name="add(a, b) — Paper Figure 7",
        description=(
            "论文的端到端示例：执行 add.remote(a, b)，其中 a 在 N1、b 在 N2。\n"
            "本示例打通 Ray 一次 remote 调用的全部环节：\n"
            "• 本地调度器接收 TaskSpec\n"
            "• 查询 GCS 找到参数位置\n"
            "• Plasma 把缺失的参数从 N2 复制到 N1\n"
            "• Worker 通过共享内存读取参数并执行\n"
            "• 结果写回本地对象存储并注册到 GCS\n"
            "• ray.get() 再经 GCS 找到结果并返回"
        ),
        paper_mapping=(
            "**对应位置**：Figure 7（§4.5 “Putting Everything Together”）。\n\n"
            "这是论文用来展示完整消息流的样板示例：\n\n"
            "• Driver 把 TaskSpec 提交给本地调度器；任务的 CPU 需求由 N1 自己满足，"
            "所以 bottom-up 调度器**就地承接**，不上抛全局调度器（§4.3）。\n"
            "• 参数 b 不在本地，Plasma 按 GCS 中登记的位置把它点对点复制到 N1"
            "（§4.2.3 replicate-on-read）。\n"
            "• Worker 执行后，结果对象写回本地存储并把位置注册回 GCS Object Table。\n"
            "• Driver 调用 ray.get() 时同样通过 GCS 找到结果位置后取回。\n\n"
            "**核心要点**：所有有状态组件之间只通过 GCS 通信 —— "
            "这正是让调度器和 Worker 都保持无状态、可水平扩展的根本设计（§4.2）。"
        ),
        num_nodes=2,
        node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker+GPU)"},
        node_resources={
            "N1": {"CPU": 4, "GPU": 0},
            "N2": {"CPU": 4, "GPU": 1},
        },
        operations=[
            # Step 0: Register function with GCS (requires GPU)
            RegisterFunction(
                function_name="add",
                num_returns=1,
                resources={"CPU": 1, "GPU": 1},
            ),
            # Put objects a and b on different nodes
            PutOp(object_id="a", value=1, node="N1"),
            PutOp(object_id="b", value=2, node="N2"),
            # Call add.remote(a, b) from N1
            # N1 cannot satisfy GPU requirement → forwards to global scheduler
            # Global scheduler queries GCS, finds b@N2, selects N2
            RemoteCallOp(
                function_name="add",
                args=["a", "b"],
                calling_node="N1",
                label="add(a,b)",
                result_label="c",
                result_id="c",
            ),
            # Get the result - triggers GCS pub-sub callback
            GetOp(object_id="c", calling_node="N1"),
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
        name="RL Training — Paper Figure 4",
        description=(
            "论文的 RL 训练示例（Figure 3 & 4）：\n"
            "train_policy() 作为顶层任务，先创建一个 policy 和若干 simulator actor，"
            "并行跑 rollout，再用结果更新 policy，循环两轮。\n\n"
            "本示例集中演示三件事：\n"
            "• **Actor** 与有状态方法调用\n"
            "• **Stateful edge**（有状态边）形成的串行链\n"
            "• 嵌套 remote 调用拉出来的**动态任务图**"
        ),
        paper_mapping=(
            "**对应位置**：Figure 1（用户 Python 代码）+ Figure 2（生成的动态任务图），"
            "覆盖 §2 “Motivation and Requirements” 与 §3 “Programming and Computation Model”。\n\n"
            "本示例同时展示 Ray 在 task 模型之上的两种互补原语：\n\n"
            "• **无状态 task**（create_policy / update_policy）—— 产生 **data edge**（数据边）。\n"
            "• **有状态 actor**（Simulator）—— 同一 actor 上的连续方法调用产生 **stateful edge**（有状态边）。"
            "这条链既保证串行执行，也让 Ray 在 actor 故障时可以按链回放恢复（§4.2.1）。\n\n"
            "update_policy 的 GPU 需求会触发全局调度器决策（§4.3），打分过程在事件日志里直接可见。\n\n"
            "完整任务图（task + actor 方法链 + 结果对象）正是论文中 "
            "Ray 区别于纯 BSP 或纯 actor 系统的核心论据。"
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
            # === T0's body: all subsequent calls are nested within T0 ===
            # These calls are made from within train_policy, not from the driver
            # 1. T1: create_policy() — creates initial policy (policy1 = obj_0)
            RemoteCallOp(
                function_name="create_policy",
                args=[],
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → T1
                parent_task="task_0",  # Nested within T0
                label="T1: create_policy",
                result_label="policy1",
            ),
            # 2. A10, A20: Create simulator actors
            # A10 on N2 (GPU×2), A20 on N3 (GPU×1) — distributed across nodes
            ActorCreateOp(
                class_name="Simulator",
                actor_id="A10",  # Actor 1 on N2
                node="N2",
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A10
                parent_task="task_0",  # Nested within T0
            ),
            ActorCreateOp(
                class_name="Simulator",
                actor_id="A20",  # Actor 2 on N3
                node="N3",
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A20
                parent_task="task_0",  # Nested within T0
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
                parent_task="task_0",  # Nested within T0
                label="A11: rollout",
                result_label="rollout11",
            ),
            ActorMethodCallOp(
                actor_id="A20",
                method_name="rollout",
                args=["obj_0"],  # policy1
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A21
                parent_task="task_0",  # Nested within T0
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
                parent_task="task_0",  # Nested within T0
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
                parent_task="task_0",  # Nested within T0
                label="A12: rollout",
                result_label="rollout12",
            ),
            ActorMethodCallOp(
                actor_id="A20",
                method_name="rollout",
                args=["obj_3"],  # policy2
                calling_node="N1",
                calling_task="task_0",  # Control edge: T0 → A22
                parent_task="task_0",  # Nested within T0
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
                parent_task="task_0",  # Nested within T0
                label="T3: update_policy",
                result_label="policy3",
            ),
            # 7. Get the final result
            GetOp(object_id="obj_6", calling_node="N1"),
        ],
    )


# def create_simple_local_example() -> RayProgram:
#     """Local scheduling — the bottom-up common case."""
#     return RayProgram(
#         name="Local Scheduling — Bottom-Up",
#         description=(
#             "演示 Ray bottom-up 调度器最常见的一种情形：\n"
#             "本地节点资源足够、参数也在本地时，本地调度器直接处理任务，不会上抛给全局调度器。\n\n"
#             "这正是论文把它称为 “bottom-up” 的原因 —— 绝大多数任务在叶子节点就消化掉了。"
#         ),
#         paper_mapping=(
#             "**对应位置**：§4.3 “Bottom-Up Distributed Scheduler” + Figure 4（分层调度）。\n\n"
#             "论文的核心调度论点：单一全局调度器扛不住每秒百万级 task 的吞吐，"
#             "所以每个节点的**本地调度器**默认就地承接任务，只有满足不了时才向上汇报。\n\n"
#             "本示例隔离演示这条默认路径：\n"
#             "• 数据在 N1、资源也够 —— 不转发、不走全局调度器\n"
#             "• GCS 只发生一次 task_table 写入\n\n"
#             "对比另外两个示例：`load_balancing` 演示**过载**触发上抛，"
#             "`fault_tolerance` 演示**节点崩溃**后的恢复路径。"
#         ),
#         num_nodes=2,
#         node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)"},
#         node_resources={
#             "N1": {"CPU": 4, "GPU": 0},
#             "N2": {"CPU": 4, "GPU": 0},
#         },
#         operations=[
#             RegisterFunction(function_name="double", num_returns=1, resources={"CPU": 1}),
#             PutOp(object_id="x", value=5, node="N1"),
#             RemoteCallOp(function_name="double", args=["x"], calling_node="N1"),
#             GetOp(object_id="obj_0", calling_node="N1"),
#         ],
#     )
#
#
# def create_chained_tasks_example() -> RayProgram:
#     """Chained tasks — data and control edges through a pipeline."""
#     return RayProgram(
#         name="Chained Tasks — Data & Control Edges",
#         description=(
#             "演示链式 remote 调用：上一个 task 的输出直接作为下一个 task 的输入。\n"
#             "通过这条数据流水线展示 task 之间的 **data edge**，"
#             "以及 Ray 的动态任务图如何即时捕获这些依赖。\n\n"
#             "流水线：load_data → preprocess → train → evaluate"
#         ),
#         paper_mapping=(
#             "**对应位置**：§3 “Programming and Computation Model”，重点是 **动态任务图**。\n\n"
#             "图中两类边：\n"
#             "• **Data edge**：一个 task 消费另一个 task 的 future\n"
#             "• **Control edge**：一个 task 内部再发起 remote 调用\n\n"
#             "本示例展示 `f.remote(g.remote(...))` 这种调用如何在运行时**懒构建**任务图 —— "
#             "它不像 TensorFlow 静态图或 Spark DAG 那样需要事先声明。\n\n"
#             "每条边对应 GCS 里 object_table（data edge）或 task_table（control edge）的一行，"
#             "这也是后续 **lineage-based recovery**（§4.2.1）能成立的基础 —— "
#             "参见 `fault_tolerance` 示例。"
#         ),
#         num_nodes=3,
#         node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker+)", "N3": "N3 (Worker)"},
#         node_resources={
#             "N1": {"CPU": 4, "GPU": 0},
#             "N2": {"CPU": 4, "GPU": 1},
#             "N3": {"CPU": 4, "GPU": 2},
#         },
#         operations=[
#             RegisterFunction(function_name="load_data", num_returns=1, resources={"CPU": 1}),
#             RegisterFunction(function_name="preprocess", num_returns=1, resources={"CPU": 1}),
#             RegisterFunction(function_name="train", num_returns=1, resources={"CPU": 1, "GPU": 1}),
#             RegisterFunction(function_name="evaluate", num_returns=1, resources={"CPU": 1}),
#             RemoteCallOp(function_name="load_data", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="preprocess", args=["obj_0"], calling_node="N1"),
#             RemoteCallOp(function_name="train", args=["obj_1"], calling_node="N1"),
#             RemoteCallOp(function_name="evaluate", args=["obj_2"], calling_node="N1"),
#             GetOp(object_id="obj_3", calling_node="N1"),
#         ],
#     )
#
#
# def create_load_balancing_example() -> RayProgram:
#     """Load-based forwarding — show local scheduler escalation on overload."""
#     return RayProgram(
#         name="Load Balancing — Local Overload Forwarding",
#         description=(
#             "Driver 从 N1 一次性提交 7 个轻量 task。\n\n"
#             "N1 的本地调度器按 bottom-up 原则**就地承接**，"
#             "直到自己的队列长度达到过载阈值（5）：\n"
#             "• 前 5 个 task 留在 N1\n"
#             "• 第 6、7 个被上抛给全局调度器\n"
#             "• 全局调度器按心跳上报的负载，把它们分散到 N2 和 N3\n\n"
#             "可以与 RL 示例对照看：那里是 **GPU 资源不足**触发上抛，这里是 **队列过载**触发上抛。"
#         ),
#         paper_mapping=(
#             "**对应位置**：§4.3 “Bottom-Up Distributed Scheduler” 最后一段。\n\n"
#             "论文写到：本地调度器在两种情况下会上抛 —— "
#             "**节点过载** 或 **资源无法满足**。本示例隔离演示**过载**这一条路径。\n\n"
#             "全局调度器随后按论文公式打分：\n"
#             "`estimated_waiting_time = queue_time + transfer_time`\n"
#             "事件 detail 里直接显示了每个候选节点的分数。\n\n"
#             "心跳保持全局调度器的 `node_loads` 表新鲜 —— 哪个节点队列变长，下次决策就避开它。\n"
#             "这正是论文 §6 评测中 Ray 能做到百万 task/s 吞吐、调度器不成瓶颈的根本原因。"
#         ),
#         num_nodes=3,
#         node_labels={"N1": "N1 (Driver)", "N2": "N2 (Worker)", "N3": "N3 (Worker)"},
#         node_resources={
#             "N1": {"CPU": 4, "GPU": 0},
#             "N2": {"CPU": 4, "GPU": 0},
#             "N3": {"CPU": 4, "GPU": 0},
#         },
#         operations=[
#             RegisterFunction(function_name="quick", num_returns=1, resources={"CPU": 1}),
#             # Open burst window so the queue actually grows visibly — without
#             # this the synchronous engine would drain the queue between every
#             # submission and the audience would never see N1 'overloaded'.
#             BurstStart(note="Driver dispatches a burst of 7 quick.remote() calls from N1."),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             RemoteCallOp(function_name="quick", args=[], calling_node="N1"),
#             BurstEnd(note="All 7 tasks have been admitted somewhere — now the workers run."),
#             GetOp(object_id="obj_6", calling_node="N1"),
#         ],
#     )


def create_hot_object_example() -> RayProgram:
    """Hot object — Plasma replicate-on-read + same-node zero-copy."""
    return RayProgram(
        name="Hot Object — Plasma Replicate-on-Read vs Zero-Copy",
        description=(
            "Driver 在 N1 上 put 一个大对象 X，随后三次 remote 调用都读 X：\n"
            "• 一次跑在 N1（同节点 → zero-copy mmap，无传输）\n"
            "• 一次跑在 N2（触发 Plasma replicate-on-read）\n"
            "• 一次跑在 N3（同样触发 replicate-on-read）\n\n"
            "可以直观对比两条路径：\n"
            "**共享内存读取**（没有传输箭头）与 **点对点复制**（N1 → N2 / N3 的传输箭头）。"
        ),
        paper_mapping=(
            "**对应位置**：§4.2.3 “In-Memory Distributed Object Store”（即 Plasma）。\n\n"
            "重点展示对象存储的两个核心特性：\n\n"
            "• **同节点 zero-copy 读取** —— 对象常驻共享内存，"
            "同一节点上任何 worker 都可以直接 mmap 访问，没有序列化、没有拷贝开销。\n"
            "• **按需复制（replicate-on-read）** —— "
            "当任务被调度到没有该对象的节点，对象存储自动点对点拉一份并缓存到本地；"
            "之后这台机器上的任务都能继续 zero-copy 读取。\n\n"
            "论文认为，正是这套机制让 Ray 在多个 worker 之间共享 "
            "“热对象”（模型权重、数据集等）时无需引入 parameter server，"
            "对应 §6 评测中的 Figure 8b。"
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
            "跨节点跑一条 4 阶段流水线：load_data → preprocess → train → evaluate。\n\n"
            "在 train 阶段于 N2 上完成后，**N2 突然崩溃**，对象存储中的内容全部丢失。\n\n"
            "Driver 随后对下游的 evaluate 结果发起 ray.get()：\n"
            "• Ray 沿 GCS 中记录的依赖链（object_table.created_by → task_table.spec）回溯\n"
            "• 在幸存节点上**重跑**丢失的那个 task\n"
            "• ray.get() 不感知失败，透明地返回最终结果"
        ),
        paper_mapping=(
            "**对应位置**：§4.2.1 “Fault Tolerance” + Figure 11，"
            "即基于 **lineage**（任务依赖谱系）的恢复机制。\n\n"
            "Ray 的关键设计：所有有状态组件都把更新发布到 GCS"
            "（object_table / task_table / actor_table），所以当 worker 甚至整个节点宕掉时，"
            "集群丢的只是“数据字节”，而不是“生成它的配方”。\n\n"
            "恢复一个对象的流程：\n"
            "• 在 GCS 中查到 `created_by` 对应的 task\n"
            "• 递归确认这个 task 的所有参数也是“活的”（必要时再回放它们的依赖）\n"
            "• 在幸存节点上重跑那个确定性函数\n\n"
            "论文也强调：lineage 恢复只对 **task 输出**自动生效；"
            "如果是 ray.put() 在已宕节点上写入的值，因为没有生产它的 task，无法重建。\n\n"
            "这一场景正是 §4.2 中 “GCS 作为单一事实源” 架构的存在意义。"
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
    # "local": create_simple_local_example,
    # "chained": create_chained_tasks_example,
    # "load_balancing": create_load_balancing_example,
    "hot_object": create_hot_object_example,
    "fault_tolerance": create_fault_tolerance_example,
}
