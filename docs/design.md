# 算法与系统设计

## 数据和数值契约

`Matrix(rows,dim,values)` 验证正维度、形状乘法溢出、数据总数及有限值；行数可为零，id 使用 uint32。矩阵连续行主序存储，每行不单独分配。`row(i)` 在越界时抛异常；返回指针在矩阵内容和生命周期不变时有效。

平方距离为 `sum_j (double(x[j])-double(q[j]))²`，依次累加。先转 double 能避免有限 float 在 float 减法时溢出。这里的“精确”指同一浮点距离契约下与完整排序一致，不是无限精度实数。编译配置不允许 fast-math，并禁用乘加收缩；跨编译器最后几位行为仍须验证。

比较器先比较 distance²，再比较 id。不能引入 epsilon：容差比较可能破坏严格弱序，使排序和堆的契约失效。距离相等与类别标签相同是两回事。

## 有界 Top-k 堆

堆根是保留候选中的最差项。父节点不优于子节点，容量始终不超过 k。

```text
push(candidate):
  capacity=0 -> discard
  heap not full -> append; sift up the worse element
  candidate is not better than root -> discard
  replace root; sift down the worse child
```

一次 push 最坏 O(log k)，根读取 O(1)，最终输出排序 O(k log k)。`invariant()` 用于测试，并非计时热路径。完整排序参考保留所有 N 个候选；扫描引擎只保留 k 个。

## KD-tree 构建与精确剪枝

每个节点存 `[begin,end)` id 区间，另有 d 维最小/最大坐标数组。选跨度最大的轴，用 `nth_element` 按坐标及 id 确定中位数划分；小区间成为叶节点。节点使用整数索引，避免递归 push_back 引起引用失效。原矩阵不重排，只分区 id。

对查询 q 和盒 B=[l,u]，每一维的最短偏差为：

```text
delta_j = l_j-q_j  (q_j < l_j)
          q_j-u_j  (q_j > u_j)
          0        (otherwise)
LB(q,B) = sum_j delta_j²
```

盒内任意点的每维平方偏差均不小于该项，所以 LB 是点距离的下界。使用同样的非负 double 累加顺序；不能换成可能高估下界的近似距离。堆未满时不按当前根剪枝；堆满后仅当 `LB > worst.distance²` 才剪枝。使用 `>=` 会漏掉等距但 id 更小的点。

先访问下界较小的子盒，尽早收紧阈值，再判断另一子盒。叶节点逐点计算并入堆。访问开销约 O(Bd + Pd + U log k + k log k)，B 为计算下界的盒数，P 为计算距离的点数，U 为堆更新数；树高平衡不保证查询对数时间。

每层扫描节点区间计算盒/跨度；平衡划分下构建通常 O(Nd log(N/leaf))，`nth_element` 非并行重载只承诺平均线性分区。索引约 O(N + Md)，M 为节点数，每节点保存 2d 个 float 端点。`index_bytes()` 按容器 capacity 估算，排除原矩阵和分配器开销。

## 批查询线程池

`BatchExecutor` 持有只读数据/索引共享所有权，构造时创建持久 worker。任务队列由 mutex 和 condition_variable 保护；取出任务后释放锁，执行 packaged_task，再由 future 向调用者传播异常。

每个 search 将查询分成不超过 `min(thread_count,Q)` 个连续区间；每条查询独享堆、统计和结果位置。1T 路径在调用线程执行，因此与多线程使用同一 API，但不付出队列调度成本。静态划分任务少，却可能因剪枝成本不同而负载不均。

生命周期约束：

1. 先为 futures 预留空间，再提交任务。提交失败或计算抛异常后，收集所有已提交 future，再传播首个异常，保证被引用的局部输出仍存活。
2. 构造中途创建线程失败，停止并 join 已创建线程；避免 joinable thread 析构触发 terminate。
3. 析构设置停止标志、唤醒 worker、排空已提交任务，在锁外 join。
4. 同一 executor 允许并发 search；禁止 search 与析构并发。数据、索引及其可变别名不能在查询期间被写入、赋值或 move。

没有按 cache-line 对齐、绑核或动态调度；不能声称已经消除 false sharing。Amdahl 定律描述串行开销的限制，但实际最优线程数由负载和机器决定。

## 文件与命令行边界

fvecs 每行是 little-endian int32 维度头及 d 个 IEEE binary32 数值。逐行检查维度一致、长度、数值有限性、大小上限和截断。使用 memcpy 做位复制，避免错误的指针类型双关；格式限制不等同 CLI 参数限制。

Windows 从 wmain 接收 Unicode 参数，转换为 UTF-8，再用 filesystem::u8path 打开文件。CLI 拒绝未知、重复、负值或缺值参数，并阻止输出覆盖输入。

核心操作不静默恢复非法输入。异常传播到 CLI 后返回非零状态，实验脚本在身份、配置、行数、checksum 或正确性不一致时不发布成功汇总。

## 验证分层

人工二维小例子证明排序与等距规则；堆测试逐次检查不变量；随机差分与完整排序比较；小整数特征用独立 NumPy 参考补充共用距离代码的盲点。I/O 测试构造逐字节错误文件；并发测试覆盖重复调用、共享 executor、无查询、k 边界和异常后的可继续使用。

运行次数是测试覆盖证据，不构成形式证明。精确性还依赖比较规则、堆不变量与盒下界推导；并发安全还依赖调用者满足生命周期约束。
