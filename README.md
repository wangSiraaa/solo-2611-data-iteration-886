# 可修订加权记录源、可重复加权采样与同源隔离

本仓库实现可独立运行的模块：

- `python -m wrsource.source`：SQLite 支撑的可修订加权记录源，稳定 `row_id`、
  分组、权重 1..100、正文、递增修订号、不可变源版本快照与固定版本分页。
- `python -m wrsource.dataset`：可重复的加权采样数据集生产器，全量构建、按变更
  清单增量刷新、真实 JSONL + manifest 输出与对照源 API 的校验。
- `python -m wrsource.evidence`：**可修订的“出自同一原件”具名证据台账**，独立
  证据修订号、并行边、撤销、任意旧版重开与带证据 id 的同源路径证人。
- `python -m wrsource.training`：**训练暴露账**，按训练计划只追加登记真正发出的
  原始行身份；不能从今天的源倒推旧训练名单，撤回的训练记录仍是已见原件。
- `python -m wrsource.package`：**评测包采样器与封存登记处**，固定
  `(训练计划, 源版本, 证据版)` 三元组，剔除同源候选后按原精确加权排序与组名额
  补位；已签包反查、干净包重发、资格解释与防篡改指纹。
- `python -m wrsource.loan`：**借阅隔离时间线**，在一段连续证据修订上逐版判定，
  只报告真正的进入/退出隔离动作；单包临时查询与多包外部追查共用同一修订扫描。

另一个模块 `wrsource/sampling.py` 只负责**精确**抽样键算术与选择；
`wrsource/reference.py` 是测试用的小集合独立参考实现；`wrsource/graph.py` 是
无 I/O 的同源图（邻接、传递闭包、BFS 证人路径）。没有模型训练，也没有人工标注
界面，全部用合成文本演示。

## 抽样规则（题目定义，照此实现）

对每个活跃行、给定 256 位 seed：

```
h = SHA-256(seed[32] || uint64_be(len(utf8(row_id))) || utf8(row_id))
a = int(h) + 1                         # u = a / D,  D = 2^256 + 1
priority = u ** (1 / weight)
```

按 priority 降序选 `quota` 行，完全相等时按 `row_id` 码点序。`quota <= 0` 给空
样本，活跃行少于 quota 时全选。不同 seed 是不同请求。抽签键只绑定
`(seed, row_id)`，因此正文修改、改组、移动分片都不会重抽该行的键；权重修改会
改变该行优先级。

### 精确比较（不能先转 float）

两个优先级的比较等价于整数比较：

```
p1 > p2  <=>  (a1/D)^(1/w1) > (a2/D)^(1/w2)
         <=>  a1^w2 * D^w1  >  a2^w1 * D^w2        (两边同取 w1*w2 次幂)
```

实现采用三级比较器（`sampling.compare_priority`）：

1. **float 对数筛选**：仅当 `|log p1 - log p2| > 1e-9` 时直接判定。阈值远大于
   `math.log` 在 `[1, 2^256]` 上的实际误差，只会给出必然正确的结论。
2. **60 位 Decimal 对数**：误差约 `1e-55`，仅当差值 `> 1e-50` 时判定。
3. **精确整数幂交叉比较**：`a1^w2 * D^w1` 与 `a2^w1 * D^w2` 直接比大整数，作为
   最终裁决。返回 0 只表示**实数意义下精确相等**，此时再按 `row_id` 码点序。

因此极近键不会被 float 压成相等。测试构造了 `a` 仅差 1、对数在 float 上完全
相等的情形，精确路径仍能区分；并用 300 位 Decimal 对随机与近邻键做交叉校验。
manifest 为每行保存 `hash_hex`、`a`、`weight`，并为每个超额组保存 `cut` 边界
证书（`a_sel^w_cut * D^w_sel` 与 `a_cut^w_sel * D^w_cut` 两个大整数），可离线
复核比较，无需相信浮点快路。

## 源服务设计

### 表结构

- `rows`：每行的**当前**可变状态，含稳定 `row_id`、group、weight、正文摘要、
  `rev`、`active`、`shard`。`row_id` 一旦分配终身不变。
- `bodies`：内容寻址、只追加的正文库。旧版本正文永不回收（本实现不清理），因此
  每个已发布源版本都保留**真实可读取**的正文。
- `versions` / `version_rows`：不可变整版快照。快照只存 group、weight、正文摘要、
  `rev`，**不存 shard**。
- `batch_log`：原始批次与结果，用于审计。

### 事务与修订

一个批次的每个操作先在内存中折叠为**每行最终状态**，然后在一个
`BEGIN IMMEDIATE` 事务里写当前状态、必要时发布新快照；任何错误整体回滚。因此
“批内按最终状态原子提交”。

- 逻辑内容（active/group/weight/body 摘要）改变 → `rev += 1`，并发布新源版本。
  同一批次对同一行多次修改只算一次修订。
- 只改 `shard`（物理位置）→ `rev` 不变，**不发布新版本**，旧的固定版本内容不变。
  这正是“分片只是物理存放位置，同一逻辑版本内改变分片不改变行身份或内容”。
- 批内先 `add` 后 `retract` 的新行视为从未存在（返回 `discarded_row_ids`）；
  先撤回后新增是合法的“最终状态为活跃”。

### 固定版本分页

分页键是 `(version_id, row_id)`，按 `row_id` 的 UTF-8 字节序（与码点序一致）。
因为版本是不可变快照，读者持有旧版本游标时，写者发布新版本不会让分页漏行或重复。
游标编码了版本号，用错版本会被拒绝。读者在分页过程中穿插写入当前状态，读取的仍
是同一快照。跨分片的行按 `row_id` 统一排序，分片不参与行身份或内容。

### 为何选 SQLite

单文件、WAL 并发读者/单写者、事务与 `PRIMARY KEY` 唯一性天然满足“稳定 row_id +
原子批次 + 不可变版本”。10 万行量级写入约 2s，整版快照读取与分页都在亚秒到数秒
级，远低于题目规模。

## 数据集生产器设计

### 合并：每分片产出全部候选，父进程按组统一选择

- 从**一次一致元数据读取** `all_rows_meta(version)` 取得所有活跃行及其当前 shard，
  在父进程按 shard 分组后分发。即使并发移动分片，也不会漏行或重复。
- 每个 worker 只计算该分片所有行的候选（`a`、`log_screen`）并**全部返回**。
- 父进程把所有分片候选按 `group` 合并成完整候选池，再对整组执行精确 top-quota。
- **绝不拼接各分片样本，也不只保留一个全局候选数**。因此无论分片数与遍历顺序，
  合并结果与单流参考完全一致（测试用随机改分片、并行/串行对照验证）。
  `--parallel` 使用进程池；缺省单进程。二者输出逐字节相同。

### 扫描 vs 候选持久索引

本实现对**需要的组**做一次全量扫描并精确排序，不在数据库里持久化全局候选顺序
索引。理由：

- 抽签键依赖 seed 与权重；权重可变，任何持久索引都要按 `(seed, weight)` 失效，
  维护成本高且易与源版本不一致。
- 当前规模（10 万行、每组 quota ≤ 1000）下，整库精确排序在亚秒到数秒级，重扫
  受影响组足够快。
- 重扫可证明与单流参考一致，避免持久索引的陈旧风险。

若规模继续增长，可缓存 `(row_id, seed) -> a`（与正文/分片无关，仅在 seed 变化时
失效），并用每组堆/树维护 top-quota；权重变化仍要求对应组重排序。当前实现刻意
选择简单且可验证的方案。

### 精确比较缓存

`Candidate` 缓存 `a`（精确键）与 `log_screen`（对数快路值），排序时大量比较走
O(1) 的 float 快路，只在极近时回退 Decimal 与大整数幂。`cut` 证书只在 manifest
生成时对每组边界计算一次，把昂贵的大整数运算限制在真正的边界与极近键上。

### 增量刷新与跨组修订

`refresh` 计算 `diff_versions(old, new)`，得到 added/removed/changed 行及其
**旧组与新组**。受影响的组（新增、撤回、改组的新旧两组、改权、改正文，以及 quota
变化的组）全部在**新版本上完整重扫**；未受影响组复用旧选择。因为未受影响组的行、
权重、分组、正文、quota 都没变，复用与重算在数学上完全相等。

关键点：撤回样本内一行时，是在**新版本的完整组候选**上重新取 top-quota，补入的
是真正的下一名，而不是从旧样本里重抽。改组同时进入旧组和新组的受影响集合，名额
在两组间正确迁移。测试覆盖：撤回首名后与同版本全量重建逐字节相同；改组后两组分别
与独立参考一致；增量刷新与同版本全量构建的 manifest 和 JSONL 完全相同。

### 输出内容

`samples.jsonl` 每行含：`group`、`rank`、`row_id`、`weight`、`hash_hex`、`a`、
`log_screen`、`body_sha256`、`body_bytes`、`source_version`、`seed_hex`、**真实
`body`**。`manifest.json` 含 `format`、`source_version`、`seed_hex`、`quotas`、
`totals`、`records_file`、`records_sha256`、以及每组 `eligible`、`quota`、
`selected_count`、`selected`（带精确比较材料）与 `cut` 边界证书。

`verify` 会重算 `records_sha256`、逐行回查源 API 的正文与摘要、按 seed 重算
`hash_hex`，并做一次同版本全量重建比对身份、顺序与真实正文。因此**每行都从源
API 回查验证**，而不是只写抽样元数据。

## 同源证据台账（`wrsource.evidence` + `wrsource.graph`）

源服务修订的是**行内容**，证据台账独立修订的是**两个稳定行身份出自同一原件**
的具名证据，两条历史不共享修订号；修订 0 恒为空图。

- 证据是**无向边**，带全局唯一的具名 `evidence_id`，端点按行身份规范化存储。
- 一版的同源等价关系 = 该版仍有效证据的**传递闭包**（并查集式连通分量）。
- 同一对端点允许**并行证据**：两份独立证据同时支持一条关系，撤销其中一份时另一
  份仍维持连通，直到最后一份被撤销才真正断开。
- 行身份不依赖正文状态：正文已撤回的行（如示例中的 M）仍是图节点，可以连结另外
  两份出处；正文相同的两条记录不会自动同源，只有显式证据才建边。
- `evidence_ops` 只追加，任何旧修订都可重开。`snapshot(rev)` 重建该版闭包并按
  修订缓存；`same_origin` 回答是否同源，`origin_path` 返回 BFS 证人路径（节点序列
  + 沿途每条证据 id），BFS 邻接按 `(对端, 证据 id)` 排序，结果确定。

## 训练暴露账（`wrsource.training`）

训练产品真正发出的原始行身份按**训练计划**登记，计划之间互不影响：

- `register_batch` 只追加批次，记录批次序号、发出时的源版本/证据修订（元数据）
  与行身份清单；同一行重复发出不改变其首次批次归属。
- 暴露集合**不能从今天的源倒推**：源里后来撤回/改正的行，其历史暴露仍在账上。
- `retract_exposure` 把训练记录标记为撤回，但默认查询仍包含该行——原件曾经被
  模型见过，隔离判定一律按“曾暴露”处理，理由中带 `training_record_retracted`。
- `exposed_rows(plan, as_of_batch=N)` 支持窗口化追查：只看窗口前已经发生的批次，
  窗口内新增训练批次不会追溯移动一份在借包。

## 评测包：固定三元组、剔除、精确补位（`wrsource.package`）

出包时一次性固定 `(training_plan, source_version, evidence_revision)`（以及种子、
组名额、训练批次截止点），全部写入 manifest：

1. 候选 = 固定源版本中的活跃行（撤回正文的行自然不在候选里，但其证据节点仍参与
   闭包计算）。
2. 对每条候选，在固定证据快照上求其连通分量；分量包含计划下任一曾暴露行（含已撤
   回的训练记录）即剔除，并保存每条暴露链的证人路径作为依据。
3. 幸存者沿用 `wrsource.sampling` 的**原精确加权排序**，按组取 top-quota，被剔除
   的名额由真正的下一名补位（不是对旧样本重抽）；边界仍出精确整数比较证书。
4. `explain_eligibility` 让采样器解释任意记录为何仍有资格：无暴露行与其同分量即
   合格，结论可仅凭 manifest 中的固定三元组离线复算。

manifest 经规范化 JSON 做 SHA-256 指纹后存入只追加的 `PackageRegistry`，打开时
重算指纹，篡改即拒。

### 已签包复查与干净包重发

- **已签包不可变**。`recheck_package` 用（通常更新的）证据版对包内**实际签署的
  行身份**逐条裁决：新进入暴露分量的行列入隔离项并附依据，其余列入仍干净；旧包
  当时的判断原样保留。
- `reissue_clean_package` 用同一种子与名额在新证据版重新抽足，新包带
  `parent_package_id` 回链旧包，并给出 `dropped_rows`（区分隔离 / 源端已消失）与
  `backfilled_rows` 补位账；新旧两个签封包并存。
- 一个来源修正可能先分裂大组、再被另一证据重新连回：复查只看指定计划的暴露集，
  其它计划的暴露不影响本计划；库中可同时存在许多计划与封存包。

## 借阅隔离时间线（`wrsource.loan`）

输入：训练计划、在借包（只取签封时的实际行身份）、连续证据修订窗口
`[start_rev, end_rev]` 与窗口前的训练批次截止点。

- 暴露集在整个窗口内冻结为截止点之前的批次；同源关系按每个修订当时仍有效的证据
  闭包逐版重算（多包追查时同一修订的快照只构建一次并共享）。
- 输出窗口起点的隔离状态（含每行人/不行的依据）以及之后**每次真正的状态翻转**：
  `enter` 带暴露链证人，`exit` 带触发解封的撤销操作。已经隔离再补并行边、两份
  证据撤掉其中一份，都不产生事件。
- 题面序列（起点 X、B 分离；新建证据 a → 新建并行证据 b → 撤 a → 撤 b）严格产生
  两个动作：rev 1 进入、rev 4 退出；对没有 X 暴露的另一计划，同一包零动作。
- 工作人员临时查一包（`loan timeline`）与外部追查多包（`loan investigate`）走同一
  驱动函数，因此两种入口对同一修订序列给出完全一致的应执行动作。

## 代码结构

```
wrsource/
  sampling.py   精确键、比较器、top-quota、边界证书（无 I/O）
  source.py     版本化源服务 + 分页 + CLI（可独立运行）
  dataset.py    全量构建 / 增量刷新 / 校验 + CLI（可独立运行）
  reference.py  测试用独立精确参考实现
  graph.py      同源图：邻接、连通分量、BFS 证人路径（无 I/O）
  evidence.py   具名证据台账：并行边、撤销、任意旧版重开 + CLI
  training.py   训练暴露账：按计划只追加、撤回留痕、as-of 查询 + CLI
  package.py    评测包采样、签封登记处、资格解释、复查/重发 + CLI
  loan.py       借阅隔离时间线：逐版翻转事件、单包/批量同一驱动 + CLI
tests/
  test_sampling.py        键构造、双向一致性、极近键、边界证书、参考对照
  test_source.py          原子批次、折叠、修订号、分片移动、分页并发、重启
  test_dataset.py         全量/参考对照、零/超名额、正文版本、分片不变性、
                          并行一致、撤回首名补位、改组、增量==全量、篡改检测、重开
  test_evidence.py        闭包、撤回正文节点、并行边、分裂再连回、旧版重开
  test_training.py        批次、as-of、撤回留痕、计划独立
  test_package_loan.py    题面三选样、同文不同源、签封反查、补位重发、
                          撤回训练记录仍隔离、借阅时间线、批量追查、截止批次
demo/run_demo.py              源 + 采样端到端演示（含 --large 10 万行）
demo/run_provenance_demo.py   同源/暴露/评测包/借阅隔离端到端演示
```

## 运行

```bash
# 测试
python3 -m unittest discover -s tests -v

# 演示
python3 demo/run_demo.py
python3 demo/run_demo.py --large --parallel
python3 demo/run_provenance_demo.py

# 源服务 CLI
python3 -m wrsource.source init --db src.sqlite
python3 -m wrsource.source batch --db src.sqlite --ops ops.json
python3 -m wrsource.source versions --db src.sqlite
python3 -m wrsource.source page --db src.sqlite --version 1 --limit 100
python3 -m wrsource.source page --db src.sqlite --version 1 --cursor <CURSOR>
python3 -m wrsource.source diff --db src.sqlite --old 1 --new 2
python3 -m wrsource.source verify-bodies --db src.sqlite --version 1
python3 -m wrsource.source dump --db src.sqlite --version 1 --out dump.jsonl

# 数据集 CLI
python3 -m wrsource.dataset build --db src.sqlite --version 1 \
    --seed-hex <64 hex> --quota g0=1000 --quota g1=1000 --out ds_v1
python3 -m wrsource.dataset refresh --db src.sqlite --old ds_v1 \
    --new-version 2 --out ds_v2
python3 -m wrsource.dataset verify --db src.sqlite --dataset ds_v2
python3 -m wrsource.dataset compare --a ds_v2 --b ds_v2_full

# 同源证据 CLI
python3 -m wrsource.evidence init --db evidence.sqlite
python3 -m wrsource.evidence add --db evidence.sqlite \
    --evidence-id ev-1 --a X --b M --note "同原件"
python3 -m wrsource.evidence revoke --db evidence.sqlite --evidence-id ev-1
python3 -m wrsource.evidence same-origin --db evidence.sqlite \
    --a X --b B --revision 3
python3 -m wrsource.evidence path --db evidence.sqlite --a X --b B
python3 -m wrsource.evidence log --db evidence.sqlite

# 训练暴露账 CLI
python3 -m wrsource.training init --db training.sqlite
python3 -m wrsource.training create-plan --db training.sqlite --plan T
python3 -m wrsource.training register --db training.sqlite --plan T \
    --rows X,Y --source-version 5 --evidence-revision 2
python3 -m wrsource.training retract --db training.sqlite --plan T --row-id X
python3 -m wrsource.training exposed --db training.sqlite --plan T \
    --as-of-batch 1

# 评测包 CLI
python3 -m wrsource.package issue --source-db src.sqlite \
    --training-db training.sqlite --evidence-db evidence.sqlite \
    --registry-db packages.sqlite --source-version 5 \
    --seed-hex <64 hex> --plan T --evidence-revision 3 --quota g1=10
python3 -m wrsource.package list --registry-db packages.sqlite
python3 -m wrsource.package open --registry-db packages.sqlite \
    --package-id <PKG>
python3 -m wrsource.package explain --training-db training.sqlite \
    --evidence-db evidence.sqlite --plan T --row-id B --evidence-revision 3
python3 -m wrsource.package recheck --registry-db packages.sqlite \
    --training-db training.sqlite --evidence-db evidence.sqlite \
    --package-id <PKG>
python3 -m wrsource.package reissue --source-db src.sqlite \
    --training-db training.sqlite --evidence-db evidence.sqlite \
    --registry-db packages.sqlite --package-id <PKG>

# 借阅隔离时间线 CLI（单包临时查询 / 多包外部追查）
python3 -m wrsource.loan timeline --registry-db packages.sqlite \
    --training-db training.sqlite --evidence-db evidence.sqlite \
    --plan T --start-rev 0 --end-rev 4 --cutoff-batch 1 \
    --package-id <PKG> --loan-id loan-1
python3 -m wrsource.loan investigate --registry-db packages.sqlite \
    --training-db training.sqlite --evidence-db evidence.sqlite \
    --plan T --start-rev 0 --end-rev 4 --cutoff-batch 1 \
    --loans-json loans.json
```

规模验证：10 万行、100 组、每组 quota 1000（合计 10 万样本）时，写入约 2s，
并行全量构建约 2.5s，增量刷新受影响组约 3s，校验约 3s。
