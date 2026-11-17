# 有限加权词形变换 + 特征合一句法（Python 3.11，纯标准库）

本工程把只能“直接查词 / 加一个 s / 无特征 CFG”的基线，扩展为两个独立、可组合的工具：

| 模块 | 入口 | 职责 |
| --- | --- | --- |
| `morph.py` | `Morphology(entries, automaton, atoms)` | 无环变换自动机上的**有限加权词形变换**，保留同一词的全部合法分析（同形多路径、多 entry、各自权重） |
| `syntax.py` | `analyze_sentence(grammar, candidates)` | **特征合一句法**：有限原子域上接地展开的加权 CFG chart，输出 `log_partition` / 最优树 / 逐词形候选后验 |
| `pipeline.py` | `analyze_tokens(morphology, grammar, tokens)` | 组合入口：对每个 token 调本次 `Morphology.analyze`，把**全部**实际候选交给同一个 `analyze_sentence` |

仅依赖 Python 3.11 标准库（`math`、`hashlib`、`json`、`itertools`、`unittest`），
无需语料下载、神经网络、在线服务或 NLTK；NLTK 等成熟库是题目允许但非必需的选项。

## 运行方法

```bash
python demo.py                       # 端到端演示，结果写入 fixtures/demo_results.json
python -m unittest discover -s tests # 26 个手算测试 + 随机暴力枚举交叉验证
```

## 表示与代价（为什么这样设计）

题目所有值域都是有限小集合：原子 ≤ 4、属性 ≤ 3、规则变量 ≤ 3、自动机最长路径 4 条边、
句子 ≤ 8 token、每位置 ≤ 8 候选。因此选择**完整枚举 + 成熟加权 CFG 技术**，不写通用合一器：

1. **词形 = 接受路径预枚举。** 自动机无环且深度 ≤ 4，构造时对每个 entry 从 `start`
   做 DFS，枚举全部接受路径（`start` 本身可接受；accept 状态可继续走边形成另一条接受
   路径）。边语义严格按顺序执行：先检查当前特征包含且等于 `requires` 全部键值，
   `old` 非空时还要求当前串以 `old` 结尾并去尾接 `new`（`old` 为空则不删原文直接接 `new`），
   再在结果前后接 `prefix`/`suffix`，最后 `set_features` 按键覆盖。条件不满足的边不可走、
   不改输入强行接受。路径质量 = `entry.weight × ∏ edge.weight`（float64，不归一）。
   `analyze` 只做“最终串逐字相等”过滤，`generate` 只做最终特征子集过滤，所以不同 entry、
   不同路径即使同形同特征也作为不同分析保留，质量分别进入句法。
   代价上界：entries(30) × 分支^深度(≤4)，枚举一次后缓存。

2. **句法 = 规则预接地（grounding）+ 加权 CKY chart。** 每条规则的变量在 `grammar.atoms`
   上枚举全部赋值（至多 4³ = 64 种/规则），合一在接地阶段完成：接地后所有符号只含原子。
   chart 阶段的匹配退化为“类别相同 + 子节点包含并满足符号全部特征键值”，子节点多出的
   特征不冲突、不向父传播，父节点特征恰为本次 lhs 的实例化映射。同一规则的不同变量赋值
   是不同规则使用、分别计质量；不同规则的变量因各自接地而天然隔离。
   - **inside**：`I(label, span)` 累加覆盖该 span 的全部子推导质量，得到分区
     `Z = Σ I(root)`；同时每个 `(span, label)` 维护一棵真正最大质量子树（规则节点用子节点
     的最大树组合，与 inside 的质量和分别维护，避免“用 inside 和拼最优树”的错误）。
   - **outside**：`O(label, span)` 自顶向下传播上下文。叶候选 `c` 的后验
     `= O(label, span)·w_c / Z`，即“整句中使用该候选的全部推导质量 / 分区”，
     **不是每位置局部 softmax**，同形同特征的不同身份也不并成一个叶。
   - **一元规则**按类别依赖图的拓扑序在每个 span 单趟闭合（child 先于 parent）；
     outside 用反向拓扑序。构造语法时用 Kahn 算法检测类别环，**声明的环即非法语法直接
     `ValueError`**，即使它在实际数据上不可达，不依赖搜索碰巧终止。
   - 二元递归合法；固定非空 token 序列配合无环一元类别图，合法推导有限。

## 输出约定

- 词形分析/生成记录字段：`analysis_id`（由完整语义资料 SHA1 生成，重查稳定、与输入顺序
  无关）、`entry_id`、`transitions`（有序 id）、`surface`、`cat`、完整最终 `features`、
  `weight`。
- 句法返回：`status`（`ok`/`no_parse`）、`log_partition`、`best_log_weight`、
  `best`（含实际 `rule_id`/变量 `assignment`/叶 `analysis_id` 与逐节点 `weight`，
  可独立重算）、`marginals`（每位置每 analysis_id 的后验）。无解析时
  `log_partition`/`best_log_weight` 为 `-inf`、`best=None`、`marginals=None`，
  不伪造均匀分布；空候选列表是合法的无解析输入。

## 测试如何核对（不让分析器与生成器互相 round-trip 自证）

- `tests/test_morph_syntax.py`：在独立小词典上**逐条手列接受路径**（零变换、尾替换、
  前缀、accept 继续扩展、同形多路径、门控不可用、尾串不匹配、环/形状拒绝）；
  在极小语法上**完整展开规则赋值与全部推导**手算分区、后验、最优树，另含一元 DAG
  链/菱形、变量共享与隔离、外部候选、真正并列树、缺特征不等于通配等。
- `tests/test_bruteforce_crosscheck.py`：用与 chart **不共享算法代码**的递归 DFS
  暴力枚举全部接地推导，在随机小语法/候选上逐案核对分区、`exp(log_*)`、最优树质量与
  逐叶后验（含后验按位置求和为 1）。
- 演示中的 `hand_check` 字段与 README 列出可手算复现的数字，例如：
  sheep 单/复同形（权重 1.0/2.0）配复数动词时分区 `ln 2`、复数身份后验 1.0、单数身份 0.0；
  再增加一个权重 0.5 的同形复数分析后，两个复数身份后验变为 0.8 与 0.2，
  而沿用旧候选的消费处仍得到旧分区 `ln 2`。

## 边界与错误处理

形状、非法身份、引用未知状态/entry、变量出现在 start、原子不在本次声明值域、
自动机环或最长路径 > 4、一元类别环（含自环）、各类数量/长度/权重越界、未配对 surrogate
等均抛 `ValueError`；合法但无词形结果返回 `[]`，合法但无句法结果返回 `no_parse`。
所有质量按 float64 解释，词形容差 atol=1e-12/rtol=1e-10，
分区/最优/后验容差 atol=rtol=1e-9。组合入口每 token 限 8 个分析，超限抛错而**不截断、
不去重**；独立 `analyze` 调用可返回更多分析。

## 文件

```
morph.py                     词形：校验 + 无环自动机接受路径枚举/过滤
syntax.py                    句法：语法校验、接地展开、inside/outside chart
pipeline.py                  组合入口（本次词形全候选 -> 同一句法入口）
demo.py                      可重复运行的结构化演示
fixtures/demo_data.json      演示词典/自动机/语法
fixtures/demo_results.json   demo.py 产出的结构化结果（含手算核对值）
fixtures/cases.json          原始基线数据（普通 cats/sleep、fish 同形、party 尾替换）
tests/test_morph_syntax.py   手算核对测试
tests/test_bruteforce_crosscheck.py  独立暴力枚举交叉验证
```

本项目使用教学规模的合成语法，其权重/概率不声称任何真实语言的解析精度。
