---
title: LTE PHY LOG 逐行字典与协议值解析（MT8000A 实战版）
published: 2026-09-03
updated: 2026-09-03
description: 对 586 万行 zCAT LTE PHY STD-LOG 做全文件逐行审计，建立 20 个打印模块、611 个消息 ID、2337 个字段的取值域字典，并逐项核对 RSRP/RSRQ 映射、EARFCN 换算、寻呼 PF/PO、Ts 三元组与 Q8 定点等 3GPP 协议值。
image: ''
tags: [LTE, PHY, MT8000A, Log分析, 3GPP, 消息ID, zCAT]
category: 协议笔记
draft: false
lang: zh-CN
password: cpetest
---

> 适用对象：需要结合 MT8000A 仪表 Trace/Measure、PS-PRIMARY 和 zCAT LTE PHY STD-LOG 定位自动化挂测问题的测试与开发人员。  
> 样例日志：`loglte_phich.txt`，5,865,468 行、约 1.3 GB，平台 `MC86001`、固件 `version=.0.6.6`。本文是独立的新文档：在流程手册基础上，增加了全文件逐行审计、20 个打印模块的逐项解析和全部 611 个消息 ID 的字典。  
> 版本提醒：文中的 zCAT 打印名来自当前样例固件；不同平台、分支和 LOG 开关可能改名或不打印。判定必须同时依靠仪表、协议栈和 PHY 三条证据链。

## 本文档怎么用

全文分三部分，按“先会查、再会判、最后能自动化”组织：

| 部分 | 章节 | 内容 | 什么时候看 |
|---|---|---|---|
| 一、日志本身 | §0 | 逐行审计结果、一行十二列怎么读、可信度标注、20 个模块总览 | 第一次接触这份日志 |
| 二、排障方法 | §1–§22 | 15 分钟快速流程、按信道（PDCCH/PDSCH/PUCCH/PUSCH/PHICH/SRS/CSI）和场景（测量、SCell、重配、切换、性能）的判据、端到端流程 | 手上有一个具体故障 |
| 三、逐项字典 | §23–§26 + 附录 A | **协议值与厂商字段值字典**、下行 56 个 ID、上行 138 个 ID、搜索/射频/测量 417 个 ID、全部 611 个 ID 的全量表 | 看到一条不认识的日志，或要确认某个值的含义 |

**最快的查法**：拿到一条日志 → 记下它的“消息ID” → 直接搜附录 A（全部 611 个 ID，含典型原文与用途）→ 需要展开就跳到 §24/§25/§26 对应模块 → 值的协议含义查 §23。

- [§23 协议值与厂商字段值字典](#23-协议值与厂商字段值字典)：某个数字在 3GPP 上代表什么、换个值又代表什么。
- [§24 下行模块逐项字典](#24-下行模块逐项字典pbch--dla--dls--csi--rxp56-个消息-id)：PBCH / DLA / DLS / CSI / RXP。
- [§25 上行模块逐项字典](#25-上行模块逐项字典ula--uls--rapc--lpc138-个消息-id)：ULA / ULS / RAPC / LPC。
- [§26 搜索、同步、射频与测量逐项字典](#26-搜索同步射频与测量逐项字典csrc--csrs--csrm--mulm--mc--cmn--rfc--dfe--接口与自定义417-个消息-id)：其余 11 个模块，含 7 个必备换算与完整会话时间线。
- [附录 A：611 个消息 ID 全量字典](#附录-a611-个消息-id-全量字典)、[附录 B：全量结构与字段取值域覆盖证明](#附录-b全量结构与字段取值域覆盖证明)。

## 0. “查看每一行”的完成标准与审计结果

这里的“查看每一行”不是把 586 万条高度重复的子帧打印原样粘贴进文档，而是：程序从文件首字节流式读取到末字节，对每一条数据行解析全部固定列，再按 `打印模块 + 消息ID + 数值归一化模板` 聚合；随后对每个消息 ID 保留出现次数、首末位置、典型原文及模板变体。这样既没有跳过任何行，又能形成真正可学习、可检索的字典。

本次审计结果：

| 项目 | 结果 |
|---|---:|
| 物理行数（含表头） | 5,865,468 |
| 数据行数 / 成功解析行数 | 5,865,467 / 5,865,467 |
| 坏行或缺列 | 0 |
| 序号断点 | 0 |
| 打印模块 | 20 |
| 唯一 `模块+消息ID` | 611 |
| 数值归一化后的结构变体 | 5,656 |
| 原文件 SHA-256 | `4f3989a4d8bc04a97f79bcf979818562fb807107e51b83040395a0172e46e896` |

配套索引：

- `generated_lte_log_index/scan_metadata.json`：逐行审计结果与完整性摘要。
- `generated_lte_log_index/module_summary.tsv`：20 个模块的行数、ID 数和模板数。
- `generated_lte_log_index/message_id_index.tsv`：611 个消息 ID 的首末位置、次数和典型原文。
- `generated_lte_log_index/normalized_template_index.tsv`：5,656 个结构变体；动态数字被占位符替换，字段名仍保留。
- `tools/build_lte_log_inventory.py`：可对同格式的新日志重新生成索引。

### 0.1 日志一行的十二列怎么读

```text
序号 | 超帧号 | 时间 | 双待 | 消息名 | 打印级别 | 打印模块 |
打印内容 | 消息ID | 长度 | UE时间 | 核序号
```

- `序号`：文件内递增记录号，不是空口帧号。
- `超帧号`：本平台的时间标签，常呈 `大周期.SFN.子帧`；真正关联 HARQ 时还要优先使用正文中的 `AbsSF/HarqId`。
- `时间`：本次抓取的相对/显示时间，可用于和仪表粗对齐。
- `打印模块`：日志生产模块，不必等同于 3GPP 层名。
- `打印内容`：函数、状态和字段；协议字段与内部寄存器可能混在同一行。
- `消息ID`：当前固件的打印格式编号，是本字典做全量覆盖的主键。
- `长度`：该日志记录的内部负载长度，不是 PDSCH/PUSCH TBS。
- `UE时间/核序号`：设备时间和处理核标签；样例主要为 `UE3`。

### 0.2 本文如何标注可信度

- **协议定义**：可在对应 3GPP TS 中找到明确语义或值域。
- **当前日志实证**：通过前后行为在本文件中验证，例如 PHICH `value=1` 后停止重传。
- **平台推断**：根据函数名、字段和正常流程推断，换固件前需要重新标定。
- **未知私有值**：寄存器、bitfield 或枚举缺少厂商符号表；本文只说明观察方法，不强行命名。

### 0.4 本版做过哪些数值核对（不是"看上去像"，是算过的）

字典里凡标"协议值"的地方，都用本文件的数据回算过一遍。下面是可复核的核对清单：

| 核对项 | 方法 | 结论 |
|---|---|---|
| EARFCN ↔ 频率 | 用 TS 36.101 公式算 5 个频点，与日志 `FreqPoint` 比对 | 全部吻合（1650/1275/1900/3201/6225） |
| 时间三元组 | `帧.时隙.Ts` 展开与同行的绝对 Ts 字段比对 | 一致（如 `0.2.6272` = 36992 Ts） |
| SIB1 位置与周期 | 窗口起点是否满足 `SFN mod 2 = 0`、子帧 5、20 ms 重复 | 335 条全部满足 |
| RSRP/RSRQ 上报映射 | 同一时刻的 dBm 值与上报值代入 36.133 公式 | `−73 dBm → 68`、`−6 dB → 28` 完全吻合；平台 NV 的 `wStandardPoint=141` 与协议基点一致 |
| 内部对数域标度 | 用 5 组 (内部值, dBm) 拟合 | 约 84 单位 = 1 dB |
| AFC 定点格式 | 用成对打印的 `[定点值, 取整值]` 反推 | Q8；`CalcPpm` 为 ppm×1024 的 Q8，与 `TotalOffsetHz` 自洽 |
| 寻呼 PF/PO | 用 TS 36.304 公式代入 `T=128, N=128, UeID=277` | 算得 PF `SFN mod 128 = 21`、PO 子帧 9，与日志一致 |
| 全字段取值域 | 对**全部 611 个**消息 ID 的 2,337 个字段做全文件 min/max/取值分布统计 | 用于判断"这个值正常吗"，见各章实测范围与附录 B.5 |
| 结构变体 | 5,656 个结构逐个归类，区分"数组长度不同"与"真有分支" | 真正的语义分支只有 5 处，已逐条确认，见附录 B.4 |
| 逐行覆盖 | 独立重跑一次，统计"落到已枚举结构"的行数 | 5,865,467 / 5,865,467 = 100%，见附录 B.2 |
| `RficMcuStat` 告警 | 统计 `Before == Current` 的比例 | 162,830 条中仅 21 条为真，99.99% 是误报 |
| 发射功率上限 | 统计 `0x6F36 TxPwr` 分布 | 范围 −24.0…+23.0 dBm，其中 23.0 dBm 出现 50 次（Power Class 3 顶格） |
| 服务小区 RSRQ | 统计 `0x8646` 的 RSRQ 上报值分布 | 351 条全部为 34（上限），与同时刻 `RSSI=[0;0]` 相关，已在 §26.6 标注为存疑项 |
| SR 与 PHICH 结论 | 全文件复核 `0x780D / 0x7B6B / 0x7B09` | SR 计数最大 2、`Fail` 恒为 0；PHICH `value=1` 27 次、`value=0` 1 次 |

### 0.3 20 个打印模块总览

模块名是固件路由标签；同一行正文可能由另一子模块产生，例如 `RFC` 列中出现 `DFE|...`。下表中的“职责”是由本日志函数族推断出的学习入口，不是3GPP标准模块划分。

| 模块 | 行数 / ID数 | 在本日志中的主要职责 | 第一批搜索词 |
|---|---:|---|---|
| `RFC` | 3,341,955 / 72 | RF/RFFE/RFIC事件、收发时隙和频点配置、RF状态监控；还承载部分DFE时序打印 | `RfReq`, `RxCfg`, `TxCfg`, `RficMcuStat`, `AsyncInfo` |
| `DFE` | 1,136,797 / 58 | 数字前端、AGC、DC/CFO、路径和采样时序 | `AgcMeanpwr`, `Gain`, `Dc`, `Cfo`, `OffsetCfg` |
| `MC` | 588,078 / 41 | PHY公共调度/场景资源和低功耗控制，含RLM/统计入口 | `Lpm Ctrl`, `RLMState`, `StdLoglInfo` |
| `CSRS` | 158,895 / 54 | PSS/SSS、频扫、搜索同步和搜索RF窗口 | `Pss`, `Sss`, `Freqscan`, `Search Req` |
| `DLA` | 126,053 / 6 | 下行硬件配置与DCI/HI报告搬运 | `Che Reg`, `Cdtr Rpt`, `DciRpt`, `HiRpt` |
| `CMN` | 102,664 / 65 | PHY内部消息、公共/专用配置编排和多载波上下文 | `PHY->PHY`, `DEDISDL`, `CgInfo`, `PUB\|` |
| `RXP` | 87,483 / 10 | 接收侧CRS功率、SNR、CIR和定时调整 | `CRsPwr`, `SNRInfo`, `CIRADJ` |
| `ULA` | 76,419 / 48 | 上行发射参数、PUCCH/PUSCH/SRS资源映射与功控 | `LtxConfigure`, `Pucch`, `PuschPow`, `Srs` |
| `CSRM` | 73,806 / 21 | 同/异频测量搜索、RSRP/RSRQ过滤和测量上报 | `#INTRA#MEAS`, `#INTER#MEAS`, `FilterInfo` |
| `MULM` | 63,170 / 4 | 多频/测量gap从流程协调 | `MulmSlaveMeasureFlow`, `GapRfState` |
| `CSI` | 56,222 / 9 | CQI/PMI/RI、RLM输入和周期/非周期反馈计算 | `CSI_FLOW`, `APER_Comb`, `RLM_Calc` |
| `CSRC` | 42,820 / 91 | 小区搜索控制、候选小区数据库、载波增加/删除入口 | `CellToDB`, `AddCarrier`, `FindCell`, `CcSrc` |
| `PBCH` | 6,434 / 17 | PBCH配置、MIB译码、CRC和驻留前信息 | `PBCH RESULT`, `PBCH CRC OK`, `MIB` |
| `ULS` | 1,678 / 32 | DCI0解析、UL grant、HARQ状态和PHICH处理 | `DecodeDci0`, `ReportUlGrant`, `Rec PHICH`, `TA timer` |
| `RAPC` | 1,535 / 56 | PRACH/随机接入时间、preamble、RAR窗口及Msg3流程 | `Preamble`, `RA-RNTI`, `RAR`, `Msg3` |
| `DLS` | 814 / 14 | PDSCH/SIB/RAR译码结果、TB CRC与统计 | `DCIInfo`, `SIBDecode`, `RarDecode`, `DecStat` |
| `PS_PHY接口消息` | 436 / 2 | PHY与协议栈之间消息收发包络 | `Phy--->Ps`, `Ps--->Phy`, `MsgID` |
| `USER_DEF_VD_12` | 136 / 8 | 项目自定义消息/状态调试，需结合内部消息表 | `RecvMsg`, `State`, `dwMsg_id` |
| `USER_DEF_VD_2` | 54 / 1 | 当前抓取中为详细 `DLS\|DCIInfo` | `TbCrcHw`, `HarqId`, `MCS`, `TBS` |
| `LPC` | 18 / 2 | DVFS/低功耗频率请求，通常不是空口协议字段 | `Dvfs Req`, `CpuFreq`, `Axi`, `Ddr` |

阅读优先级：先看 `PBCH/DLA/DLS/ULA/ULS/RAPC/CSI/CSRM` 的空口主链，再用 `CSRS/CSRC/RXP/DFE/RFC` 下钻搜索、信号质量和硬件原因；`CMN/MC/PS_PHY接口消息` 用于确认跨模块配置是否真正传递。

快速导航：

- [15 分钟快速排障](#3-15-分钟快速排障流程)
- [小区搜索、SIB、随机接入](#4-小区搜索mibsib-与随机接入)
- [PDCCH](#5-pdcch--dci-分析) / [PDSCH](#6-pdsch-分析) / [PUCCH](#7-pucch-分析) / [PUSCH](#8-pusch-分析) / [PHICH](#9-phich-分析)
- [SRS](#10-srs-分析) / [CSI](#11-csicqipmiri分析) / [测量](#12-测量分析)
- [SCell](#13-sccscell-增加删除与激活) / [重配置](#14-重配置问题) / [切换](#15-切换分析)
- [性能与峰值](#16-性能pdschpusch-峰值与误码)
- [当前日志结论](#18-当前-loglte_phichtxt-实例结论) / [命令速查](#19-大日志检索命令速查) / [报告模板](#20-自动化挂测问题报告模板)
- [规范、Context7 与官方资料](#21-规范context7-与官方资料)
- 逐项字典：[§23 协议值字典](#23-协议值与厂商字段值字典) / [§24 下行](#24-下行模块逐项字典pbch--dla--dls--csi--rxp56-个消息-id) / [§25 上行](#25-上行模块逐项字典ula--uls--rapc--lpc138-个消息-id) / [§26 搜索射频测量](#26-搜索同步射频与测量逐项字典csrc--csrs--csrm--mulm--mc--cmn--rfc--dfe--接口与自定义417-个消息-id) / [附录 A 全量 ID 表](#附录-a611-个消息-id-全量字典)

## 1. 先记住这条总原则

任何空口问题都按以下链路找“最后一个正确节点”和“第一个错误节点”：

```text
仪表配置/发送
  → UE PDCCH/DCI 检测
  → UE PDSCH 解调译码
  → UE MAC/RRC 处理
  → UE PUCCH/PUSCH/SRS 发射
  → 仪表接收与解码
  → 下一条网络消息
```

不要用单条打印下结论。例如：

- 仪表报 `NACK`：优先查 PDSCH CRC，但也可能是 HARQ 反馈映射或时序问题。
- 仪表报 `DTX`：优先查 DCI 漏检、UE 未调度反馈、TA/功率/PUCCH 资源，但不能直接等同于“UE 没发”。
- UE 报 PHICH `value=0`：要看随后是否进入重传；字段值必须用行为验证。
- 只看到 SCell 已配置，不能证明已激活；配置、激活、真正承载数据是三个不同状态。

## 2. 证据优先级与必备材料

### 2.1 三条证据链

| 证据 | 主要用途 | 典型内容 |
|---|---|---|
| MT8000A Trace/Measure | 网络实际发了什么、收到什么 | RRC、DCI、DL/UL HARQ、ACK/NACK/DTX、TBS、码率、吞吐 |
| PS-PRIMARY | RRC/MAC 状态机是否收到并完成 | dedicated、重配完成、SCell MAC CE、测量上报、RLF |
| LTE PHY STD-LOG | 每个子帧实际检测/译码/发射结果 | DCIInfo、TbCrcHw、SNR、PUSCH grant、PUCCH、PHICH、SRS |

结论至少需要两条证据链互相印证。涉及“仪表没收到 UE 上行”的问题，必须同时保留仪表接收结果和 UE 发射侧日志。

### 2.2 每次挂测必须保存

1. MT8000A 工程、脚本、Cell 参数和软件版本。
2. Trace LOG、Measure LOG；上行重点保留 `Measure_BTS1_PHYMAC(UL_HARQ).csv`（名称可能随版本变化）。
3. PS-PRIMARY、LTE PHY STD-LOG、崩溃/复位日志。
4. UE 软件版本、射频校准/NV 版本、测试时间和失败用例编号。
5. 线缆、衰减器、功分器、端口连接图与外部衰减值。
6. 失败前至少 10 s、失败后至少 5 s 的完整日志，不要只截错误那一行。

### 2.3 时间对齐

- 优先用空口事件对齐：`RRC CONNECTION RECONFIGURATION`、DCI0、RAR、某个唯一 HARQ ID。
- PHY 行头时间可能是“大周期.SFN.子帧”，SFN 会回绕；跨回绕时不能只按数字大小排序。
- 硬件中断类打印常延后。当前样例中 PHICH 内部字段 `AbsSF=8906` 打印在 `1102.890.7`；分析时以内部 `AbsSF` 为准。
- `DLS|DCIInfo` 的 `TbCrcHw` 可能指向前一笔已配置 TB。用 `Cfg/Int`、`HarqId`、`NDI/RV` 和前后子帧共同配对，不要只看行头。

## 3. 15 分钟快速排障流程

### 第 1 步：判断停在哪一层

- 未搜网/无 MIB：PSS/SSS/PBCH。
- MIB 有、SIB 无：SIB DCI/PDSCH。
- `RRCConnectionRequest` 后停：随机接入、Msg3/Msg4 或下行 RRC。
- 已连接、重配不完成：下行重配接收、参数处理、上行 Complete。
- 吞吐低：先看 CA/MIMO/调制/资源，再看 BLER、重传、功率与 SNR。
- 切换失败：测量 → Measurement Report → mobilityControlInfo → 目标小区搜索 → 目标 RACH → Complete。

### 第 2 步：先看四条总览日志

```text
StdLoglInfoDl
StdLoglInfoUl
DLS|DecStat
DLS|DCIInfo
```

它们先回答：有没有调度、平均吞吐、BLER/重传是否异常、问题属于哪一个 CC/TB/HARQ。

### 第 3 步：按现象下钻

| 现象 | 第一组搜索词 | 第二组交叉验证 |
|---|---|---|
| DL NACK | `DCIInfo`, `TbCrcHw`, `DecStat` | `SNRInfo`, MCS/TBS/RE/层数、仪表 DL HARQ |
| DL DTX | `DciValid`, `DtchInt`, `DCIInfo` | PUCCH ACK 生成、TA、仪表是否真正下发 DCI/PDSCH |
| UL NACK | `DecodeDci0`, `ReportUlGrant`, `Rec PHICH` | 仪表 UL_HARQ、UL 码率、PUSCH 功率 |
| UL DTX | `LtxConfigure`, `PuschPow`, `TA timer` | RB/DMRS/频点/TA/线缆、仪表接收功率 |
| SCell 无数据 | `AddCarrier`, `CgInfo`, `ACT_DEACT` | `CcActBitMap`, `DCIInfo[1]`, `StdLoglInfoDl:CC[1]` |
| 切换失败 | `Handover`, `#INTER#MEAS`, `RAPC` | RRC mobilityControlInfo、目标 PCI/EARFCN、T304 |

## 4. 小区搜索、MIB/SIB 与随机接入

### 4.1 小区搜索

主搜索词：

```text
LogPssHwOriValue
Sss(
PBCH CRC OK
no cell!!!!
Search Finished
```

正常 PBCH 例：

```text
PBCH CRC OK: Earcfn=1650, CellID=503, dwCrcOk=1,
RBNum=100, dwAntNum=4, BchPhich=0x2
```

字段：

- `Earcfn`：下行频点。
- `CellID`：PCI。
- `RBNum`：下行带宽；100/75/50/25/15/6 RB 对应 20/15/10/5/3/1.4 MHz。
- `dwAntNum`：PBCH 推断的 CRS 发射天线端口数。
- `BchPhich`：MIB 中 PHICH 配置的厂商编码；应与后续 PHICH 解码配置一致。

PSS 判断：PCI mod 3 决定 PSS 的 `N_ID_2`。在 `LogPssHwOriValue` 中查看对应 `Id0/Id1/Id2` 的峰值，但“1000 左右”只能作为当前平台经验阈值，必须和噪声底、AGC、其它 Id 峰值比较，不能作为跨平台固定门限。

判定流程：

1. 没有 PSS 峰值：查频点、带宽、RF 路径、外衰、AGC、仪表 Cell 是否 ON。
2. 有 PSS、无 SSS：查频偏/CFO、时序、PCI、CP 检测和干扰。
3. PSS/SSS 有、PBCH CRC 连续失败：查带宽/MIB、天线端口、SNR/CFO、PBCH 配置。
4. PBCH CRC OK 后仍未驻留：查 SIB DCI/PDSCH 和协议栈接收确认。

### 4.2 SIB

PHY 搜索：

```text
DciValid=0x8
DLS|CommDecCfg;...RNTI=0xFFFF
DLS|SIBDecode
```

`DLS|SIBDecode:Cfg,Int,Ack,Nack` 的使用方法：

- `Cfg`：配置给译码器的次数。
- `Int`：硬件完成中断次数。
- `Ack/Nack`：CRC 成功/失败累计。
- 短时允许最新一笔 `Cfg=Int+1`；持续扩大表示配置后无中断或硬件流程异常。
- `Int=Ack+Nack` 应大体成立；若长期不成立，先确认计数窗口是否刚复位或有其它状态值。

### 4.3 随机接入

PHY 模块搜索 `RAPC`：

- Msg1：`PreambleID`、`Preamble Trans Power`、`PREAMBLE_PROCESS_EV`。
- RAR 窗口：`Enable RA-RNTI`、`RAR Detect start/stop`。
- Msg2/RAR：`DLS|RarDecode`、RAR grant。
- Msg3：`TPU_INT1_RARGrantProcess`、PUSCH grant、Msg3 HARQ。
- Msg4：竞争解决消息及上层 `wResult`。

PS-PRIMARY 的 `wResult`：

- 1：非竞争随机接入成功。
- 2：竞争随机接入成功。
- 3：竞争解决定时器超时。
- 4：Preamble 发送达到最大次数。

失败分层：

- 没有 Msg1 发射：RACH 配置、PRACH 资源、RF TX、功率。
- 有 Msg1、无 RAR：仪表是否检出 PRACH；同时查 UE RA-RNTI 窗口是否正确。
- 有 RAR、无 Msg3：RAR grant 解析、PUSCH 配置、TA、UL RF。
- Msg3 后无 Msg4：仪表 Msg3 CRC、竞争解决、下行 DCI/PDSCH。

## 5. PDCCH / DCI 分析

### 5.1 主日志

```text
DLA|INFO > CInt >>> Cdtr Rpt
DciValid
DLS|DecStat
DLS|DCIInfo
DtchCfg / DtchInt
```

当前版本 `DciValid` 位图：

| 位 | 含义 | 例值 |
|---|---|---|
| bit0 | DCI 0，上行 grant | `0x1` |
| bit3 | DCI 1A/1C，SIB | `0x8` |
| bit5 | Paging | `0x20` |
| bit7 | RA 相关 | `0x80` |
| bit9 | DCI 3/3A | `0x200` |
| bit13 | 其它专用 DCI | `0x2000` |
| bit15 | DCI 4 | `0x8000` |

常见 DCI 语义速记：DCI 0 是 UL grant；1/1A/1C 主要是单码字或公共下行分配；2/2A/2B/2C 主要用于空间复用/双码字下行；3/3A 是 TPC；4 用于支持 UL MIMO 的上行调度。最终以仪表解码出的 DCI format 和该 Release 的 TS 36.212/36.213 为准。

同时看：

- `CcIdx`：载波索引。
- `Cfi`：控制区 OFDM 符号数的检测结果。
- `RntiEnInd`：当前打开的 RNTI 检测类型。
- `Candid/Commid`：专用/公共搜索空间候选统计，主要用于研发下钻。

### 5.2 DCI 漏检判据

1. 从 MT8000A Trace 确认该子帧确实发送了 DCI、DCI 格式、RNTI、CCE 聚合级别。
2. UE 同子帧 `DciValid` 是否有对应位；注意 PHY 打印延迟。
3. 看 `DtchCfg` 与 `DtchInt`，或 `DLS|DCIInfo` 的 `Cfg/Int`。持续 `Cfg>Int` 才是强异常，最新一笔在途不能算漏中断。
4. `DecStat` 的各 DCI 格式计数是否增长。
5. 若 DCI 无而 CFI/SNR 正常：查 RNTI、搜索空间、聚合级别、盲检能力、PDCCH 功率。
6. 若 CFI 也异常或控制区 SNR 差：优先查 RF、CFO、时序、干扰。

### 5.3 不要混淆

- `DciValid=0` 不代表该子帧必然异常；仪表可能本来就没有给该 UE 发 DCI。
- DCI 正确只证明获得调度信息，不证明 PDSCH 一定 CRC OK。
- 仪表报 PDSCH `DTX` 时，也可能是下行解对但 UE 的 PUCCH 反馈未被仪表检出。

## 6. PDSCH 分析

### 6.1 逐子帧主日志

```text
DLS|DCIInfo[0]
DLS|DCIInfo[1]
DLS|DCIInfo[2]
DLS|DecStat
StdLoglInfoDl
RX|SNRInfo
```

通常 `[0]/[1]/[2]` 对应 PCC/SCC1/SCC2，但必须用 `CcIdx`、频点和重配日志确认。

`DLS|DCIInfo` 关键字段：

| 字段 | 含义 | 判断重点 |
|---|---|---|
| `TM` | 传输模式 | 与重配及仪表一致 |
| `RbNum/RbS` | RB 数/起始 RB | 是否与仪表调度一致 |
| `REs` | 可用资源单元数 | 码率计算 |
| `MCS1/2` | 两个码字的 MCS | 是否超 UE 能力/码率过高 |
| `TBS1/2` | TB 比特数 | `TBS2=0` 表示单 TB |
| `HarqId` | HARQ 进程 | 串联新传与重传 |
| `NDI1/2` | 新数据指示 | 翻转通常代表新传 |
| `RV1/2` | 冗余版本/厂商打包字段 | 同 HARQ 跟踪重传 |
| `TbCrcHw` | TB CRC 结果 | 当前样例 0=无结果，1=失败，2=成功 |
| `TB1LayerNum/TB2LayerNum` | 每码字层数 | 优先于手工解析 Tpmi |
| `TB1Mod/TB2Mod` | 调制阶数 Qm | 2/4/6/8 对应 QPSK/16/64/256QAM |
| `MCSTable` | 使用的 MCS 表 | 判断是否启用 256QAM 表 |
| `Cfg/Int` | 配置/完成次数 | 判断硬件中断是否丢失 |

旧版无显式层数字段时，常从 `Tpmi=0x0p0q00ab` 读取两个 TB 层数 `p/q` 和 PMI `ab`；不同芯片打包可能不同，必须先用已知单层/双层用例标定。

常见 TM 速记：TM1 单天线端口；TM2 发射分集；TM3 开环空间复用；TM4 闭环空间复用；TM7/8/9/10 使用 UE-specific RS 支持波束赋形或更高阶多天线。日志 TM、RRC transmissionMode、仪表 MIMO 模式三者必须一致。

### 6.2 CRC 与 HARQ

当前样例实证：

- `TbCrcHw=0x00020000`：TB1 成功。
- `TbCrcHw=0x00010000`：TB1 失败。
- `TbCrcHw=0x00000000`：尚无有效 CRC 或该 TB 不存在。

两 TB 时按 `0x000m000n` 分别读取 TB1/TB2，但先结合 `TBS1/TBS2` 确认 TB 是否存在。

FDD 常见 DL HARQ RTT 为 8 ms，同一 `HarqId` 的新传/重传常相隔 8 个子帧。RV 常见序列为 0、2、3、1，但调度器和场景可能不同，不要把序列当作绝对规则。判断第几次才成功：

1. 固定 `CcIdx + HarqId + TB`。
2. 看 NDI 是否保持；保持时通常是同一数据的重传。
3. 按 RV 和子帧排序。
4. 找首次 `TbCrcHw=2`；此前 `=1` 的数量即失败传输次数。

### 6.3 SNR

日志：

```text
RX|SNRInfo CC_HW_CH[0010] ... SNR[a,b,c,d]
RX|SNRInfo CC_HW_CH[0011] ... SNR[e,f,g,h]
```

在当前 2Tx 映射中，尾号 0/1 是两组 RX 分支；每行四个值可按两个 TX 到两个 RX 的组合观察。每个 RX 的有效 SNR 可先取对应 TX 路径的最大值，但最终要看接收算法和 MIMO 模式。4Tx 或其它芯片映射可能不同。

排查顺序：

1. 单个 RX 差：天线/线缆/端口映射、LNA/校准。
2. 所有 RX 差：仪表功率、外衰、频偏、干扰。
3. SNR 高但 CRC 差：码率/MCS/层数/PMI、信道估计、参数不一致。
4. 只某 SCC 差：该 CC 的频点、RF 路径、SCell 是否真正激活。

### 6.4 码率

仪表精确法：

- 下行 Trace 选中 `LTE_PHY_DATA_REQ LTE_DL_SCH 1`。
- 读取 `A` 与 `UINT`，计算 `A/UINT`。
- 经验上不超过约 0.93；超过时先降低 MCS/层数并核对 256QAM 表。

zCAT 估算法，对每个 TB 分开：

```text
R_eff ≈ TBS_bits / (REs × TB_layer_num × Qm)
```

优先使用 `TB1Mod/TB2Mod` 作为 Qm；不要在启用 256QAM 时继续套用旧 64QAM MCS 表。估算未计所有打孔、参考信号和编码细节，适合定位明显超码率，不替代仪表的 `A/UINT`。

### 6.5 仪表报 NACK/DTX 的决策

- 仪表 NACK + UE `TbCrcHw=1`：真实下行译码失败，查 SNR、码率、MCS、层数、重传。
- 仪表 NACK + UE `TbCrcHw=2`：查 HARQ 时间配对、PUCCH ACK 值、CA ACK bundling/multiplexing。
- 仪表 DTX + UE 无 DCI：查 PDCCH/DCI。
- 仪表 DTX + UE DCI 有但无 CRC：查 PDSCH 配置/中断。
- 仪表 DTX + UE CRC 成功且已生成 ACK：查 PUCCH 资源、TA、功率、SRS 冲突、仪表接收。

## 7. PUCCH 分析

PUCCH 承载 HARQ-ACK、SR 和周期 CSI。先确认“应该发什么”，再确认“是否选对格式/资源/功率并真正发出”。

传统 LTE 常见格式：Format 1 用于 SR；1a/1b 用于 1/2 bit HARQ-ACK；2/2a/2b 主要承载 CQI/PMI/RI，并可带 HARQ-ACK；Format 3 支持更多 HARQ-ACK 比特。后续 Release 还有扩展格式，分析时以 RRC 配置和本版本 `DeterminePucchFmt` 枚举为准。

主日志：

```text
zPHY_euls_DeterminePucchFmt
zPHY_eula_GetPucchHarqAckInfo
zPHY_eula_GetPucchHarqAckLen
zPHY_eula_FDD_PucchAckParasCalc
zPHY_eula_LtxParasResMappingPucch
zPHY_eulpc_PucchPowCtrl
zPHY_eula_LtxConfigure
```

关键字段：

- `pbHasHarqAck`：本子帧是否有 HARQ-ACK 待发。
- `dwPucchHarqAckLen/Value`：ACK 比特长度和值。
- `pucchformatflag`：最终格式。
- `dwAckLen/dwOAckInfo`：LTX 配置中的 ACK 位数/内容。
- `CqiLen/QCQILen/RiLen`：CSI 是否复用在 PUCCH/PUSCH。
- `PucchPow`、`PathLoss`、`Pcmax`、`reachMaxPow`：功率是否饱和。
- `RegCfgErr/CddValueErr/LtxStatus`：硬件配置/发射状态。

FDD 基础关联：某 PDSCH 的 ACK 通常在 n+4 子帧反馈；CA、TDD、bundling/multiplexing 需按对应配置和 DAI 处理，不能一律套 n+4。

常见故障：

1. `pbHasHarqAck=0`：上游未产生 ACK，先查 PDSCH/DCI 和 HARQ 映射。
2. `pbHasHarqAck=1` 但 `dwAckLen=0`：PUCCH 格式/CA ACK 打包异常。
3. UE 配置完成但仪表 DTX：查 TA、PUCCH RB/资源索引、功率、外衰和端口。
4. 与 SRS 同子帧：查 `Set shorten format for PUCCH`、`AckNackSrsSimulTrans`，确认脚本与 UE 能力一致。
5. 上行已失步：`TA timer is not running`、TA 超时、长时间无有效上行；先恢复同步再讨论反馈值。

## 8. PUSCH 分析

### 8.1 完整链路

```text
PDCCH DCI0
  → zPHY_euls_DecodeDci0
  → zPHY_euls_DeterminePuschTransType
  → zPHY_euls_ReportUlGrantParas
  → zPHY_eulpc_PuschPowCalcProc
  → zPHY_eula_LtxConfigure
  → 仪表 UL HARQ / PHICH
```

`DecodeDci0`：

- `wCurCCNum/CIF`：在哪个 CC 调度。
- `HarqID`、`NDI`：新传/重传关系。
- `RBstart0/Lcrb0`：资源位置与长度。
- `MCS`、`TPC`、`DMRS`：调制编码、功控命令和 DMRS cyclic shift。
- `CSI`：是否触发 aperiodic CSI。
- `SrsReq`：是否触发 aperiodic SRS。

`ReportUlGrantParas`：

- `GrantType`：RAR/SPS/动态等。
- `datasize(BYTES)`、`TBS`：本次 TB 大小。
- `TransType`：1 新传、2 自适应重传、3 非自适应重传（以本日志枚举为准）。
- `wABSPhichSubFrmNo`：预期 PHICH 子帧，是关联 PHICH 的首选字段。

`PuschPowCalcProc`：

- `Pcmax`：本次最大允许发射功率。
- `PathLoss/Alpha/Popusch/Fi`：开环与闭环功控组成。
- `PuschPowVal`：计算后的功率。
- `reachMaxPow`：功率受限时的关键标志，但需用已知用例标定枚举值。
- `MPR/AMPR`：因调制、RB 占用和频段产生的回退。
- `PHR`：功率余量相关值。

### 8.2 ACK/NACK/DTX

- 仪表 ACK：PUSCH 译码正确；UE 应收到 PHICH ACK 并停止该 HARQ 重传。
- 仪表 NACK：PUSCH CRC 失败；UE 应收到 PHICH NACK 并按 grant/同步 HARQ 重传。
- 仪表 DTX：仪表未检出有效 PUSCH；查 UE 是否发射、TA、功率、RB、DMRS、频点和端口。

仪表码率：Trace 选中 `LTE_PHY_DATA_IND`，读取 `A/UINT`；经验上不超过约 0.93。超过时降低 UL MCS，随后再查是否仍 NACK。

### 8.3 UL CA

- LTE 常见终端最多支持 UL 2CC；能力与仪表授权/配置必须同时满足。
- 仅 UL 1CC 时通常是 PCC。
- UL 2CC 时检查 PCC + 一个 SCC，PS-PRIMARY `dedicated` 的 SCC common 配置中 `UlConfigCtrlFlag=1`。
- PHY 再用 `CgInfo ... UlFlag=1`、`DecodeDci0 wCurCCNum/CIF` 和 `StdLoglInfoUl:CC[x]` 验证真正有上行调度。

## 9. PHICH 分析

PHICH 是 eNB 对 UE PUSCH 的 ACK/NACK。不要与 UE 对 PDSCH 的 PUCCH HARQ-ACK 混淆。

主日志：

```text
zPHY_eula_SchdPhichRecInSad
DLA|CDTR_Reg DciRpt ... HiRpt
zPHY_euls_HARQProcess()-->Rec PHICH
Non-AdaptReTrans / AdaptReTrans / DO not Trans
```

关联流程：

1. 从 `ReportUlGrantToPS` 记录 `HarqId`、PUSCH 子帧和 `wABSPhichSubFrmNo`。
2. `SchdPhichRecInSad` 检查 `IPrbLowestIdx`、`NDmrs`、`IPhich` 与预期 PHICH 资源。
3. 到 `AbsSF` 查 `HiRpt` 和 `Rec PHICH:value`。
4. 用后续行为验证映射：停止重传是 ACK，进入重传是 NACK。
5. 与 MT8000A UL_HARQ 的 ACK/NACK/DTX 对齐。

当前样例已验证：

- `value=1` 后打印 `DO not Trans`，因此 1=ACK。
- `value=0` 后打印 `Non-AdaptReTrans`，因此 0=NACK。
- `TotalPuschNackTB(adapt+nonadapt)` 在 ACK 时仍递增，绝对值明显不可信；不要用它判断本次 PHICH。

一组可复核的实际序列：

```text
1103.864.5  Rec PHICH:value=0, Harq Id=4, AbsSF=8644
1103.864.5  Non-AdaptReTrans, HarqId=4, 下一次 PUSCH=8648, PHICH=8652
1103.865.3  Rec PHICH:value=1, Harq Id=4, AbsSF=8652
1103.865.3  DO not Trans, UlHarqTransCnt=2
```

这组行为比任何未标定的累计计数器更可靠。

FDD 常见时序：PUSCH n → PHICH n+4；NACK 后同步 UL HARQ 常在 n+8 重传。TDD 必须查 UL/DL 配置表。

PHICH 不一致矩阵：

| 仪表实际结果 | UE PHICH | 后续行为 | 优先方向 |
|---|---|---|---|
| ACK | ACK | 停止重传 | 正常 |
| NACK | NACK | 重传 | 空口 PUSCH 质量/码率 |
| ACK | NACK | 无谓重传 | PHICH 资源、功率、干扰、Ng/duration |
| NACK | ACK | 丢包且不重传 | PHICH 误判，最高优先级 |
| DTX | 任意 | 不稳定 | 先查 PUSCH 是否真正发出/被检出 |

## 10. SRS 分析

SRS 分成配置、触发/周期到达、资源冲突处理、实际发射四步。

### 10.1 配置

RRC/PS 检查：

- `soundingRS-UL-ConfigCommon`。
- `soundingRS-UL-ConfigDedicated`。
- SRS bandwidth/config index、频域位置、cyclic shift、comb、是否允许与 ACK/NACK 同时发送。

PHY：

```text
zPHY_eula_UpdataSrsBGParas
zPHY_eula_UpdataSrsBGParas_Cell
```

字段：

- `Active Subframe`、`SrsSubframeConfig`：小区级可用子帧。
- `RbStart/RbEnd`：SRS 覆盖频域。
- `Msrsb/Nb`、`Csrs`：带宽树和小区级带宽配置。
- `BandWidth`：UL 系统带宽。

### 10.2 触发与冲突

```text
zPHY_eula_CommSrsProc
DetermineSrsCellSpecStateInPusch
Set shorten format for PUCCH
DecodeDci0 ... SrsReq
```

- `LastSymbol is reserved for SRS`：该子帧末符号已为 SRS 预留，不等同于一定发了 UE-specific SRS。
- `ScheduleLocation=0[Pusch]`：与 PUSCH 同子帧，需要检查缩短 PUSCH。
- `SrsRb` 与 `PuschRB`：检查频域冲突处理。
- `SrsReq`：aperiodic SRS 触发位。

### 10.3 实际发射判定

1. 配置日志有效。
2. 周期/触发子帧正确。
3. LTX 对应子帧有 SRS channel/type 配置且 `RegCfgErr=0`。
4. RF TX 未关闭、TA 有效、功率未受限。
5. MT8000A 对应 UL/SRS 测量有能量或有效估计。

只看到 `LastSymbol is reserved` 不能宣称 SRS 发射成功。

## 11. CSI（CQI/PMI/RI）分析

### 11.1 三段链路

1. RRC 配置：periodic/aperiodic、report mode、CSI process、PCell/SCell、CSI-RS/CRS 资源。
2. PHY 计算：信道/SNR → CQI/PMI/RI。
3. UCI 上报：PUCCH 或 PUSCH，仪表正确接收。

### 11.2 主日志

```text
CSI_FLOW:First_FdBkCfg_IN_CAIdx
CSI_FLOW:First_FdBkCfg_INT1_CAIdx
CSI_FLOW:AperRepJudge_IN
CSI_FLOW:RLM_FdBkFirCfg
CSI_FLOW:RLM_Calc_In
CSI_FLOW:PCellCSI_En_IN
APER_Comb
zPHY_eula_LtxConfigure ... CqiLen/QCQILen/RiLen
DecodeDci0 ... CSI
```

字段：

- `wTransMode`：必须与 RRC TM 一致。
- `wCsiEn_aop/wAperiodTrigger`：aperiodic CSI 是否启用/触发。
- `g_wScellComFlag`：SCell CSI 相关状态。
- `wTxAnPortsNum/wUeRxAttennaNum`：PMI/RI 计算的天线维度。
- `dwRLMValue/sdwSnrValue`：内部链路质量输入；`0xffffffff`、`524287` 常是无效哨兵值，不能当成极好信道。
- `riBitLen*/riValue`：RI 打包长度和值。
- `CqiLen/QCQILen/RiLen`：最终 UCI 是否实际带上 CSI；全部为 0 表示该次发射没有 CSI。

### 11.3 故障流程

- 未配置：回到 RRC `cqi-ReportConfig`/CSI process。
- 配置有、没有周期/触发：查 config index、DCI0 CSI request、DRX/gap。
- 触发有、计算输入为无效哨兵：查参考信号、CSI-RS/CRS、SCell 激活、测量窗口。
- 计算有、`CqiLen/RiLen=0`：查 report mode/打包/PUCCH-PUSCH 复用。
- UE UCI 有、仪表没收到：查 PUCCH/PUSCH 资源、功率、TA。
- 仪表收到但值不合理：与 `SNRInfo`、层数、PMI、实际 BLER 做闭环验证。

## 12. 测量分析

### 12.1 同频

PHY：

```text
#INTRA#MEAS
#CSRM:RX:RSRPoffsetCal
FilterInfo
```

PS-PRIMARY：

```text
LTE_P_INTRA_MEAS_IND_EV
```

`#INTRA#MEAS` 字段：`Earfcn`、`ID`、`Rsrp`、`Rsrq`、`MeasAge`、`SearchAge`。

当前编码的 RSRP 转换：

```text
RSRP_dBm ≈ encoded_Rsrp - 141
```

例如 `Rsrp=69` 对应约 -72 dBm。边界值 0/97 是范围，不应按单点解释。RSRQ 常用编码的区间下界可按 `value/2 - 20 dB` 理解，边界值同样是范围；内部 `#CSRM` 原始定点值不要直接套此公式。

### 12.2 异频

PHY：

```text
#INTER#MEAS
InterMeasdebug_CtrlMeasFilterReq
MulmSlaveMeasureFlow
MeasProcStart / Measure Finished
```

PS-PRIMARY：

```text
LTE_P_INTER_MEAS_IND_EV
```

异频失败分层：

1. RRC 没有 measurement object/report config/measId。
2. gap 未配置或 `bGapCfgState` 不对。
3. gap 有但 RF 未切到目标频点。
4. PSS/SSS 搜索无结果。
5. 搜到邻区但 CSRM 无 RSRP/RSRQ。
6. PHY 有结果但 PS 无 `INTER_MEAS_IND`。
7. PS 有结果但未触发 A3/A5：查 offset、hysteresis、TTT 和 L3 filter。

事件速记：A1 服务小区高于门限；A2 服务小区低于门限；A3 邻区相对服务小区达到 offset；A4 邻区高于绝对门限；A5 同时要求服务小区低于门限 1、邻区高于门限 2。必须把 hysteresis 和 timeToTrigger 一起算入，瞬时越线不等于应立即上报。

### 12.3 测量不能只看瞬时值

- 至少观察多个周期的稳定值和 `MeasAge/SearchAge`。
- 比较仪表配置的 RS power、线损与 UE RSRP，固定条件下突变通常是 RF/AGC/路径问题。
- 切换问题必须同时确认测量配置、测量结果、Measurement Report 三步。

## 13. SCC/SCell 增加、删除与激活

### 13.1 增加 SCell

RRC/PS 链路：

```text
RRC CONNECTION RECONFIGURATION (sCellToAddModList)
LTE_P_DEDICATED_CONFIG_REQ_EV
T_zEurrc_RRCConnectionReconfigurationComplete
RRC CONNECTION RECONFIGURATION COMPLETE
```

PHY：

```text
CMN|DEDISDL[StartDediProc]
PUB| Rst:Core=... CcNum=2
L1l_SchedCsrc_AddCarrier
CMN|CgInfo: dwServCellIdx=1
RFC|RxCfg / TxCfg
```

成功至少满足：

1. `CcNum` 从 1 增加到 2。
2. `AddCarrier` 的 `dwServCellIdx/PCI/EARFCN/BW` 与仪表一致。
3. `CgInfo` 的 `UlFlag` 与脚本一致。
4. RF/DFE/CSR/DL/UL/CSI 模块配置全部完成，`DEDISDL ... AllDoneFlg` 完成。
5. RRC Complete 被仪表收到。

### 13.2 删除 SCell

RRC 看 `sCellToReleaseList`。PHY 预期看到 release/del carrier 流程、`CcNum` 减少、对应 RF/DFE/DL/UL 资源释放。当前样例没有独立删除场景，实际打印名需在目标版本用正常删除用例标定；不能把全栈 `RESET` 当成正常 SCell 删除。

### 13.3 激活/去激活

PS-PRIMARY：

```text
LTE_P_ACT_DEACT_SCELL_CTRL_ELEMNT_IND_EV
ActDeactSCellInfo
```

MAC CE 中 `Ci=1` 激活 `sCellIndex=i`，`Ci=0` 去激活；未配置的 index 忽略。例：`ActDeactSCellInfo=6` → `0000 0110` → 激活 index 1 和 2。

PHY 二次验证：

```text
SccMeasStateCfg ... bScellExist/bScellActive
DFE|IntraCaAgcCollect ... CcActBitMap/CcActive
RX|SNRInfo 对应 CC
DLS|DCIInfo[1]
StdLoglInfoDl:CC[1]
```

状态区别：

- 已配置：`CcNum` 包含该 SCell。
- 已激活：MAC CE 和 PHY active bitmap 有效。
- 已承载：该 CC 有 DCI/PDSCH、TBS 和吞吐计数。

只满足前一项不能推出后一项。

## 14. 重配置问题

先排除 RLF：

```text
EL2_EURRC_RADIOLINK_FAIL_IND_EV
```

然后把仪表 RRC 与 PS `dedicated` 同时间对齐：

1. 仪表发 `RRC CONNECTION RECONFIGURATION`。
2. UE 收到 `LTE_P_DEDICATED_CONFIG_REQ_EV`。
3. PHY 各模块完成 dedicated 配置。
4. UE 产生 `T_zEurrc_RRCConnectionReconfigurationComplete`。
5. 仪表收到 `RRC CONNECTION RECONFIGURATION COMPLETE`。

定位：

- 第 1 有、第 2 无：下行 DCI/PDSCH/RRC 解码。
- 第 2 有、第 3 卡住：参数不支持、资源分配、RF/PHY 配置。
- 第 3 有、第 4 无：RRC/能力/参数处理。
- 第 4 有、第 5 无：上行 PUCCH/PUSCH、TA、功率或仪表接收。

PCC 重配重点：TM、256QAM、天线端口、PUCCH、SRS、CSI。SCell 重配重点：PCI/EARFCN/BW、UL enable、跨载波调度、SCell index。

## 15. 切换分析

不要把包含字符串 `HandoverReqPro` 的内部处理函数直接当作切换成功证据；某些版本的 dedicated/SCell 重配也复用该函数。

完整切换链：

```text
测量配置
 → 邻区测量结果
 → Measurement Report
 → RRC Reconfiguration + mobilityControlInfo
 → 目标频点/PCI 搜索与同步
 → 目标 PBCH（按场景）
 → 目标随机接入
 → RRC Reconfiguration Complete
 → 目标小区业务恢复
```

每段检查：

1. 测量：`#INTER#MEAS`/`LTE_P_INTER_MEAS_IND_EV`，事件条件、TTT 是否满足。
2. 命令：Trace 中 `mobilityControlInfo` 的 target PCI/EARFCN、T304、newUE-Identity、RACH dedicated。
3. PHY：目标频点 RF、PSS/SSS、`g_bHandoverCsrCnf` 等版本相关状态。
4. RACH：若有 dedicated preamble，应走非竞争接入；核对 preamble ID。
5. 完成：Complete 是否从目标小区上行并被仪表收到。
6. 恢复：目标小区 PDCCH/PDSCH/PUSCH 是否开始，源小区资源是否释放。

失败分类：

- 没有 Measurement Report：测量配置/门限/gap。
- 有报告无 HO command：仪表脚本/网络决策。
- 有 command 无目标同步：频点/PCI/RF/搜索。
- 目标同步有、RACH 失败：PRACH/功率/TA/RAR。
- RACH 成功无 Complete：上行 RRC/安全/重配处理。
- Complete 后无业务：目标 bearer/调度/CA 配置。

## 16. 性能、PDSCH/PUSCH 峰值与误码

### 16.1 总览日志

`StdLoglInfoDl`：

- `AvgPHYTb.kbps`：PHY 平均 TB 吞吐。
- `1TB/2TB`：单/双 TB 调度次数。
- `TB1Dec/TB2Dec`：各 TB 译码统计，使用前先以已知 ACK/NACK 用例标定两个数组元素。
- `aveRbNum`、`MCS`、`CQI`、`RI`：资源与链路自适应状态。
- `Rv_Num`、`hq_fail`：重传与 HARQ 失败。

`StdLoglInfoUl`：

- `AvgPHY.Kbps`、`avgTB.byte`。
- `trans:[newTx,AdReTx,NadReTx,HqFai]`。
- `aveRbNum*10`、`MCS*10`、`UlBLER*10`。
- `SR[trig,reTx,Fail]`、`RACH[trig,succ,msg12Fail,msg34Fail]`。

累计计数要取时间窗口差值；刚复位、跨场景或出现明显大初值时，不要直接用绝对值。

### 16.2 峰值吞吐计算

本手册把“PDSCH/PUSCH 峰值”解释为 PHY 峰值吞吐。若你指峰值功率，应另看仪表 UL power 和 UE `PuschPow/PucchPow/Pcmax`。

DL 有效 PHY 吞吐：

```text
DL_Mbps = Σ(成功的新传 TBS_bits) / (窗口_ms × 1000)
```

UL 有效 PHY 吞吐（日志 datasize 为字节时）：

```text
UL_Mbps = Σ(成功的新传 datasize_bytes × 8) / (窗口_ms × 1000)
```

CA 时对全部 active CC 求和；重传不能重复计入有效吞吐。更便捷时直接用稳定窗口的 `StdLoglInfoDl/Ul`，再用逐 TB 计算抽样验证。

### 16.3 峰值上不去的顺序

1. UE capability：Category、DL/UL CA、MIMO 层数、256QAM/UL 64QAM。
2. 仪表许可与硬件：实际可用 LTE CC、MIMO、UL CA 能力。
3. SCell：已配置、已激活、各 CC 真正有调度。
4. 资源：RB 是否满配、是否每个可用子帧都调度。
5. 调制/层数：`TBxMod`、`TBxLayerNum` 是否达到目标。
6. 码率：`A/UINT`、zCAT 估算是否合理。
7. BLER/重传：高 BLER 会直接吞掉峰值。
8. RF：SNR、各 RX 路径、功率受限、MPR/AMPR。
9. 上层：PHY 吞吐正常但 IP 低，查 RLC/PDCP/TCP、服务器和 PC 网卡。

### 16.4 性能判定建议

- 先做单 CC、低 MCS、单层基线，再逐步开 CA、MIMO、256QAM。
- 每次只改变一个变量，保存稳定 10–30 s 窗口。
- 目标不是“零 NACK”，而是 BLER、重传和吞吐符合用例预期。
- 仪表未配置重传的特殊用例，不能套用现网 1 新传 + 3 重传的经验。

## 17. MT8000A 侧操作要点

Anritsu 公共资料确认 MT8000A/SmartStudio NR 支持 5G/LTE 基站与核心网仿真、LTE CA/MIMO 和吞吐测试；具体 Trace 字段、Measure 文件名、菜单位置随软件许可和版本变化。

建议固定工作法：

1. 在 Sequence/Message Log 标记 attach、reconfiguration、handover 的起止点。
2. Trace 同时保留 RRC、PHY DATA REQ/IND、DL/UL HARQ。
3. Measure 导出每 BTS/CC 的 DL/UL HARQ、吞吐、接收功率。
4. 每次改脚本后导出完整 Cell 参数，不只截图差异项。
5. 先以仪表“实际发出/收到”的 Trace 为事实，GUI 目标配置只代表意图。
6. 自动化挂测失败时记录首次异常子帧，避免后续级联 NACK/DTX 淹没根因。

公开资料没有给出你当前软件版本全部内部 Trace 字段定义；`A/UINT`、Measure CSV 名称和 zCAT 打印的解释需以项目内部字典/已知正常用例继续标定。

## 18. 当前 `loglte_phich.txt` 实例结论

### 18.1 已确认的正常链路

- 小区搜索：存在 `PBCH CRC OK`，例 `Earcfn=1650, CellID=503, RBNum=100, dwAntNum=4`。
- 后续连接态 PCell 示例为 EARFCN 1275、PCI 30、75 RB。
- PDSCH `DLS|DCIInfo` 共 54 条：`TbCrcHw` 中 49 次成功、1 次失败、4 次无结果。打印窗口有限，不能代表全程 BLER。
- PHICH：27 次 `value=1`、1 次 `value=0`。唯一 NACK 后进入 `Non-AdaptReTrans`，下一次 PHICH 为 ACK 并 `DO not Trans`；说明捕获到的 PHICH 处理链能闭环。
- SCell 添加：`1104.156.6` 出现 `CcNum=2`、`L1l_SchedCsrc_AddCarrier`，随后 `CgInfo dwServCellIdx=1 ... UlFlag=1`；证明配置层添加成功。

### 18.2 需要重点核实但不能直接定根因

1. `TA timer is not running` 共 1099 次，范围约 `1102.881.6` 到 `1105.526.9`。如果仪表同时间出现 PUCCH/PUSCH DTX，这是高优先级线索；若上行仍持续 ACK，则可能是无有效 TA 时段的重复告警。
2. CSI `RLM_Calc_In` 有 2652 次 `dwRLMValue=0xffffffff`，只有 25 次非该值；没有看到 `CqiLen/RiLen` 非零。可能是用例未配置 CSI，也可能是 CSI 输入/上报链未建立，需先查仪表 RRC 配置。
3. 有 86 条 `#INTRA#MEAS`，没有 `#INTER#MEAS`；该日志不能证明异频测量或异频切换链正常。
4. `RficMcuStat|McuHang` 告警约 162837 条，且部分行的 before/current 计数看起来仍变化。它可能是 RFIC/日志判定异常或告警风暴，不能仅凭名字归因 PHICH；应让 RF/平台负责人结合 RF 中断、复位、吞吐掉零时刻确认。
5. 没有发现明确的 PS `LTE_P_ACT_DEACT_SCELL_CTRL_ELEMNT_IND_EV`；PHY active bitmap 的短时变化不能代替 MAC CE 证据，因此不能宣称 SCell 激活流程已完整验证。

### 18.3 当前 PHICH 小结

这份日志没有显示“持续 PHICH 解错”：捕获到的 28 次 PHICH 中仅 1 次 NACK，且重传后恢复。若自动化用例仍报 UL NACK/DTX，下一步应按失败时刻提取 MT8000A `UL_HARQ`，与 `wABSPhichSubFrmNo + HarqId + RB/DMRS` 对齐，而不是看 `TotalPuschNackTB` 的大数值。

## 19. 大日志检索命令速查

文件很大，优先用 `rg -m` 限制输出：

```bash
# 小区搜索/MIB/SIB
rg -m 100 'LogPssHwOriValue|PBCH CRC OK|SIBDecode|no cell!!!!' loglte_phich.txt

# PDCCH/PDSCH
rg -m 200 'DciValid|DLS\|DCIInfo|DLS\|DecStat|TbCrcHw|SNRInfo' loglte_phich.txt

# PUCCH/PUSCH
rg -m 200 'DeterminePucchFmt|PucchHarqAck|DecodeDci0|ReportUlGrant|PuschPow' loglte_phich.txt

# PHICH 完整链
rg -m 300 'SchdPhichRecInSad|Rec PHICH|Non-AdaptReTrans|DO not Trans' loglte_phich.txt

# SRS/CSI
rg -m 300 'UpdataSrs|CommSrsProc|SrsReq|CSI_FLOW|APER_Comb|CqiLen|RiLen' loglte_phich.txt

# 测量/SCell/切换
rg -m 300 '#INTRA#MEAS|#INTER#MEAS|AddCarrier|CgInfo|ActDeact|Handover' loglte_phich.txt

# 只保留帧号、模块、内容三列
rg -m 100 'Rec PHICH' loglte_phich.txt | cut -f2,7,8
```

逐个 HARQ 分析时，把帧号、CC、`HarqId`、NDI、RV、TBS、CRC、反馈整理成表，不要在原始 1.3 GB 文本中来回跳。

## 20. 自动化挂测问题报告模板

```text
用例/时间/版本：
MT8000A 工程与软件版本：
UE 软件/PHY/RF NV 版本：
PCC/SCC：EARFCN、PCI、BW、DL/UL、TM、MIMO、QAM：
失败现象与首次异常时间：

仪表证据：
- 首次异常前一条正确消息：
- 首次异常消息/子帧：
- DL/UL HARQ：ACK/NACK/DTX：
- DCI/TBS/MCS/RB/码率 A/UINT：

PS 证据：
- 是否收到 dedicated/reconfig/HO/SCell CE：
- 是否生成 Complete/Measurement Report/RLF：

PHY 证据：
- CC/HarqId/AbsSF：
- DciValid、Cfg/Int：
- TBS、MCS、层数、Qm、TbCrcHw：
- SNR/RSRP/CFO：
- PUCCH/PUSCH/SRS LTX 与功率/TA：
- PHICH value 与后续重传行为：

结论：最后正确节点 → 第一个错误节点
已排除：
下一步最小验证：只改变一个变量
```

## 21. 规范、Context7 与官方资料

### 21.1 模块与规范入口

厂商字段名和枚举值必须先用当前固件的已知正常/异常用例标定；3GPP 规范用于确认空口语义、时序和过程，不能反向证明某个私有字段的编码。

| 分析模块 | 首要规范 | 用它确认什么 |
|---|---|---|
| PSS/SSS、PBCH、PDCCH、PDSCH、PUCCH、PUSCH、PHICH、SRS | TS 36.211 | 物理信道、信号、资源映射与调制 |
| DCI、UCI、HARQ-ACK、信道编码 | TS 36.212 | 控制信息格式、复用和编码 |
| HARQ 时序、功控、CSI、SRS、资源分配 | TS 36.213 | UE 物理层过程与参数解释 |
| RSRP、RSRQ 等测量量 | TS 36.214 | 物理层测量定义 |
| 测量性能与无线资源管理要求 | TS 36.133 | 测量精度、时延和 RRM 要求 |
| 随机接入、MAC CE、SCell 激活/去激活 | TS 36.321 | MAC 过程和控制元素 |
| RRC 重配、SCell 增删、测量事件、切换 | TS 36.331 | 配置 IE、触发条件和 RRC 状态机 |

推荐核对顺序：先在 MT8000A Trace 中确认“网络实际发送内容”，再按上表找到规范语义，最后回到 zCAT 日志用行为标定私有字段。不要只根据字段名猜测 ACK/NACK、单位或位序。

### 21.2 Context7 连接与覆盖验证

2026-09-02 已在本机 Codex 全局配置中启用 Context7：

```bash
codex mcp add context7 -- npx -y @upstash/context7-mcp
codex mcp get context7
codex mcp list
```

本机协议级验证结果：`@upstash/context7-mcp` v4.0.4 可以初始化，并暴露以下工具：

```text
resolve-library-id
query-docs
```

新版服务器使用 `query-docs`；若旧流程写的是 `get-library-docs`，应以实际 `tools/list` 返回名称为准。桌面端、CLI 与 IDE 共用 `~/.codex/config.toml`，但已经运行的任务不会热加载新工具；重启 Codex 或新建任务后用 `/mcp` 或 `codex mcp list` 确认。

本次先调用 `resolve-library-id`，解析到官方高信誉库：

```text
/websites/3gpp_specifications-technologies
```

随后分别查询 PDCCH/PDSCH/PHICH、PUSCH/PUCCH/SRS/CSI、SCell、测量/切换。当前 Context7 库只返回 3GPP 门户、规范发布流程和 Release 概览，没有返回 TS 36.211/36.213/36.331 正文条款。因此：

1. Context7 可用于发现资料入口和检查资料是否更新。
2. 当前不能把它当作 LTE 规范正文检索器，也不能用低相关检索结果支持具体 HARQ 时序或字段解释。
3. 当 Context7 无正文覆盖时，继续使用 3GPP 官方 Specification Details/原始规范，并在结论中注明覆盖限制。
4. Context7 结果的 `Source Reputation` 不是协议结论本身；必须检查返回段落是否真正回答当前问题。

适合后续任务的查询模板：

```text
1) resolve-library-id
   libraryName: 3GPP
   query: LTE TS 36.213 <单一专题>

2) query-docs
   libraryId: /websites/3gpp_specifications-technologies
   query: <只问一个概念，例如 FDD uplink HARQ timing>

3) 若没有规范正文：打开对应 3GPP 官方 TS 页面交叉核验，并记录“Context7 无正文覆盖”。
```

### 21.3 官方资料

- [3GPP TS 36.211：Physical channels and modulation](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2425)
- [3GPP TS 36.212：Multiplexing and channel coding](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2426)
- [3GPP TS 36.213：Physical layer procedures](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2427)
- [3GPP TS 36.214：Physical layer measurements](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2428)
- [3GPP TS 36.133：RRM requirements](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2420)
- [3GPP TS 36.321：MAC protocol](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2437)
- [3GPP TS 36.331：RRC protocol](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2440)
- [Anritsu MT8000A 官方产品页](https://www.anritsu.com/zh-cn/test-measurement/products/mt8000a)
- [Anritsu MT8000A Product Introduction（含 LTE/RTD 选件）](https://dl.cdn-anritsu.com/en-en/test-measurement/files/Product-Introductions/Product-Introduction/mt8000a-el11400.pdf)
- [Anritsu SmartStudio NR Product Introduction（含 LTE CA/MIMO 能力示例）](https://dl.cdn-anritsu.com/en-en/test-measurement/files/Product-Introductions/Product-Introduction/mt8000a-ssnr-el1100.pdf)
- [OpenAI 官方：Codex MCP 配置与 Context7 示例](https://developers.openai.com/codex/mcp)

本版已经完成 Context7 与官方 Web 的双重查询。Context7 当前缺少 LTE TS 正文覆盖，因此具体协议结论仍以 3GPP 官方规范入口为准；该限制不影响本手册基于实际日志行为得出的厂商字段标定结果。

## 22. 把所有模块串起来：一次 LTE 会话的端到端流程

这一章用于建立全局认识。实际排障时始终沿时间方向寻找“最后一个正确节点”和“第一个错误节点”，不要一开始就在某个 PHY 模块里孤立地找异常。

### 22.1 一张总流程图

```text
仪表开小区
  │
  ├─ PSS/SSS 搜索、频率/时间同步
  │    日志：LogPssHwOriValue、Sss、Search Finished
  │
  ├─ PBCH 解 MIB
  │    日志：PBCH CRC OK、Earcfn、CellID、RBNum、dwAntNum
  │
  ├─ PDCCH 找 SIB DCI ──→ PDSCH 解 SIB
  │    日志：DciValid bit3、DCIInfo、TbCrcHw、SIBDecode
  │
  ├─ 随机接入
  │    PRACH Msg1 → RAR Msg2 → PUSCH Msg3 → PDSCH Msg4
  │    日志：RAPC、RAR Ack、wResult、DecodeDci0、ReportUlGrant
  │
  ├─ RRC 建连/重配
  │    PDCCH → PDSCH承载RRC消息 → UE处理 → PUSCH承载Complete
  │    日志：LTE_P_DEDICATED_CONFIG_REQ_EV、DEDISDL、ReconfigurationComplete
  │
  ├─ 连接态业务形成两个HARQ闭环
  │    │
  │    ├─ 下行：PDCCH/DCI → PDSCH → CRC → PUCCH ACK/NACK
  │    │
  │    └─ 上行：PUCCH SR/BSR → PDCCH DCI0 → PUSCH → PHICH ACK/NACK
  │
  ├─ 无线质量反馈形成调度闭环
  │    CSI(CQI/PMI/RI) + SRS + RSRP/RSRQ
  │       → 仪表/基站调整MCS、层数、RB、功率和载波调度
  │
  ├─ CA/SCell
  │    RRC添加SCell → MAC CE激活 → SCC产生DCI/PDSCH/PUSCH
  │    去激活/删除时按相反方向释放
  │
  ├─ 移动性
  │    邻区测量 → Measurement Report → HO Command
  │       → 目标小区同步 → 目标RACH → Complete → 业务恢复
  │
  └─ 性能闭环
       配置/能力 → 调度资源 → MCS/层数 → CRC/BLER/HARQ
          → 有效TBS → PDSCH/PUSCH吞吐峰值
```

如果流程在某个箭头后没有下一步打印，箭头两端就是最需要对齐的模块。例如 PDSCH CRC 已成功但仪表仍报 DTX，问题通常已经从“下行解码”移动到“PUCCH 反馈链”。

### 22.2 阶段一：从开小区到 RRC_CONNECTED

| 顺序 | 仪表/空口动作 | UE 应有结果 | 首要日志 | 没有下一步时查什么 |
|---|---|---|---|---|
| 1 | 仪表发 PSS/SSS | 找到频点、PCI并完成同步 | `LogPssHwOriValue`、`Sss` | 功率、频点、PCI mod 3、CFO、线缆 |
| 2 | 仪表发 PBCH | PBCH CRC OK，得到 MIB/带宽 | `PBCH CRC OK`、`RBNum` | PBCH功率、天线端口、同步稳定性 |
| 3 | 仪表调度 SIB | 找到 SIB DCI并解开 PDSCH | `DciValid bit3`、`DCIInfo`、`TbCrcHw` | 先分 PDCCH漏检还是PDSCH CRC失败 |
| 4 | UE 发 PRACH Msg1 | 仪表检出前导 | `RAPC` Msg1 | PRACH配置、频点、功率、TA前状态 |
| 5 | 仪表发 RAR Msg2 | UE正确解RAR、获得TA和UL grant | RAR `Ack`、RA相关DCI | RA-RNTI、RAR窗口、PDCCH/PDSCH |
| 6 | UE 发 PUSCH Msg3 | 仪表解到 RRCConnectionRequest | `DecodeDci0`、`ReportUlGrant`、`LtxConfigure` | RB/DMRS、TA、PUSCH功率、UL DTX/NACK |
| 7 | 仪表发 Msg4 | UE完成竞争解决 | `wResult=2`等结果 | Msg4 DCI/PDSCH、竞争解决定时器 |
| 8 | RRC消息交互完成 | UE进入连接态 | PS RRC日志、RRC Complete | 按下行消息和上行Complete分界 |

这里最常见的误区是：看到 `RrcConnectionRequest` 就认为随机接入全部成功。它只证明 Msg3 至少在协议侧被处理过；仍需确认 Msg4/竞争解决和最终 RRC 状态。

### 22.3 阶段二：下行业务闭环

```text
仪表调度PDCCH
  → UE检测DCI
  → 按DCI参数接收PDSCH
  → TB译码/CRC
  → 生成HARQ-ACK
  → PUCCH或PUSCH发ACK/NACK
  → 仪表决定停止或重传
```

逐步对齐：

1. PDCCH：仪表 Trace 的 DCI format/RNTI/CCE，对齐 `DciValid`、`DtchCfg/Int`、`DecStat`。
2. PDSCH：按 `CcIdx + HarqId + NDI + RV + TB` 串联 `DCIInfo`，查看 `TBS/MCS/REs/Layer/Qm/TbCrcHw`。
3. 解码环境：用 `SNRInfo`、码率估算和仪表 `A/UINT` 判断 RF 差还是码率过高。
4. PUCCH：CRC成功后检查 `pbHasHarqAck`、`AckLen/Value`、格式、资源、功率和 `LtxStatus`。
5. 仪表：确认相应 DL HARQ 显示 ACK/NACK/DTX，以及是否真的安排重传。

三种典型断点：

- 没有 DCI：PDCCH/DCI 漏检，PDSCH 模块还没有开始工作。
- DCI 有、`TbCrcHw=1`：PDSCH真实译码失败，查SNR、码率、MIMO和参数一致性。
- `TbCrcHw=2`、UE已生成ACK、仪表报DTX：下行已经解对，转查PUCCH资源、TA、功率和仪表接收。

### 22.4 阶段三：上行业务闭环

```text
UE有上行数据
  → PUCCH发送SR（或已有grant/BSR）
  → 仪表在PDCCH发送DCI0
  → UE解析UL grant
  → PUSCH编码、功控并发射
  → 仪表得到ACK/NACK/DTX
  → 仪表发送PHICH
  → UE停止传输或执行自适应/非自适应重传
```

逐步对齐：

1. 没有 grant：先确认 SR 是否触发并通过 PUCCH 发出，再看仪表是否下发 DCI0。
2. 有 DCI0：用 `DecodeDci0` 核对 `HarqId/NDI/RBstart/Lcrb/MCS/DMRS/TPC`。
3. 准备发射：用 `ReportUlGrantParas` 核对 `datasize/TransType/wABSPhichSubFrmNo`。
4. 实际发射：用 `PuschPowCalcProc + LtxConfigure` 检查 `PuschPow/Pcmax/reachMaxPow/RegCfgErr/LtxStatus`。
5. 仪表接收：UL_HARQ 的 ACK/NACK/DTX 是基站侧事实；DTX 与 NACK 的方向不同。
6. PHICH：按 `wABSPhichSubFrmNo + HarqId` 找 `Rec PHICH:value`，再用 `DO not Trans` 或重传行为验证ACK/NACK映射。

```text
仪表 NACK + UE收到PHICH NACK + UE重传：PUSCH质量/码率问题
仪表 ACK  + UE收到PHICH NACK + UE重传：PHICH接收或资源映射问题
仪表 NACK + UE收到PHICH ACK  + UE不重传：高风险PHICH误判
仪表 DTX  + UE侧无LTX：调度/状态机问题
仪表 DTX  + UE侧有LTX：TA、功率、RB/DMRS、频点或RF路径问题
```

### 22.5 CSI、SRS 和测量如何影响上下行业务

这三个模块通常不直接承载用户数据，但共同决定调度器“给多少资源、用多高MCS、上几层、在哪个频点发”。

```text
下行参考信号
  → UE估计信道
  → CSI计算CQI/PMI/RI
  → 通过PUCCH/PUSCH上报
  → 仪表调整PDSCH MCS、PMI、RI和层数

UE发送SRS
  → 仪表估计上行信道
  → 调整PUSCH RB、MCS、频域位置和功控

CRS/邻区信号
  → UE得到RSRP/RSRQ
  → L1/L3滤波及事件判断
  → Measurement Report
  → 重配、SCell策略或切换决策
```

关联判断：

- CSI 未配置或 `CqiLen/RiLen=0`：PDSCH仍可能有数据，但MCS/层数可能固定或无法自适应。
- CSI 长期无效哨兵值：先查参考信号和SCell状态，再看UCI打包，不能直接认为“CQI很差”。
- SRS预留不等于真正发射：必须再看触发、冲突处理、LTX和仪表接收。
- 测量值存在不等于会切换：还要经过L3滤波、事件门限、hysteresis、TTT和Measurement Report。
- PUCCH同时连接下行HARQ、SR和周期CSI；因此PUCCH异常可能同时表现为DL DTX、无UL grant和CSI缺失。

### 22.6 重配与 SCell 的完整闭环

```text
仪表PDCCH/PDSCH发送RRC Reconfiguration
  → UE解码并上送PS
  → LTE_P_DEDICATED_CONFIG_REQ_EV
  → PHY执行DEDISDL/RF/DFE/CSI/UL/DL配置
  → CcNum增加，AddCarrier/CgInfo完成
  → UE通过PUSCH发送Reconfiguration Complete
  → 仪表通过PDSCH发送SCell Activation MAC CE
  → PS收到ACT_DEACT，PHY active bitmap改变
  → SCC出现DCIInfo、TBS、SNR和吞吐
```

SCell 必须分成三层判断：

| 层次 | 成功证据 | 失败表现 |
|---|---|---|
| 已添加/配置 | `CcNum`增加、`AddCarrier`、`CgInfo` | 参数不支持、dedicated配置卡住 |
| 已激活 | PS收到MAC CE、`CcActBitMap`有效 | 有SCell配置但SCC一直无调度 |
| 已承载数据 | `DCIInfo[1]`、SCC TBS/吞吐增长 | 已激活但仪表未给SCC资源 |

去激活只应停止该 SCell 承载，不等同于删除配置；删除应看到 `sCellToReleaseList`、`CcNum`减少和资源释放。性能上不去时，应按“添加 → 激活 → 承载”逐层确认，而不是只搜一次 `AddCarrier`。

### 22.7 测量到切换的完整闭环

```text
RRC下发measurementObject/reportConfig/measId
  → PHY按gap搜索目标EARFCN/PCI
  → 产生#INTER#MEAS RSRP/RSRQ
  → PS做L3滤波和A3/A5事件判断
  → UE通过PUSCH发送Measurement Report
  → 仪表通过PDSCH发送mobilityControlInfo
  → UE切RF并同步目标小区
  → 在目标小区发PRACH
  → RAR/Msg3/Msg4或非竞争RACH完成
  → UE发送RRC Reconfiguration Complete
  → 目标小区PDCCH/PDSCH/PUSCH恢复
```

因此切换不是单一“HO模块”，而是测量、PUSCH上报、PDSCH命令、目标小区搜索、随机接入和RRC重配六段流程的组合。定位时从左到右找第一个缺失节点：

- 无异频结果：测量对象、gap、RF切频、PSS/SSS。
- 有结果无报告：事件门限、hysteresis、TTT、L3滤波。
- 有报告无命令：仪表脚本/网络决策。
- 有命令无目标同步：目标频点、PCI、RF和搜索。
- 有同步无RACH成功：PRACH、RAR、TA、UL功率。
- RACH成功无Complete：重配处理或目标小区上行。
- Complete后无业务：目标承载、PDCCH调度或CA恢复。

### 22.8 性能与峰值是前面所有模块的最终结果

```text
UE能力/仪表许可
  × 已激活CC数量
  × 每CC可用RB和调度子帧
  × 调制阶数Qm
  × MIMO层数
  × 有效码率
  × (1 - BLER与重传损失)
  = PDSCH/PUSCH实际PHY吞吐
```

峰值不达标时按这个顺序回溯：

1. SCell 是否完成添加、激活并真正承载。
2. PDCCH 是否持续给足 RB，是否存在 DCI 漏检。
3. PDSCH/PUSCH 的 MCS、Qm、层数和 TBS 是否达到目标。
4. CSI/RI/PMI 与 SRS 是否支持目标调度，而不是固定在低阶配置。
5. `TbCrcHw`、UL_HARQ、PHICH 和 `Rv_Num` 是否显示高 BLER/重传。
6. SNR、RSRP、PUSCH/PUCCH功率、`Pcmax/MPR/AMPR` 是否受限。
7. PHY吞吐正常但应用吞吐低时，再转查RLC/PDCP/TCP和测试电脑。

### 22.9 自动化挂测时的“一条线”分析法

每次只填下面这条链，通常就能快速缩小范围：

```text
用例最后正常状态：____________________
仪表最后正确发送/接收：________________
UE最后正确处理：______________________
第一个缺失或错误节点：________________

PDCCH/DCI：有 / 无 / 未确认
PDSCH CRC：ACK / NACK / 无结果
PUCCH反馈：已生成 / 已发射 / 仪表ACK-NACK-DTX
PUSCH发射：已授权 / 已配置 / 已发射 / 仪表ACK-NACK-DTX
PHICH：ACK / NACK / 与仪表不一致
CSI/SRS/测量：有效 / 未配置 / 无效哨兵 / 未上报
SCell：未配置 / 已配置 / 已激活 / 已承载
重配或切换：停在______________________
吞吐损失来自：资源 / MCS层数 / BLER重传 / RF / 上层
```

最后把结论写成一句因果链，而不是堆日志：

```text
在子帧X仪表已发送DCI0，UE也正确解析grant并配置PUSCH；
UE LTX显示已发射，但MT8000A UL_HARQ连续DTX，同时TA timer失效，
所以第一个错误节点位于“UE上行发射 → 仪表检出”之间，优先验证TA、功率和RB/DMRS，而不是继续检查PDCCH。
```

## 23. 协议值与厂商字段值字典

本章专门回答“这个值在协议上代表什么、换成另一个值又代表什么”。先判断字段属于哪一类：3GPP空口字段可以按规范解释；`TbCrcHw/DciValid/eTXPucchType/BchPhich` 等是芯片内部编码，只能按当前版本实证解释。

### 23.1 帧号、带宽、PCI与小区身份

| 字段/值 | 协议含义 | LOG判读 |
|---|---|---|
| 1 radio frame | 10 ms，包含10个1 ms subframe | 行头最后一段通常是子帧0–9，但本平台还有外层周期 |
| SFN | 0–1023，10.24 s回绕 | 跨回绕时用外层周期或正文`AbsSF`排序 |
| 6/15/25/50/75/100 RB | 1.4/3/5/10/15/20 MHz | `RBNum`和仪表带宽必须一致 |
| PCI | 0–503 | `CellID`通常就是PCI，不是ECGI中的E-UTRAN Cell ID |
| `N_ID_2` | `PCI mod 3`，取0/1/2 | 对应PSS序列 `Id0/Id1/Id2` |
| `N_ID_1` | `floor(PCI/3)`，取0–167 | 与SSS共同恢复完整PCI |

PBCH/MIB提供下行带宽、PHICH配置和SFN高位等驻留基础信息。日志中的 `dwAntNum` 是当前实现根据PBCH/CRS得到的天线端口信息；不要把它直接等同于后续PDSCH实际层数。

### 23.2 RNTI与公共调度

| 值 | 标准用途 | 常见日志 |
|---|---|---|
| `0xFFFF` | SI-RNTI，调度系统消息 | `CommDecCfg ... RNTI=0xFFFF`, `SIBDecode` |
| `0xFFFE` | P-RNTI，调度Paging | Paging相关DCI/PDSCH |
| RA-RNTI | 标识PRACH时频机会，值由机会位置计算 | `Enable RA-RNTI`, RAR检测窗口 |
| Temporary C-RNTI | RAR中分配，随机接入期间使用 | RAR/Msg3/竞争解决 |
| C-RNTI | 连接态UE专用调度身份 | 专用PDCCH/PDSCH/PUSCH |

不能仅从一个十六进制RNTI猜类型；必须结合当时所处流程和仪表Trace。

### 23.3 DCI format与本平台 `DciValid`

| DCI格式 | 主要标准用途 |
|---|---|
| 0 | PUSCH上行授权 |
| 1 / 1A / 1C | 单码字或紧凑下行分配，1A/1C也用于公共消息场景 |
| 1B / 1D | 带预编码/功率偏置等扩展的下行调度，依能力与TM使用 |
| 2 / 2A / 2B / 2C / 2D | 空间复用、双码字及后续Release多天线增强 |
| 3 / 3A | 一组UE的TPC命令 |
| 4 | 支持UL MIMO时的上行授权 |

当前固件 `DciValid` 是**内部检测类型位图**，不是把标准DCI format数字直接放进位图：bit0=DCI0、bit3=SIB、bit5=Paging、bit7=RA、bit9=3/3A、bit13=其它专用DCI、bit15=DCI4。换芯片/版本必须重新标定。

### 23.4 TB、MCS、Qm、层数与码率

| 字段/值 | 含义 | 不同值如何看 |
|---|---|---|
| `TBS1/TBS2` | 两个transport block/码字的比特数 | `TBS2=0`通常为单TB；非0才分析TB2 |
| `MCS` | 调制与TBS表索引 | 同一索引的含义会随DL/UL及MCS table变化，启用256QAM时必须选对应表 |
| `Qm=2` | QPSK，每调制符号2 bit | 鲁棒、峰值低 |
| `Qm=4` | 16QAM | 中等SNR/吞吐 |
| `Qm=6` | 64QAM | 需要更高SNR |
| `Qm=8` | 256QAM | 需UE能力、相应MCS表和更高信号质量 |
| layer=1/2/3/4 | 该码字承载的MIMO层数 | 不能用天线数替代层数；两个TB的层数要分别看 |
| `REs` | 当前实现统计的可用于传输的resource elements | 用于估算码率，必须和同一TB配对 |

粗估每TB有效码率：`TBS/(REs × Qm × TB层数)`。MT8000A的 `A/UINT` 更接近真实编码输入/可用编码位；经验阈值0.93只用于当前测试配置，不是3GPP统一“合法/非法”边界。

### 23.5 HARQ、NDI、RV、CRC与反馈

| 字段/值 | 标准/实证含义 | 注意 |
|---|---|---|
| `HarqId` | HARQ进程编号 | 用 `CC+方向+HarqId+TB` 跟踪，不可只按时间相邻配对 |
| NDI翻转 | 通常表示新数据 | NDI保持且同一进程再次调度，通常是重传 |
| RV | 冗余版本 | 常见新传/重传顺序0→2→3→1，但调度器可少发或改变顺序 |
| ACK | 接收端CRC正确 | DL由UE经PUCCH/PUSCH回；UL由eNB经PHICH回 |
| NACK | 检到了传输但CRC错误 | 应进入HARQ重传（若用例允许） |
| DTX | 预期位置未检出有效传输/反馈 | 方向通常是时序、资源、功率或根本未发，不应等同NACK |

当前日志 `TbCrcHw` 的实证编码：有效TB的 `2=CRC成功`、`1=CRC失败`、`0=无有效结果/尚未完成`。这不是3GPP线上的三值字段。当前日志PHICH `value=1` 后 `DO not Trans`，实证为ACK；`value=0` 后 `Non-AdaptReTrans`，实证为NACK，同样属于本版本内部编码。

### 23.6 PHICH配置值

MIB中的标准PHICH配置包含：

- `phich-Duration`：`normal` 或 `extended`。
- `phich-Resource`：`oneSixth`、`half`、`one`、`two`，即 `N_g=1/6、1/2、1、2`，影响PHICH group数量。

日志 `BchPhich=0x...` 是打包后的私有值。没有厂商位定义时，不要直接把某个十六进制bit说成duration或Ng；应同时看MIB解码、仪表配置以及后续PHICH资源计算。

### 23.7 PUCCH格式、UCI与SR

| 传统LTE PUCCH格式 | 主要承载内容 |
|---|---|
| Format 1 | SR，不携带HARQ信息比特的调度请求波形 |
| Format 1a | 1 bit HARQ-ACK |
| Format 1b | 2 bit HARQ-ACK |
| Format 2 | CQI/PMI/RI类CSI |
| Format 2a / 2b | CSI加1/2 bit HARQ-ACK（具体适用受CP/配置约束） |
| Format 3 | CA等场景的多bit HARQ-ACK |

后续Release还有扩展格式。当前日志的 `eTXPucchType/ePucchFormat/PucchType` 是内部枚举，不能直接套上表数字；先用 `pbHasHarqAck、AckLen、CqiLen、RiLen` 判断实际UCI内容，再用已知用例标定枚举。

#### SR次数超限怎么判断

`SR_COUNTER` 和 `dsr-TransMax` 属于MAC过程，不由PHY维护。`dsr-TransMax` 的标准配置选项为 `n4/n8/n16/n32/n64`。完整链路是：

```text
MAC有待发送数据且没有UL grant
 → 到SR机会，命令PHY在PUCCH发SR
 → SR_COUNTER增加
 → 一直没有UL grant
 → 达到dsr-TransMax
 → MAC释放相关PUCCH/SRS资源并发起Random Access
```

PHY最多证明“在若干SR机会确实配置/发出了PUCCH”，并可观察之后是否出现DCI0/PUSCH；确认“已超限”必须搜MAC/PS的 `SR_COUNTER/dsr-TransMax/SR FAIL/Trigger RACH`。`wHNcqiNharqNsr`是PUCCH功率公式中的信息bit项，不是SR次数；`bPucchPowReachMax`是功率相关私有状态，也不是SR_COUNTER。

### 23.8 PUSCH功控中的TPC值

DCI中的2 bit TPC需要先判断当前采用累积还是绝对方式。常见PUSCH映射：

| TPC命令 | 累积方式 `δPUSCH` | 绝对方式 `δPUSCH` |
|---:|---:|---:|
| 0 | -1 dB | -4 dB |
| 1 | 0 dB | -1 dB |
| 2 | +1 dB | +1 dB |
| 3 | +3 dB | +4 dB |

PUCCH常见2 bit TPC对应 `-1/0/+1/+3 dB`。日志中的 `TPC` 若已经被固件二次换算，可能打印的是dB值而不是原始2 bit；必须与前后 `Fi/closed-loop/PuschPow` 变化验证。

`Pcmax` 是UE在当前载波/配置下可用的最大发射功率约束；`MPR/AMPR` 是调制、资源占用、频段或附加要求造成的功率回退。功率到顶时，继续提高标称功率也可能没有实际增益。

### 23.9 CQI、PMI、RI

CQI是UE推荐的下行传输质量索引，不是直接的dB SNR。常用CQI table 1如下；启用256QAM或其它Release扩展时必须换相应表：

| CQI | 调制 | 码率×1024 | 频谱效率(bit/RE) |
|---:|---|---:|---:|
| 0 | out of range | 0 | 0 |
| 1 | QPSK | 78 | 0.1523 |
| 2 | QPSK | 120 | 0.2344 |
| 3 | QPSK | 193 | 0.3770 |
| 4 | QPSK | 308 | 0.6016 |
| 5 | QPSK | 449 | 0.8770 |
| 6 | QPSK | 602 | 1.1758 |
| 7 | 16QAM | 378 | 1.4766 |
| 8 | 16QAM | 490 | 1.9141 |
| 9 | 16QAM | 616 | 2.4063 |
| 10 | 64QAM | 466 | 2.7305 |
| 11 | 64QAM | 567 | 3.3223 |
| 12 | 64QAM | 666 | 3.9023 |
| 13 | 64QAM | 772 | 4.5234 |
| 14 | 64QAM | 873 | 5.1152 |
| 15 | 64QAM | 948 | 5.5547 |

- RI表示推荐rank/空间层数，但日志 `riValue` 可能是0-based内部编码；必须用已知Rank-1/Rank-2场景标定。
- PMI是码本预编码矩阵索引，值域取决于天线端口数、rank、codebook subset和report mode，没有跨配置通用的“PMI=某值就是某波束”。
- `CqiLen/RiLen=0`只说明本次UCI没携带对应字段，不能推出“计算结果为0”。
- `0xffffffff`、`524287` 等在本日志中常表现为内部无效哨兵；不能解释为极高CQI/SNR。

### 23.10 SRS字段

- 周期SRS由RRC `srs-ConfigIndex`等参数共同映射到周期与offset；FDD/TDD表不同，不能把原始index直接当周期毫秒数。
- 非周期SRS由特定DCI中的SRS request触发，是否支持取决于Release/能力/配置。
- `SrsSubframeConfig` 是小区级可用于SRS的子帧集合；dedicated参数再决定某UE何时/在哪发。
- `cyclicShift/comb/frequencyDomainPosition/bandwidth` 共同决定序列与频域资源；任一与仪表不一致都可能导致仪表DTX。
- “last symbol reserved”只证明资源预留；真正发射还需UE-specific周期或trigger、LTX状态和仪表接收证据。

### 23.11 RSRP/RSRQ报告值

RRC测量上报中的常用整数编码不是直接dBm/dB：

| 编码 | RSRP范围 |
|---:|---|
| 0 | `< -140 dBm` |
| 1–96 | `[value-141, value-140) dBm` |
| 97 | `≥ -44 dBm` |

例如 `Rsrp=69` 表示约 `[-72,-71) dBm`，排障时常简写为 `69-141=-72 dBm`。

| 编码 | RSRQ范围 |
|---:|---|
| 0 | `< -19.5 dB` |
| 1–33 | `[value/2-20, value/2-19.5) dB` |
| 34 | `≥ -3 dB` |

只有确认字段是RRC report range编码时才能套公式；`CSRM/RXP/DFE`内部定点值、线性功率或0.1 dB量化值不能套用。

### 23.12 A1/A2/A3/A4/A5与切换

| 事件 | 进入条件的直观含义 | 常见用途 |
|---|---|---|
| A1 | serving变好并高于门限 | 停止某些测量 |
| A2 | serving变差并低于门限 | 启动异频/异系统测量 |
| A3 | neighbor相对serving好到一定offset | 同频/异频切换常用 |
| A4 | neighbor高于绝对门限 | 目标覆盖判断 |
| A5 | serving低于门限1且neighbor高于门限2 | 带双门限的切换触发 |

真正判断必须加入测量对象offset、cell individual offset、hysteresis、L3 filter和timeToTrigger。一次瞬时越线不等于协议栈应立即发Measurement Report。

### 23.13 SCell MAC CE位图

Activation/Deactivation MAC CE中的 `C_i` 对应已配置的 `SCellIndex=i`：`1=激活`，`0=去激活`；未配置index的bit忽略。当前项目示例 `ActDeactSCellInfo=6 = 0b00000110` 表示index 1和2置1。仍需注意：

- 这是MAC层激活，不是RRC添加；RRC `sCellToAddModList` 先建立配置。
- 去激活不会自动删除配置；删除由 `sCellToReleaseList` 完成。
- 私有 `CcActBitMap` 的bit序必须用当前版本已知用例标定，再与MAC CE比对。

### 23.14 当前日志常见私有状态值

| 私有字段 | 本文件可确认的值 | 不能过度解释的部分 |
|---|---|---|
| `TbCrcHw` | 有效TB：2成功、1失败、0无结果 | 多TB十六进制打包位序需结合TBS和已知用例 |
| `DciValid` | 已标定的内部检测类型位图 | 不是标准DCI format原值 |
| PHICH `value` | 1=ACK、0=NACK，由后续停止/重传验证 | 换版本必须重验 |
| `TransType` | 当前资料：1新传、2自适应重传、3非自适应重传 | 以本版本函数上下文为准 |
| `Cfg/Int` | 配置次数/完成中断次数 | 最新一笔在途时允许暂差1；需看差值是否持续扩大 |
| `RegCfgErr/LtxStatus` | 硬件配置/发射状态入口 | 具体非零bit需源码或内部字典 |
| `BchPhich/Tpmi/eTXPucchType` | 厂商打包字段 | 不可把十六进制位序当3GPP公共定义 |

### 23.15 频点、带宽与时间单位（本文件已验算）

| 量 | 协议/平台定义 | 换算与验证 |
|---|---|---|
| EARFCN → 频率 | `F_DL = F_DL_low + 0.1×(N_DL − N_Offs-DL)`（TS 36.101） | 1650@band3 = 1850.0 MHz；1275@band3 = 1812.5 MHz；3201@band7 = 2665.1 MHz；6225@band20 = 798.5 MHz——全部与日志频点吻合 |
| 频点单位（搜索/测量侧） | `CSRC/CSRS/MC/CSRM` 的 `Freq`、`FreqPoint` | **100 kHz**，`18500` = 1850.0 MHz |
| 频点单位（射频侧） | `RFC/DFE` 的 `Freq`、`Freq100Hz` | **100 Hz**，`18500000` = 1850.0 MHz |
| 带宽表示（射频侧） | `RFC` 的 `Bw` | **100 kHz**，`14` = 1.4 MHz（同步用的中心 6 RB）、`150` = 15 MHz |
| 带宽表示（基带侧） | `DFE` 的 `BandWidth`、MIB 的 `Band` | **RB 数**，6/15/25/50/75/100 对应 1.4/3/5/10/15/20 MHz |
| 基本时间单位 | `T_s = 1/30.72 MHz ≈ 32.552 ns`（TS 36.211） | 1 时隙 = 15360 T_s，1 子帧 = 30720 T_s，1 帧 = 307200 T_s |
| 时间三元组 | `帧.时隙(0–19).时隙内Ts` | `FB=0.2.6272` → `2×15360+6272 = 36992`，与同行 `CellFBTsBefAdj=36992` 一致 |
| 紧凑时间编码 | `CurTime / SchedPos / SibPos / RcvSubFrm` | `SFN×10 + 子帧`，`2445` = SFN 244 子帧 5 |
| SIB1 位置 | TS 36.331：`SFN mod 2 = 0` 的子帧 #5，每 20 ms 重复 | 日志窗口起点 2445/2525/2545 全部满足；`CurSiPeriod=2` 的单位是无线帧 |

详细推导与出处见 §26.2。

### 23.16 平台定点数与内部对数域

| 表示 | 换算 | 出现位置 |
|---|---|---|
| 内部对数域 | **约 84 个单位 = 1 dB**（log2 定点） | `swRsrpLog`、`swAveRsrp`、`swAveRsrq`、`swRssiLog1`、`swValLog2` |
| Q8 定点 | `值/256` | `0x5C16 AfcReq` 的 `OffsetHz/TmpPpm/CordPpm/CordHz`（每项都成对打印 `[定点值, 取整值]`） |
| ppm×1024 的 Q8 | `值/262144` | `0x5C16 CalcPpm`：`−156764` = `−0.598 ppm`，与同行 `TotalOffsetHz=−1106`（@1850 MHz）自洽 |
| 线性数字增益 | `128 = 1×` | `RxDagcLin`、`CsrmDagcGainLin` |
| 直接可读的物理量 | 无需换算 | `AgcGain`(dB)、`TxPwr`(dBm)、`0x5881` 的 CFO(Hz)、`0x866E` 的 SNR(dB)、`0x68C6` 的 `AveLogRsrp`(dBm) |

**无效哨兵**（当成真实值会得出荒唐结论）：对数域 `−32768`、dBm 域 `−386`、未配置 `−1`、`CurSNR=100000`、累加器 `0`。

### 23.17 寻呼帧与寻呼时机（PF/PO）

TS 36.304 §7.1：

- `T` = UE 的 DRX 周期（无线帧），取 `defaultPagingCycle`（rf32/64/128/256）与 UE 专用值的较小者。
- `nB ∈ {4T, 2T, T, T/2, T/4, T/8, T/16, T/32}`；`N = min(T, nB)`；`Ns = max(1, nB/T)`。
- **PF**：`SFN mod T = (T div N) × (UE_ID mod N)`，`UE_ID = IMSI mod 1024`。
- **PO**：`i_s = floor(UE_ID/N) mod Ns`，再按 FDD/TDD 查表得子帧号（FDD 且 `Ns=1` 时为子帧 9）。

本文件 `0x960A`：`T=128, nB=128, N=128, Ns=1, UeID=277` → `277 mod 128 = 21`，日志 `PO=917:9` 且 `917 mod 128 = 21`、子帧 9，**与公式完全一致**。“寻呼收不到”类问题可用这一条直接判断 UE 侧算得对不对。

### 23.18 小区选择/重选判据（S 准则与 R 值）

TS 36.304：

- `Srxlev = Q_rxlevmeas − (Q_rxlevmin + Q_rxlevminoffset) − P_compensation − Qoffset_temp`
- `Squal = Q_qualmeas − (Q_qualmin + Q_qualminoffset) − Qoffset_temp`
- 两者均 `> 0` 才满足 S 准则（可驻留）；重选按 R 值排序：`R_s = Q_meas,s + Q_hyst − Qoffset_temp`，`R_n = Q_meas,n − Qoffset − Qoffset_temp`。

日志对应：`0x9E88 IdleServCellInfo` 直接打印 `ScellSrxlev / ScellSqual / scellRvalue`，`0x9E99` 给出频点重选优先级。**当 UE “搜到小区却不驻留”时，先看这两个值是否 ≤ 0，再去查 SIB1/SIB3 的门限配置。**

### 23.19 脱网原因码（本固件自带枚举）

`USER_DEF_VD_12` 的 `0x9EAB OOSCause` 在行内给出完整枚举，是判定 out of service 的直接证据：

| 值 | 含义 |
|---:|---|
| 0 | 10 s 超时 |
| 1 | out-of-sync 门限 |
| 2 | island 门限 |
| 3 | PHY 门限 |
| 4 | 小区被 barred |
| 5 | 无测量结果 |
| 6 | 10 s 超时或小区被 barred |
| 7 | 10 s 超时或无测量结果 |
| 8 | 10 s 超时 / 无测量结果 / 锁定小区被 barred |

本文件在序号 3608641（超帧 1106.406.0，约 00:07:39）出现 `OOSCause:6`，此后进入长时间反复搜网，见 §26.13 的完整时间线。

## 24. 下行模块逐项字典（PBCH / DLA / DLS / CSI / RXP，56 个消息 ID）

本章覆盖下行接收链路的 5 个模块共 **56 个 message_id**：PBCH 的 MIB 译码、DLA 的 PCFICH/PDCCH/DCI、DLS 的 PDSCH 与 TB/CB CRC、CSI 的反馈计算与 RLM 输入、RXP 的 CRS 功率/SNR/CFO/CIR。

> 数据源：`loglte_phich.txt`；索引：`generated_lte_log_index/message_id_index.tsv` 与 `normalized_template_index.tsv`。关键字段均从原始日志抽样复核；没有厂商源码或寄存器手册支撑的位域，统一标为“厂商私有/不可直接解码”。

### 24.1 先建立四种“值”的边界

阅读这类 PHY LOG 时，最危险的错误是把所有数字都当成 3GPP 协议值。本文统一使用下面四类置信标签：

| 标签 | 含义 | 例子 |
|---|---|---|
| 协议值 | 3GPP 空口中有明确含义和取值域 | `RV=0/1/2/3`、`SI-RNTI=0xFFFF`、Qm=2/4/6/8 |
| 日志明示枚举 | 厂商日志自己打印了映射，可按本固件理解 | `State=0[0:Idle,1:WaitLlr,...]`、`DciValid` 后的 bit 注释 |
| 相关性推断 | 从相邻消息和变化规律可以判断用途，但不能保证数值编码 | `wCsiEn_aop=0` 很可能表示当前未使能相应路径 |
| 私有打包值 | 寄存器、地址、内部定点数或位域，无资料不能拆位 | `CcTbRlt`、`HarqCtrlInfo`、`dwRLMValue`、`PeakValue` |

必须遵守：

1. `0` 不总是“关闭”，也可能是合法编号、空结果、默认值或复位值。
2. 十六进制非零不自动等于“成功”；例如 `ErrIntRpt`、`DciValid`、地址字段含义完全不同。
3. 统计计数器要看差分，不能只看绝对值；计数器可能在重配、掉网或模块复位时清零。
4. 同一名字在不同模块可能不是同一单位；例如 RXP 的 `RSRPxx` 是内部量，不是协议上报的 dBm。
5. 对打包寄存器只使用日志显式拆出的字段；不要凭数值形状猜测 bit 位。

### 24.2 下行接收总链路

```text
PSS/SSS 搜索、定时与 PCI
    ↓（本章从 PBCH 开始，PSS/SSS 见第 26 章）
PBCH 解 MIB
    ├─ 下行带宽 N_RB_DL
    ├─ PHICH duration/resource
    ├─ SFN 高位
    └─ PBCH CRC 掩码推断 CRS 天线端口数
    ↓
建立 PCell 公共配置与 PDCCH 搜索空间
    ↓
DLA：信道估计配置 → PCFICH/CFI → PDCCH 盲检 → DCI 有效位图
    ↓
DLS：按 DCI 配置 PDSCH → 解调/解码 → TB/CB CRC → MAC PDU
    ↓                                      ↓
生成下行 HARQ ACK/NACK                  更新 CQI/PMI/RI、RLM、SNR
    ↓                                      ↓
PUCCH/PUSCH 上报                         调度、MCS、层数与失步判断

另一条下行控制支路：
PUSCH 已发送 → 按资源定位 PHICH → 解 HI ACK/NACK → 上行 HARQ 重传或停止
```

定位自动化挂测时，不要从某一条“红字”直接下结论，应逐门检查：

```text
是否搜到正确 PCI/频点
→ PBCH/MIB 是否成功
→ PDCCH 是否在预期子帧检出正确 DCI
→ PDSCH 配置是否与仪表一致
→ TB CRC 是否成功
→ 下行 ACK/NACK 是否真正发出
→ 若是上行问题，PHICH 是否按同一 HARQ ID 返回
→ SNR/CFO/CIR/RLM 是否支持上述结论
```

### 24.3 PBCH：从小区搜索结果进入可调度状态

#### 24.3.1 本日志中的完整成功片段

原始日志序号 546～560 展示了一个可以作为模板的成功过程：

```text
PBCH RESULT ... CrcOk=0, dwBranch=0, wPbchIntCnt=1
ConfigPbchReg ... Branch=1
PBCH RESULT ... CrcOk=0, dwBranch=1, wPbchIntCnt=2
ConfigPbchReg ... Branch=2
PBCH RESULT ... MIB_CRC=0xa8f00001, CrcOk=1,
                BchMib=0xa8f000, dwBranch=2, SFN=242, wPbchIntCnt=3
MIB OK!!
PBCH CRC OK: Earcfn=1650, CellID=503, RBNum=100,
             dwAntNum=4, BchPhich=0x2
UpRxState > RxCurrState=3, RxPreState=2 [2:Pbch,3:Nomal]
```

正确理解：前两次 `CrcOk=0` 并不等于最终搜网失败。算法在更换 `Branch`/天线端口假设或积累分支后第三次成功，随后才由汇总打印给出 `RBNum=100` 与 `dwAntNum=4`。

#### 24.3.2 MIB 中真正属于协议的字段

| 协议信息 | 协议含义 | 在本日志中对应或近似对应 |
|---|---|---|
| `dl-Bandwidth` | 取值 n6/n15/n25/n50/n75/n100，分别对应 1.4/3/5/10/15/20 MHz | `RBNum=6/15/25/50/75/100`；`MibInfoCheck` 从 MIB 位域查表得到 RB 数 |
| `phich-Duration` | `normal` 或 `extended` | DLA 的 `PhichDur=0[0:Nor,1:Ext]` 是日志明示枚举；PBCH 的 `BchPhich` 是打包值，不能只凭 `0x2` 单独拆解 |
| `phich-Resource` | `oneSixth/half/one/two`，决定 PHICH group 资源 | DLA `PhichNg` 很可能是内部枚举；只有结合 MIB 解码与固件映射后才能把 0/1/2/3 映射成上述协议值 |
| `systemFrameNumber` | MIB 携带 SFN 的高 8 bit，低位由 PBCH 接收时刻恢复 | `PBCH RESULT ... SFN=242` 是已恢复后的输出，不应当作原始 8 bit 字段 |
| `spare` | 保留位 | 日志没有单独打印 |
| PBCH CRC 掩码 | CRC 掩码区分 1/2/4 个天线端口假设 | `dwAntNum=4` 为解码结论；`PBCH RESULT` 中 `dwAntNum=3` 很可能是内部假设编号，不是“3 根天线” |

补充：CP 类型、PCI、EARFCN 并不是 LTE MIB 的字段。它们由同步/搜索流程得到，再作为 PBCH 解码输入。因此 `MibReqMonitor` 中出现 `CpMode`、`CellId`、`Earfcn`，不能说这些值是“从 MIB 解出来的”。

#### 24.3.3 逐字段阅读

`MibReqMonitor`：

- `Earfcn`：目标 LTE 下行频点号。
- `CellId`：物理小区 ID（PCI，0～503）。
- `Boundry`：芯片内部采样/TPU 时间边界，不是 SFN。
- `CpMode`：本日志和 DLA 一致时 `0=NCP、1=ECP`；映射由 DLA 日志明示。
- `Ant=-1`、`Bw=-1`：请求时尚未知、待 PBCH 确认；不要解释成负天线或负带宽。
- `FrmTyp[-1,-1]`：帧型未知的内部占位；FDD 初搜阶段出现是合理的。

`PBCH RESULT`：

- `MIB_CRC`：厂商将 MIB 结果和 CRC 状态打包后的字；样本成功值最低位为 1，但只应以相邻 `CrcOk` 为准。
- `CrcOk=1/0`：本固件明示的单次 PBCH 解码成败。
- `BchMib`：MIB 原始/整理后的打包 payload，不能直接按十六进制肉眼读带宽。
- `dwBranch`：内部尝试分支。样本按 0、1、2 变化；没有资料时不能映射为特定天线端口。
- `wPbchIntCnt`：本次 PBCH 尝试/中断累计值，主要用于观察是否持续尝试。
- `SFN`：模块恢复后的系统帧号。

`ConfigPbchReg`：

- `HwIdx`、`Branch`、`AntNum`、`CellId` 是硬件配置上下文。
- `MaxIts` 很可能为译码最大迭代/尝试参数，但属于实现值。
- `NewDataInt` 为新数据中断相关标志；不要等同 NDI。
- `PbchEn` 是硬件使能位图。
- `State=0/1/2/3` 的含义由日志明确给出：Idle、WaitLlr、WaitViterbi、ViterbiWork。

`RxRsrpMoniter`：

- `RSRP00[I,Q]` 等是 PBCH 流程内部 I/Q 或相关量，不是 dBm。
- `RSSI`、`RSPRx` 同样是内部线性/累积值。可用于同一次扫描中比较分支强弱，不可直接与仪表的 -95 dBm 比较。

#### 24.3.4 PBCH 异常判据

| 现象 | 初步判断 | 下一步 |
|---|---|---|
| 有 `MibReqMonitor`，持续 `PBCH RESULT CrcOk=0`，最后 `Extra Max Branch,Report Fail` | PBCH 所有内部假设耗尽 | 回看 PSS/SSS、频偏、PCI、接收功率、仪表端口数与 MIB 配置 |
| 直接没有 `PBCH RESULT` | PBCH 硬件未启动、RF 未开、事件丢失或状态机未进入 | 检查 `ConfigPbchReg`、RF Open、`UpRxState`、错误日志 |
| `MIB OK` 有，但没有 `PBCH CRC OK`/上层确认 | 成功后的结果上报或状态机异常 | 看 `ErrId`、PHY→MC/PS 消息、是否立即 reset/release |
| `RBNum` 与仪表不一致 | MIB 配置、搜错小区或 payload 解释异常 | 对齐 EARFCN/PCI，查看 `BchMib` 与仪表 MIB |
| `dwAntNum` 与仪表天线端口数不一致 | PBCH CRC 掩码假设错误或仪表配置错 | 检查 CRS 端口配置、功率和 MIB/PBCH 波形 |
| 频繁 `RxCurrState 2↔3` 或反复 reset | 同步不稳或上层重复建链 | 联查 RXP CFO/CIR/SNR 与 reset 原因 |

### 24.4 DLA：PCFICH、PDCCH、DCI 与控制信道

#### 24.4.1 配置到中断结果

本日志的常用顺序是：

```text
ProDbgMsgRecvCommMsg / ProDbgStateSwitchPrint
    ↓  小区、带宽、天线、TM、PHICH 配置到位
Che Reg
    ↓  为本子帧配置信道估计/硬件寄存器
CInt >>> Cdtr Rpt
    ↓  输出 CFI、DciValid、RNTI 搜索使能、候选信息
CDTR_Reg DciRpt
    ↓  原始 PDCCH/HI 硬件报告（仅供芯片级追查）
ProDlCtrlChStatInfoMonitor
       周期统计 CFI、DCI format、HI ACK/NACK
```

连接态样本（原始日志序号 1585867～1585869）：

```text
DLS|DCIInfo[0] ... TbCrcHw=0x00020000 ...
DLA|... Cfi=3, DciValid=0x2000, ... DlWorkIndBmp=48, Candid=[2,0]
DLA|CDTR_Reg DciRpt ...
```

`DciValid=0x2000` 的 bit13 置位，按同一行日志自带注释属于 `Other` DCI。它与前后 `DLS|DCIInfo[0]` 的用户数据调度一致。这里的 bit 位是厂商汇总位图，不是 3GPP 在空口上传输的 DCI bit 布局。

#### 24.4.2 `DciValid` 位图：只按日志自带映射解释

| 位 | 掩码 | 本日志标签 | 用法 |
|---:|---:|---|---|
| 0 | `0x0001` | `0` | DCI format 0，上行授权 |
| 3 | `0x0008` | `1a1cSib` | SIB/SI 调度相关的 DCI 1A/1C |
| 5 | `0x0020` | `1a1cPch` | Paging 调度相关的 DCI 1A/1C |
| 7 | `0x0080` | `1a1cRa` | 随机接入相关的 DCI 1A/1C |
| 9 | `0x0200` | `33a` | DCI 3/3A，TPC 命令 |
| 13 | `0x2000` | `Other` | 其他 DCI，通常覆盖 UE 专用下行调度等 |
| 15 | `0x8000` | `4` | DCI format 4 |

快速例子：

```text
0x0008  → bit3，SIB/SI 相关
0x0080  → bit7，RAR/随机接入相关
0x2000  → bit13，其他 DCI；连接态 PDSCH 常见
0x0000  → 本次没有被该汇总器标成有效的 DCI
```

`DciValid=0` 不能脱离调度预期判失败：空子帧、搜索阶段或仅测量的子帧本来就可能没有 DCI。只有仪表明确在该帧发了 DCI，而 UE 对应载波/子帧连续为 0，才构成漏检证据。

#### 24.4.3 `RntiEnInd` 是搜索使能，不是解码成功

日志自带映射：

| 位 | 标签 | 常见协议身份 |
|---:|---|---|
| 0 | C | C-RNTI |
| 1 | Tc | Temporary C-RNTI |
| 2 | Sps | SPS C-RNTI |
| 3 | Si | SI-RNTI |
| 4 | P | P-RNTI |
| 5 | Ra | RA-RNTI |
| 6 | Cch | 厂商内部公共信道类标签，不能直接扩写 |
| 7 | Sch | 厂商内部调度类标签，不能直接扩写 |
| 8 | M | 厂商内部标签，不能直接扩写 |

`RntiEnInd=0xffffff` 表示大量搜索类型被使能，不表示 24 个 RNTI 都解码成功。真正检出要看 `DciValid`、DCIInfo 和相应 RNTI。

#### 24.4.4 CFI 与控制区

- 协议上的 CFI 取 1、2、3，指示 PDCCH 控制区长度；小带宽下实际控制符号数还有带宽相关规则。
- 本日志连接态常见 `Cfi=3`。
- 搜索/空结果期间可见 `Cfi=0`；0 不是正常空口 CFI，应按“无有效结果/内部默认值”处理。
- 周期统计 `CfiNum(0/1/2/Total)` 的标签不够清晰，样本 `(0,0,1322,1540)` 与逐子帧 `Cfi=3` 并不适合直接逐格映射。没有源码前只做差分趋势，不给数组索引强行命名。

#### 24.4.5 DCI format 的协议用途

| DCI format | 典型用途 | 本日志位置 |
|---|---|---|
| 0 | 单天线端口 PUSCH UL grant | `DciValid bit0`、`DecStat DCI[F0]` |
| 1/1A/1C | 下行分配；1A/1C 也用于 SI、Paging、RAR 等公共调度 | `DciValid bit3/5/7/13`、`DecStat F1/F1A` |
| 2/2A/2B/2C | 多天线/空间复用下行调度，不同版本和 TM 使用不同 format | `ProDlCtrl... DciNum`、`DecStat` 部分 format |
| 3/3A | PUCCH/PUSCH 功控命令 | `DciValid bit9` |
| 4 | 支持多天线上行传输时的 UL grant | `DciValid bit15`、统计 `DciNum(...4=)` |

`DciNum=(01=...)` 中的 `01` 是厂商打印标签，不能擅自解释成“DCI0+DCI1 的和”；应以原字段名保留。

#### 24.4.6 PDCCH 漏检分析

1. 先用 MT8000A Trace 确定具体 `SFN.subframe`、载波、RNTI、aggregation level、DCI format。
2. UE 侧看同一子帧 `Cfi` 是否有效、`DciValid` 对应 bit 是否置位。
3. 若 `DciValid` 有效，再找同子帧的 `DLS|DCIInfo[n]`，核对 RB、MCS、HARQ ID、NDI、RV。
4. 若仪表发了而 `DciValid=0`，同时看 RXP SNR、CFO、CIR、`LowSINRInd`；控制信道可能先于 PDSCH 恶化。
5. 若 `DciValid` 有但没有 DCIInfo/译码配置，问题位于 DLA→DLS 交接或调度过滤。
6. 若 DCIInfo 存在且 TB CRC 失败，不能再称为“DCI 漏检”，应转入 PDSCH 解码分析。

### 24.5 DLS：PDSCH、TB/CB CRC 与 HARQ

#### 24.5.1 最有用的跨模块主日志：`DLS|DCIInfo[n]`

该日志在本文件索引中归入 `USER_DEF_VD_2`（ID `0x628E`），不是 DLS 模块的 14 个 ID 之一，但它是分析 PDSCH 的首选入口：

```text
DLS|DCIInfo[0];TM=2,RbNum=2,RbS=4,REs=232,
Tpmi=0x02010022,TbCrcHw=0x00020000,HarqId=0,
MCS1=1,NDI1=0x01012001,RV1=0x12B60000,TBS1=56,
MCS2=23,...,TBS2=0;Cfg=1,Int=1,
CbCRC=0x00000000,...,RxNum=4,TxNum=4,
CodewordNum=1,TB1LayerNum=1,TB2LayerNum=0,
MimoScheme=1,MCSTable=0,TB1Mod=2,TB2Mod=2
```

字段分层：

| 字段 | 可安全理解的含义 | 注意事项 |
|---|---|---|
| `DCIInfo[n]` | 同一载波/子帧内第 n 条调度结果 | n 不是 HARQ ID，也不必等于载波号 |
| `TM` | 传输模式上下文 | 若与 RRC `transmissionMode` 一致可按协议理解；仍需与重配日志核对 |
| `RbNum`/`RbS` | 分配 RB 数/起始位置 | 与仪表 resource allocation 对齐；单位通常为 PRB |
| `REs` | 本次可用资源元素数 | 是否已扣除 CRS/控制区等由实现决定，码率计算需与仪表定义一致 |
| `Tpmi` | 厂商打包的层数/PMI 信息 | 优先使用同一行已拆出的 `TB1LayerNum/TB2LayerNum`，不要盲拆十六进制 |
| `TbCrcHw` | TB 硬件 CRC 结果打包 | 本固件观测：相应 nibble/字段 2=成功、1=失败、0=无结果/未使用；这是厂商枚举，不是协议值 |
| `HarqId` | HARQ 进程号 | FDD 传统下行每个 serving cell 8 个进程；TDD/新特性需按配置 |
| `MCS1/2` | TB1/TB2 的 MCS index | MCS index 要经对应 MCS table 才得到 Qm/TBS；不能直接当码率 |
| `NDI1/2` | 含 NDI 的打包字 | 协议含义是 NDI 翻转通常表示新传；此日志整个十六进制还含私有位域 |
| `RV1/2` | 含 RV 的打包字 | 协议 RV 值为 0/1/2/3，常见发送顺序 0→2→3→1；十六进制整字不可直接等同 RV |
| `TBS1/2` | 每个 TB 的大小 | 日志通常按 bit；`TBS2=0` 表示本次没有第二 TB |
| `Cfg`/`Int` | 配置次数/硬件译码中断次数 | `Cfg>Int` 持续扩大可能是中断丢失或未完成；短暂相差 1 要考虑流水线 |
| `CbCRC/1` | code block CRC 汇总/位图 | 单 CB TB 可能不使用 CB CRC；不能只因 0 就判错 |
| `RxNum/TxNum` | 接收/发射天线或端口上下文 | 与层数不同；4Tx 不等于 4 layer |
| `CodewordNum` | codeword/TB 数 | 与 `TBS2`、TB2LayerNum 交叉确认 |
| `TBxLayerNum` | 每个 TB 使用的层数 | 码率和峰值计算应使用它 |
| `TBxMod` | 调制阶数 Qm | 协议常见 2=QPSK、4=16QAM、6=64QAM、8=256QAM |

本日志仅 54 条 `DCIInfo`：`TbCrcHw=0x00020000` 49 条、`0x00010000` 1 条、`0x00000000` 4 条。这个样本说明主要数据可解，但仍应把唯一失败按同一 `HarqId` 追到重传，不能只报总体成功率。

本固件打印存在约一个子帧的流水延迟，用户已有样本显示结果行可能指向上一子帧。自动脚本应把“打印子帧”和“被指示 PDSCH 子帧”都保留，并通过 HARQ ID/NDI/RV 验证，不能硬编码所有版本都固定延后一帧。

#### 24.5.2 HARQ 正确追踪键

推荐主键：

```text
(serving cell / CC, HarqId, codeword/TB, NDI epoch)
```

追踪规则：

1. 同一 CC、HARQ ID、TB 上 NDI 翻转，通常开启新传输。
2. NDI 不变、RV 变化，通常是同一 TB 的重传。
3. 常见 RV 顺序是 0、2、3、1，但仪表场景可能禁用重传或从非零 RV 开始，不能单凭顺序删除样本。
4. TB1/TB2 要分别跟踪；TB1 ACK 不代表 TB2 ACK。
5. FDD 常见同 HARQ ID 的重传相隔 8 个子帧，但必须结合调度；CA 下每个 serving cell 独立跟踪。
6. 成功后同一 TB 不应继续重传；若仪表仍重传，检查 UE 的 ACK 是否真正经 PUCCH/PUSCH 发出。

#### 24.5.3 码率与高 MCS

近似物理码率：

```text
R ≈ TBS / (REs × Qm × layer_count)
```

这里的 `REs` 定义必须先确认。如果它已经包含所有层或已按 codeword 统计，再乘层数会重复。仪表 Trace 中 `A/UINT` 是更可靠的编码率判据；本地估算主要用于快速筛选。用户现有经验阈值为约 0.93，超过时要优先检查 MCS、RE 扣除和 256QAM table 配置。

#### 24.5.4 DLS 模块 14 个 ID 的用途

`CommDecCfg`：用于 SIB/RAR 等公共数据的译码配置。

- `RNTI=0xFFFF` 是协议定义的 SI-RNTI，强烈指向系统信息调度。
- `RNTI=0x2` 可能是 RA-RNTI；必须用同帧 RAPC/RAR 流程确认，不能仅凭小数值断言。
- `Rb/RbS/REs/MCS/RV/TBS` 为公共 PDSCH 配置。
- `DCI=<word0,word1>` 是打包 DCI，不可在无位域表时手拆。
- `TPCl` 名称/单位不明，不能写成“层数”或“TPC”。

`SIBDecode`、`RarDecode`：

```text
Cfg = 下发译码配置次数
Int = 实际译码中断次数
Ack/Nack = CRC 成功/失败累计
通常应满足 Int = Ack + Nack
```

原始日志中存在 `SIBDecode: Cfg=39, Int=39, Ack=15, Nack=24`，说明硬件中断没有丢，但多数 SIB PDSCH CRC 失败。另一个时段 `Cfg=1,Int=1,Ack=1,Nack=0` 正常。应看同一流程内增量，不要把多个搜网周期的累计计数混在一起。

`DecStat`：

- `PDSCH:ACK0/NACK0` 与 `ACK1/NACK1`：两个 TB/codeword 的累计译码结果。
- `HI:ACK/NACK`：PHICH HI 的累计结果，是对 PUSCH 的下行反馈，不是 PDSCH ACK。
- `DCI[F0/F1/F1A/F2/F2A/F2B]`：各 format 累计检出数。
- `CFI=[...]`：内部 CFI 统计桶；索引映射未打印，不能强行按 0～3 命名。
- `DtchInt`：数据信道/下行译码相关硬件中断累计；与配置数持续不相等时才有意义。

`StdLoglInfoDl`：256 子帧窗口 KPI。

- `CC[n]`：载波索引。
- `AvgPHYTb.kbps=[TB1,TB2]`：两 TB 物理层吞吐率。
- `1TB/2TB`：单 TB/双 TB 调度或样本计数。
- `TB1Dec=[success,fail]`、`TB2Dec=[success,fail]`：从数据变化推断为译码成功/失败对；该映射建议再用已知错包场景确认。
- `Rv_Num=[...]`：各 RV/发送轮次的内部计数，数组索引未显式定义。
- `hq_fail`：HARQ 最终失败计数或相关 KPI，具体条件为私有实现。
- `subFrame=256`：统计窗口长度。
- `aveRbNum`、`MCS` 在样本中可超过协议 MCS index，例如单个样本出现 `MCS=40`，很可能是定点/累积平均值；不能直接说 MCS 40。观察上很像带 Q2 或累加缩放，必须用厂商定义确认。

`DDTR_Reg BdRpt+CcTop`、`DDTR_Reg DtchTb`：芯片级译码寄存器快照。

- 可读上下文：`Hw/Tag`、`ErrorInfo`、`TB1Addr/TB2Addr`、`RptTagid`。
- `CK/RE/QmNl/K0/K1/Ncb` 名称对应 Turbo 码块、速率匹配、调制/层、循环缓冲起点等概念，但当前打印的是打包寄存器字，不能直接把十六进制当十进制参数。
- `ErrorInfo` 非零、硬件无中断、地址异常可作芯片级线索；`CfgValid=0` 在本日志成功片段也出现，故不能解释为“配置无效”。
- `CcTbRlt`、`PdschEn`、`CcTag`、`Cw*Cinit`、`HarqCtrlInfo` 都需专用位域表。

`MacPdu` 与 `DLDdrBaseRep`：

- `MacPdu ... len=7` 是 PHY 已得到可交给 MAC 的 PDU，通常是 CRC 成功后的强证据；payload 要交由 MAC parser 解释。
- `apbMacPduDataTB1/TB2` 是内存地址，不是空口数据。
- `wHarqId` 可用于与 DCIInfo 对齐。
- `g_wHarqDdrRep=1` 是内部上报/调试标志，不等于 ACK。

`Feedback`：

- `Ack0/Ack1` 是两个 TB/codeword 的下行 HARQ 反馈值队列。
- `Valid0/Valid1` 指相应位置是否有效。
- `VALID[...]` 是打包有效位图。
- 样本 `Ack0=[1,0,0,0], Valid0=[1,0,0,0]` 出现在 Msg4 成功后，支持 `1=ACK`；它仍是厂商队列编码，最终是否发出需联查 ULA PUCCH/PUSCH。

#### 24.5.5 PDSCH 异常定界

| 证据组合 | 结论方向 |
|---|---|
| 仪表有 DCI；UE `DciValid=0`、无 DCIInfo | PDCCH 漏检/搜索空间/RNTI/CFI 问题 |
| DCIInfo 有，`Cfg` 增加但 `Int` 不增加 | 译码硬件配置、中断、RF 门控或时序问题 |
| `Int` 增加，`TbCrcHw=1`/NACK 增加 | PDSCH 解调译码失败；查 SNR、码率、TM、层数、参考信号 |
| TB CRC 成功并有 `MacPdu`，上层没有消息 | PHY→MAC 交接、PDU 解析或上层状态机问题 |
| UE TB 成功，但仪表持续重传同 HARQ/NDI | ACK 生成或 PUCCH/PUSCH 反馈链问题 |
| 第二 TB 独立失败 | 重点查双码字配置、层映射、PMI/RI、TB2 MCS 与功率 |

### 24.6 PHICH：它确认的是 PUSCH，不是 PDSCH

#### 24.6.1 完整证据链

```text
PBCH/MIB：phich-Duration + phich-Resource
    ↓
DLA 公共配置：PhichNg、PhichDur
    ↓
UE 发 PUSCH，记录 CC + HARQ ID + PUSCH subframe + PRB/DMRS
    ↓
ULA 安排 PHICH 接收资源和绝对子帧
    ↓
ULS: Rec PHICH:value=1/0, AbsSF=..., Harq Id=...
    ↓
DLA ProDlCtrlChStat / DLS DecStat 的 HI ACK/NACK 累计增加
    ↓
ACK 停止该次 UL HARQ；NACK 进入自适应或非自适应重传
```

最直接的跨模块日志是：

```text
ULS|... Rec PHICH:value = 1,wCurCcNum=0,AbsSF=8906,Harq Id=2
```

本日志中这类结果共有 28 次：27 次 `value=1`、1 次 `value=0`。结合后续重传行为可验证本固件 `1=ACK、0=NACK`。唯一 NACK 后出现非自适应重传并最终 ACK，是完整恢复样本。

#### 24.6.2 相关字段

- `PhichDur=0/1`：DLA 日志明确 `0:Nor,1:Ext`。
- `PhichNg`：内部枚举/资源配置，需与 MIB `phich-Resource` 对照，不直接把数字当 Ng 实数。
- `HiAckNum/HiNackNum(2CC2TB)`：按 CC/TB 槽位统计的 HI；括号中四项的具体排列需要厂商定义，至少可做逐周期差分。
- `DecStat HI:ACK/NACK`：载波级累计。
- `AbsSF`/`wABSPhichSubFrmNo`：绝对子帧键，用于跨 1024 SFN 回卷对齐。
- `Harq Id`：必须与触发它的 PUSCH HARQ 进程一致。

务必区分：

| 名称 | 方向 | 确认对象 |
|---|---|---|
| PHICH HI ACK/NACK | 下行 | UE 先前发送的 PUSCH |
| PDSCH HARQ ACK/NACK | 上行 PUCCH/PUSCH | UE 先前接收的 PDSCH |
| `DLS\|Feedback` | UE 内部生成、待上行发送 | PDSCH，不是 PHICH |

#### 24.6.3 PHICH 异常判据

- PUSCH 已发但没有安排 `wABSPhichSubFrmNo`：UL HARQ 调度/双工时序问题。
- 安排了 PHICH，但对应 `AbsSF` 没有结果：PHICH 漏检、RF 门控或中断丢失。
- 连续 NACK 且仪表 UL HARQ 也判 NACK：先查 PUSCH 质量。
- 仪表判 PUSCH ACK，而 UE 连续解成 NACK：查 PHICH resource、`I_PRB_RA`、DMRS cyclic shift、MIB PHICH 配置、定时和功率。
- DLS `HI` 计数与 ULS 直接 PHICH 数不一致：先考虑统计窗口/复位，再查跨模块事件丢失。

### 24.7 CSI、RLM、SNR、CFO 与 CIR

#### 24.7.1 CSI 模块在本日志里打印了什么、没打印什么

CSI 模块的 9 个 ID主要是“反馈配置入口”和“RLM 计算入口”，并没有直接打印一条完整的 `CQI=x,PMI=y,RI=z` 空口上报。因此：

- 看到 `PCellCSI_En_IN` 只能证明进入 PCell CSI 处理路径。
- `wAperiodTrigger=0`、`wAperiodReportFlag=0` 表示该次调用没有看到非周期触发的可能性很大；数值编码仍是厂商内部。
- `wCsiEn_aop=0` 不应单独写成“CSI 关闭”，还需看周期 CSI 配置、ULA 的 `CqiLen/RiLen` 和仪表是否期待上报。
- `CAIdx=0` 是内部 serving-cell/CA 上下文索引，通常与 PCell 对应，但需和 SCC 配置核对。
- `wTransMode=2` 很可能对应 LTE transmission mode 2；应与 RRC dedicated config/DLS UeCategoryInfo 一致后再确认。
- `g_wScellComFlag:[0-0-0]` 是 SCell 组合内部标志，不能直接当三张 SCell 的 activation bitmap。

协议层 CSI 基本值：

| 值 | 协议含义 | 常见取值与注意点 |
|---|---|---|
| CQI | UE 建议的信道质量/调制编码能力索引 | 通常 0～15；0 表示 out of range，1～15 对应 CQI table。启用 256QAM 时表可能变化 |
| RI | 推荐 rank/层数 | 协议值通常从 rank 1 开始；厂商内部可能零基编码，日志里的 0 不可直接说 rank 0 |
| PMI | 预编码矩阵指示 | 取值域依天线端口、TM、codebook subset restriction 而变 |
| CRI | CSI-RS resource 指示（较新特性） | 本日志未看到独立字段 |

`APER_Comb`：

- `riBitLenM/riBitLenS/riBitLenS1` 为不同组合/小区分量的 RI bit 长度上下文；M/S/S1 的准确展开未给出。
- `riValue[...]` 是内部组合值。样本 `[0-2-2]` 不能直接解释成三个协议 rank。

#### 24.7.2 RLM 不是“PDSCH CRC 统计”

RLM（Radio Link Monitoring）协议上基于下行参考信号质量，判断假想 PDCCH BLER 是否达到 `Qout/Qin` 条件，再向上层产生 out-of-sync/in-sync。它与单包 PDSCH CRC 强相关但不是同一个计数器。

本日志：

```text
RLM_FdBkFirCfg:wStepIdx=0,wHwIdx=1,wTxAnPortsNum=4,wUeRxAttennaNum=4
RLM_Calc_In:dwRLMValue=0xe04b5,ptEsnrInfo->sdwSnrValue=112
```

- `wTxAnPortsNum`、`wUeRxAttennaNum` 是端口/接收天线配置。
- `sdwSnrValue` 是送入 RLM 的内部 SNR/ESNR 定点量，不是显示为 dB 的 RXP `SNR[]`。
- `dwRLMValue` 是私有打包结果，无法从十六进制直接判 Qin/Qout。

数据核验：2677 条 `RLM_Calc_In` 中，2652 条 `dwRLMValue=0xffffffff`，只有 25 条为其他值。`0xffffffff` 极可能是无效/未更新哨兵，但没有源码不能把它命名为 out-of-sync。自动化应将其标为 `INVALID_OR_SENTINEL`，再结合上层 in-sync/out-of-sync indication，而不是把 2652 条都计成 RLF。

#### 24.7.3 RXP `CRsPwr` 与 `SNRInfo`（`0x7302`，全文件 36,069 条已做全量统计）

一条完整的原文（本 ID 的字段在此固件里是固定的，前面几节引用的是截断版）：

```text
RX|SNRInfo CC_HW_CH[0010];RxN0[0x0167;0x0CA7];SNR[ -4.3, -4.8, -5.5, -6.4],
   SINR[ 63;     449,     499,     374,     284];LowSINRInd:   0;0,
   RxAntAdapt:   0,IrcCfg:0x0001ff06
```

**逐字段说明与全文件实测取值域**（36,069 条全部解析，仅 1 条格式不同）：

| 字段 | 是什么 | 本文件实测 | 怎么用 |
|---|---|---|---|
| `CC_HW_CH[0010]/[0011]` | 载波+硬件+通道的拼接上下文 | 只有这两种：`0010` 21,303 条、`0011` 14,765 条 | 两行成对出现，是同一时刻的两组 RX 分支；没有格式定义前不要硬映射成 PCC/SCC |
| `RxN0[0x____;0x____]` | 噪声功率估计（内部定点） | 高位 808 种、低位 862 种取值 | **不是 dBm/Hz**；只做同配置下的相对比较 |
| `SNR[四项]` | 四个接收分支的信噪比，**单位 dB，一位小数，可直接读** | −20.6 … **40.9**；中位 38.7–39.1；5 分位约 −14.5 | 最有用的一项。注意 **40.9 dB 是本固件的上限**，强信号下会被钳住 |
| `SINR[第1项]` | 综合/宽带 SINR 的 8 bit 量化值 | 228 种取值，其中 **255 出现 27,217 次（75.5%）** | 255 = 8 bit 饱和，强信号下无分辨力；弱信号段才有参考价值 |
| `SINR[后4项]` | 四分支的线性域 SINR 原始值 | 0 … 61,666,456 | 见下面的换算：**未饱和区可与 SNR 互相印证，饱和区只能用它** |
| `LowSINRInd` | **连续低 SINR 计数器**（不是位域） | 0×31,206、1×2,373、2×1,392、3×816、4×158，最大 9 | 相邻取值只会 +1 或清 0；`=0` 时平均 SNR 32.7 dB，`≥1` 时平均 −13 dB，分界非常干净 |
| `LowSINRInd` 后的无名字段 | 私有 | 13,676 种取值，范围含大负数 | 无资料，不解释 |
| `RxAntAdapt` | 接收天线自适应 | **恒为 0** | 本次会话未启用天线自适应 |
| `IrcCfg` | IRC（干扰抑制合并）配置 | **恒为 `0x0001ff06`** | 全程未变，可排除"IRC 配置被改动"这一类怀疑 |

**新发现：`SINR[]` 与 `SNR[]` 是同一个量的两种刻度**（原来这份文档说"不能横向比较"，现在可以了）。对全文件配对样本做拟合：

| 区间 | `10·log10(SINR) − SNR` 中位 | 离散度（5%–95%） |
|---|---:|---|
| SNR < 0 dB（未饱和） | **31.0 dB** | 28.9 – 32.1 |
| SNR > 35 dB（已饱和） | 35.3 dB | 31.7 – 37.3 |

即 **`SNR_dB ≈ 10·log10(SINR原始值) − 31`**，在未饱和区吻合到 ±1.5 dB；进入 40.9 dB 上限后 SNR 被钳住、SINR 仍继续增长，所以偏差变大且发散。

**实用结论**：

1. 强信号（SNR 顶到 ~39–40.9、`SINR[第1项]=255`）时，**用 `SNR[]` 已经分不出好坏**，要比较链路余量必须用 `SINR[后4项]` 的原始值。
2. `LowSINRInd` 非零就是明确的低质量告警，本文件里它非零的样本 SNR 全部在 −13 dB 附近——对应的是搜网/频扫时段，不是业务时段。
3. 四个分支同时掉 → 总接收电平、线损、衰落或仪表功率；只掉一两个 → RF 通道、衰减线、天线映射、Rx23 swap。
4. SNR 很高仍 TB CRC 失败 → 转查 MCS/码率、TM/PMI/RI、RB/RE 与软件配置，不是射频问题。
5. 不给通用硬阈值：不同 MCS、层数和 BLER 目标的要求不同。

与之配对的 `RX|CRsPwr` 打印：

```text
RX|CRsPwr CC_HW_CH[0010];RSRP00[345;-3],...;RSSI[1891,2041]
```

- `RSRP00/01/10/11` 每项含两个内部量，**不是协议上报的 dBm RSRP**（协议口径见 §23.11 与 §26.8.2 的 CSRM 计算链）。
- `RSP0/RSP1`、`RSSI` 是内部线性/累积功率，只适合同固件、同增益、同配置下的相对比较。

#### 24.7.4 CFO

```text
RX|CFO Info:Cc=0,Flag=0,AbsCfo=295,NewCfo=3406,OldCfo=9668,
Out=966800,Coeff=2,K=0,Extend=2,AdjCfo=14,AdjInd=1
```

- `Cc`：载波上下文。
- `NewCfo/OldCfo`：新测量值与滤波前值或新旧滤波状态；单位/缩放未知。
- `Out`：滤波输出的内部定点量。样本恰有 `OldCfo×100`，不能据单样本推广成固定单位。
- `Coeff/K/Extend`：滤波参数。
- `AdjCfo/AdjInd`：频偏调整值与调整指示；`AdjInd=1` 很可能表示本次触发调整。
- `CfoFilterCoeffAdapt UpdateFlag/Old/NewTemp/K`：温度或状态相关的 CFO 滤波系数自适应。

判据应看趋势：绝对值是否持续变大、正负来回震荡、调整后是否收敛，以及是否同时出现 PBCH/PDCCH/PDSCH 失败。无单位时不要直接写“超过 X Hz”。

#### 24.7.5 CIR 与接收窗调整

`0x7400～0x7403` 分别打印 Rx0～Rx3 对 Tx0～Tx3 的：

```text
[PeakValue, Pos, WinStart, WinEnd]
```

注意：

- `PeakValue` 是打包十六进制，出现 `ffff0000`、`00000000`、`0001xxxx` 等；它可能含符号/标志，不能按无符号整数简单排序。
- `Pos` 是内部 CIR 峰位置，单位可能是采样点或折算 timing bin。
- `WinStart/WinEnd` 是接收/搜索窗边界；0 可能是未建立窗口或默认值。
- `0x7404 CIRAdjustValue` 是各分支窗调整量与剩余量；`PreSyncState`、`Bw`、`ShiftNum` 提供上下文。
- `0x7405 MaxDelay00~33`、`ModeType`、`Fgt`、`32KCALIHw`、`RefSen`、`FixCoef` 都是时延扩展/滤波私有配置。

可用异常特征：

- 主峰 `Pos` 突跳且 CFO/SNR 同时异常；
- 峰长期在 `WinStart/WinEnd` 边缘或窗外；
- 多 RX 的峰位置互相严重矛盾；
- `CIRAdjustValue`/`Remain` 长时间不收敛；
- 重配/切换后 `Bw` 与服务小区不符；
- 不能仅因 `ffff0000` 判硬件坏，必须与成功/失败时段对比，因为本日志正常流程也大量出现该值。

### 24.8 推荐的下行自动化分析流程

#### 24.8.1 每个失败点保存的最小上下文

```text
时间戳、SFN.subframe、绝对子帧
CC/serving cell、EARFCN、PCI、带宽、天线、TM
PBCH 最近一次成功/失败
CFI、DciValid、RNTI 类型、DCI format
RB start/count、RE、MCS、Qm、层数、TBS
HARQ ID、NDI、RV、TB1/TB2 CRC
PDSCH ACK/NACK 是否生成并是否发出
PHICH 的 AbsSF、HARQ ID、value
SNR 四分支、LowSINRInd、CFO、CIR peak/window
CSI/RI/PMI 配置及 RLM sentinel/in-sync/out-of-sync
```

#### 24.8.2 决策树

```text
PBCH 成功吗？
├─ 否：PSS/SSS → CFO/CIR/SNR → PBCH branch/CRC → MIB配置
└─ 是：预期子帧 DCI 检出吗？
   ├─ 否：CFI/RNTI/search space/aggregation → RXP质量
   └─ 是：DLS 配置和中断到达吗？
      ├─ 否：DLA→DLS交接/硬件中断/RF门控
      └─ 是：TB CRC 成功吗？
         ├─ 否：码率/MCS/RE/层数/TM/PMI/RS功率/SNR
         └─ 是：MAC PDU 上报了吗？
            ├─ 否：DDR/PHY→MAC接口
            └─ 是：上层流程继续了吗？
               ├─ 否：MAC/RLC/RRC
               └─ 是：检查后续 ACK/PHICH/吞吐性能
```

### 24.9 本章 5 个模块全部 56 个 message_id 覆盖表

说明：次数来自 `message_id_index.tsv`。典型日志只保留签名，避免把私有动态值误当固定模板。

#### 24.9.1 PBCH（17 个 ID）

| ID | 次数 | 典型日志签名 | 作用、关键值 | 异常判据 |
|---|---:|---|---|---|
| `0x6A10` | 2584 | `PBCH RESULT: ... CrcOk, BchMib, dwBranch, SFN` | 每次 PBCH 硬件结果；`CrcOk 1/0` 成败，branch/ant 为内部假设 | 连续 0 并耗尽 branch；有配置无结果 |
| `0x7EC1` | 2583 | `ConfigPbchReg ... State=[Idle/WaitLlr/...]` | PBCH 寄存器/状态配置；状态枚举由日志明示 | state 卡住、配置次数与结果数长期不匹配 |
| `0x7F8D` | 423 | `Update Rx Regs > CellInfo, MimoScheme, CheStartParm` | 更新 PBCH 接收/信道估计寄存器，三项均私有打包 | 与 PCI/天线重配不同步；需芯片位域表 |
| `0x6A0D` | 273 | `RxRsrpMoniter > RSRPxx[I,Q], RSSI, RSPRx` | PBCH 内部功率/相关量；不是 dBm | 相对值塌陷、分支严重失衡且 CRC 失败 |
| `0x6A04` | 151 | `UpRxState > RxCurrState, RxPreState` | `0 Idle/1 Meas/2 Pbch/3 Nomal/4 Sync` | 长期停在 Pbch、反复 2↔其他态 |
| `0x6A0E` | 62 | `MibReqMonitor > Earfcn, CellId, Boundry, CpMode` | MIB 请求输入；-1 表示未知/占位 | 目标 EARFCN/PCI 错；请求后无配置/结果 |
| `0x6A08` | 61 | `AdjTpuTime > AdjFrame, OffsetSlot, OffsetTs` | 解 MIB 后调整内部 TPU/帧边界 | 调整反复、大幅变化并伴随同步失败 |
| `0x6A0F` | 61 | `EPBCH_UPDATE_COUNTER_CNF ... TpuOffset, AFC` | RF/TPU 计数更新确认 | 请求无确认、offset 与同步链不一致 |
| `0x7EC2` | 61 | `ScGeneration > HwIdx, CellId, ScGenEn` | 配置 PBCH scrambling/序列生成上下文 | PCI 不一致、预期使能但 `ScGenEn=0` |
| `0x6A14` | 39 | `MibInfoCheck ... BwIdxMapToNrbdl=...` | 从 MIB 带宽索引查到 RB 数；其他标志为私有 | 映射 RB 与仪表带宽不一致 |
| `0x6A11` | 38 | `MIB OK!!` | PBCH/MIB 成功里程碑 | 有 `CrcOk=1` 却无此行，查结果处理 |
| `0x6A12` | 38 | `PBCH CRC OK: Earcfn, CellID, RBNum, dwAntNum, BchPhich` | 最实用的 PBCH 汇总；RB/天线为解码结论 | 与仪表/目标小区不一致 |
| `0x6A03` | 25 | `Recv CSR Reset Msg` | 搜索/同步侧要求 PBCH reset | 非预期频繁 reset，查上游状态原因 |
| `0x6A09` | 24 | `Extra Max Branch,Report Fail` | 全部分支/最大尝试耗尽并上报失败 | 明确 PBCH 失败终点；联查 SNR/CFO/PCI |
| `0x6A01` | 7 | `Recv MC Reset/SetMode Msg` | MC 侧复位或模式切换 | 非脚本预期出现、之后不重新启动 |
| `0x6A02` | 3 | `Recv Release Msg` | 释放 PBCH 任务/资源 | 连接过程中意外 release |
| `0x6A0C` | 1 | `Error Print, ErrId=[...EventDelFail/StartMibFail]` | 内部错误；日志明确列出 1～4 的名字 | 任一出现均应保存上下文；样本 ErrId=3 |

#### 24.9.2 DLA（6 个 ID）

| ID | 次数 | 典型日志签名 | 作用、关键值 | 异常判据 |
|---|---:|---|---|---|
| `0x5E1A` | 42024 | `Che Reg: CcIdx, HwIdx, TagId, SubFrm, ...` | 每子帧信道估计/下行硬件配置；只直接用 CC/tag/subframe | 预期工作 CC 无配置；其余寄存器需位域表 |
| `0x5E2A` | 41989 | `CDTR_Reg DciRpt > ... PdcchRpt/HiRpt` | PDCCH 与 HI 原始硬件报告 | 仅用于芯片级对比；不能凭 packed word 判 format |
| `0x7F8B` | 41989 | `CInt >>> Cdtr Rpt: Cfi, DciValid, RntiEnInd` | PDCCH 主结果；DciValid/RNTI bit 映射由日志明示 | 仪表有 DCI 而相应 bit 持续为 0 |
| `0x5E06` | 38 | `State Switch ... Search/Campon ... PhichNg/Duration` | DLA 状态和公共下行上下文 | 状态不进 Campon；小区/PHICH 配置错 |
| `0x5E08` | 10 | `ProDlCtrlChStatInfoMonitor > CfiNum, HiAck/Nack, DciNum` | 周期控制信道/PHICH统计 | 统计差分与直接日志不一致；NACK激增 |
| `0x5E02` | 3 | `Recv ... EDLA_COMM_REQ ... CellId, Freq, Bw, Ant, DlTM` | 接收公共/专用下行配置 | 与 PBCH/RRC/仪表不一致，或有请求无后续工作 |

#### 24.9.3 DLS（14 个 ID）

| ID | 次数 | 典型日志签名 | 作用、关键值 | 异常判据 |
|---|---:|---|---|---|
| `0x6221` | 164 | `DDTR_Reg BdRpt+CcTop > CcTbRlt, ErrorInfo, TBAddr...` | 译码硬件报告/地址/错误寄存器；绝大多数字段私有 | `ErrorInfo` 非零、无对应结果；不要用 `CfgValid=0` 单独判错 |
| `0x6222` | 164 | `DDTR_Reg DtchTb > [CK,RE,QmNl,K0,K1,Ncb]` | TB 码块与速率匹配寄存器快照 | 需厂商位域表；仅做成功/失败场景 diff |
| `0x628D` | 113 | `CommDecCfg; RNTI, Rb, REs, MCS, RV, TBS` | SIB/RAR 等公共 PDSCH 解码配置 | 与仪表 DCI 不一致；配置后无中断 |
| `0x6382` | 110 | `SIBDecode:Cfg,Int,Ack,Nack,Data` | SIB CRC 累计；应有 `Int=Ack+Nack` | Nack连续增、Cfg-Int差扩大 |
| `0x638D` | 54 | `StdLoglInfoDl: CC, AvgPHYTb, TBDec, aveRbNum, MCS` | 256子帧窗口 DL KPI；部分平均值有内部缩放 | 吞吐为0但持续有调度；失败计数/hq_fail增长 |
| `0x6407` | 51 | `DLS_Task: Ccb, NirDivC, KPIplus, NumCB, HarqIDmin, Kmimo` | 软缓冲/Turbo译码任务资源上下文 | 配置与 UE category/TM 不匹配；十六进制字段私有 |
| `0x6297` | 50 | `DLDdrBaseRep: wHarqId, MacPduDataTB1/2` | 已解 PDU 的 DDR 地址和 HARQ 上下文 | CRC成功却地址/上报缺失；地址本身不是 payload |
| `0x6391` | 50 | `MacPdu ... len=` | 打印已交付 MAC 的 PDU 摘要 | TB成功无 MacPdu；len异常需联查 MAC |
| `0x6386` | 17 | `DecStat[CC]: PDSCH ACK/NACK; HI; DCI; CFI; DtchInt` | 下行解码/PDCCH/PHICH累计总表 | 差分异常、计数停滞、NACK持续增长 |
| `0x6387` | 17 | `DecStat; TB1Nack[...]; TB2Nack[...]` | 两 TB 的 NACK 分桶 | 数组索引未定义；只做同版本趋势/差分 |
| `0x6206` | 15 | `UeCategoryInfo: UECat, Nsoft, AltCqiTable, TransMode` | UE能力、软缓冲、MCS/CQI表、TM译码上下文 | 与能力/RRC重配不一致；重配后未刷新 |
| `0x6202` | 3 | `Recv Common Msg: CellId, BW, Ant, Pb` | DLS公共小区配置；BW样本为PRB数 | 与 PBCH/仪表不一致；有配置无后续译码 |
| `0x6384` | 3 | `RarDecode:Cfg,Int,Ack,Nack` | RAR PDSCH译码累计 | `Int≠Ack+Nack`、Nack或无中断 |
| `0x6385` | 3 | `Feedback: Ack0/1, Valid0/1, VALID` | 生成的 PDSCH HARQ反馈队列 | CRC成功却无valid ACK；需继续查上行是否发出 |

#### 24.9.4 CSI（9 个 ID）

| ID | 次数 | 典型日志签名 | 作用、关键值 | 异常判据 |
|---|---:|---|---|---|
| `0x5723` | 13385 | `First_FdBkCfg_IN` | 进入首次/主 CSI feedback 配置入口 | 单独无成败语义；预期流程完全不进入才异常 |
| `0x5745` | 13385 | `AperRepJudge_IN: AperiodReportFlag, Sym4IntFlag, Sym9IntFlag` | 判断非周期 CSI 与符号中断上下文 | 仪表触发 A-CSI 但 flag 始终不变化 |
| `0x5760` | 13385 | `First_FdBkCfg_IN_CAIdx, AperiodTrigger, CsiEn_aop` | 按 CA/serving cell 配置反馈入口 | CAIdx/触发/使能与调度不符 |
| `0x5763` | 2682 | `APER_Comb: riBitLen..., riValue[...]` | 非周期 CSI 组合中的 RI 长度/内部值 | bit长度与配置小区/天线不匹配；具体编码私有 |
| `0x574B` | 2677 | `RLM_FdBkFirCfg: StepIdx, HwIdx, TxAnPortsNum, UeRxAttennaNum` | RLM/CSI 首次硬件反馈配置 | 端口/接收天线数与服务小区不一致 |
| `0x574D` | 2677 | `RLM_Calc_In: dwRLMValue, sdwSnrValue` | RLM计算输入/结果 | `0xffffffff` 为疑似哨兵；持续哨兵须结合 in/out-sync |
| `0x5756` | 2677 | `FbHRConfig_wRDiagInd, HTransIdx, RTransDiagIdx` | H/R 变换或诊断相关内部配置 | 私有枚举，只能场景 diff |
| `0x5759` | 2677 | `PCellCSI_En_IN` | 进入 PCell CSI enable/处理路径 | 预期 PCell CSI 时完全缺失 |
| `0x575B` | 2677 | `First_FdBkCfg_INT1: CAIdx, TransMode, CsiEn_aop, ScellComFlag` | 首次反馈中断阶段1，含 TM/CA/SCell上下文 | TM/CA与RRC配置不一致；flag含义私有 |

#### 24.9.5 RXP（10 个 ID）

| ID | 次数 | 典型日志签名 | 作用、关键值 | 异常判据 |
|---|---:|---|---|---|
| `0x7300` | 36069 | `CRsPwr: RSRPxx, RSP0/1, RSSI` | CRS相关内部功率估计；不是dBm | 相对塌陷/通道失衡，与SNR/CRC联判 |
| `0x7302` | 36069 | `SNRInfo: RxN0, SNR[4], SINR, LowSINRInd` | 最可读的四路SNR及低SINR指示 | SNR下降、非零LowSINRInd、分支失衡 |
| `0x7381` | 8395 | `CFO Info: New/Old/Out/Coeff/AdjCfo/AdjInd` | 频偏估计、滤波与调整 | 不收敛、震荡、与同步/CRC失败同时出现 |
| `0x7400` | 1596 | `CIRADJ: Rx0Tx0~3 [PeakValue/Pos/Win]` | Rx0到各Tx端口的CIR峰/窗口 | 峰位突跳、窗外；packed peak不可直接比大小 |
| `0x7401` | 1596 | `CIRADJ: Rx1Tx0~3 ...` | Rx1 CIR | 同上，并与Rx0交叉比较 |
| `0x7402` | 1271 | `CIRADJ: Rx2Tx0~3 ...` | Rx2 CIR；仅4Rx阶段有更多打印 | 同上；缺失也可能因仅2Rx配置 |
| `0x7403` | 1271 | `CIRADJ: Rx3Tx0~3 ...` | Rx3 CIR | 同上 |
| `0x7404` | 600 | `CIRADJ0: CIRAdjustValue, Remain, PreSyncState, Bw` | 接收窗调整量/残差和带宽上下文 | 调整不收敛、BW错、预同步状态异常 |
| `0x7405` | 600 | `CIRADJ1: MaxDelay, ModeType, Fgt, RefSen, FixCoef` | 时延扩展与CIR滤波私有配置 | 仅做版本/场景diff；不能凭单值判错 |
| `0x7382` | 16 | `CfoFilterCoeffAdapt: UpdateFlag, Old/NewTemp, K` | CFO滤波系数/温度自适应 | 温变后不更新且CFO发散；单位私有 |

### 24.10 本日志已经验证的下行事实

1. PBCH 成功样本明确：`EARFCN=1650, PCI=503, RBNum=100, dwAntNum=4`；成功前有两次 branch CRC 失败，说明单次失败不能提前终止分析。
2. 连接态另一个 PCell 配置为 `EARFCN=1275, PCI=30, BW=75 PRB, Ant=4`。
3. 54 条 `DLS|DCIInfo` 中 TB 硬件 CRC：49 成功、1 失败、4 无结果/未使用。
4. PHICH 直接结果 28 条：27 ACK、1 NACK；NACK 后通过重传恢复。
5. 2677 条 RLM 计算里 2652 条为 `dwRLMValue=0xffffffff`。这应被标成私有哨兵/无效候选，不能自动解释为无线链路失步。
6. RXP 在正常数据期能看到两组 `CC_HW_CH[0010]/[0011]` 与约 34～40 dB 的四项 SNR；在其他搜索期也有 -10 dB 量级样本，因此分析必须按服务小区/时间窗分组。
7. `DLS|Feedback` 的 ACK 样本出现在 Msg4 解码成功后；它证明 UE 内部已生成 ACK，但是否上行发出仍需 ULA/PUCCH 日志闭环。

以上事实只适用于本文件和当前固件日志格式；自动化解析器应将固件版本、日志模板 hash 和字段映射版本一同保存。

## 25. 上行模块逐项字典（ULA / ULS / RAPC / LPC，138 个消息 ID）

本章覆盖上行发射与随机接入的 4 个模块共 **138 个 message_id**：RAPC 的 PRACH/RAR/Msg3/Msg4、ULS 的 DCI0 与 UL HARQ/PHICH/TA、ULA 的 PUCCH/PUSCH/SRS 资源与功控、LPC 的低功耗请求。

> 分析对象：`loglte_phich.txt`；索引来自 `generated_lte_log_index/`。本章的“原始行号”是物理文本行号，日志第一列“序号”比物理行号小 2。

### 25.1 覆盖范围、完整性与使用边界

本章覆盖原始文件中所有下列模块日志，而不是只挑“看起来有用”的几行：

| 模块 | 原始行数 | message_id 数 | 本章定位 |
|---|---:|---:|---|
| ULA | 76,419 | 48 | 上行发射参数、PUCCH/PUSCH/SRS、功率、发射完成 |
| ULS | 1,678 | 32 | DCI 0/RAR Grant、UL HARQ、PHICH、TA、上行统计 |
| RAPC | 1,535 | 56 | PRACH、RAR、Msg3/Msg4、随机接入状态机 |
| LPC | 18 | 2 | 低功耗/唤醒和 DVFS；不是 LTE 空口协议本身 |
| **合计** | **79,650** | **138** | 附录保证每个 message_id 至少出现一次 |

本文件中的“原始行号”是物理文本行号；日志第一列“序号”比物理行号小 2。典型流程用 `frame.subframe` 对齐，不能只依赖墙钟时间，因为墙钟时间精度只有秒级。

#### 25.1.1 三类字段必须分开理解

1. **协议字段**：例如 DCI 0 的 `NDI/TPC/MCS/CSI request`、RAR 的 `TA/Temporary C-RNTI/UL Grant`、PHICH ACK/NACK。这些能按 3GPP 语义解释。
2. **厂商枚举**：例如 `ChanType/eTXPuschType/eTXPucchType/RntiType/NewTranCondition`。只有日志或源码定义，不能把数字直接当作 3GPP 枚举。本章只在上下文能够闭环时给出“本版本观察映射”。
3. **硬件/内部值**：例如 `KValue/Cv/dwPucchScale/DataSrc/RegCfgErr`。可用于同版本前后对比，但通常不能转换成空口字段。

#### 25.1.2 本版本存在打印字段错位，必须先知道

本日志至少有两类明显的格式串/参数错位：

- `0x7B38 zPHY_euls_DecodeModuleCodeSchem` 中，`mcs:0,Qm:20,TBS:6,CurIRV:48936` 与同一 Grant 的 `MCS=20`、`TBS=6117 bytes` 对齐后可确认：打印标签并非字面含义。对本版本更合理的读法是：第一个 `0` 是 CC/内部索引，`20` 是 MCS，`6` 是调制阶数 Qm，`48936` 才是 TBS bit 数，因为 `48936/8=6117`。
- `0x6D01/0x6D86` 中名为 `b...ReachMax` 的“布尔值”出现 91、85、14 等不可能值，`MPR/AMPR/Pemax` 位置也出现类似 RSRP 的负数。说明字段从某处开始整体错位。此版本可优先信任日志开头的最终功率值，后半段字段必须结合相邻功控日志或源码确认。

因此，本章不会为了“看起来完整”而强行解释不可靠的私有字段。

### 25.2 上行端到端总图

```text
上行数据到达 MAC
    |
    +-- 已有 UL Grant -----------------------------+
    |                                              |
    +-- 无 UL Grant -> BSR 触发 -> SR pending      |
                       |                           |
                       v                           |
              PUCCH Format 1 发 SR                 |
                       |                           |
                       v                           |
              PDCCH DCI Format 0 -----------------+
                                                   |
                                                   v
                                         ULS 解 Grant/RIV/MCS
                                                   |
                                                   v
                                         ULA 取数、功控、发 PUSCH
                                                   |
                                                   v
                                FDD: PUSCH n -> PHICH n+4
                                      | ACK              | NACK
                                      v                  v
                                    结束        自适应/非自适应重传

若 SR 达到 dsr-TransMax 或上行失步：MAC 触发随机接入
    -> Msg1 PRACH -> Msg2 RAR -> Msg3 PUSCH -> PHICH -> Msg4 -> 接入完成

辅助链：
  CSI request/周期 CSI -> PUCCH 或随 PUSCH 复用，影响下行调度
  SRS -> eNB 估计上行信道，影响 UL RB/MCS/频选调度
  TA -> 保证 PUCCH/PUSCH/SRS 到达时间对齐
  功控 -> 保证 MT8000A 能正确检测 PRACH/PUCCH/PUSCH/SRS
```

排查时不要从“某一条 Err”开始猜。先锁定业务阶段，再按 `请求/配置 -> 资源 -> 发射 -> 仪表接收 -> 反馈 -> 重传/结束` 闭环。

### 25.3 PRACH：Msg1 到 Msg4 的完整逐行流程

#### 25.3.1 Msg1：收到接入请求并选 PRACH 资源

本日志第一次成功接入从 `1102.881.6` 开始：

```text
0x9812  Recv ... ACCESS_REQ ... CCCH-SDU Content RA Start
0x9931  RAPC state = E_RAPC_STEP1_2
0x9814  Preamble Format=0
0x9954  wPreamTransCounter=1
0x990F  RF is not opened yet, delay ... 10ms
0x9910  Preamble select TimeStart: Frame=882, SubFrame=9
0x9911  wCfgAvailNum=1, awCfgAvailSubFrame[0]=1
0x9908  Resource Select Result: FrameDes=883, SubFrameDes=1
0x9909  RA-RNTI Value=2
```

字段解释：

- `CCCH-SDU Content RA Start`：竞争随机接入，Msg3 将承载 CCCH 数据。
- `E_RAPC_STEP1_2`：厂商状态机，表示 Msg1/Msg2 阶段；不是 3GPP 信令值。
- `Preamble Format=0`：LTE PRACH 长序列格式 0。协议还定义格式 1/2/3，TDD 还可能有短格式 4；不同格式具有不同 CP/重复长度和覆盖能力。
- `wPreamTransCounter`：本次随机接入过程中的前导发送计数。本文件观察到 1 到 7，7 达到最大，说明脚本/网络很可能配置 `preambleTransMax=n7`。
- `FrameDes/SubFrameDes`：选中的 PRACH 发射时刻。本例为 883.1。
- FDD 的 `RA-RNTI = 1 + t_id + 10*f_id`。本例 PRACH 子帧 `t_id=1`、第一频域 PRACH 资源 `f_id=0`，所以 RA-RNTI=2，与日志一致。

#### 25.3.2 Msg1：前导 ID、频域位置、序列与发射功率

在 `1102.882.8`，实际配置 Msg1：

```text
0x9802  Recv ... PREAMBLE_PROCESS_EV
0x9915  Invalid input of ... PreambleGroupSelect
0x994A  Preamble Group=0
0x993B  Random number=4 between 0~52
0x994B  PreambleID=4
0x994D  NPrbRa Value=4
0x991A  Preamble LogicU=22
0x991D  PathLoss=91, RsrpFilterValue=-73
0x991E  Preamble trans ... PrachPower=-13
0x991F  Pathloss=91, PrachPower=-13.0
0x994E  PValue=1, Cv=104, KValue=1955594240, Power=-13
0x9948  SubFrame=1, PreaFormat=0, Cv=104, ... PreamPower=-13
```

字段解释和判据：

- LTE 每个 PRACH 配置最多有 64 个前导，`PreambleID` 通常为 0..63。本例随机池为 0..52，选中 4。
- `Preamble Group=0` 可理解为厂商内部的 Group A/默认组索引；需结合 MAC 的 `sizeOfRA-PreamblesGroupA/messageSizeGroupA` 才能判断协议上的 A/B 组。
- `NPrbRa` 很可能是 PRACH 的频域起始/资源位置。本版本观察值 4；它不是 RA-RNTI，也不是前导 ID。
- `LogicU` 是用于生成 PRACH Zadoff-Chu 序列的根序列编号/逻辑映射结果；`Cv` 是循环移位相关内部值；`KValue`、`PValue` 是硬件配置，不能按 3GPP 枚举直接解码。
- PRACH 功率基本关系是 `min(P_CMAX, PREAMBLE_RECEIVED_TARGET_POWER + PathLoss)`，目标功率还包括格式偏置和每次重发的 `powerRampingStep`。本例路径损耗 91 dB、发射功率 -13 dBm；失败场景后期路径损耗 152 dB 时功率顶到 23 dBm。
- `0x9915` 虽然带 `Err/Invalid input`，但本文件 54 次前导流程每次都出现，且 3 次最终成功，因此它是该版本的非致命回退打印，不能单独判定随机接入失败。紧跟着能选出 Group、PreambleID 并发射才是关键。

#### 25.3.3 RAR 接收窗和 Msg2

```text
0x9965  RAR_DETECT_ENABLE_EV: Frame=883, SubFrame=4
0x9951  Add RAR detect start ... success
0x9952  Add RAR detect stop ... wAbsSfn=8847
0x9803  RAR Detect start at Frame 883, Subframe 4
0x9804  Enable RA-RNTI flag
...
DLS 0x6384  RarDecode:Cfg:1,Int:1,Ack:1,Nack:0
0x9815  Recieve RAR PDU from DLS at 884.0
0x9922  RA_PID match: PreambleId=4, RA_PID=4
0x9924  TA command=0, Temporary C-RNTI=121, RarUlGrant=284
```

- `Cfg/Int/Ack` 是 RAR PDSCH 配置、硬件中断、CRC 正确的闭环；只看 RA-RNTI DCI 不足以证明 RAR 数据解对。
- `RA_PID match` 表示 RAR 中的 RAPID 与本 UE 发的 `PreambleID` 相同，避免把同一 RA-RNTI 窗口中别的 UE 的 RAR 当成自己的。
- RAR `TA command` 为 11 bit 初始 TA 命令。`0` 表示本例不增加初始时间提前量；初始 TA 的换算与后续 6-bit TA MAC CE 不同。
- `Temporary C-RNTI=121` 是 Msg3/竞争解决期间使用的临时标识。
- `RarUlGrant=284` 是原始 20-bit UL Grant 的厂商十进制表示，下一步由 ULS 解码。

#### 25.3.4 RAR UL Grant、Msg3 与 PHICH

```text
0x995C  ConResoTimer=48, MaxHarqMsg3Tx=4, RarUlGrant=284,
        bContensionFlag=1, wDataValid=1
0x7B1D  RARINFO: HopFlag=0,RIV=0,MCS=8,TPC=7,
        UlDelay=0,CQIFlag=0,sf=8839
0x7B47  MSG3 LTX Start Frame=884 SubFrm=3
0x7A07  ABS Msg3 PUSCH SubFrame=8845
0x7B07  RntiType=2,... HarqTransType=1,Msg3ID=1,MaxMsg3Tx=4
0x7B20  harqid=1 TBS=15
0x7B21  GrantType=2,... datasize=15, TransType=1
0x7806  Get PS DATA ... DataSendSize=15
0x780B  schedule PHICH at 8849
0x781C  Subfrm=5, tTxChannelType=2,eTXPuschType=1
0x7B09  Rec PHICH:value=1, AbsSF=8849, Harq Id=1
0x7B0D  DO not Trans. UlHarqTransCnt=1
```

RAR UL Grant 字段：

- `HopFlag=0`：Msg3 不使用 PUSCH 跳频。
- `RIV=0`：在当前 UL 带宽下解为 RB start 0、长度 1 RB。
- `MCS=8`：调制编码索引。该版本随后打印实际 Qm=2、TBS=120 bit，即 15 byte。
- RAR 的 3-bit TPC 与 DCI 0 的 2-bit TPC 不是同一张表。RAR TPC 0..7 通常对应 -6、-4、-2、0、+2、+4、+6、+8 dB；本例 `TPC=7`，相邻 `0x6C1B` 也打印 `Tpc Value=8`。
- `UlDelay=0`：不要求延迟 PUSCH；`CQIFlag=0`：Msg3 不附带 aperiodic CQI。
- `ConResoTimer=48`：竞争解决定时器配置为 48 个子帧量级；`MaxHarqMsg3Tx=4`：Msg3 HARQ 最大次数 4。
- FDD 正常链路中 PUSCH 884.5 对应 PHICH 884.9，即 n+4。`value=1` 在本版本为 ACK，因此不再传；`value=0` 为 NACK。

#### 25.3.5 Msg4 与随机接入完成

```text
0x7B42  MSG3 Contension Start SubFrm=8846
0x9807  Contention Resolution Window start at 884.6
0x9808  Enable T-C-RNTI
0x9818  Recv ... MSG4_PDU_DETECT_EV at 888.4
0x981A  Need to response Ack, Set Msg4 Ack Flag TRUE
0x995E  Delete contention stop event
0x995F  Disable T-C-RNTI
0x981C  ABORT_ACCESS_REQ ... Random Access Success
0x9940  Random Access result: Content RA Success
```

`0x981A` 中的 Ack 是对下行 Msg4 的 HARQ-ACK，实际会走 PUCCH/PUSCH UCI；它与 PHICH 对 Msg3 的 ACK 是相反方向的两类反馈，不能混为一谈。

#### 25.3.6 Msg1/Msg2 失败、功率爬升和最大次数

```text
0x9805  RAR Detect stop ... failed to receive RAR PDU
0x9806  Disable RA-RNTI
0x9954  wPreamTransCounter=7
0x9942  Random Access result: Preamble Trans Counter Exceed Maximum
0x9955  wPreamTransCounter reach max
```

本文件共有 54 次 RAR 窗启动、50 次未收到 RAR、3 次收到并成功；有 1 次明确达到前导最大次数。失败时按以下顺序定界：

1. 看 `PreambleID/NPrbRa/PrachPower` 是否生成；没有则 UE 内部配置问题。
2. 看 ULA/RFC 是否真正 `TxEn=1` 且发射完成；只有 RAPC 配置不等于上天线。
3. 看 MT8000A 是否检测到 PRACH、RAPID 是否一致。
4. 仪表已发 RAR 时，查 UE `DciValid bit7`、`RarDecode Cfg/Int/Ack` 和 `RA_PID match`。
5. 路径损耗升到 152 dB、PRACH 功率已达 23 dBm仍无 RAR，优先查线损、仪表 UL 接收电平、频率/时序/PRACH 配置，不要继续提高功率。

### 25.4 SR、BSR 与 DCI 0

#### 25.4.1 SR 的正向证据

典型 SR 流程位于 `1102.903.3`：

```text
0x7B63  eTXPucchType=0,ePucchFormat=0,wAckToSendLen=0
0x782E  dwPucchHarqAckLen=0,dwPucchHarqAckValue=0
0x7811  wN1Pucch=41,...
0x780D  SR Send at sf=9033, SR send counter=1,... sr_index=30
0x7812  PUCCH RB Slot0=3, Slot1=71
0x781C  Subfrm=5, ChanType=7
0x7738  eChannelType=6,... LtxStatus=0, RegCfgErr=0
```

本版本上下文映射：

- `ePucchFormat=0` 与协议 PUCCH Format 1（SR）对应；它是厂商从 0 开始的格式枚举，不是“没有 PUCCH”。
- `wN1Pucch=41` 在 SR 场景很可能对应 `sr-PUCCH-ResourceIndex`。
- `sr_index=30` 很可能是 `sr-ConfigIndex I_SR`。FDD `I_SR=30` 对应 20 ms 周期中的一个偏移；日志在当前子帧提前配置，实际 `LtxConfigure Subfrm=5`，所以不能直接用打印处 `sf=9033` 的末位做模运算。
- `SR send counter` 本文件只出现 1 和 2：17 次为 1，1 次为 2；说明确实有一次 SR 重发，但没有达到上限。
- `RegCfgErr=0/CddValueErr=0` 是硬件配置无错；`LtxStatus` 是私有完成状态，本文件 0 为常态，不能单凭 0/1 宣称网络已收到。

#### 25.4.2 BSR 在 PHY 能看到什么

BSR 是 MAC Control Element，不是独立物理信道。本文件 ULA/ULS/RAPC/LPC 没有任何明确 `BSR/Buffer Status Report` 日志。PHY 通常只能看到：

```text
SR 触发/发送
-> DCI 0 UL Grant
-> ReportUlGrant datasize/TBS
-> ULA Get PS DATA
-> PUSCH 发射
```

这条链只能证明“有上行 MAC PDU 被发出”，不能证明该 PDU 内一定包含哪一种 BSR，也不能读出 LCG buffer size。BSR 类型（Regular/Periodic/Padding）、LCG ID 和索引值必须查 MAC/PS 日志或 MT8000A MAC 解码。

#### 25.4.3 DCI Format 0 逐字段

典型动态 Grant：

```text
0x7B1C DCI0 Decoded:
  wCurCCNum=0,wDci0Length=28,HarqID=7,CIF=0,
  HopFlag=0,HopInfo=0,RBstart0=0,Lcrb0=75,
  MCS=20,NDI=0,TPC=2,DMRS=5,
  cUlIndex=0,cDAI=0,CSI=0,SrsReq=0,Ratype=0,
  aldwDci0=0x002568a8,Comm=0
```

| 字段 | 协议含义 | 本日志如何判断 |
|---|---|---|
| `wCurCCNum` | 被调度 UL serving cell/内部 CC | 0 为 PCell；CA 时需结合 CIF/Scell 配置 |
| `wDci0Length` | DCI payload/内部对齐长度 | 27/28 与带宽、CIF/版本配置有关，不能跨配置硬比 |
| `HarqID` | UL HARQ 进程 | FDD 常见 0..7；相同进程用 NDI 判断新传/重传 |
| `CIF` | Carrier Indicator Field | 0 常指本载波；启用跨载波调度时按 RRC CIF 映射 |
| `HopFlag/HopInfo` | PUSCH 跳频开关/信息 | 0 表示不跳；1 时必须按 hopping 配置解释 RB |
| `RBstart0/Lcrb0` | 解码后的起始 RB/连续 RB 数 | 起点+长度不得越过 UL 带宽；RA type 1 还需满足其映射规则 |
| `MCS` | 调制编码索引 | 映射取决于配置的 PUSCH MCS 表；本文件 8/11/20 对应内部输出 Qm 2/4/6 |
| `NDI` | New Data Indicator | 同一 HARQ ID 发生翻转通常表示新传；不翻转通常表示重传 |
| `TPC` | PUSCH 功控命令 | 累积模式下 0/1/2/3 常映射 -1/0/+1/+3 dB；绝对模式映射不同 |
| `DMRS` | DM-RS cyclic shift/OCC 相关域 | 影响 PUSCH DM-RS 和 PHICH 资源推导；不是“DMRS是否存在” |
| `cUlIndex` | TDD UL index | FDD 通常为 0 |
| `cDAI` | TDD DAI | FDD 通常为 0 |
| `CSI` | Aperiodic CSI request | 0 不请求；1 请求。请求不等于最终已带 CSI 发射 |
| `SrsReq` | Aperiodic SRS request | 0 未触发；非零值需结合版本和 SRS trigger set |
| `Ratype` | UL resource allocation type | 0/1 的具体映射依版本/配置；type 1 需走对应 RIV/cluster 解码 |
| `aldwDci0` | 原始 DCI bit word | 用于和仪表 bit 级对齐；字节序/补零是厂商实现 |
| `Comm` | 厂商内部公共/专用标志 | 非 3GPP 字段，待源码确认 |

#### 25.4.4 DCI 0 两条明确异常

```text
0x7B39 invalid DCI0 Grant: RB!! RIV=500,RB=32,RbStart=28
0x7B06 Invalid, Not toggled, but over the harq length, NDI_STATE=0 ...
```

- `0x7B39` 表示本地 Grant 合法性检查失败，随后没有正常 `0x7B1C -> 0x7B20 -> PUSCH` 闭环。要核对当时 CC 的 UL BW、RA type、CIF 和仪表 bit 打包；不要只做 `28+32<75` 的简单算术，因为 type 1/实际 serving cell 映射可能还有约束。
- `0x7B06` 表示 HARQ 状态已越过有效长度但 NDI 未按预期翻转。该次代码仍继续建立新传，因此它是状态异常告警，不是单条即“PUSCH没发”；需要看随后是否有 `ReportUlGrant/Get PS DATA/LtxConfigure/PHICH`。

### 25.5 PUCCH：SR、HARQ-ACK、资源与功率

#### 25.5.1 本版本 PUCCH 格式枚举

由 `wAckToSendLen` 和实际用途交叉验证：

| `ePucchFormat` | 观察到的 `wAckToSendLen` | 本版本最可能对应协议格式 | 用途 |
|---:|---:|---|---|
| 0 | 0 | Format 1 | SR |
| 1 | 1 | Format 1a | 1 bit HARQ-ACK |
| 2 | 2 | Format 1b | 2 bit HARQ-ACK |

协议还定义 Format 2/2a/2b（CSI，某些情形叠加 ACK）以及 Format 3（CA 多 bit ACK）。本文件没有足够正样本建立这些格式与厂商枚举的映射。

#### 25.5.2 下行 HARQ-ACK 的典型链

```text
0x7823  wAckToSendLen=1, bPCellOnly=1
0x7B63  eTXPucchType=2,ePucchFormat=1,wAckToSendLen=1
0x782D  awn1pucch ..., acAckValue1,... acAckValid 1,...
0x782E  dwPucchHarqAckLen=1,dwPucchHarqAckValue=1
0x7838  PUCCH format=1, PucchRLength=12
0x7811  wN1Pucch=4,wNcsAn=6,wDeltaPucchShift=2
0x7812  RB Slot0=2, Slot1=72
0x783B  dwPucchScale1/2=...
0x7738  eChannelType=6, RegCfgErr=0
```

- `dwPucchHarqAckLen` 0/1/2 是 ACK bit 数；`Value` 是打包 bit。此版本值 1 与 ACK 对齐，0 可为 NACK，但必须同时确认 `AckValid`，否则 0 也可能只是无有效 bit。
- `awn1pucch[]/acAckValue[]/acAckValid[]` 是载波聚合/信道选择候选资源数组；没有源码时不要把数组每一项机械映射到某个 SCell。
- `bPCellOnly=1` 表示本次 ACK 集合只有 PCell；CA 多载波时格式和资源选择会改变。
- `wN1Pucch` 是最终 PUCCH n1 资源索引；ACK 场景可由下行 DCI 的 CCE 位置和 `n1PUCCH-AN` 推导，SR 场景来自专用 SR 资源。
- `RB Slot0/Slot1` 显示 PUCCH 在两个 slot 间跳到系统带宽两端。75 RB 带宽中本文件常见 2/72、3/71 及反向顺序。
- `dwPucchScale*` 是基带幅度字，不是 dBm；只能同平台同版本对比。

#### 25.5.3 PUCCH 与 SRS 的 shortened format

```text
0x7923 Set shorten format for PUCCH:
  PucchType=9,format=2,subframe=1,
  AckNackSrsSimulTrans=...,PcellSrsCellSpecSfFlg=1
```

当该子帧是 cell-specific SRS 子帧时，PUCCH 可能采用 shortened format，为末符号留出 SRS/保护。`PcellSrsCellSpecSfFlg=1` 是重要上下文。这里 `PucchType=9`、`AckNackSrsSimulTrans=0..9` 显然不是协议布尔值，属于厂商位图/枚举，不能按字面把 3、5、9解释为“允许”。

#### 25.5.4 PUCCH 功率

协议概念公式：

```text
P_PUCCH = min(P_CMAX,
              P0_PUCCH + PL + h(nCQI,nHARQ,nSR)
              + DeltaF_PUCCH + DeltaTxD + g(i))
```

对应日志：

```text
0x6C13  PUCCH TPC source/index/value
0x6C1E  Gi, PucchTpcPositive/Negative
0x6D01  Current has Pucch Trans: Pcmax, PucchPow[...], ...
0x6D02  Current no Pucch Trans: ...
0x781A  RFC final power configuration
```

本文件中可稳妥读取的是 `Pcmax`、`PucchPow[第1链]` 和相邻 `RfcConfigure` 最终功率。`0x6D01` 后半段出现 `bPucchPowReachMax=91` 这种不可能的布尔值，证明字段错位，不能据此说“达到最大功率”。真正判断顶功率应同时满足：计算功率接近 `Pcmax`、正确版本的 reach-max flag、仪表接收电平/波形异常。

`0x7846 ... Tpc:-1` 虽然打印级别含 `Err`，-1 dB 本身是合法的累积 TPC 步进，且本文件出现 26 次仍能正常发 PUCCH；不能单条判故障。

### 25.6 PUSCH：Grant、调制编码、发射与 HARQ

#### 25.6.1 从 DCI 0 到 PUSCH 上天线

```text
0x7B1C  DCI0 Decoded ... RBstart/Lcrb/MCS/NDI/TPC/DMRS
0x6C17  process PUSCH TPC
0x7A12  wUlHarqId
0x7B07  HARQEntity ... HarqTransType
0x7B38  code scheme / Qm / TBS（注意标签错位）
0x7B69  是否把 HARQ-ACK 复用到 PUSCH
0x7B62  eTXPuschType, eTXPucchType, ePuschSend
0x6C09/0x6C28/0x6C0B/0x6C0F  MPR/A-MPR/P_CMAX
0x6D86  PUSCH power calculation
0x7B20/0x7B21/0x7B22  上报 Grant/TBS 给 PS
0x7806  ULA 从 PS 取得 TB 数据
0x780B  排程 PHICH 资源和时刻
0x781C  配置 PUSCH/UCI 到 LTX
0x781A  配置 RF 发射功率
0x7738  发射完成/硬件错误状态
```

任何一步断链都有不同结论：

- 有 DCI 0、无 `0x7B20`：Grant/RIV/HARQ 校验失败。
- 有 `0x7B20`、无 `0x7806`：PS/MAC 未及时供数或接口问题。
- `0x7729 Has no PS data`：明确的新传取数失败；本文件仅 1 次。
- 有数据、无 `LtxConfigure/TxEn`：ULA/RF 发射配置问题。
- UE 完成发射但仪表 DTX：查 TA、功率、频率、RB/DMRS、线损。

#### 25.6.2 GrantType 与传输类型

`0x7B21` 已在日志内给出 GrantType 枚举：

| 值 | 日志定义 | 本文件实例 |
|---:|---|---|
| 0 | None | 无 |
| 1 | RAR content | 无正样本，含义待源码确认 |
| 2 | RAR nonConten | Msg3/RAR Grant，15 byte |
| 3 | SPS | 无 |
| 4 | Dynamic | DCI 0 动态调度 |
| 5 | Configed | 无，可能指 configured grant/内部预配置 |

`TransType` 在 `0x7A11/0x7B22` 中明确：0 NONE、1 NEW、2 Adaptive retransmission、3 Non-adaptive retransmission。`255` 是未使用 TB/字段无效哨兵。

#### 25.6.3 `0x7B38` 的正确读法

三个可验证例子：

| DCI/Grant | `0x7B38` 原打印 | `0x7B20` | 校验 |
|---|---|---:|---|
| RAR MCS 8, 1 RB | `mcs:0,Qm:8,TBS:2,CurIRV:120` | 15 B | 120/8=15 |
| DCI MCS 11 | `mcs:0,Qm:11,TBS:4,CurIRV:12960` | 1620 B | 12960/8=1620 |
| DCI MCS 20 | `mcs:0,Qm:20,TBS:6,CurIRV:48936` | 6117 B | 48936/8=6117 |

所以在本固件中：

- 标签 `Qm` 后的值实际是 MCS；
- 标签 `TBS` 后的 2/4/6 实际是调制阶数 Qm；
- 标签 `CurIRV` 后的是 TBS bit；
- `mcs:0` 更像 CC/内部索引；
- `NextRiv` 很可能实际是 RV/下一冗余版本，但本文件仅见 0，不能定论。

这是自动化解析器必须加入的版本适配，不能按字段名直接入库。

#### 25.6.4 UCI 复用到 PUSCH

```text
0x7B69  pbHasHarqAck=1
0x7B62  eTXPuschType=3,... eTXPucchType=0,
        bHasHarqAck=1,ePuschSend=1
0x781C  ChanType=6,tTxChannelType=2,eTXPuschType=1/3,...
```

本版本观察上，`eTXPuschType=1` 是普通 PUSCH，`=3` 与“PUSCH 携带 HARQ-ACK”相关；这是厂商枚举推断。若 `bSimultaneousPucchPuschFlag=0`，同子帧有 PUSCH 时 UCI 通常复用在 PUSCH，而不是另发 PUCCH。

#### 25.6.5 FDD UL HARQ 与 PHICH

本文件唯一 NACK 闭环：

```text
PUSCH newTx:   wAbsSubFrm=8640, harqId=4
PHICH:         AbsSF=8644, value=0
0x7B0C:        Non-AdaptReTrans, HarqId=4, count=1
PUSCH reTx:    wAbsSubFrm=8648, TransType=3
PHICH:         AbsSF=8652, value=1
0x7B0D:        DO not Trans, count=2
```

规律：FDD PUSCH n 在 n+4 接收 PHICH；NACK 后无新 DCI 0 时，n+8 使用相同 HARQ 进程和资源做非自适应重传。若重传前收到新 DCI 0 并更改资源/MCS，则是自适应重传。

`0x780B` 中：

- `awIPrbLowestIdx`：PUSCH 最低 PRB；
- `awNDmrs`：DM-RS cyclic shift 派生量；
- 二者参与 PHICH group/sequence 资源计算；
- `wABSPhichSubFrmNo`：期望 PHICH 的绝对子帧；
- `wValidNum`：有效 TB/资源槽数，私有数组结构。

本文件 `0x7B09` 共 28 条：27 ACK、1 NACK；NACK 后成功非自适应重传。注意 UE 的 PHICH 日志证明“UE解到的反馈”，MT8000A UL HARQ CSV 才直接反映仪表是否正确解码 PUSCH。

### 25.7 CSI/UCI

#### 25.7.1 Aperiodic CSI request

```text
0x7B60 Aperiodic CQI request bit configured:
  PdcchSf=6,Kvalue=4,PuschSf=0
0x7B1C ... CSI=1 ...
```

`CSI=1` 表示 DCI 0 请求 aperiodic CSI，通常随相应 PUSCH 上报。`Kvalue=4` 是本实现从 PDCCH 调度时刻到 UL 处理/目标时刻的偏移参数；具体以 FDD/TDD 调度关系为准。

但是，本文件 15,054 条 `0x781C` 中 `CqiLen/QCQILen/RiLen` 全部为 0，`0x7B62 eCurCsiType` 也全为 0。故只能确认“收到过请求”，不能用本日志证明 CSI bit 已生成并上天线。尤其第二个 CSI 请求紧接 `0x7B39 invalid DCI0 Grant`，明确没有正常 PUSCH 闭环。

#### 25.7.2 UCI 字段

`0x781C` 可能出现：

- `dwAckLen/dwOAckInfo`：HARQ-ACK bit 长度/打包值；
- `CqiLen/QCQILen`：CQI/PMI 编码前后或原始/编码 bit 长度，具体命名待源码；
- `RiLen/dwORiInfo/dwQRiInfo`：RI 长度和原始/编码信息；
- `wQACK/wQACKSymb`：PUSCH 上 ACK 编码资源/符号数相关内部参数。

本文件这些字段全部为 0，不能建立非零值字典。分析其他日志时，应把它们与 `CSI request`、CSI 模块输出和仪表 UCI 解码三方对齐，而不是只看一个长度。

### 25.8 SRS

#### 25.8.1 Cell-specific SRS 背景配置

```text
0x791B  SRS Para Calculating in Cell 0 Start
0x7902  Active Subframe:[1 1 1 1 1 1 1 1 1 1]
        RatMode=1,SrsSubframeConfig=0
0x7903  RbStart=7,RbEnd=66,
        Msrsb[60 20 4 4],Nb[1 3 5 1],BandWidth=75,Csrs=2
```

- `SrsSubframeConfig=0` 是 cell-specific SRS 子帧配置索引。本 FDD 样本计算出 10 个子帧位置全部 active。
- `Csrs=2` 对应 cell-specific SRS bandwidth configuration `C_SRS`。
- `Msrsb/Nb` 是不同 SRS bandwidth level 的带宽/分支数查表结果；本例 75 RB 系统带宽、最宽 SRS 区域约覆盖 RB 7..66。
- 这些只证明“小区允许 SRS 的区域/时机”，不证明本 UE 已配置并实际发送 user-specific SRS。

#### 25.8.2 子帧预留与冲突处理

```text
0x791F LastSymbol is reserved for SRS Configuration ...
       ScheduleLocation=0[0:Pusch 1:other]
0x7922 SRS CellState is Cancel ...
       SrsRb[7 66] PuschRB[...] 
0x7923 Set shorten format for PUCCH ... PcellSrsCellSpecSfFlg=1
```

- `ScheduleLocation=0` 表示当前调度对象是 PUSCH，`=1` 是其他路径；这是日志自带说明。
- `LastSymbol is reserved` 表示按 cell-specific 配置处理最后一个 SC-FDMA symbol，不等价于 user-specific SRS 已发。
- `0x7922` 文案“Cancel because no resource conflicted”语义矛盾，只能确认执行了 PUSCH/SRS 资源冲突判定；SRS 是否取消需源码或实际 TX channel/仪表结果。
- `SrsReq=0` 在本文件所有 DCI 0 中都为 0，未发现 aperiodic SRS 正触发。

完整证明 SRS 发射至少需要：RRC user-specific `srs-ConfigIndex/B_SRS/b_hop/n_RRC/transmissionComb/cyclicShift` 配置、正确 SRS opportunity、ULA 非取消/实际 TX 配置，以及 MT8000A 检测结果。本文件缺少后三者中的明确正证据。

### 25.9 上行功率、P_CMAX、MPR/AMPR 与 PHR

#### 25.9.1 P_CMAX 链

```text
0x6C03  Band/UL EARFCN -> DeltaTc
0x6C09  SingleCarrierMprDeterm: Mpr, BW, modulation, RB
0x6C28  NS_1/Band/BW/RB -> AMPR
0x6C0B  additionalSpectrumEmission/Band -> AMPR
0x6C0F  Pcmax, MPR, AMPR, Pemax, Powerclass
```

协议含义：`P_CMAX` 受 UE power class、网络 `p-Max/P_EMAX`、MPR、A-MPR 和 band-edge `DeltaT_C` 限制。本例 power class/Pemax 常为 23 dBm；满带宽高调制时 `P_CMAX` 降到 18~22 dBm。

本版本 `wMpr` 很可能是 0.25 dB 定点：例如 `wMpr=12` 与约 3 dB 降额、`wMpr=18` 与约 4.5 dB 降额相符，但应以平台源码确认。不要把 12 直接当 12 dB。

#### 25.9.2 PUSCH 功率

协议概念公式：

```text
P_PUSCH = min(P_CMAX,
              10log10(M_PUSCH) + P0_PUSCH + alpha*PL
              + Delta_TF + f(i))
```

日志对应：`0x6D86` 的 `Pcmax/LogMpusch/DeltaTF/PathLoss/Alpha/Popusch/Fi/PuschPowVal`。本版本 `Alpha=128` 很可能表示 1.0，`Alpha=102` 约表示 0.8（Q7 定点）。`Fi_Pusch` 是闭环 TPC 累加状态。

但 `0x6D86` 后半段也存在字段错位，`reachMaxPow` 出现 14 等非布尔值；因此自动化应优先使用 `PuschPowVal`、`0x6D8B After adjust` 和 `0x781A RfcConfigure` 三条互相校验。

#### 25.9.3 TPC 与闭环累加

- `0x6C17`：DCI 0 或 DCI 3/3A 的 PUSCH TPC 源；本文件 DCI 0 TPC 0..3 均出现。
- `0x6C1D`：`PuschTpc` 转为 dB 步进后更新 `Fi_Pusch`；本文件看到 -1、+1、+3。
- `0x6C13`：下行 DCI 或 DCI 3/3A 给 PUCCH 的 TPC；`-1` 哨兵表示该来源无效，另一个来源可能有效。
- `0x6C1E`：更新 PUCCH accumulator `Gi`。
- `0x6C1B`：RAR TPC；本例 index 7 -> +8 dB，初始化 `Fi/Gi=8`。

#### 25.9.4 PHR

```text
0x6C20 Type1PhrCalc: swType1PhrVal=31..49
```

`swType1PhrVal` 很可能是 MAC PHR 6-bit 编码值，而不是直接 dB。常用映射中中间区间可近似理解为 `PH[dB] = index - 23`，边界值有饱和区；最终应与 MAC PHR CE 的编码表核对。低 PHR/持续顶功率通常意味着上行覆盖受限，但本文件 31..49 没有直接显示负余量。

### 25.10 TA：时间对齐状态

相关日志：

```text
0x7A0D  TA timer is not running
0x7B51  TA stop, wTagId=0
CMN     ProTASchedFlow: TACmdType,... wTimeAdvance
RAR     TA command=...
```

判读：

- RAR TA 是初始 11-bit TA；后续 TA MAC CE 是相对调整，LTE 常以 `(TA-31)*16*T_s` 更新，值 31 表示不变。
- `wTagId=0` 是 PCell 所属 Timing Advance Group；CA 可有多个 TAG。
- `TA timer is not running` 本文件有 1,099 条，主要在随机接入/重新接入窗口按子帧重复打印。初始接入尚未建立 TA timer 时它可以是预期现象，不能看到一条就定故障。
- 若已处于稳定 RRC_CONNECTED 且需要发 PUCCH/PUSCH时仍持续出现，同时仪表报 UL DTX，才高度怀疑 timeAlignmentTimer 未启动/已超时、TA MAC CE 未处理或脚本 TA 配置问题。
- TA timer 失效后 MAC 会停止相关上行传输并可能重新随机接入；最终原因仍需 MAC/PS 日志确认。

### 25.11 PHICH 专项

PHICH 是 **下行** 信道，但反馈的是 **上行 PUSCH**：

| 日志值 | 本版本含义 | 后续动作 |
|---:|---|---|
| `value=1` | ACK | `0x7B0D DO not Trans` |
| `value=0` | NACK | `0x7B0C Non-AdaptReTrans`，或等待新 DCI 做 adaptive reTx |

分析闭环：

```text
DCI0/RAR Grant
-> 0x7A11 schd_pusch=n, schd_phich=n+4
-> 0x780B IPrbLowest/nDMRS/PHICH resource
-> ULA PUSCH TX complete
-> DLA PDCCH/PHICH decode context
-> 0x7B09 value/HarqId/AbsSF
-> 0x7B0D stop or 0x7B0C retransmit
```

PHICH 与下行 PDSCH 的 PUCCH ACK 是两个相反方向的反馈：

- PUSCH -> eNB/MT8000A 解码 -> PHICH ACK/NACK -> UE；
- PDSCH -> UE 解码 -> PUCCH/PUSCH HARQ-ACK -> eNB/MT8000A。

因此 MT8000A `UL_HARQ ACK/NACK/DTX`、UE `Rec PHICH value` 和 UE 下一次 PUSCH 动作需要三方一致。若仪表判 ACK 而 UE读 NACK，查 PHICH资源/下行解码；若仪表判 DTX，先查 UE是否真正发PUSCH以及TA/功率。

### 25.12 性能与峰值分析

`0x7B6B StdLoglInfoUl` 是本组最有价值的周期统计：

```text
CC[0],AvgPHY.Kbps=1146,avgTB.byte=6117,
trans:[newTx:6,AdReTx:0,NadReTx:0,HqFai:0],
aveRbNum*10=750,MCS*10=[200,0],UlBLER*10:0,
SR[trig:1;reTx:1;Fail:0],
RACH:[trig:0;succ:0;msg12Fail:0;msg34Fail:0]
```

字段：

- `AvgPHY.Kbps`：统计窗内 PHY 层平均吞吐，不一定等于应用层吞吐。
- `avgTB.byte`：平均 TB 字节数。
- `newTx/AdReTx/NadReTx/HqFai`：新传、自适应重传、非自适应重传、HARQ 最终失败。
- `aveRbNum*10=750` 表示平均 75.0 RB；`MCS*10=200` 表示 MCS 20.0。
- `UlBLER*10=1000` 在该实现可读作 100.0%；本文件有一个小统计窗因一次 NACK/重传显示 1000。
- `SR trig/reTx/Fail`：SR 触发、重发、失败聚合计数。
- `RACH trig/succ/msg12Fail/msg34Fail`：随机接入阶段统计。

PUSCH 瞬时/峰值的可靠算法：

```text
每个成功 UL TB 的有效 PHY bit = 0x7B20 TBS(bytes) * 8
统计窗速率 = Σ(最终由 PHICH ACK 的新传 TB bit) / 窗长
```

不要把重传 TBS 再算为新有效数据。脚本峰值排查还要看：RB 是否满带宽、MCS/Qm 是否达到目标、DCI 是否连续、PUSCH 是否带 UCI/SRS 导致 RE 损失、PHICH NACK率、P_CMAX/MPR 是否限功率。

### 25.13 “SR 超限”在 PHY LOG 能看到什么、不能看到什么

#### 25.13.1 能看到

1. `0x780D SR Send`：PHY 已为 SR 配置 PUCCH，含 `counter/sr_index/时刻`。
2. `0x7B63/0x7811/0x7812/0x7738`：PUCCH 格式、资源映射、发射硬件结果。
3. `0x7B6B SR[trig,reTx,Fail]`：本固件的周期统计。本文件出现 `reTx=1`，未出现 `Fail>0`。
4. SR 之后是否出现 DCI 0、UL Grant 和 PUSCH。
5. SR 多次无 Grant 后是否进入 RAPC/PRACH，可作为“可能 SR 失败”的跨模块旁证。

#### 25.13.2 不能仅靠 PHY 确认

`SR_COUNTER >= dsr-TransMax` 是 MAC 状态机条件。`dsr-TransMax` 来自 RRC/MAC 配置，常见枚举 n4/n8/n16/n32/n64；MAC 达到上限后通知 RRC、释放 SR 相关资源并启动随机接入。PHY 的 `SR send counter` 可能只是本地发射计数，复位时机和 MAC `SR_COUNTER` 不一定相同。

因此，以下结论都不充分：

- 只看到连续 PUCCH，就说 SR 超限；它可能是 HARQ-ACK/CQI。
- 只看到 `Nsr`，就当 SR 次数；功控公式里的 `nSR` 是 UCI bit/功控参数。
- 只看到 `bPucchPowReachMax`，就说 SR 超限；功率顶格和次数上限是两件事，而且本版本该字段还存在错位。
- 只看到随后 RACH，就断定一定是 SR 超限；RLF、TA 失效、PDCCH order、切换也能触发 RACH。

最终闭环应为：

```text
MAC: SR pending -> SR_COUNTER 递增 -> 达到 dsr-TransMax
PHY: 对应次数的 0x780D + PUCCH TX
仪表: SR detect/DTX，是否下发 DCI0
MAC/RRC: SR failure cause -> initiate RA
RAPC: 0x9812 ACCESS_REQ，原因与时间吻合
```

本文件结论：18 条 `0x780D` 中 counter 最大仅 2；`0x7B6B` 的 `SR Fail` 全为 0。因此没有 PHY 正证据证明 SR 超限。

### 25.14 LPC：为什么也要看，但不能当空口结论

- `0x6602 LTE Dvfs Req`：CPU/AXI/DDR/PS/L2 频率请求与结果。它用于判断处理器是否降频、资源不足或唤醒异常；不是 LTE 协议字段。此文件 14 条结果均维持 CPU 1600、AXI 800，未显示调频失败。
- `0x8F97 E_ACCESS_T300_TIMER`：为接入过程设置保持唤醒时间。`dwStayAwakeTime=4020` 是厂商睡眠调度值，不能直接等同 RRC T300=4020 ms。若接入过程中 RF/CPU误睡，可用它与 RAPC 起点对齐。

### 25.15 自动化挂测的上行固定分析顺序

1. 锁定首个异常帧和最后一个正常帧。
2. 看 `0x7B6B`：SR/RACH/BLER/重传是否先异常。
3. 无 UL Grant：查 SR `0x780D -> PUCCH资源 -> TX完成 -> MT8000A检测 -> DCI0`。
4. 有 DCI0：查 `0x7B1C` 合法性、`0x7B07` HARQ状态、`0x7B20` TBS。
5. 有 Grant：查 `0x7806` 是否取到数据；`0x7729` 是否无数据。
6. 有数据：查功率链、`0x781C/0x781A/0x7738` 是否真正发射。
7. 查 MT8000A UL HARQ；再对齐 `0x7B09 PHICH` 和重传动作。
8. 同时检查 TA `0x7A0D/0x7B51`、SRS冲突、UCI复用和 P_CMAX/MPR。
9. 若转 RACH，按 Msg1/2/3/4 四段定位，而不是笼统写“随机接入失败”。
10. 结论必须写清：请求是否到、资源是否有、UE是否发、仪表是否收、反馈是否到、下一步状态机是否正确。

## 26. 搜索、同步、射频与测量逐项字典（CSRC / CSRS / CSRM / MULM / MC / CMN / RFC / DFE / 接口与自定义，417 个消息 ID）

本章覆盖其余 11 个模块共 **417 个 message_id**，占全文件 93.9% 的行数。其中 §26.2 的 7 个编码约定是读懂这一组日志的前提，§26.13 给出由全文件锚点还原的完整会话时间线。

> 数据源：`loglte_phich.txt`（5,865,467 数据行，SHA-256 `4f3989a4…46e896`，平台 `MC86001`，固件 `version=.0.6.6`）。索引：`generated_lte_log_index/`。
>
> 本章覆盖 11 个模块、**417 个 message_id**，与第 24 章（56 个）、第 25 章（138 个）合计 611 个，等于全文件唯一 `模块+消息ID` 总数。

### 26.1 覆盖范围

| 模块 | 行数 | ID 数 | 占全文件 | 本章定位 |
|---|---:|---:|---:|---|
| `RFC` | 3,341,955 | 72 | 57.0% | RFIC/RFFE 底层事件、收发通道与频点、收发功率、AFC 控制字、发射 TA 微调 |
| `DFE` | 1,136,797 | 58 | 19.4% | 数字前端：AGC/DAGC、DC/IQ、AFC 输入、FFT 窗、带宽切换 |
| `MC` | 588,078 | 41 | 10.0% | PHY 主控：低功耗场景、TPU 定时、SIB1/SI 状态机、寻呼、预同步、RLM |
| `CSRS` | 158,895 | 54 | 2.7% | PSS/SSS/CFO 的执行侧与硬件配置 |
| `CMN` | 102,664 | 65 | 1.8% | PHY 内部消息包络、公共/专用配置事务、RF 资源申请、开关 RF |
| `CSRM` | 73,806 | 21 | 1.3% | 测量执行：硬件测量窗、RSRP/RSRQ/RSSI 计算与平均 |
| `MULM` | 63,170 | 4 | 1.1% | 多频/gap 从流程协调 |
| `CSRC` | 42,820 | 91 | 0.7% | 小区搜索控制、频扫状态机、候选小区数据库、L3 过滤、SCell |
| `PS_PHY接口消息` | 436 | 2 | 0.01% | PHY↔协议栈消息包络 |
| `USER_DEF_VD_12` | 136 | 8 | 0.00% | 空闲态测量上报与脱网原因 |
| `USER_DEF_VD_2` | 54 | 1 | 0.00% | 详细 `DLS\|DCIInfo` 副本 |
| **合计** | **5,508,811** | **417** | **93.9%** | 全文件 94% 的行属于本组，但其中 87% 是每子帧/每时隙的周期性硬件打印 |

**第一条使用原则**：本组绝大多数行是周期性状态打印，逐条读没有意义；正确用法是**先按 message_id 定位到关心的环节，再在异常时刻的前后 ±2 个无线帧内横向对齐**。哪些 ID 值得逐条读、哪些只能看差分，本章每一节都给出结论。

### 26.2 读这组日志前必须掌握的 7 个编码约定

这一节的每条结论都在本文件中做了数值验证，验证行号写在括号里。不掌握这 7 条，后面所有数字都会读错。

#### 26.2.1 时间三元组 `[帧].[时隙].[Ts]`

`FB=0. 2.6272`、`CurrMrtr= 97.10.10572`、`PssAdjustBoundry= 0. 5.12800` 这类字段是**帧号 . 时隙号(0–19) . 时隙内 Ts 偏移**。

- `Ts = 1/30.72 MHz ≈ 32.552 ns`（3GPP TS 36.211 基本时间单位）。
- 1 时隙 = 15360 Ts = 0.5 ms；1 子帧 = 30720 Ts；1 无线帧 = 307200 Ts。
- 验证：`0x5B06` 打印 `FB=0. 2.6272` 且同一行 `CellFBTsBefAdj=36992`，而 `2×15360+6272 = 36992`（L1574766）。`0x5809` 打印 `PssAdjustBoundry= 0. 5.12800( 89600Ts)`，`5×15360+12800 = 89600`（L1112902）。

因此“两个小区边界差多少”要换算成 Ts 再比较，不能直接比字符串。

#### 26.2.2 `CurTime / SchedPos / SibPos = SFN×10 + 子帧号`

`0x9E12/0x9E13/0x9E11/0x9E09/0x9E10` 的时间字段是这种紧凑编码：`2445` = SFN 244、子帧 5。

- 验证：`0x9E12` 的窗口起点依次为 2445、2525、2545（L1626/L4366/L5420），SFN 全部是偶数、子帧全部是 5，正好落在 3GPP TS 36.331 规定的 **SIB1 传输位置：`SFN mod 2 = 0` 的子帧 #5**。
- 同一组打印里 `CurSiPeriod = 2` 的单位是**无线帧**，2 帧 = 20 ms，正是 SIB1 的重复周期（80 ms 周期内 4 次重复，RV 不同）。全文件 335 条中 333 条是 2，另有 16 和 32 各 1 条（真正的 SI 消息周期，对应 si-Periodicity rf16/rf32）。

#### 26.2.3 频点有两种单位，混用会差 1000 倍

| 打印族 | 字段 | 单位 | 例子 |
|---|---|---|---|
| `CSRC`/`CSRS`/`MC`/`CSRM` | `Freq`、`FreqPoint`、`CurFreq` | **100 kHz** | `18500` = 1850.0 MHz |
| `RFC`/`DFE` | `Freq`、`Freq100Hz`、`Freq100Khz` 里的大数 | **100 Hz** | `18500000` = 1850.0 MHz |
| 全部 | `Earfcn`、`dwEarfcn` | E-UTRA 绝对频点号 | `1650` |

EARFCN → 频率按 TS 36.101：`F_DL = F_DL_low + 0.1×(N_DL − N_Offs-DL)`。本文件的实测对应关系全部成立：

| EARFCN | Band | 计算 | 日志频点 | 出处 |
|---|---|---|---|---|
| 1650 | 3 | 1805 + 0.1×(1650−1200) = 1850.0 MHz | `18500` | L169 `0x9E0D` |
| 1275 | 3 | 1805 + 0.1×(1275−1200) = 1812.5 MHz | `18125` | L1566344 `0x5908` |
| 1900 | 3 | 1805 + 0.1×(1900−1200) = 1875.0 MHz | `18750` | `0x8583` |
| 3201 | 7 | 2620 + 0.1×(3201−2750) = 2665.1 MHz | `26651` | L1109594 `0x5806` |
| 6225 | 20 | 791 + 0.1×(6225−6150) = 798.5 MHz | `7985` | L1124708 `0x5806` |

上行同理：`0x702A TxCfg` 的 `Freq=17175000` = 1717.5 MHz，对应 band 3 上行 EARFCN 19275，与下行 1275 相差 95 MHz 的双工间隔，说明 UL/DL 配对正确。**排查“配错频点”时先做这一步换算，再去看仪表。**

#### 26.2.4 内部对数域：约 84 个单位 = 1 dB

`swRsrpLog`、`swAveRsrp`、`swRssiLog1`、`aswRsrq`、`swValLog2` 这类以 `Log` / `Log2` 结尾的量是 log2 域定点数。用同一时刻的成对打印拟合：

| 内部值（`0x68C6 swAveRsrp`） | 同行 dBm（`AveLogRsrp`） | 换算 `值/84` |
|---:|---:|---:|
| −6124 | −73 | −72.9 |
| −5452 | −65 | −64.9 |
| −6975 | −83 | −83.0 |
| −6631 | −79 | −78.9 |
| −7042 | −84 | −83.8 |

结论：`dB ≈ 内部值 / 84`（等价于每 dB 约 84 LSB 的 log2 定点）。同样适用于 RSRQ：`swAveRsrq = −470 → −5.6 dB`，同行 `AveLogRsrq = −6`。**这是推断值不是协议值**，换固件必须重新拟合；但它让你能把一堆五位数直接读成 dB。

注意区分两类 RSSI：`swRssiLog1`（已做 AGC 补偿，`/84` 就是 dBm）和 `swTempRssiLog`/`TempRssi`（未补偿的原始累加值）。`0x8676` 同时打印两者：`RSSI:[-60.9], TempRssi[-15124,0]`（L1576107），`-15124/84 = -180`，显然不是 dBm——原始值必须加上当时的 AGC 增益才有物理意义。

#### 26.2.5 Q8 与 ×1024 定点：AFC 全链

`0x5C16 AfcReq` 的每个字段都是 `[定点值, 取整值]` 成对打印：

| 字段 | 定点值 | 取整值 | 换算 |
|---|---:|---:|---|
| `OffsetHz` | 3776 | 14 | 3776/256 = 14.75 Hz（Q8） |
| `TmpPpm` | 2090 | 8 | 2090/256 = 8.16 |
| `CordPpm` | 4004 | 15 | 4004/256 = 15.64 |
| `CordHz` | 7233 | 28 | 7233/256 = 28.25 Hz |
| `CalcPpm` | −156764 | −612 | −156764/256 = −612.4（该字段是 **ppm×1024** 的 Q8，即 −0.598 ppm） |

同行 `TotalOffsetHz = −1106`，在 1850 MHz 上 −1106 Hz = −0.598 ppm，与 `CalcPpm` 完全自洽（L886）。**看见五六位数的 ppm 不要惊慌，先除 256，必要时再除 1024。**

#### 26.2.6 增益类字段的单位

| 字段 | 单位 | 本文件实测范围 |
|---|---|---|
| `AgcGain`（`0x6F1D`、`0x5C8E`、`0x5CA0`） | dB（模拟前端总增益） | 42 … 120 dB（321,664 个采样） |
| `AgcdBGain`（`0x5C8A/0x5C88/0x5C89`） | dB | 65（顶格）… 105（底格） |
| `RxDagcLin` / `DagcGainLin` | 线性定点（128 = 1×） | 102、128 |
| `MeanPwr` / `MaxAgcMeanPwr` / `Target` | 私有对数刻度 | Target ∈ {3280, 3430, 3620} |
| `TxPwr`（`0x6F36`） | **dBm，直接可读** | −24.0 … +23.0 dBm |

#### 26.2.7 RSRP / RSRQ 的“上报域”与“dBm 域”

3GPP TS 36.133 的上报映射：

- **RSRP**：`RSRP_00 < −140 dBm`，`RSRP_xx` 对应 `[−141+x, −140+x) dBm`，x = 0…97。即 **`上报值 = dBm + 141`**。
- **RSRQ**：`RSRQ_00 < −19.5 dB`，`RSRQ_xx` 对应 `[−20+0.5x, −19.5+0.5x) dB`，x = 0…34（Rel-12 起有扩展范围 −30…46，分辨率 0.5 dB）。即 **`上报值 ≈ 2×(dB) + 40`**。

本平台实证：

- `0x688C MeasNvInfo` 打印 `wStandardPoint = [141,141,…]`（L915）——**141 就是 RSRP 的上报基点常数**，与协议一致。
- 同一时刻：`0x68C6 AveLogRsrp = −73 dBm / AveLogRsrq = −6 dB`（L1575247），`0x850B CtrlICPReportResult` 上报 `Rsrp=68, Rsrq=28`（L1576109）。代入公式：`−73+141 = 68` ✓，`2×(−6)+40 = 28` ✓。
- `0x8668 FilterRSRP = 68.5, RSRQ = 29.7` 是**带一位小数的上报域滤波值**，不是 dBm。

**所以：看到 60~80 的“RSRP”是上报值（对应 −81…−61 dBm），看到 −60~−120 的才是 dBm。两者混淆会让门限判断整体偏 141 dB。**

### 26.3 本组模块的协作总链路

```text
上电/复位
  CMN: L1L_P_RESET_REQ → SetMode → 各模块 ResultMap=0x9FF（全部就绪）
    ↓
MC: 决定"要搜哪一个频点/带宽/场景" → 0x904F SendRfResReq
    ↓
CMN: RF 资源事务 0x9010 TRY-RX → 0x9012/0x9013 Req-Rx → 0x9016 Req-RxRst
    ↓
RFC: 通道选择与配置 0x702C/0x702D RxCfg → RFIC/RFFE 写寄存器 0x6F0F/0x6F48
    ↓
DFE: AGC 收敛 0x5C8E/0x5C80 → DC 估计 0x5C0F → 采样率/滤波 0x5C06
    ↓
CSRC: 搜索/频扫状态机 0x843E/0x845A → 分派给 CSRS
    ↓
CSRS: PSS 0x7D83/0x5A00 → 得 N_ID2 与定时 → SSS 0x7D86/0x5B06 → 得 N_ID1、半帧、CP
    ↓                                    ↓
    CFO 0x7D85/0x5881 ──→ DFE AFC 0x5C16 ──→ RFC 0x6F4E/0x6F10 写 DCXO 控制字
    ↓
CSRS/CSRC: TPU 定时对齐 0x5809/0x581C → MC 0x904C Macro Req
    ↓
PBCH（下行组）: MIB → 带宽/PHICH/天线端口 → MC 0x9E0F MibCnfMonitor
    ↓
MC SIR 状态机: 0x9E06 SIB1 请求 → 0x9E11/0x9E12/0x9E13 窗口 → 0x9E09 SI CRC OK
    ↓                                                        ↘ 失败: 0x9E0B / 0x9E1D
CSRC: 0x8449 CampOn=1（驻留）
    ↓
CMN: 0x8F9F ACCESS_REQ → RAPC（上行组）→ 0x8F07 DEDICATED First Config（进入连接态）
    ↓
连接态并行运行：
  CSRM 测量 0x7E83→0x68BC→0x68C8→0x68C3→0x68C7→0x68C6
  CSRC L3 过滤与上报 0x8646→0x8658→0x8667→0x8668→0x8663
  MULM gap 协调 0x848C
  MC 寻呼/预同步/RLM 0x960A / 0x9610 / 0x9C01
  RFC/DFE 持续跟踪 AGC、AFC、发射功率
```

### 26.4 MC：PHY 主控（41 个 ID）

MC 决定“现在做什么”，是排查“为什么没去做某件事”的第一站。

#### 26.4.1 状态与场景

| ID | 打印 | 是什么 / 关键值 |
|---|---|---|
| `0x8FBB` | `INF >LTE PHY State: RatMode:(1,1),Lv1State:2,Lv2State=1,Lv3MasterState=2,Lv3SlaveState=0` | PHY 三级状态。`RatMode=1` 为 FDD。 |
| `0x8F88` | `Lv3MasterState:4 [0=CLOSE,1=INIT,2=IDLE,3=CONN,4=CONN_SUSPEND]` | **日志自带枚举**，判断“UE 现在到底在不在连接态”的最直接依据。本文件中 3 条：CONN_SUSPEND 1 次、CONN 2 次。 |
| `0x8F07` | `DEDICATED First Config at Frame 895, SubFrame 6, E_ZPHY_STATE_CONN` | 专用配置首次下发＝真正进入连接态的时刻，全文件仅 1 条（L1590426）。 |
| `0x9052` | `TPU\|LTEA Lpm Ctrl: eScenario=48, bIsNeedOpen=0, DfeIdx=4, CsrIdx=1, DlIdx=1, UlIdx=1, TpuIdx=1` | 低功耗场景切换，**410,795 条、占全文件 7%**。`eScenario` 17 种取值（top：14×162,993、48×162,716、83×84,070），是厂商私有场景号，不要猜；有价值的是 `bIsNeedOpen`（是否需要开硬件）和各 `*Idx`（本次用哪套硬件实例）。 |
| `0x904E` | `PRINT\|Core=3 dwMsgId=1058, 0-4[…] 5-9[…]` | 每子帧的 20 时隙任务表快照，162,960 条。只在怀疑“某个时隙没排上任务”时看。 |
| `0x9055` | `L1l_RcvUnknownMsg > ThreadId=4, MsgId=1069, FileNo=93, Line=9638` | **收到未知消息**，718 条。数量多但本文件里始终是同一 MsgId，属于版本内消息未实现，不是空口问题；如果 MsgId 变化或与异常时刻重合才需要追。 |

#### 26.4.2 TPU 定时调整

TPU 是本平台的时间基准单元。所有“UE 的帧边界要往前/往后挪”的动作都在这里。

| ID | 打印要点 | 判读 |
|---|---|---|
| `0x904C` | `Macro Req … OldMrtrOffset=[488,14,3968], NewMrtrOffset=[488,0,3936], tAdjTime=[242,0,12904]` | 宏调整请求，`[frame, hsf, tc:15khz]`。新旧 offset 之差就是这次挪的量。 |
| `0x904A` | `Macro Adj Cnf Int TpuHwIdx=1, Cur[Frame,Hsf]=242,0` | 宏调整完成中断。**Req 有 186 条、Cnf 也有 186 条**，一一对应说明没有丢调整。 |
| `0x904D` | `Micro Req: AdjTc=3, [IsAdd=0, IsSub=1], Old/New MrtrOffset` | 微调，每次只动几个 Tc；持续单向微调说明晶振有残余频差，应结合 AFC 看。 |
| `0x9048` | `Tpu ModeSet: eCpMode=0[0:Ncp 1:Ecp], eWorkMode=1[0:NR 1:LTE], eScsMu=0[0:15Khz…]` | 模式设置，日志自带枚举。 |
| `0x9049` | `Tpu --->Reset TpuHwIdx=0` | TPU 复位，13 条；出现在每次系统复位处。 |

#### 26.4.3 SIB1 / SI 读取状态机（`SIR|` 系列，本文件最有价值的 MC 打印）

这是一条完整、可逐条读的链，20 个 ID 构成状态机的每一步：

```text
0x9E06 Sib1MsgMonitor   收到 PS 的 READ_SIB1_REQ（Earfcn/CellId/ProcId）
0x9E0E SibParaMonitor   更新 SIB 参数（邻区/服务小区、SchedNum、WinLen）
0x9E03 SndMibReq        先向 PBCH 要 MIB（Boundry = 帧边界 Ts）
0x9E0F MibCnfMonitor    MIB 结果：Crc / Band(RB) / Ant / Phich / BchSfn
0x9E0C RfcMonitor       为读 SI 配置 RF（Bw 枚举 0:20M…5:1.4M）
0x9E11 SchedParaMonitor 算出本次 SI 窗口位置 SchedPos、Period、WinLen
0x9E12 StartWinMonitor  窗口开始，RntiEn=1（打开 SI-RNTI 搜索）
0x9E10 RxRcvCtrlMonitor 接收控制（RcvInd = 1:Close 2:Open）
0x9E09 SibReportMonitor DLS 报 SI CRC OK：SibPos / SibTbs / ReptNum
0x9E13 EndWinMonitor    窗口结束，RntiEn=0
0x9E0B ErrMonitor       任一步失败
0x9E1D AbortSi          "SIB Fail!!!"，整个 SI 读取放弃
0x9E01 MainCtrlFlow     收到 ABORT_SI_READ_REQ
```

字段判读：

- `0x9E0F` 的 `Crc = 1/0` 是 MIB 是否正确。全文件 56 条中 **38 条 Crc=1、18 条 Crc=0**；`Crc=0` 的行 `Band/Ant/Phich` 全为 0，说明失败时字段无效，不能当“天线数=0”来读。
- `Band` 其实是**下行 RB 数**：取值 {100, 75, 50}，对应 20/15/10 MHz。`Ant` 取值 {2, 4} 是 CRS 天线端口数。
- `Phich` 取值 {0, 2, 6}：MIB 的 `phich-Config` 是 3 bit（1 bit duration + 2 bit resource，resource 枚举 `oneSixth/half/one/two` 即 N_g = 1/6, 1/2, 1, 2）。按位拆 `2 = 0b010`、`6 = 0b110` 时，高位像是 duration、低两位像是 resource，但**没有厂商位定义就只能当整体标签**；`0x8FD3` 里同一小区打印 `PhichCfg=6`，与 PBCH 模块的 `BchPhich` 应当一致，交叉核对即可，不要单独拆位下结论。
- `0x9E0B ErrMonitor` 的三个枚举都在行内给出：`MainState = 1:Sib1 / 2:Si / 3:Abort`，`StepState = 1:Msg 2:Mib 3:Rfc 4:Tpu 5:AdjOver 6:Sched`。**全文件 46 条，`ErrCode` 恒为 3；MainState 分布 1×30、3×15、2×1；StepState 分布 6(Sched)×25、0×15、2(Mib)×6。**含义：绝大多数失败发生在“已经拿到 MIB、进入调度/窗口阶段却始终没解出 SIB1”。
- `0x9E1D` 15 条 `SIB Fail`，与 `MainState=3(Abort)` 的 15 条一一对应。

**判据**：读 SIB1 失败时，先看 `0x9E0F Crc`。Crc=1 而后续报错 → 不是同步/PBCH 问题，而是 SI-RNTI 的 PDCCH 或 PDSCH 没解出来（转下行组的 PDCCH/PDSCH 章节）；Crc=0 → 回到 PSS/SSS/CFO/AGC。

#### 26.4.4 寻呼参数：可以完整验算的一段协议

`0x960A`（L1581816）：

```text
zPHY_emc_CalPagingParam(NbValue=2; T=128; nB=128; N=128; UeID=277; Ns=1;
  SubframIndex=0; RatMode=1; PagingState=1, wActivedPageCycle=128,
  PO= 917:9, tIdlePiStartTime= 916:9, PoGapCflict=0, wModDstSfn=[21,917])
```

按 TS 36.304 §7.1：

- `T` = UE 的 DRX 周期（无线帧数），由 `defaultPagingCycle`（rf32/64/128/256）与 UE 专用值取小；这里 T=128 帧 = **1.28 s**。
- `nB` ∈ {4T, 2T, T, T/2, T/4, T/8, T/16, T/32}；`N = min(T, nB)`，`Ns = max(1, nB/T)`。这里 nB=T=128 → N=128、Ns=1，与打印一致。
- **PF（寻呼帧）**：`SFN mod T = (T div N) × (UE_ID mod N)` = `1 × (277 mod 128)` = **21**。日志 `wModDstSfn=[21,917]`、`PO=917:9`，而 `917 mod 128 = 21` ✓。
- **PO（寻呼时机子帧）**：`i_s = floor(UE_ID/N) mod Ns = 0`；FDD 且 Ns=1 时查表得子帧 **9**，日志 `PO= …:9` ✓。
- `UE_ID = IMSI mod 1024` = 277。

配套打印：`0x960D`（PO 中断真正到来）、`0x9605`（PO 参数更新）。**如果测试用例是“寻呼不响应”，这三条能直接证明 UE 算出的 PF/PO 与网络是否一致**——比抓空口快得多。

#### 26.4.5 预同步与 CIR（`PreSync` 族）

DRX/空闲态醒来前，UE 要先把定时和频偏拉回来，这组打印就是这个过程：

| ID | 打印 | 判读 |
|---|---|---|
| `0x9610` | `PreSync[0x02];Sys=1,Udc=-1,T=6;PO=9;Rf=0,Agc=0,Fss=1,Cfo=1;Subf=3,4,5;Idx=1,2,2` | 预同步进度位图。`Rf/Agc/Fss/Cfo` 是四个子步骤的状态计数（本文件 Rf∈{0,3}、Fss∈{1,2,3}、Cfo∈{1,2,3}）。`Subf=3,4,5` 是安排在 PO 前的三个子帧。 |
| `0x9612` | `zPHY_emc_tRxCirPreSyncStart, RxCirPreSyncState=0! CsrNoUseRxMeasFlag=1` | 预同步启动 |
| `0x9611` | `no need to do RxCirPreSync !!!` | 判定不需要预同步（2 条） |
| `0x960B` | `PI Task START … CIR Adjust Total Value in One DRX for Ant0..3: [-167,-46,-62,-62]` | **一个 DRX 周期内的累计 CIR（定时）调整量**，单位 Ts。数值持续单向增大说明定时在漂。 |
| `0x9614/0x9615` | `PreSyncAccNum / AbsSumCirAdjVal / AbsSumCfoAdjVal / CurAdjTs` | 预同步累计统计 |
| `0x960F` | `PreSync[SleepSched]; SetRxCirPreSyncFlag=0!!!` | 睡眠调度改写标志 |
| `0x9618` | `Cannot Colse RF at RAPC, SIB1 or SI PLMN PROC` | **不能关 RF 的原因**，直接说明当前有随机接入或 SI 读取在跑 |
| `0x960C/0x9613` | `RF RxOffset: TxTab / TxOffset / MeasTab / TpuMrtrOffset=35288` | 收发与测量表的时间偏置 |

#### 26.4.6 RLM（无线链路监测）`0x9C01`

```text
RLMState:0; N[Cnt:0, 310311:1-1]; CalcRst:0; Drx[State0,Cyc:0,CalcFlg:0,Act:0,QC:…];
SlpCnt:0, Win[Flg:1, Con&DrxCnt:40, 0]; FI&THFilter:0,0; Final[TH:34000, QC:24960716]; CurSNR:100000
```

- 协议背景（TS 36.133 §7.6）：UE 用 CRS 估计**假想 PDCCH 的 BLER**，超过 `Q_out`（10%）判 out-of-sync，回落到 `Q_in`（2%）判 in-sync；连续 N310 个 out-of-sync 启动 T310，期间连续 N311 个 in-sync 则恢复，否则宣告 RLF。**这些计数与门限在 RRC 配置里，PHY 只给出“质量是否过门限”。**
- 本文件实证：2,676 条中 `RLMState` **恒为 0**、`TH` **恒为 34000**、`CurSNR` 有 2,667 条是 **100000**（无效哨兵），只有 9 条是真实值（112、4894、10881、12976、21532…）。
- 结论：**本次会话 RLM 没有进入实质评估**，不能用这条日志去论证“链路好/坏”。要说“失步”，必须有协议栈的 out-of-sync 指示或 T310 记录佐证。

#### 26.4.7 MC 其余 ID

`0x9E10 RxRcvCtrlMonitor`（`RcvInd = 1:Close 2:Open`，374 条，全为 1）、`0x904F SendRfResReq`（向 RF 要资源，`dwReqType=0x100c1` 是私有位图）、`0x9E05 DelyTpuAdjust`（延迟事件注册）、`0x9E07/0x9E08 SiMsgMonitor`（SI 调度请求）、`0x8A1E`（DRX 下 DL HARQ 与 PDCCH 标志，1 条）、`0x9E01 MainCtrlFlow`。用途见附录 A。

### 26.5 CMN：配置事务与 RF 资源（65 个 ID）

CMN 回答两个问题：**配置有没有真的下发到各模块**，以及**RF 硬件有没有真的给到这条载波**。

#### 26.5.1 PHY 内部消息包络

| ID | 打印 | 用法 |
|---|---|---|
| `0x9000` | `PUB\|PHY->PHY SEND:MsgId=0x2b9c(11164) NS=1[N L LV NV W] SrcCore/SrcThread → DecCore/DecThread` | 51,681 条，PHY 内部消息总线。**排“消息发了没有/被谁收”时按 MsgId 过滤**，例如 `0x2b9c` 是 BCH 解码确认。 |
| `0x8F94` / `0x8F96` | `Lte Phy Recv Msg: MsgID=0xD306` / `ClearSyncMsg` | PHY 收协议栈消息与同步消息清理 |
| `0x902B` / `0x902C` | `SendIcpToPS` / `WakeUpPS`（`PS sleep flag=0(1:sleep;0:work)`） | PHY 主动唤醒协议栈 |

#### 26.5.2 配置事务：SDL / DEDISDL / PUB CG

本平台把“一次配置下发”做成事务，分公共（SDL）、专用（DEDISDL）、载波组（ProcCG）三类：

```text
0x9F11 PUB|ProcCG Start   AddCCNum / RelCCNum / RelSrvCellMap ← 本次要加/删几条载波
0x9F12 PUB| Idx=0: SrvCell=1 Band=3 BW=10 …                  ← 每条载波的参数
0x9F13/0x9F14 ProcCG PubProc / Rst                           ← 分发到各核并回结果
0x9F16/0x9F17 ProcCC ModProc / Rst                           ← 载波修改
0x9021 CMN|SDL: SdlIdx / ProcId / CurStep / TableNum         ← 公共配置分步下发
0x9007/0x900D CMN|DEDISDL StartDediProc / DediProc-Mod       ← 专用配置，ModId=4[CSR UL DL RF CSI]
0x9008 CMN|SDL[EndSDL]                                       ← 事务结束
0x901F/0x8FF8/0x8FFA/0x8FFD/0x8FFE/0x8FFF                    ← 释放路径
0x902A CMN|CgInfo: dwServCellIdx / Band / dwBW / UlFlag / HwAssign
```

**判读方法**：`ModDoneMap=[0x0 0x0 0x0 0x10] AllDoneFlg=1` 表示各模块完成位图；`AllDoneFlg=0` 且长期不变，就是某个模块没回复——这正是“重配置不生效”的根因位置。本文件 5 次 `ProcCG Start`，其中 L2447295 一次 `AddCCNum=1`（加 SCell），L2481018 一次 `RelCCNum=1`（删 SCell），其余为 0/0 的重配。

#### 26.5.3 RF 资源申请状态机

```text
0x9010 CMN|RF[TRY-RX]     试探：CacheReqP / CurCacheNum / State / Ret
0x9012 CMN|RF[Req-Rx]     申请：Step / Scene / RxNum / State / FreeMrtr
0x9013 CMN|RF[Req-Rx]     参数：SrvCellId / Band / BWEnum / AntNum / Hw / Freq / Dup(0:TDD,1:FDD)
0x9016 CMN|RF[Req-RxRst]  结果：Return / CcNum / [SvCellId Ant SwitchFlg|IntraFlg|V4Flg] / Error[]
0x9014/0x9015/0x9017      发射侧对应三步
0x9018/0x901A CMN|RF[SLOTL-RX/TX]  时隙级 RF 归属与冲突：ConflictFlg
0x900A CMN|RF[TRY-RTX]    收发同时申请
```

**关键字段**：`0x9016` 的 `Error=[0,0,0,0,0]` 全 0 才是成功；`0x9018` 的 `ConflictFlg` 非 0 表示同一时隙有多个用户抢 RF（典型场景：测量 gap 与业务冲突）。本文件 728 条 `SLOTL-RX` 的 `ConflictFlg` 均为 0。

#### 26.5.4 开关 RF 与子帧调度

| ID | 打印 | 判读 |
|---|---|---|
| `0x8FEB` | `SchedMc\|INFO > OpenRF >>> Time: 241-9 > Open DL in SubFrm:0, OpenCcEn:0, State:2, TaskInfo:[SIB]=1,[RAPC]=0,[PLMN]=0,[PreSync]=0, DrxCloseRf=0, DLbmp=[…]` | **42,275 条**。`TaskInfo` 直接说明这次开 RF 是为谁开的（SIB / 随机接入 / PLMN 搜索 / 预同步）——这是把 RF 行为和业务意图对应起来的关键。`OpenCcEn` 1/0 分布 42,040 / 235。 |
| `0x8FE5` | `CloseRF >>> Stop RF Rx Receive by Carrier 0, RxMask:[1]` | 关 RF，392 条 |
| `0x8FE8` | `Config RF Work Info > CcIdx / CfgSubFrm / Freq / BandNum / Bw / UDMode / SpecMode / CpMode` | RF 工作参数 |
| `0x8FEA` | `Calc Open RF Time: 242-9, tReferTime: 242-14(Slot)+4(Slot), CalcOver=1, RfOpenEn=0` | 提前量计算；`RfOpenEn=0` 表示这次算完不开 |
| `0x5001` | `SetRxMask:CcIdx[0]:(1->0); Mode=5[0:Mc,1:Rx,2:Csrc,3:Csrs,4:Csrm,5:Pbch,6:Sib]` | **接收权归属切换**，日志自带枚举，221 条。谁持有 Rx 就由谁决定 RF 行为。 |

#### 26.5.5 小区信息、TA 与复位

- `0x8FD3 MC_Task: GetCellInfo(1[1:Ok;Other:Err]); Earfcn=1275, CellId=30, Cp=0, TxAntNum=4, SysDlBw=75, PhichCfg=6, BCH Boundry=39.2.4574` —— **一行给出驻留小区的全部关键参数**，是每次驻留成功的“身份证”，全文件 3 条（L1581480、L2495577、L2561644）。
- `0x8FAF/0x8FB0 ProTASchedFlow: TACmdType=0/1, wSubFN, wAdjSubFN, wTimeAdvance=31, TxOffset=0` —— TA 命令调度。协议上 TA 命令 `T_A` 取值 0–63，调整量 `(T_A−31)×16 T_s`，**31 = 不变**；本文件 `wTimeAdvance=31` 正是“保持不变”，`0x8FB0` 的 0 则对应最大提前调整。TA 绝对值在随机接入 RAR 中是 11 bit（0–1282）。
- 复位链：`0x9020 RELSDL[Start]（收到 L1L_P_RESET_REQ）` → `0x8FB2 ProResetFlow Start (ResultMap=0x1)` → `0x8FB4 Down (ResultMap=0x9FF)`；SetMode 同理 `0x8FB7 → 0x8FB9`。**`0x9FF` 是“全部子模块就绪”的位图**，出现别的值就说明有模块没回。
- `0x8F89 Recv ZPS_LTE_P_MAC_RESET_REQ_EV, C-RNTI_En=0` —— MAC 复位，4 条，出现在链路重建/释放处。
- `0x8F9F / 0x8FA2 MC Send ACCESS_REQ Msg to RAPC` —— **发起随机接入的时刻**，与上行组 RAPC 的 `0x9812` 对齐即可判断“请求有没有到”。
- `0x9028 Warning > COMM Receive In NeibourCell SI Read!!` —— 在读邻区 SI 时收到公共消息，1 条，属竞态告警。

### 26.6 CSRC：搜索控制、频扫、L3 过滤与 SCell（91 个 ID）

#### 26.6.1 搜索与测量的入口/出口

| ID | 打印 | 判读 |
|---|---|---|
| `0x843E` | `Recv ZPHY_EMC_ECSR_CELL_SEARCH_REQ CsrMainState=0, dwEarfcn=3201, wSearchType=2[1:Strongst 2:List 3:Appointed], CellNum=0, wIniSearchTime=1` | 搜索请求入口。`wSearchType` 自带枚举：全网最强/按列表/指定小区。 |
| `0x845A` | `Recv ZPHY_EMC_ECSR_FREQ_SCAN_REQ` | 频扫请求，15 条 |
| `0x851B` | `Search Finished: Result:-1 / 0, CellEarfcn=…` | **搜索结束与结果**：`0` 成功、`-1` 失败。30 条，是画时间线最好用的锚点。 |
| `0x851C` | `Measure Finished: Result: 0` | 测量结束 |
| `0x8506` | `Recv PBCH_SUCC_IND/PBCH_FAIL_IND = 1` | PBCH 成功/失败指示（1/0），24 条 |
| `0x850B` | `CtrlICPReportResult >RIGHT: Earfcn 1275, CellNum 1, searchType=2, [0]ID=30,Rsrp=68,Rsrq=28 …` | **初搜结果上报**，最多 3 个小区，用的是上报域数值 |
| `0x8526` | `Cell Rank: wTempIndex=0, CellID=30, dwEarfcn=1275, RSRP=-73` | 候选小区排序，这里的 RSRP 是 **dBm** |
| `0x851E`/`0x8449` | `Recv E_L1L_MC_CSR_COMMON_CONFIG_REQ wCampOn=1[0:EPHY NO 1:ephy] PiPeriod=128` | **驻留标志**：`wCampOn=1` 表示 PHY 认为已驻留，`PiPeriod` 即寻呼周期帧数 |
| `0x8448`/`0x844D`/`0x845E`/`0x8459` | `ABORT_CELL_SEARCH / ABORT_MEAS / STOP_INTER_SEARCH_MEAS / REL_REQ` | 各类中止；出现即说明上层主动打断，不要当“搜索失败” |

#### 26.6.2 频扫状态机（`<FreqScan>` 族，共 20 余个 ID）

`0x5917` 一行给出状态机全貌：

```text
<FreqScan>[Result]Band=1, FreqPoint=21105, MaxValue=11772, Gain=110 110, PssFlag=7,
  FsState=0[0:MP 1:500K 2:100K 3:Re500K 4:AGC 5:Re100K 6:Discrete PSS], CsFlag=0, DisFreqScan=0
```

阶段顺序与对应打印：

| 阶段 | FsState | 主要打印 | 作用 |
|---|---|---|---|
| 宽带功率扫描 | 0 (MP) | `0x5924 [MP report]`（BW、频点区间、`dwMaxValMean`）、`0x5927 MpInfo` | 用大带宽粗测各段能量，先排掉空频段 |
| 500 kHz 栅格 | 1 | `0x5911 [500K]`、`0x5910`（含 band38/41 频率表）、`0x592B [Redo 500K]` | 缩小到 500 kHz 精度 |
| 100 kHz 栅格 | 2 | `0x5912 [100K]`（左右频点、`MaxPeakVal`、GAIN） | 缩到 LTE 100 kHz 频率栅格 |
| 增益标定 | 4 (AGC) | `0x592F [AGC Start]`、`0x592C [GAIN]`、`0x592D [AGC]`（`NotSyncAGCDone`）、`0x5906`（AGC 增益计算错误） | 为该频点设增益，否则峰值不可比 |
| 离散 PSS | 6 | `0x5934 Pss100KResult`、`0x5943 Scan-Procinfo` | 用 PSS 确认这个频点上真有 LTE 小区 |
| 交给小区搜索 | — | `0x5903 FreqScan add CellSearch ok!!! (SSSValue, wGain)`、`0x5904 [Fail] (dwMaxPeakValue=0)`、`0x590B Delete_[PointValid]` | 成功则进入 PSS/SSS 全搜索；失败/删点则换下一个频点 |
| 上报 | — | `0x5908 FreqScan Report to ps`（Earfcn/BandNo/Value/Gain/补偿值）、`0x5909 End: Earfcn Num=0` | 把候选频点交给协议栈 |

**判读要点**：

- 峰值 `MaxValue`/`MaxPeakVal`/`SSSValue` 只有在**同一 Gain** 下才可比；`0x5908` 的 `FingerValueComp / FingerValueAfterComp` 就是增益补偿后的值，跨频点比较必须用补偿后的。
- 本文件 `0x5917` 3,256 条，扫过 20 个 band（top：42、3、43、41、66、1），`Gain` 只有 8 种取值（110×2084、0×891、105×215…）。
- `0x5909 … Earfcn Num = 0` 出现 3 次，含义是**这一轮频扫一个可用频点都没报出来**——这是脱网后反复搜网的直接证据。

#### 26.6.3 候选小区数据库

| ID | 打印 | 判读 |
|---|---|---|
| `0x8583` | `Add CellToDB(01): Pos:1, Earfcn=1900, CellId=226, Finger=995, FB=0.11.14528, FbAge=0, Throld=[80,4,0,0,80,4], MeasAge=2560, NewFlag=0, CellNum=2` | 新小区入库。`Finger` 是相关峰值，`FB` 是帧边界（§2.1 三元组），`*Age` 是老化计数（越大越旧） |
| `0x8584` | `RefreshDB: Flag=2 0, Earfcn=1900, CellId=226, Age=[667,645], Throld=[512,100], State=[0xff,0xff,0xff]` | 刷新与老化 |
| `0x8645` | `SearchMeasAgeThrold: Intra=[3712,128,512], Inter=[512,50,100]` | 同/异频的老化门限 |
| `0x8671` | `FindCell > no cell!!!! Earfcn=0, CellId=0` | 查库未命中，229 条；单独出现不代表故障 |
| `0x852C` | `ProWriteBch2CsrDb>PBCH: Cell_1Info=[2452,125,535.19.11744,1]` | 把 PBCH 结果写回小区库 |
| `0x847E` | `Csrc_UpdateBackBchBnd: dwEarfcn=2452, DestTimeMrtr=535.19.10132` | 更新回退小区的 BCH 边界 |

#### 26.6.4 服务小区测量与 L3 过滤（连接态最重要的一段）

链路（同一时刻的 5 条打印，L1593512–L1593515 与 L1604476）：

```text
0x8646 WriteServingCellResult > SrvCellInfo-ID[30] RSRP=[70][34] RSSI=[-22639][-22594] wAgeEventCnt=5
0x8658 FilterInfo[0] >: wCellNum=1, cell0:[30,69], wNeiReport=0
0x8667 FilterIntraDebugInfo: AdaptFilterType=2, RSRP_K=[8,64], RSRQ_K=[7,76], SF=[76], S=[76,76 76,76]
0x8668 FilterIntraDebugInfo: CellId[0]=30, FilterRSRP=68.5, RSRQ=29.7
0x8663 #INTRA#MEAS:PCC Intra-Meas:Cell INFO > Earfcn=1275, Cell[0], ID=30, Rsrp=69, Rsrq=34, MeasAge=1, SearchAge=21
```

- `0x8646` 的 `RSRP=[a][b]` 是**[RSRP 上报值][RSRQ 上报值]**，不是两个 RSRP；`RSSI` 是未补偿的内部对数值（§2.4）。
- L3 滤波（TS 36.331 §5.5.3.2）：`F_n = (1−a)·F_{n−1} + a·M_n`，`a = 1/2^(k/4)`，`k = filterCoefficient`（fc0…fc19）。`0x8667` 的 `RSRP_K/RSRQ_K` 是本平台的滤波内部系数/计数（本文件中第一位随时间递增、第二位递减），**不能直接当协议 `filterCoefficient` 读**；判断“滤波是不是太慢”应看 `0x8668 FilterRSRP` 相对 `0x8646` 原始值的追赶速度。
- `0x8663 #INTRA#MEAS` 是最终对外口径的同频测量结果（上报域）。`MeasAge/SearchAge` 是结果新鲜度，`SearchAge` 很大（本文件最大 2176）说明这个小区很久没重新搜过，其定时可能已经不准。

**本文件的一个可复现异常（需与仪表交叉验证）**：

- `0x8646` 全部 351 条的 RSRQ 上报值**恒为 34**，即 −3 dB，是 Rel-8 legacy 范围的上限；`0x68C3 CalRsrp` 的 `RSRQ*2` 有 308 条为 71（相当于 +35.5 dB，超出任何合法 RSRQ 值域）。
- 同一时刻 `0x68BC MeasHwInfo` 的 `RSSI=[0;0]`（硬件 RSSI 累加器为 0），而更早的空闲态测量里 `RSSI=[559;225]` 有效、RSRQ 稳定在 −6 dB（`0x68C6 AveLogRsrq=-6`，`0x850B Rsrq=28`）。
- 物理上：4 端口小区在只有 CRS 有功率时 RSRQ 约 −6 dB，满负载时趋向 −10.8 dB（`RSRQ = N·RSRP/RSSI`，同符号 12N 个子载波）。**连接态跑业务却报 −3 dB，方向反了**，最可能是 RSSI 累加路径在该模式下没取到值，导致 RSRQ 被算大后钳到上限。
- 影响：单小区传导测试不影响业务，但**任何用 RSRQ 做门限的用例（A2/A3、重选、异频启测）都会误判**。核对方法：读 MT8000A 的 RSRQ 显示，或在同一时刻比对 `0x68C6 AveLogRsrq`（该值仍然合理）。

#### 26.6.5 SNR 与 RSSI

- `0x866E zPHY_ecsrc_CsrSNR >> Snr[0]=39.1, Snr[1]=39.6, Snr[2]=39.8, Snr[3]=36.1` —— **四天线 CRS SNR，单位 dB，一位小数**。全文件 340 个采样，范围 −17.7…40.6 dB。传导环境下 35–40 dB 属正常高值。
- `0x8675 CSRC:SNR_dB[-12;-13;-13;-12]` —— 另一处 SNR 打印（10 条，−17…−12 dB），出现在频扫/空频点场景，**不要和 `0x866E` 混为一谈**。
- `0x8676 IntraRSSI: RSSI:[-60.9], TempRssi[-15124,0]` —— 左边是补偿后的 dBm，右边是原始对数值；206 个采样范围 −123.6…−4.0 dBm（−4 dBm 属频扫时的强信号/近端）。

#### 26.6.6 SCell 与多载波

`0x8445 AddCarrier`（`wCurCcIdx=1, wValidCc=3, dwServCellIdx=1`）→ `0x8586 AddScellInfoToDatabase (wAddModNum=1, wServCellNum=2)` → `0x858A (wii=1, Earfcn=1900, Pci=47, ScellState[1 0])` → `0x858B SccMeasStateCfg (bScellExist / bScellActive / wSccCellId=47)` → `0x8640 WriteServingCellResult SCellInfo-ID[1900-47] RSRP=[76][24]` → `0x8665 #INTRA#MEAS:SCC Intra-Meas`。

**读法**：`ScellState[1 0]` 与 `bScellActive` 区分**已配置**与**已激活**（激活由 MAC CE 完成，见总文档 §13）。本文件 SCell 只在 L2447375–L2481022 之间短暂存在，随后被 `RelCCNum=1` 释放。

### 26.7 CSRS：PSS/SSS/CFO 的执行侧（54 个 ID）

#### 26.7.1 协议背景（判读前提）

- **PSS**：位于每半帧最后一个 OFDM 符号（FDD 为子帧 0 和 5），占中心 **62 个子载波**（连同保护共 6 RB / 1.08 MHz），由 Zadoff-Chu 序列生成，根 `u = 25 / 29 / 34` 分别对应 `N_ID^(2) = 0 / 1 / 2`。→ 日志的 `Id0 / Id1 / Id2` 就是这三个假设的相关峰。
- **SSS**：紧邻 PSS 前一个符号，携带 `N_ID^(1)` (0–167)，且子帧 0 与子帧 5 的序列对调，因此**SSS 同时给出半帧位置**（日志 `HfIndic`）。
- `PCI = 3×N_ID^(1) + N_ID^(2)`，范围 0–503。
- CP 类型（NCP/ECP）在这一步用相关能量判定，日志 `CPMode=0(NCP)/1(ECP)`。

#### 26.7.2 PSS 链

```text
0x580F GetPssStartTime   PssHwCfgMode / PssStartMrtr / LocalMrtr        ← 什么时候开始
0x7D83 PssCfg(00)        CurrMrtr / HFB / Config / NumHF / ThrePower / ThreNoise / MaskW / IdEn / SearchEn
0x7D84 PssCfg(00)        时钟、F0CfgTimes、F0HalfFrameCnt、Busy、Done、ErrFlag
0x7D89 PssCfgHwMT(0)     写进硬件的寄存器镜像（Top/DataSource/Config/FrameBoundry/…）
0x580C PssProc           过程状态：ResultFlg / SwStatus / Config / Read / Done / RfOpen / GapDistan / PssBusy
0x7D88 LogPssHwOriValue  硬件原始峰值：Id0=[23,3198], Id1=[20,3176], Id2=[26,3216], dwHalfNum=10
0x5A00 Pss(3)(00)        整理后的三候选：Id0=[23(3262),19(3253),18(3225)]; Id1=…; Id2=…; wFingerCnt=8
0x8483 PssConvertFinger  NoisePower / NoiseCount / MaxPower（判决门限的来源）
0x5809 PSS_TPU_ADJUST    把 PSS 得到的边界写给 TPU（§2.1 的 Ts 换算）
0x581C TPU 回 CNF        NewTpuOffset
```

**判读**：`Id0/Id1/Id2` 每项是 `峰值(位置Ts)`。三者中峰值最大且明显高于 `NoisePower` 的即为 `N_ID^(2)`；本文件 L1112787 的 `Id2=26` 略高于 `Id0=23`、`Id1=20`，差距很小——这正是弱信号频点上误判 PCI 的典型形态，后续 SSS 会把它否掉。`wFingerCnt=8` 表示保留 8 个时间候选（多径/多小区）。

#### 26.7.3 SSS 链

```text
0x7D86/0x7D87 SssCfg      Id2 / En / WorkSta / ChEsti / Thre1~3 / NoiseKill / WinPos0~7 / BufferTime
0x7D8D/0x7D8E/0x7D99      SssCfgHwMT 硬件镜像（含 SpecCellidDet 指定 PCI 检测）
0x5811 SssProc            过程状态：ProcStatus / FreqChange / RfOpen / GapDistan
0x5B00 SssStartFinger     每个 finger 的 CellId / PeakValue / OffsetTime / HwOffsetTime / FingerOffset
0x5B05 SssFingerReorder   8 个 finger 按时延重排
0x5B0A SssGetThreshold    MaxFinger / AccuNum / CommThreshold / Proc0~7[CellId/Finger]
0x5B04 GetSssReadFlagInfor SssReadFlag[0:FALSE,1:TRUE] / SssCfgTimes / SssReadTimes
0x5B06 Sss(3)(0)(00)[30]  最终结论：Earfcn / CellId / FB / HfIndic / CPMode / Total(积累时长) / Finger / Threshold
0x5B0C GetRfcEnableInfo    本次搜索占用的 RF 窗口
```

`0x5B06` 是**小区搜索的结论行**，一行同时给出 PCI、帧边界、半帧指示、CP 类型、相关峰与门限。本文件 173 条中：`CellId` 覆盖 5–503，`HfIndic` 0/1 各半，`CPMode=1(ECP)` 出现 23 次（弱信号频点上的误判，正常网络为 NCP），`Total` 只有 10 ms 和 50 ms 两种积累档——**弱信号时用 50 ms 档**，这也解释了为什么搜网慢。判决条件是 `Finger > Threshold`（例：`Finger=1178 > Threshold=736`）。

#### 26.7.4 CFO（频偏）

```text
0x7D85 CfoCfg      HFBNcp/HFBEcp / SymMapNcp / SymMapEcp / SearchWinLen / Accnum
0x7D96 CfoCfg[MT]  硬件镜像
0x5883 CfoResultMerge  Done=1, wIndex, NcpIQ-EcpIQ=[1938,28,5461,291]   ← 原始 IQ 相关累加
0x5881 CSRS_ICS: CFO Result  Frequency offset=-126 Hz, CFOCordicValue=0 Hz, CPMode=0, TimeShift=0
0x5810 CsCfoProcEnd
```

`0x5881` 给出的是 **Hz**，直接可读。本文件 67 次估计，范围 **−7496 … +7280 Hz**（在 1850 MHz 上约 ±4 ppm），初搜阶段大、锁定后收敛到几十 Hz。这个值随后进入 DFE AFC（§9.4）。

#### 26.7.5 搜索资源与 RF 窗口

`0x5802 CsRfcCfgComm`（57,051 条，最频繁的 CSRS 打印：每子帧的搜索 RF 状态、DFE 状态、AGC 模式）、`0x590F FreqscanRfcCfg`（35,818 条）、`0x5801 CsrRfcConfig`、`0x580B CaFreqChange`、`0x5806 Search Req`（`State=0[0:Ini,1:Intra,2:Inter,3:Fast]` 与 PCC/SCC/Inter 三组频点）、`0x5813 GetHwConfigMode`、`0x581A GetSearchHwIdxIdle`、`0x7D91 CheckCsBufferHw`（27,074 条缓冲状态）、`0x7D8C TopHwReset`、`0x7D90 CSRAddr`（各硬件块基址）、`0x7DA0 SearchProcReset`、`0x5820 TaskEntry`。这些属**执行侧状态**，通常只在“搜索卡住不动”时按时间顺序翻一遍，看卡在哪个 `Busy/Done/State`。

### 26.8 CSRM + MULM：测量执行（21 + 4 个 ID）

#### 26.8.1 测量配置与硬件窗口

```text
0x844B (CSRC) Recv MEAS_CONFIG_REQ        ← 上层下发测量配置（频点列表）
0x6706 MeasProcStart                       eCoreId / wCCcIdx / tMeasState / IntraFreq / InterFreq / wNextSchTime
0x7E8E CsrmValidCcIdx / 0x7E88 CsrmMeasSeek  每次测量前的载波与 gap 状态（各 26,284 条）
0x7E85 CsrmCalcRfOpenTime                  算出 RF 需要提前多久打开
0x7E86 CsrmCfgRfcData (bRfOn / wOpenSf)
0x6704 WriteTDDRfcEventTab                 写测量事件表（含 dwMeasTabOffset / MeasMode）
0x7E83/0x7E87 MeasCfg / MeasCfgMT          CellNum / CellInfo(PCI 列表) / MeasEn / Mode(0:Normal,1:SingleSym) / BandWidth / Quantity
0x7E84 MeasHwInfo                          dwDoneFlag / Slot.Sym / SymCnt / RspCnt   ← 硬件是否真的测了
0x7EB9 MeasStartSubFrame
```

`0x7E83` 的 `CellInfo=(0x1e,0,…)` 就是要测的 PCI 列表（0x1e = 30）；`MeasEn=1` 且 `0x7E84 dwDoneFlag=1` 才算这次测量真正完成。

#### 26.8.2 RSRP/RSRQ/RSSI 的五步计算链（可逐步验算）

以 L1575240–L1575247 的同一次测量为例：

| 步骤 | 打印 | 内容 |
|---|---|---|
| ① 硬件累加 | `0x68BC MeasHwInfo` | `RSRP00=[107;18], RSRP01=[131;16], RSRP10=[63;10], RSRP11=[24;5], RSSI=[559;225]` —— 每天线/每支路的 RS 功率累加与计数 |
| ② 取模与对数 | `0x68C8` | `adwMod=[1756964,…]`（线性模）→ `swTempRsrpLog=[1800,1530,−32768,−32768]`（对数域，−32768 = 无效哨兵），`wRsNumLog=1174`，`tResult.swRsrq=−470` |
| ③ 加增益/偏置转 dBm | `0x68C3 CalRsrp` | `AgcAnt=[85,76,380,9]` + `sawOffset=[−25,−21,0,0]`（NV 校准）→ `RSRP=[−75,−73,−386,−386] dBm`（−386 = 无效），`RSRQ*2=[−13,−12,…]`，`swRssiLog1=[−5084,−4992]` |
| ④ 逐量上报 | `0x68C7` | 四条连打，依次是 RSRP、RSRQ、RSSI0、RSSI1 的对数值（−6124 / −470 / −5084 / −4992） |
| ⑤ 平均输出 | `0x68C6` | `AveLogRsrp=−73 dBm, AveLogRsrq=−6 dB, swAveRsrp=[−6124], swAveRsrq=[−470], swAveRssi=[−15124,−14848]` |

配套：`0x6889 #CSRM:RX:RSRPoffsetCal`（把 dBm 与上报域并排打印）、`0x6895 ServCellRxMeas`（服务小区 Bandwidth/Earfcn/PCI 三元组，`−1` 表示该位未配置）、`0x688C MeasNvInfo`（**NV 校准：`OffsetAnt0/1` 各频段偏置、`wStandardPoint=141`**，17,745 条）。

**无效哨兵必须记住**：`−32768`（对数域）、`−386`（dBm）、`−1`（未配置）、`0`（累加器未取到）。把哨兵当真实值平均，会得到荒唐的结论。

#### 26.8.3 判据

1. `0x7E84 dwDoneFlag != 1` → 硬件没测完，后面的 RSRP 都不可信。
2. `0x68BC RSSI=[0;0]` → RSRQ 一定要打问号（§6.4）。
3. `0x68C3` 中某天线 `RSRP=−386` 且长期不变 → 该通道未使能或未连接，与 `0x9013 AntNum`、`0x702D RxAntPathId` 对照。
4. 四天线 RSRP 差 > 10 dB → 通道不平衡（本文件 `[−73,−71,−62,−62]` 差 11 dB，属天线 2/3 增益路径不同，需与 `AgcAnt=[67,62,48,54]` 一起看）。
5. `AveLogRsrq` 低于 −19.5 dB → 上报时会被钳到 0，**协议上报值和内部值会不一致**，本文件内部值最低到 −27 dB。

#### 26.8.4 MULM：多频/gap 从流程（4 个 ID）

- `0x848C MulmSlaveMeasureFlow`（**63,170 条**）：`PlmnSearchMeasFlag/cnt=(0xffff,300)`、`GapRfState`、`SlaveSyn/fun_State`、`CfoCnt`、`40msGapCnt`。逐条读没有意义；有价值的是 `GapRfState`（本文件 0×60,988、1×2,152，即约 3.4% 的时间 gap 占用 RF）与各计数的**差分**。
- `0x9401 Csr Slave State Change to:0[0=SLAVE_IDLE;1=SLAVE_SEAR_MEAS_IDLE;2=SLAVE_ASYN_SEAR_MEAS;3=SLAVE_SYN_SEAR_MEAS;4=SLAVE_FREQSCAN;5=SLAVE_PBCH]` —— 日志自带完整枚举，13 条，是从流程状态跳转的锚点。
- `0x9402 Csr Slave SYN State Change to:0[0=E_CSR_SLAVE_ASYN;1=E_CSR_SLAVE_SYNING;2=E_CSR_SLAVE_SYN]`。
- `0x9412 REV ZPHY_EMULM_EMC_ECSR_SET_MODE_REQ IratMode=0/1` —— 异系统模式设置（4 条）。

**测量 gap 判据**：`GapRfState=1` 期间不应有本载波的 PDSCH 期望；如果仪表在 gap 内下发数据而 UE 没收，是配置问题不是 UE 问题。

### 26.9 DFE：数字前端与 AGC/AFC（58 个 ID）

DFE 回答“信号进来之后，增益、直流、频偏、采样是否处理正确”。它是**下行解调质量差**这一类问题的第二现场（第一现场是 RXP 的 SNR/CIR）。

#### 26.9.1 AGC 的四条并行路径

本平台不是一个 AGC，而是四种场合各一套；**看错路径会得到相反结论**：

| 路径 | 主要打印 | 使用场合 |
|---|---|---|
| 同步态 AGC | `0x5C8E SyncAgcInfo`（47,723 条） | 已同步、正常收发时的闭环 |
| 非同步 AGC | `0x5C94 AsyncAgcPwr` / `0x5C95 AsyncAgcGain`（各 37,798 条）、`0x5C18/0x5C19 AsyncAgc` | 搜索/频扫，还没有帧定时 |
| 快速 AGC | `0x5C87 FastAgc`（29,868 条） | 突发场景，按子帧快速收敛 |
| 半静态 AGC | `0x5C8A/0x5C8B/0x5C8C/0x5C88/0x5C89 SemiStaticAgc` | PSS/CSR 阶段按最大均值一次性算增益 |

`0x5C8E` 字段：

```text
AgcWorkState=13, AgcLen=0x4, FreqPoint=18500, BandWidth=6, Lf=0x400, Target=3280,
MeanPwr[3247,3349,3248,3246], Gain=[114,104,113,116], AgcdoneFlag=[1,0], PosIdx=5, MeanpwrValid=1
```

- `BandWidth` 是 **RB 数**，取值 {6, 50, 75, 100}。**`6` 就是同步/PBCH 阶段只用中心 6 RB**（PSS/SSS/PBCH 的协议带宽），`75` 是本文件驻留小区的 15 MHz 全带宽。看到 6 不要以为是 1.4 MHz 小区。
- `AgcWorkState` 取值 {3,4,6,9,12,13}（9×35,559、6×6,729、4×5,166…），是厂商状态机编号；配合 `0x7012 SetAgcState` 的跳变看流程。
- `Target` ∈ {3280, 3430, 3620}：AGC 收敛目标（私有刻度）。`MeanPwr` 应收敛到 `Target` 附近，长期偏离即 AGC 未锁定。
- `MeanpwrValid=0`（本文件仅 6 条）时该行功率无效。

**饱和/欠量程判据（日志自带上下限提示）**：

- `0x5C88 SemiStaticAgc: Greater than Max Value … AgcdBGain=65` —— 输入过强，增益压到下限 **65 dB**（14 条）。
- `0x5C89 … Less than Min Value … AgcdBGain=105` —— 输入过弱，增益顶到上限 **105 dB**（5 条）。
- 这两条一旦密集出现，说明仪表功率设置超出 UE 前端量程，先调仪表再谈解调。

#### 26.9.2 模拟增益 + 数字增益

`0x5CA0 TotalGainInfo`（115,860 条）把四类增益并排打印：

```text
AgcGain=[114,104,113,116]        ← 模拟前端 dB（与 RFC 0x6F1D 的 AgcGain 一致）
RaDagcGainLin=[102,102,102,102]  ← 随机接入用数字增益（线性，128=1×）
CsrmDagcGainLin=[128,…]          ← 测量用
CsrsDagcGainLin=[128,…]          ← 搜索用
```

配套：`0x5CA5 RxDagcHandle`（`DagcPwrNormal` 与 `RxDagcLin/RxDagcLog2`）、`0x5C91/0x5C90 CSRS DAGC`（搜索通道 DAGC 计算）、`0x5C9D GetTotalAGCGain`（含 `wTotalAgcDagcRsrpOffset`，**这正是 §8.2 第 ③ 步把内部功率换成 dBm 所用的偏置**）、`0x5CA7/0x5CA8/0x5CAC IntraCaAgc*`（载波聚合时多 CC 的增益协调，`MasterCc` 决定谁做主）。

#### 26.9.3 带宽/频点切换时的增益重载

`0x5C9A ChangeBandwithCalAGCGain: Freq=18500, Bw=[6(cur),100(next)], CurGain=[114,…], NextGain=[96,…]` —— 带宽从 6 RB 变 100 RB，增益要降约 18 dB（功率随带宽增大）。相关：`0x5C99 AgcReload`、`0x5C98 NextSfAgcInfo`、`0x5C9B ChangeNotSyncToSyncState`、`0x5C9C SyncToNotSyncSetAgc`、`0x5C9E FindSaveAgcInfo`（按频点/带宽缓存增益，加快回切）、`0x5C97 CSRSetAGCGain`、`0x5C9F SetFSNewState`。

**判读**：切换后第一个子帧的解调失败往往不是故障，而是增益重载还没生效；要看 `NextGain` 是否被正确应用（下一条 `0x5C8E` 的 `Gain`）。

#### 26.9.4 AFC：从 CFO 估计到晶振控制字

```text
CSRS 0x5881  CFO Result = −126 Hz                （频偏估计，§7.4）
     ↓
DFE 0x5C27 AfcInComingPara: dwAfcMode=1, AfcType=1, Earfcn=1650, FreqOffset=[14,3776], Source=1
DFE 0x5C16 AfcReq: OffsetHz=[3776,14] TmpPpm=[2090,8] CalcPpm=[−156764,−612]
                   CordPpm=[4004,15] CordHz=[7233,28] TotalOffsetHz=−1106     （§2.5 定点换算）
     ↓
RFC 0x6F4E CalcAfcCw: swPpmx1024[902], pdwCoarseCw[53], pdwFineCw[957], numSeg[13], idx[9]
RFC 0x6F10 Drv-Afc_Cfg: Adj100Ppt=−613, InitCw=[50,1268], CalcCfgCw=[53,957], CfgTs=223296
RFC 0x7018 PccAfcCwChgInfo: AfcCw=[0x3bd(pre), 0x3be(Cur)], CordPpmQ8=[…], AfcCwChgFlag=0
RFC 0x6F4D DCXOTemp: sdTmpDegree[38073], sdInitAfcppm[−1326], sdCoef0..3    （温补曲线）
```

- `TotalOffsetHz` 本文件范围 **−8233 … +7569 Hz**，`0x5C16` 共 4,755 次，最常见值 −693 Hz（172 次）——即锁定后残余频偏约 **−0.37 ppm**，正常。
- `AfcCwChgFlag=1` 表示这次真的改了控制字；长期 `0` 而频偏又大，说明 AFC 环没闭合。
- `0x6F4D` 的温度补偿：`sdTmpDegree=38073`（私有刻度）配合 `sdCoef*` 的多项式；温漂问题（长时间挂测后掉网）要看它是否随时间变化。

#### 26.9.5 DC/IQ、FFT 窗与前端配置

| ID | 打印 | 判读 |
|---|---|---|
| `0x5C0F DcEstiRet` | `EstiI=[161,290,−106,−308] EstiQ=[−186,−36,34,−122]` | 各天线直流估计。数值持续很大或某天线明显离群 → 该通道有直流泄漏，会抬高噪底 |
| `0x5C22 DcOffset_Clear` | 切频点时清直流 | 换频点后第一次估计不准是正常的 |
| `0x5C0A Config Cordic Offset` | `CordicOfsHz=13, SpRate=1920000, fcRotVal=−29080` | 数字频偏旋转补偿（85,522 条） |
| `0x5C0D FftPWinOffset` | `FftCfg=[28,72], ReCpBitMap=0x3fff, WinOfs=[0,0,0,0]` | FFT 窗位置；`WinOfs` 非 0 说明在补偿多径/定时误差 |
| `0x5C06 CmnCfg` | `ccSpRate=0(0:1.92M,1:3.84M,…,6:122.88M), ccAntNum=3(0-3), workmode=1(0:NR,1:LTE), scs=0, fcEn=1` | **日志自带完整枚举**：采样率、天线数、制式、子载波间隔 |
| `0x5C02 DfeReq` | `eUid[0](0:DL,1:CSR), eState[0](1:Open,0:Close), SfPattern=[0,15], SyncState=2` | DFE 开关请求，谁在用前端 |
| `0x5C08 CsrReq` | `CsrAntNum / eBwRb=6 / AgcMode / bCellSearch=1 / SyncMode / eCellMeas / MeasBwRb` | 搜索用前端请求 |
| `0x5C07 RxReq` / `0x5C2E ConnectCc2Che` | 接收链与信道估计器绑定 | |
| `0x5C10 MeasFftConfig` | 测量专用 FFT（`MeasRbNum=6`） | 测量只用 6 RB，与 §9.1 的 BandWidth=6 同理 |
| `0x5C29 DfeDebug` / `0x5CA9 AsyncIntMrtr` / `0x5C31 SsIntMrtr`(RFC 列) | 时间戳与调试 ID | 只用于对时序 |
| `0x5C2F Dfe Reset` / `0x5C2C EvtTabOffsetInit` / `0x5C28 IntCfg` / `0x5C1A Lpm` | 复位、事件表、中断与低功耗 | |

### 26.10 RFC：射频前端（72 个 ID，占全文件 57%）

#### 26.10.1 三层结构

| 层 | 主要 ID | 行数 | 能不能读 |
|---|---|---:|---|
| RFFE/RFIC 寄存器事件 | `0x6F48 drvRffe_evt`、`0x6F0F drvRfic_evt`、`0x6F13 RfTu` | 198,153 / 177,967 / 94,397 | **不能逐条解读**：内容是 `地址-数据` 对，需要厂商寄存器手册。只用于确认“确实在写 RF”和统计写入量 |
| DBB/SerDes 链路 | `0x6F44 Ppi`、`0x6F45 RxDlcClc/TxDlcClc`、`0x6F46 RxFifo/TxFifo`、`0x6F59 FrameNum/FrameLenErr`、`0x6F5F DBBV4`、`0x6F54/0x6F55 Rx/TxLaneInfo` | 各约 162,861 | 看**错误计数是否增长**：`RxLaErr`、`FrameLenErr`、`DlcCrc/ClcCrc`、`TXJitter`。全 0 即链路正常 |
| 业务级配置与测量 | `0x702C/0x702D RxCfg`、`0x702A/0x702B TxCfg`、`0x6F1D RfRxPwr`、`0x6F36 RfTxPwr`、`0x7018 AFC` 等 | 数百至数千 | **这一层要逐条读** |

#### 26.10.2 收发通道配置

```text
0x702C RFC|RxCfg Cc[0]: Opt[2], Ret=178257920, Band=3, Bw=14(100k), Freq=18500000(100Hz),
       Duplex=1(0:TDD,1:FDD), TxRxInd=[0,0], Mdm=2, IntraCcNum=1, RxAntNum=4, DbbOrder=[1,2,3,4]
0x702D RFC|RxCfg Cc[0]: RxCcPathId=70, RxAntPathId=[43,47,56,52], RxPort=[5,6,7,8], Pdidx=4
0x702A RFC|TxCfg Cc[0]: Opt[6], Band=3, Bw=150(100k), Freq=17175000(100Hz), Duplex=1,
       TxRxInd=[1,1], Srs=[1,2], TxAntNum=1, Scene=[0,6]
0x702B RFC|TxCfg Cc[0]: TxCcPathId=12, TxAntPathId=[13,0], TxPort=[10,0], AnaIndex=[1,0]
```

- **`Bw` 单位是 100 kHz**：`14` = 1.4 MHz（同步阶段的 6 RB 带宽），`150` = 15 MHz（本小区全带宽）。与 §9.1 的 RB 表示法互相印证。
- `Duplex=1` 必须与小区一致；`RxAntNum=4 / TxAntNum=1` 说明 4 收 1 发，与 `0x9013 AntNum`、CSRM 的四天线 RSRP 一致。
- `Srs=[1,2]` 出现在 TxCfg 里，说明该次配置考虑了 SRS 天线切换。
- 查询/资源类：`0x6F20/0x6F1F RfQueryCcInfoRx/Tx`、`0x6F27/0x6F25 RfMultiCcResInfo`、`0x6F26 RfSingleCcResInfoRx`、`0x7029/0x7028 RFC|Cc [0] Opt/Uid`、`0x6F01 RfReq`（9,369 条，含 `Sence`、`pll`、`port`、`bandGp`）。

#### 26.10.3 接收功率链

`0x6F1D RfRxPwr|Band[3] Freq[18500000] RfTime[241 1226472] LnaLvl[0 0 0 0] DgaIdx[26 16 27 24] PgaIdx[19 19 19 19] AgcLvl[12 12 12 12] AgcGain[114 104 113 116]`（80,416 条）

- `AgcGain` 是四通道总增益 dB（本文件 42–120 dB），与 DFE `0x5C8E` 的 `Gain` 同源。
- `LnaLvl`（LNA 档）、`DgaIdx`/`PgaIdx`（数字/可编程增益索引）、`AgcLvl`（AGC 档位）共同构成增益分配。**四通道 `LnaLvl` 不一致或某通道 `DgaIdx` 顶格**，即通道不平衡的硬件侧证据。
- `0x6F58 RficRssiInfo|FrmRssiPre/Post, AdcRssiPre/Post, AcsFlag, BlockFlag`（80,123 条）：**`BlockFlag=1` 表示检测到阻塞（强干扰）**，`AcsFlag` 为邻道选择性标志。本文件两者恒为 0。
- `0x6F0A GetRxLnaRffeCw` / `0x6F08 GetRxAswRffeCw` / `0x6F56 GetPubSwRffeCw`：从 NV 取 LNA/天线开关/公共开关的 RFFE 命令字；排“某频段收不到”时看 `RxPath`/`Sw` 是否为 0。

#### 26.10.4 发射功率（与上行功控直接相关）

`0x6F36 RfTxPwr|Band[3], TxPwr[−13.0,−13.0][−11.25,0.0], AntOrd[1 2], AntPath[13 0], Pa[2 0], Apt[176 0], Apc[13036 0], Lol[−6 5][0 0], AswSta[1], Bit[0x8], RssiCfg[883 122880 0 0 0]`

- **`TxPwr` 单位是 dBm，可直接读**。本文件 148 次发射配置，范围 **−24.0 … +23.0 dBm**，其中 **23.0 dBm 出现 50 次**——正好是 Power Class 3 的标称最大发射功率（23 dBm ±2 dB，TS 36.101 §6.2.2）。这是判断“UE 是否已经顶格发射”的**硬证据**，比上行功控日志里的标志位更可靠。
- `Pa` 是功放档位（本文件只有 0 和 2 两档，91/57 次）；`Apt` 是包络/平均功率跟踪的电压索引（170–184）。切档瞬间的功率突变属正常。
- 配套：`0x6F09 GetTxAswRffeCw`、`0x6F0B GetTxPaRffeCw`、`0x6F0C GetTxAptRffeCw`、`0x6F50 TxTempComp_AntPath`（温度补偿）、`0x6F52 GetSrsTxAntPath`（SRS 天线路径选择）、`0x7013 TxEnEvt`、`0x700E DbbTxReq`、`0x700F TxDbbCfg`、`0x7010 TxDbbInit`。

**上行没功率/仪表收不到时的固定顺序**：`0x702A/0x702B TxCfg`（频点/带宽/通道对不对） → `0x6F36 TxPwr`（算出来多少 dBm） → `0x6F09/0x6F0B`（开关和 PA 命令字有没有下） → `0x7013 TxEnEvt`（发射使能事件） → 再回上行组看 PUSCH/PUCCH 是否真的映射了数据。

#### 26.10.5 发射定时微调（TA 的硬件末端）

- `0x702E L1l_Rfc_LTXTxTaConfig > cResult=1, swTxTa=−2, wSmallDelay=0, wStaRegUpFlag=1` —— 把 TA 落到发射硬件，`swTxTa` 是微调量（本文件 −2 / +2）。
- `0x703E RxoffsetAcumulatorInfo … Config TA to LTX! MainAcumulator=5, Acumulator=[3,−3,−6,−6], MainAnt=0` —— 各天线接收定时累计偏差，用 `MainAnt` 那一路去驱动发射定时。
- `0x703F TAoffsetCfg CC[0]: SfNum=3 TAOffset=0`、`0x7040 RxMicroOffsetCfg`、`0x6F05 Sched-RxTimeOfs`、`0x6F07 Sched-TxTimeOfs`、`0x6F14 Sched-MeasTimeOfs`、`0x6F43 Sched-RfcTxOffset` —— 收/发/测量三套时间偏置。**上行时序问题（仪表报 UL timing error）先看这一组是否与 MAC TA 命令一致。**

#### 26.10.6 错误与告警（本组唯一必须逐条看的错误打印）

| ID | 打印 | 本文件实证与判读 |
|---|---|---|
| `0x6F1E` | `RF ERR\|code=−2212, parm=[26651000,7,0,0][0,0,0,0]` | **31 条**。第一个参数是频率（100 Hz，2665.1 MHz），第二个是 band(7)。含义：该频段/频点的 RF 请求失败——与 §13 时间线中“band 7 初搜失败”完全吻合。RF ERR 是明确错误，必须解释。 |
| `0x6F4F` | `RficMcuStat\|McuHang, WARNING! Rfic may hang, AliveCnt has not changed! Before:552063996, Current:552068876` | **162,830 条，但其中只有 21 条 `Before == Current`**。也就是说 **99.99% 是误报**：文字说“没变”，实际两个计数明显在变。判据应改为“只在 Before==Current 时告警”，否则整份日志会被这条淹没。 |
| `0x5C30`（RFC 列） | `DFE\|EsIntDelay! CC[0]: ExpectSlot=0, CurSlot=7` | 中断延迟，2 条；偶发可忽略，频繁出现说明负载过重 |
| `0x7016` | `receive MC Rest MSG!` | RFC 收到复位 |

#### 26.10.7 其余 RFC 打印

`0x7041 RFSDMerge`（收发频点/天线的当前与下一状态，163,359 条）、`0x7005 DbbStateDl` / `0x700B DbbStateUl` / `0x700C RxDbbCfg` / `0x7006 MeasDbbCfg`（数字基带时隙与事件位图）、`0x7011 AsyncInfo`（`SyncState/FsNewFlag/FastAgcFlag/CsrsWorkFlag`，47,499 条，**判断当前处于同步还是异步状态最快的一行**）、`0x7001 AlterSpRate`（采样率切换）、`0x7012 SetAgcState`、`0x6F30/0x6F2F RficInfoRx/Tx`、`0x6F34/0x6F33 RfV4ReCfgDlc*`、`0x6F53 UserNVInfo`（**平台与固件版本，163,359 条，每子帧都打**：`platform=MC86001 version=.0.6.6`）、`0x960C/0x9613`（RF 偏置，与 MC 同名打印）。

### 26.11 接口与自定义模块

#### 26.11.1 `PS_PHY接口消息`（2 个 ID）

- `0x8F83 INTF >Send MSG Phy--->Ps, ID=0xD40B, Size=4, at:Frame 241, Subbframe 1.{68(53565,2048)}`（251 条）
- `0x8F82 INTF >Recv MSG Ps--->Phy, ID=0xD306, MsgIdx=13, at:Frame 241, Subbframe 4! C-RNTI_En=0`（185 条）

**用法**：这是 PHY 与协议栈的边界。当“PHY 没做某件事”时，先在这里确认**协议栈到底有没有下发**；`ID` 是平台私有消息号，需要版本接口表，但**收发方向、帧号子帧号和数量**本身就足以定位“是没发还是没收”。

#### 26.11.2 `USER_DEF_VD_12`（8 个 ID）：空闲态测量与脱网

```text
0x9E91 L1mcrRecvMsgInfo: dwMsg_id=53569, L1mcrState=0        （收消息）
0x9E85 L1mcrSendPhy/PSMsgInfo: dwMsg_id=53605                （发消息）
0x9E84 RecvLteIntraMsg: LteCurrentTime=2780, bSuspendPsMsg=0, bIsIdleGapReprot=0
0x9E86 IdleServMeasReport[Freq(0)]: FreqNum=1, Narfcn=1275, wCellNum=1
0x9E87 IdleServMeasReport[Freq(0)]: cell[0]: id:30, rsrp:78, rsrq:28, sinr:60, beamIdx:0
0x9E88 IdleServCellInfo: Narfcn=1275, wCellPhyId=30, wScs=0, rsrp=78, rsrq=28, sinr=60,
       ScellSrxlev=78, ScellSqual=28, scellRvalue=78
0x9E99 LteMaterSendMaskMsgStratFreqInfo: Narfcn=1275, wReselPrio=50, wNewMask=1
0x9EAB OOSCause:6(0:10s expired 1:oosthresh 2:islandthresh 3:phythresh 4:barscell
       5:nomeas 6:10s or bar 7:10s or nomeas 8:10s or nomeas or bar locked cell)
```

- `0x9E87/0x9E88` 的 `rsrp/rsrq` 是**上报域**：`rsrp=78` → −63 dBm，`rsrq=28` → −6 dB（用 §2.7 的公式）。`sinr=60` 是平台刻度。
- `ScellSrxlev / ScellSqual` 对应 TS 36.304 的小区选择判据 **S 准则**：`Srxlev = Q_rxlevmeas − (Q_rxlevmin + Q_rxlevminoffset) − P_compensation − Qoffset_temp`，`Squal = Q_qualmeas − (Q_qualmin + Q_qualminoffset) − Qoffset_temp`；两者都 > 0 才满足驻留条件。`scellRvalue` 是重选排序用的 R 值。
- `0x9E99 wReselPrio=50` 是频点重选优先级（RRC `cellReselectionPriority` 0–7 的平台内部放大表示，需按版本核对）。
- **`0x9EAB OOSCause` 自带完整枚举**，是本文件最重要的单条日志之一：`OOSCause:6` = “10 s 超时或小区被 bar”。它出现在 L3608641（超帧 1106.406.0，约 00:07:39），标志 UE 正式判定**脱离服务（out of service）**，此后进入反复搜网。

#### 26.11.3 `USER_DEF_VD_2`（1 个 ID）

`0x628E` 是 `DLS|DCIInfo` 的详细副本（54 条），字段与下行组同名。本文件实证取值：`TM=4`(30 条)/`2`(24 条)、`RbNum≤68`、`MCS1≤13`、`TBS1` 最大 **9528 bit**、`MimoScheme∈{1,5}`、`MCSTable∈{0,1}`、`TB1Mod∈{2,4}`（QPSK/16QAM）、`TB2Mod=2`、`RxNum=TxNum=4`、`CodewordNum=1`、`TB1LayerNum=1`。

**判读**：`TM=4`（闭环空间复用）却只有 1 个码字 1 层，说明当时按 rank-1 调度；`TBS1=9528 bit` 配 68 RB / 15 MHz，属于该配置下的常规值。详细的 MCS/TBS/HARQ 解释见下行组。

### 26.12 可以直接写进自动化脚本的判据

按“先证伪、后归因”的顺序排列，每条都给出可 grep 的锚点：

| # | 判据 | 锚点 | 阈值/条件 |
|---|---|---|---|
| 1 | 前端量程是否合适 | `0x5C88` / `0x5C89` | 任一在 1 s 内出现 > 5 次 → 仪表功率超出量程 |
| 2 | AGC 是否收敛 | `0x5C8E` | `\|MeanPwr − Target\| > 300` 持续 > 20 ms |
| 3 | 频偏是否收敛 | `0x5C16 TotalOffsetHz` | 稳态 `> ±2000 Hz`（≈1 ppm @1.8 GHz）报警 |
| 4 | RF 是否真的开 | `0x8FEB` 的 `TaskInfo` | 期望业务时 `[SIB]/[RAPC]/[PLMN]` 全 0 且无 DL 开启 → RF 没为业务开 |
| 5 | RF 资源申请是否成功 | `0x9016 Error=[…]` | 任一非 0 |
| 6 | 配置事务是否完成 | `0x900D/0x901F AllDoneFlg` | 0 且 > 100 ms 未变 |
| 7 | 搜索结果 | `0x851B Result` | `-1` 为失败；连续 ≥3 次失败即判定搜网异常 |
| 8 | 频扫是否有产出 | `0x5909 Earfcn Num` | `= 0` → 本轮频扫无候选 |
| 9 | MIB 是否解出 | `0x9E0F Crc` | `0` 且连续 → 同步/信道问题；`1` 但 SIB1 失败 → 转 PDCCH/PDSCH |
| 10 | SIB1 失败定位 | `0x9E0B MainState/StepState` | `StepState=6(Sched)` 占多数 → SI 调度/解码侧 |
| 11 | 测量是否完成 | `0x7E84 dwDoneFlag` | `≠1` |
| 12 | RSRQ 是否可信 | `0x68BC RSSI` | `=[0;0]` → 本次 RSRQ 打问号 |
| 13 | 通道是否平衡 | `0x68C3 RSRP[]` / `0x6F1D LnaLvl` | 四天线 RSRP 极差 > 10 dB 或 LnaLvl 不一致 |
| 14 | 是否顶格发射 | `0x6F36 TxPwr` | `= 23.0` 且持续 → 上行受限 |
| 15 | RF 是否报错 | `0x6F1E RF ERR` | 任意一条都要解释（带 band 与频率） |
| 16 | RFIC 是否真挂 | `0x6F4F` | **仅当 `Before == Current`**（本文件 21/162830） |
| 17 | 是否脱网 | `0x9EAB OOSCause` | 出现即记录原因码 |
| 18 | gap 是否与业务冲突 | `0x848C GapRfState` / `0x9018 ConflictFlg` | gap 期间不应期望本载波数据 |
| 19 | 寻呼位置是否算对 | `0x960A` | 按 §4.4 公式复算 PF/PO |
| 20 | 链路层是否有误码 | `0x6F45/0x6F46/0x6F59` | `RxLaErr/FrameLenErr/DlcCrc` 由 0 变非 0 |

### 26.13 本文件实证结论与完整会话时间线

从 611 个 ID 的首末位置和上述锚点还原出的时间线（序号为文件内记录号，超帧号格式 `外层.SFN.子帧`）：

| 序号 | 超帧号 | 时刻 | 事件 | 依据 |
|---:|---|---|---|---|
| 99–7773 | 1090.241 – 1091.260 | 00:05:50 | 在 EARFCN 1650（1850 MHz, band 3）对 PCI 503/336/287/502 逐个尝试读 SIB1，全部失败 | `0x9E06` × N，`0x9E0B ErrCode=3` |
| 9719 | 1091.268 | 00:05:50 | 收到 `L1L_P_RESET_REQ`，PHY 复位并重设模式（`ResultMap=0x9FF` 全就绪） | `0x9020` → `0x8FB9` |
| 1108627 | 1098.756 | 00:06:52 | 第二次复位 + SetMode，`IratMode` 由 0 变 1 | `0x9020`、`0x9412` |
| 1109594–1139646 | 1098.760 – 1100.842 | 00:06:52 | 初始小区搜索：EARFCN 3201（band 7, 2665.1 MHz）与 6225（band 20, 798.5 MHz）均 `Result:-1` 失败；band 7 同时出现 31 条 `RF ERR code=−2212` | `0x5806`、`0x851B`、`0x6F1E` |
| 1139747 | 1100.842 | 00:06:52 | 转入**全频段频扫**：MP → 500K → 100K → AGC → PSS 五阶段，扫过 20 个 band | `0x845A`、`0x5917`、`0x5924`、`0x5911`、`0x5912` |
| 1574780 | 1101.902 | 00:07:03 | 频扫命中 **EARFCN 1275（1812.5 MHz）PCI 30**，搜索 `Result:0` | `0x851B`、`0x5B06` |
| 1575881–1576172 | 1101.907 – 1101.908 | 00:07:03 | 测量完成 → `PBCH_SUCC_IND=1` → 向 PBCH/MC 请求 SIB1 | `0x851C`、`0x8506`、`0x9E06` |
| 1581480 | 1102.880 | 00:07:03 | **驻留成功**：`Earfcn=1275, CellId=30, TxAntNum=4, SysDlBw=75(15 MHz), PhichCfg=6`；`wCampOn=1, PiPeriod=128` | `0x8FD3`、`0x8449` |
| 1581816 | 1102.881 | 00:07:03 | 寻呼参数计算：T=128、N=128、UeID=277 → PF SFN mod 128 = 21、PO 子帧 9（与协议公式一致） | `0x960A` |
| 1581856 | 1102.881 | 00:07:03 | MC 向 RAPC 发 `ACCESS_REQ`，启动随机接入 | `0x8F9F` |
| 1590349–1590426 | 1102.895 | 00:07:04 | `ProcCG` 载波组配置 → **`DEDICATED First Config` 进入连接态** | `0x9F11`、`0x8F07` |
| 1590481 | 1102.895 | 00:07:04 | `SIB Fail`（MainState=3 Abort）——因进入专用配置而主动中止 SI 读取，**属正常** | `0x9E1D`、`0x9E0B` |
| ~1593000+ | 1102.9xx | 00:07:04 | 连接态测量开始；此后 `0x8646` 的 RSRQ 上报值恒为 34（见 §6.4 的存疑点） | `0x68BC/0x68C3/0x8646` |
| 2447295–2447383 | 1104.156 | 00:07:17 | **增加 SCell**：`AddCCNum=1`，EARFCN 1900（1875 MHz）PCI 47，`wServCellNum=2` | `0x9F11`、`0x8445`、`0x858A` |
| 2470638 | 1104.177 | 00:07:17 | SCC 同频测量：SCell RSRP 上报 77（−64 dBm） | `0x8665` |
| 2480978–2481104 | 1104.186 | 00:07:17 | **释放 SCell**（`RelCCNum=1`），`Lv3MasterState=4 CONN_SUSPEND` | `0x9F11`、`0x9F15`、`0x8F88` |
| 2491294–2495726 | 1104.209 – 1104.219 | 00:07:17 | 重新搜索/测量/PBCH/SIB1 并再次驻留、再次 `ACCESS_REQ` | `0x851B`…`0x8FA2` |
| 2550105–2561992 | 1104.318 – 1104.352 | 00:07:19 | 又一轮复位（`Lv3MasterState=3 CONN`）→ 搜索 → 驻留 → 测量配置 → 接入 | `0x9020`、`0x844B` |
| 2615969 / 3167662 | 1104.451 / 1105.527 | 00:07:20 / 00:07:31 | 两轮 `MAC_RESET_REQ`（各 2 条） | `0x8F89` |
| 3109089 | 1105.427 | 00:07:28 | 再次 `ACCESS_REQ` | `0x8F9F` |
| **3608641** | **1106.406** | **00:07:39** | **`OOSCause:6` —— 判定脱离服务（10 s 超时或小区被 bar）** | `0x9EAB` |
| 3617065–3624603 | 1107.428 – 1107.448 | 00:07:40 | 回原频点 1275 搜索，连续 `Result:-1` | `0x851B` |
| 3624710–3905179 | 1107.448 – 1109.113 | 00:07:40 – 00:07:47 | 连续三轮频扫；`0x5909 Earfcn Num = 0`（一个候选都没报出来） | `0x845A`、`0x5909` |
| 4002083–4035671 | 1111.350 – 1112.961 | 00:07:49 – 00:07:51 | 命中 EARFCN 2452（band 5）PCI 125/123/219 → PBCH OK → **SIB1 全部失败** → 再频扫 | `0x851B`、`0x8506`、`0x9E1D` |
| 4320242–4394861 | 1114.587 – 1117.811 | 00:07:56 – 00:08:00 | 依次尝试 1850 / 1500 / 1600 / 1650：1500 与 1600 `PBCH_FAIL_IND=0`；1650 上 PCI 503/417/453/303 的 SIB1 全部 `ErrCode=3` | `0x8506`、`0x9E06`、`0x9E0B` |
| 5249897–5594301 | 1144.755 – 1152.547 | 00:08:20 – 00:08:33 | 继续在 3590 / 8842 / 3745 等频点重复“搜到 → PBCH OK → SIB1 Fail → 再频扫”的循环，直到日志结束 | `0x851B`、`0x9E1D`、`0x845A` |
| 5865466 | 1152.678 | 00:08:33 | 日志结束，最后一行仍是每子帧的 `UserNVInfo` | `0x6F53` |

**由此得出的三条结论**：

1. **本文件不是一份"业务正常"的日志**：只有 00:07:03–00:07:39 约 36 秒处于驻留/连接态（其间下行、上行、PHICH 链路正常，见下行组与上行组），其余约 2 分钟都在复位、搜网和读 SIB1 失败。分析任何 PHY 指标前，必须先确认取样窗口落在哪一段。
2. **失败的一致形态是"能同步、不能读 SIB1"**：多个频点上 PSS/SSS/PBCH 都成功（`0x8506=1`、`0x9E0F Crc=1`），但 SIB1 始终失败且 `StepState` 多为 6(Sched)。这把嫌疑集中在 **SI-RNTI 的 PDCCH 检测与 SIB1 的 PDSCH 解码**，而不是同步或射频；下一步应查下行组的 PDCCH/PDSCH 与仪表的 SIB1 发送配置（是否真的在发、周期与窗长是否匹配）。
3. **射频侧无硬故障证据**：`RF ERR` 仅 31 条且集中在 band 7 初搜；`McuHang` 16 万条中 99.99% 是误报；DBB 链路错误计数全 0；AGC/AFC 均在正常范围（残余频偏约 −0.37 ppm）。因此不应把本次失败归因于 RF 硬件。

## 附录 A：611 个消息 ID 全量字典

本附录覆盖全文件 611 个 `模块+消息ID`，每个恰好出现一次。其中 367 个的“是什么/主要作用”为逐条人工核定（结论见 §24–§26），其余 244 个按函数名与字段族规则分类。`结构数` 是将动态数字/十六进制数替换后仍不同的文本结构数；大于1可能表示同一ID含分支或数组布局变化。典型原文只用于识别日志，所有动态值的完整分布请查 `normalized_template_index.tsv`。用途提示由函数名和字段族分类，厂商私有寄存器不能据此反推标准枚举。

### A.1 `CMN`（65个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x9000` | 51681 | 1 | `PUB\|PHY->PHY SEND:MsgId=0x2b9c(11164) NS=1[N L LV NV W] SrcCore=0x0 SrcThread=0x0 DecCore=0x3 DecThread=0x4 MsgBuffLen=0` | PHY 内部消息总线包络：MsgId 与源/目的核线程；按 MsgId 过滤可追消息去向。 |
| `0x8FEB` | 42275 | 2 | `SchedMc\|INFO > OpenRF >>> Time: 241- 9 > Open DL in SubFrm: 0,OpenCcEn:0,State:2,TaskInfo:[SIB]=1,[RAPC]=0,[PLMN]=0,[PreSync]=0,DrxCloseRf=0,DLbmp=[0x00000000,0x00000000][0,0,2]` | 开 RF：TaskInfo 说明这次为谁开（SIB/RAPC/PLMN/PreSync），DLbmp 为下行位图。 |
| `0x8FCB` | 2245 | 1 | `SetWorkingSysBand[0],CAlterRate = 0,eSysBandwidth = [5 --> 0] [20M:0],sdwAdjustTs = 0` | 工作系统带宽切换（eSysBandwidth 前后值）。 |
| `0x9021` | 2148 | 1 | `CMN\|SDL:SdlIdx=3,ProcId = 255,ActiveFlg=1,CurStep=0,TableNum=4,KeyPara = [ 0, 3, 1058, 1, 0, 0,]` | 公共配置事务 SDL 分步下发（SdlIdx/CurStep/TableNum）。 |
| `0x9018` | 728 | 1 | `CMN\|RF[SLOTL-RX]:RtxMasterType=0 OrgState=7 State=7 FreeMrtr_CurSlotOrSf=954645044 SwitchFreeMrtr=954884408 ConflictFlg=0 state:7,tCurFreeTime = 15.6400,dwCurPpIdx=1,Frame=241,eScene=3,wPpAlter=0` | 时隙级 RF 归属（接收）；ConflictFlg 非 0 表示同一时隙抢 RF。 |
| `0x8FE5` | 392 | 1 | `SchedMc\|INFO > CloseRF >>> 241- 1 > Stop RF Rx Receive by Carrier 0,RxMask:[1]CurWorkInd:0` | 停止 RF 接收（按载波与 RxMask）。 |
| `0x902C` | 375 | 1 | `zPHY_emc_WakeUpPS> PHY2PS message Write Msg Ret value = 0,Msg len=0,PS sleep flag=0(1:sleep;0:work),LTE Mode=1.` | 唤醒协议栈；PS sleep flag=1 表示对端在睡。 |
| `0x8FBB` | 260 | 1 | `INF >LTE PHY State: RatMode:(1 ,1),Lv1State:2,Lv2State=1,Lv3MasterState=2,Lv3SlaveState=0.` | PHY 三级状态总览（RatMode=1 为 FDD）。 |
| `0x9012` | 249 | 1 | `CMN\|RF[Req-Rx]: Step=1 MasterType=0 Scene=3 RxNum=1 State=1 HwHasChoiseFlg=65729 FreeMrtr=[954884408 954884408]` | RF 接收资源申请步骤与场景。 |
| `0x9016` | 249 | 1 | `CMN\|RF[Req-RxRst]:Step=1 State=7 Return=4 FreeMrtr=954884408 CcNum=1 [SvCellId Ant SwitchFlg\|IntraFlg\|V4Flg]=[0 4 0x0][1 4 0x20][0 0 0x0][0 0 0x0][0 0 0x0][0 0 0x0],Error = [0,0,0,0,0]` | RF 接收申请结果；Error 数组全 0 才是成功。 |
| `0x9008` | 245 | 1 | `CMN\|SDL[EndSDL]:MasterType=0 Core=3` | SDL 事务结束。 |
| `0x9013` | 245 | 1 | `CMN\|RF[Req-Rx]: SrvCellId=0 Core=3 SpFlg=1 Scs=0 Band=3 BWEnum=0 AntNum=255 Hw=4 Freq=18500 Dup=1 CompTs=0 IntraCc[Flg=0 Id=0 Freq=0 BW=0 P=0 PSC=0] DbbOd[1 2 3 4]` | RF 接收资源参数：band/带宽枚举/天线数/硬件/频点/双工。 |
| `0x9010` | 232 | 1 | `CMN\|RF[TRY-RX]: MasterType=0 CacheReqP=0x0 CurCacheNum=0 State=0 ,Ret = 0` | RF 接收资源试探申请（缓存与状态）。 |
| `0x5001` | 221 | 1 | `SetRxMask:CcIdx[0]:(1->0);Mode=5[0:Mc,1:Rx,2:Csrc,3:Csrs,4:Csrm,5:Pbch,6:Sib].` | 接收权归属切换；Mode=0:Mc 1:Rx 2:Csrc 3:Csrs 4:Csrm 5:Pbch 6:Sib（自带枚举）。 |
| `0x8F94` | 184 | 1 | `INTF > 13-0, Lte Phy Recv Msg: MsgID =0x D306, CF =0x 0 2000!!` | PHY 收到协议栈消息（MsgID）。 |
| `0x8F96` | 184 | 1 | `INTF >ClearSyncMsg, CF=0x 0 2000, wIndex=0,dwPhyMsgId=54022,dwMsgSize=12.` | 清理同步消息。 |
| `0x902B` | 127 | 1 | `zPHY_emc_SendIcpToPS> PHY2PS message Write Msg Ret value = 0,Msg len=0.` | PHY 向协议栈发 ICP 消息。 |
| `0x8FD1` | 108 | 1 | `AlterRateRefreshFB[0],CAlterRate = 0,eSysBandwidth = [5 --> 5] [20M:0],tBoundary = 0.19.11424` | 带宽切换后刷新帧边界。 |
| `0x8FE8` | 106 | 1 | `SchedMc\|INFO > Config RF Work Info > CcIdx=0,CfgSubFrm=9,Freq=18500,BandNum=3,Bw=5,UDMode=0,SpecMode=0,CpMode=0` | 配置 RF 工作参数：频点、band、带宽、双工与 CP。 |
| `0x8FC1` | 102 | 3 | `INF >LTE PHY SID Info: FrameNum= 241, SubFrameNum= 4, CellId= 0, DlFreq= 0, UlFreq= 0, SubFrmAssign=-1, SpeSubFrmPat=-1, CfoCirMaskFlag=[1],CampOn=0!` | PHY SID 信息：帧/子帧、小区、上下行频点与 CampOn。 |
| `0x8FEA` | 45 | 3 | `SchedMc\|INFO > 242- 7 >>> CcIdx=0, Calc Open RF Time: 242- 9,tReferTime: 242-14(Slot) + 4(Slot),CalcOver=1,RfOpenEn=0` | 计算开 RF 的提前时间；RfOpenEn=0 表示算完不开。 |
| `0x900D` | 28 | 1 | `CMN\|DEDISDL[DediProc-Mod]:MasterType=0 Core=3 ModId=4[CSR UL DL RF CSI] ModDoneMap=[0x0 0x0 0x0 0x10] AllDoneFlg=1` | 专用配置各模块完成位图；AllDoneFlg=1 才算下发成功。 |
| `0x8FAF` | 24 | 1 | `zPHY_emc_ProTASchedFlow > TACmdType=0,wSubFN=8993, wCrutSubFN=6, wAdjSubFN=9,wTimeAdvance=31, TxOffset=0` | TA 调度流程；wTimeAdvance=31 对应 TA 命令“保持不变”。 |
| `0x9014` | 17 | 1 | `CMN\|RF[Req-Tx]: Step=3 MasterType=0 Scene=1 RxNum=0 State=1 HwHasChoiseFlg=1 FreeMrtr=[988307896 988307896]` | RF 发射资源申请步骤。 |
| `0x9017` | 17 | 1 | `CMN\|RF[Req-TxRst]:Step=3 State=0 Return=0 FreeMrtr=0 CcNum=0 [SvCellId UlSul Ant SwitchFlg\|IntraFlg\|V4Flg]=[0 0 0 0x0][0 0 0 0x0][0 0 0 0x0],Error = [0,0,0,0,0]` | RF 发射申请结果；看 Error 数组。 |
| `0x901A` | 16 | 1 | `CMN\|RF[SLOTL-TX]:RtxMasterType=0 OrgState=7 State=7 FreeMrtr_CurSlotOrSf=1130032848 SwitchFreeMrtr=1130268324 ConflictFlg=0,tR2IMrtr=880.13.6400,wPpAlter=0` | 时隙级 RF 归属（发射）。 |
| `0x900A` | 13 | 1 | `CMN\|RF[TRY-RTX]: MasterType=0 CacheReqP=0x0 Rx[CurCacheNum=0 State=0] Tx[CurCacheNum=0 State=0]` | 收发同时申请 RF。 |
| `0x9015` | 13 | 1 | `CMN\|RF[Req-Tx]: SrvCellId=0 UlSul=0 Core=3 SpFlg=1 Scs=0 Band=3 BWEnum=4 AntNum=255 Hw=1 Freq=17175 Dup=1 CompTs=0 IntraCc[Flg=0 Id=0 Freq=0 BW=0 P=0 PSC=0] DbbOd[1 2 3 4]` | RF 发射资源参数（含上行频点）。 |
| `0x9F04` | 10 | 1 | `PUB\|AllTask:Sence=11 RefTaskAddrOffset=0 TaskNum=1 [0 301012][0 0][0 0][0 0][0 0][0 0] [AddrOffset N \|N \|CoreId \|DapsId \|NS \|MasterType \|Type(addcg proccc delcg)\|State(idle wait set)]` | 任务表总览（场景与任务数）。 |
| `0x901F` | 7 | 1 | `CMN\|SDL[RelProc-Mod]:MasterType=0 Core=3 ModId=4[CSR UL DL RF CSI] ModDoneMap=[0x0 0x0 0x0 0x90] CurCoreDone=0 AllDoneFlg=0` | 释放流程中各模块完成位图。 |
| `0x9020` | 6 | 1 | `PS-MSG\|RELSDL[Start]: RtxMstType=0 Core=3 F_Slot:53531-0, ----->>>>>Nr Phy Recv Msg: L1L_P_RESET_REQ MsgID =0x 0,RelScellProcFlg=0` | 收到 L1L_P_RESET_REQ，开始释放/复位事务。 |
| `0x9022` | 6 | 1 | `CMN\|RELSDL[StartMstRel]:RtxMstType=0 Core=3` | PHY 公共消息或配置编排。 |
| `0x8FF8` | 5 | 1 | `CMN\|SDL[CalcRelCcInfo]:RtxMstType=0 Core=3 RelOrStopCoreNum=0 Core0[ID=0 RelCcNum=0 [0 0] StopCcNum=0 [0 0]] Core1[ID=0 RelCcNum=0 [0 0] StopCcNum=0 [0 0]] Core2[ID=0 RelCcNum=0 [0 0] StopCcNum=0 [0 0]]` | PHY 公共消息或配置编排。 |
| `0x8FFA` | 5 | 1 | `CMN\|SDL[PrtPubRst]:MasterType=0 Core=3 OUT:[CoreNum=1 [CoreId CCMap]=[3 1][0 0][0 0][0 0] RelCoreNum=0 [CoreId]=[0][0][0][0]]` | PHY 公共消息或配置编排。 |
| `0x902A` | 5 | 1 | `CMN\|CgInfo: dwServCellIdx=0,eCoreId=3,dwSpCellFlg=1,Band=3,dwBW=15,RxFlg=0,UlFlag=1,IntraCc=[0,0],HwAssign = [1,4,1,1,1]DLBund = [0,0,0],` | 载波组上下文：服务小区索引、band、带宽、上行标志与硬件分配。 |
| `0x9F11` | 5 | 1 | `PUB\|ProcCG Start:Type=0[S0M S0S S1M S1S LTEV NRV] Ns=1[Nr Lte Ltev Nrv] DapsId=0 MmW=0 RelCCNum=0 AddCCNum=0 RelSrvCellMap=0x0` | 载波组处理开始：AddCCNum/RelCCNum 表明本次加/删几条载波。 |
| `0x9F13` | 5 | 1 | `PUB\|ProcCG PubProc:MstType=0 Ns=1 Core=3 Daps=0 CellNum=1 CoreNum=1 RelCoreNum=0 RelCoreMap=0x0 RelOrModCCNum=0` | 载波组分发到各核。 |
| `0x9F14` | 5 | 1 | `PUB\| Rst:Core=3 CcNum=1 CC0[SrvCell=0 Type=0 Band=3 Bw=1 Scs=0 2RxFlg=0 Ul=[1 0] Sp=1 DLBd=0 HW=81848180 81810000] CC1[SrvCell=0 Type=0 Band=0 Bw=0 Scs=0 2RxFlg=0 Ul=[0 0] Sp=0 DLBd=0 HW=0 0]HW[[TPU DFE CSR P…` | 载波组处理结果（每 CC 的硬件分配）。 |
| `0x9F16` | 5 | 1 | `PUB\|ProcCC ModProc:MasterType=0 Ns=1 DapsId=0 TaskAddrOffset=0 MasterCore=3 CGCore=3 CurCore=0 CoreProMap=0x0` | 载波修改流程。 |
| `0x9F17` | 5 | 1 | `PUB\| Rst:Core=3 CcNum=1 CC0[SrvCell=0 Type=0 Band=3 Bw=1 Scs=0 2RxFlg=0 Ul=[1 0] Sp=1 DLBd=0 HW=81848180 81810000] CC1[SrvCell=0 Type=0 Band=0 Bw=0 Scs=0 2RxFlg=0 Ul=[0 0] Sp=0 DLBd=0 HW=0 0] HW[[TPU DFE CSR …` | 载波修改结果。 |
| `0x8A00` | 4 | 1 | `Dedicate: CtrlFlag = 0,DRXINFO[0 -1 0 0 0 0 0] ActvieFlag = 0 C-RNTI = 0, LTEA PHY SYS INFO: PS[0] TH[0] SLEEP[0] ZSPCLK[0x 0]` | PHY 公共消息或配置编排。 |
| `0x8F89` | 4 | 1 | `INF >Recv ZPS_LTE_P_MAC_RESET_REQ_EV, C-RNTI_En= 0, g_zPHY_tWakeupTimerInfo.dwAwakeTimer[E_ACCESS_T300_TIMER] = 0,dwStayAwakeTime[E_ACCESS_T300_TIMER]=0.` | 收到 MAC_RESET_REQ（含 C-RNTI 使能与唤醒定时器）。 |
| `0x8FB5` | 4 | 1 | `DBG >zPHY_emc_ProResetFlow > Recevie SubSystem Soft Reset Cnf Msg(MsgId=0x2B15) not at L1_INI State, PhyLv1State=2, McCtrlState=0!` | 在非 L1_INI 状态收到子系统软复位确认（时序告警）。 |
| `0x8FB7` | 4 | 1 | `DBG >zPHY_emc_ProSetModeFlow > Start, g_zPHY_emc_tMcCtrlParam.wSetModeResultMap = 0x0!` | SetMode 流程开始。 |
| `0x8FB9` | 4 | 1 | `DBG >zPHY_emc_ProSetModeFlow > Down, g_zPHY_emc_tMcCtrlParam.wSetModeResultMap = 0x9FF!` | SetMode 流程完成；0x9FF 表示全部子模块就绪。 |
| `0x9007` | 4 | 1 | `CMN\|DEDISDL[StartDediProc]:RtxMstType=0 Core=3 CoreNum=1 Core0[ID=3 CcNum=1 [0 0]] Core1[ID=0 CcNum=0 [0 0]] Core2[ID=0 CcNum=0 [0 0]]` | 专用配置事务开始（各核载波清单）。 |
| `0x8F88` | 3 | 1 | `INF >LTE PHY SID Info: Lv3MasterState:4 [0=CLOSE, 1=INIT, 2=IDLE, 3=CONN,4=CONN_SUSPEND], CfoCirMaskFlag=[1].` | Lv3 主状态：0=CLOSE 1=INIT 2=IDLE 3=CONN 4=CONN_SUSPEND（自带枚举）。 |
| `0x8F9F` | 3 | 1 | `DBG > zPHY_emc_ProAccessMsg > MC Send ACCESS_REQ Msg to RAPC, at:Frame 881,Subbframe 6，dwRfcOpenEventRegsted = 1, RfcRxOpenFlag= 0!` | MC 向 RAPC 发 ACCESS_REQ：发起随机接入的时刻。 |
| `0x8FA5` | 3 | 1 | `DBG >zPHY_emc_ProReleaseFlow: Init SID & SAD!` | PHY 公共消息或配置编排。 |
| `0x8FAA` | 3 | 1 | `DBG >ProTimingCtrlFlow>Serve Cell Info: CellId= 30, DlFreq= 1275, BCH Boundry= 39. 2. 4574!inttype=0,` | 服务小区定时控制信息（含 BCH 边界）。 |
| `0x8FB0` | 3 | 1 | `zPHY_emc_ProTASchedFlow > TACmdType=1,wSubFN=8839, wCrutSubFN=1, wAdjSubFN=5,wTimeAdvance=0, TxOffset=0` | TA 调度流程（另一命令类型）。 |
| `0x8FB1` | 3 | 1 | `DBG > 268- 7, -- > Del Open RF Event, RfcOpenEvent Regsted= 0, RxOpenFlag= 0, RxCurrState = 2, PBCH_BLINDING_STATE = 2` | PHY 公共消息或配置编排。 |
| `0x8FB2` | 3 | 1 | `DBG >zPHY_emc_ProResetFlow > Start, g_zPHY_emc_tMcCtrlParam.wResetResultMap = 0x1!` | 复位流程开始（ResultMap 初值）。 |
| `0x8FB4` | 3 | 1 | `DBG >zPHY_emc_ProResetFlow > Down, g_zPHY_emc_tMcCtrlParam.wResetResultMap = 0x9FF!` | 复位流程完成；0x9FF 表示全部子模块就绪。 |
| `0x8FD3` | 3 | 1 | `INF >MC_Task: GetCellInfo(1[1:Ok;Other:Err]);Mode=(1,1),Earfcn= 1275,CellId= 30,Cp=0,TxAntNum=4,SysDlBw= 75,PhichCfg=6 BCH Boundry= 39. 2. 4574!.` | 驻留小区身份证：Earfcn/CellId/CP/TxAntNum/SysDlBw/PhichCfg/BCH 边界。 |
| `0x8FD9` | 3 | 1 | `DBG > zPHY_emc_TaskEntry > MC Send comm_ho Msg to PHY MDLS,at:Frame 881,Subbframe 6, wResult 0,C-RNTI = 0!` | PHY 公共消息或配置编排。 |
| `0x8FE4` | 3 | 1 | `DBG > zPHY_emc_TaskEntry > tx Earfcn: 17175 FRE:17175` | PHY 公共消息或配置编排。 |
| `0x8FFE` | 2 | 1 | `CMN\|SDL[WaitRelCnf]:MasterType=0 Core=3 MsgId=1058[E_L1_MSG_ASS2MST_REL_CNF=14400]` | PHY 公共消息或配置编排。 |
| `0x8F9E` | 1 | 1 | `DBG > zPHY_emc_ProAccessMsg > store the RAR message and wait for the COMMON message!` | 暂存 RAR 消息等待 COMMON 消息。 |
| `0x8FA2` | 1 | 1 | `DBG > zPHY_emc_TaskEntry > Recv ZPHY_ETPU_EMC_COMM_HO_MSG_DISPATH, MC Send ACCESS_REQ Msg to RAPC,at:Frame 219,Subbframe 5!` | 收到切换消息分发后发 ACCESS_REQ。 |
| `0x8FFD` | 1 | 1 | `CMN\|SDL[StartRelProc]:MasterType=0 Core=3 RelOrStopCoreNum=1 ID=[3 0 0]` | PHY 公共消息或配置编排。 |
| `0x8FFF` | 1 | 1 | `CMN\|SDL[EndRelProc]:MasterType=0 Core=3` | PHY 公共消息或配置编排。 |
| `0x9028` | 1 | 1 | `Warning > zPHY_emc_ProSyncMsgSend > COMM Receive In NeibourCell SI Read!! CommInSiProc = 0` | 读邻区 SI 时收到公共消息的竞态告警。 |
| `0x9F12` | 1 | 1 | `PUB\| Idx=0 :SrvCell=1 Band=3 BW=10 Scs=0 2RxFlg=0 HasUl=1 SpCell=0` | 载波组中每条载波的参数（band/带宽/是否 SpCell）。 |
| `0x9F15` | 1 | 1 | `PUB\| RelOrModCC:[servcellidx map]=[1 0xff][0 0x0][0 0x0][0 0x0][0 0x0][0 0x0]` | 本次释放/修改的载波位图。 |

### A.2 `CSI`（9个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x5723` | 13385 | 1 | `CSI_FLOW: First_FdBkCfg_IN` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x5745` | 13385 | 1 | `CSI_FLOW:AperRepJudge_IN:wAperiodReportFlag:0,g_Sym4IntFlag:1,g_Sym9IntFlag:0` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x5760` | 13385 | 1 | `CSI_FLOW:First_FdBkCfg_IN_CAIdx:0,wAperiodTrigger:0,wCsiEn_aop:0,MultiComFlag[2]:0` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x5763` | 2682 | 1 | `APER_Comb:riBitLenM: 0,riBitLenS: 4,riBitLenS1: 2,riValue[ 0- 2- 2]` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x574B` | 2677 | 1 | `CSI_FLOW:RLM_FdBkFirCfg:wStepIdx:0,wHwIdx:1,wTxAnPortsNum:4,wUeRxAttennaNum:4` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x574D` | 2677 | 1 | `CSI_FLOW:RLM_Calc_In:dwRLMValue:0xe04b5,ptEsnrInfo->sdwSnrValue:112` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x5756` | 2677 | 1 | `CSI_FLOW: FbHRConfig_wRDiagInd:1, wHTransIdx:0,wRTransDiagIdx:0` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x5759` | 2677 | 1 | `CSI_FLOW:PCellCSI_En_IN` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x575B` | 2677 | 1 | `CSI_FLOW:First_FdBkCfg_INT1_CAIdx:0,wTransMode:2,wCsiEn_aop:0,g_wScellComFlag:[0-0-0]` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |

### A.3 `CSRC`（91个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x8485` | 13343 | 1 | `CcSrcWorkCheck ValidNum=4, cCcIdx=0, ValidCc=1, wSearchStartDone=1, SearchMeasWorkFlag=1` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x8484` | 13048 | 1 | `SfProcWorkFlag bGapCfgState=0, bInterFreqChange=1, wInterSearchMeasWorkFlag=0` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x8420` | 4304 | 1 | `SchedCsrc_T1To2Dist,time1: 463 2,time2: 0 0,distance: -9262` | 调度时间差计算（内部）。 |
| `0x5917` | 3256 | 1 | `<FreqScan>[Result]Band=1,FreqPoint=21105,MaxValue=11772 Gain=110 110 PssFlag = 7 FsState = 0[0:MP 1:500K 2:100K 3:Re500K 4:AGC 5:Re100K 6:Discrete PSS] CsFlag = 0,DisFreqScan = 0` | 频扫结果行：FsState=0:MP 1:500K 2:100K 3:Re500K 4:AGC 5:Re100K 6:离散PSS（自带枚举）；峰值必须同增益比较。 |
| `0x8478` | 2077 | 1 | `Csrc_Idx: CurCCcIndex=0, ServeCellIdx=0, PubCcIdx=0, TpuIdx=1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8678` | 1379 | 1 | `InterMeasdebug_CtrlMeasFilterReq:wCarIndex=0,MainState=9,wIntraInterInd0=0,wIntraInterInd1=0,wIntraCount=1,wIntraMeasCnt=0,neireport=0` | 异频测量过滤控制流程状态。 |
| `0x8490` | 584 | 1 | `CSR_NormalTemp=37` | CSR 模块温度读数。 |
| `0x8668` | 361 | 1 | `FilterIntraDebugInfo:CellId[0] = 30,FilterRSRP = 68.5,RSRQ = 29.7` | 过滤后的 RSRP/RSRQ（上报域，带一位小数）。 |
| `0x8658` | 359 | 1 | `FilterInfo[0] >:wCellNum =1, cell0:[30,69] ,cell1:[0,0], wNeiReport =0` | 过滤后小区列表（PCI 与上报值）。 |
| `0x8667` | 358 | 1 | `FilterIntraDebugInfo:wCarIndex = 0[0 1],dwEarfcn = 1275,AdaptFilterType = 2,wFlagCounter = 0,RSRP_K = [8,64],RSRQ_K = [7,76],SF = [76] S=[76,76 76,76] IN[0 0]` | L3 过滤内部系数与计数；不要当协议 filterCoefficient 直接读。 |
| `0x8646` | 351 | 1 | `WriteServingCellResult >wCarIndex = 0,SrvCellInfo-ID[30]RSRP=[70][34] RSSI=[-22639][-22594] PreRead = 0 Rxflag = 0,ScheduleInfoCnt = 0 wAgeEventCnt =5` | 服务小区测量结果写入：RSRP=[上报值][RSRQ上报值]，RSSI 为未补偿内部值。 |
| `0x8671` | 229 | 4 | `L1l_SchedCsrc_FindCell > no cell!!!! Earfcn= 0, CellId= 0.` | 查库未命中（no cell）；单独出现不代表故障。 |
| `0x847E` | 226 | 4 | `Csrc_UpdateBackBchBnd: dwEarfcn=2452, BackMibParadwEarfcn=2452, DestTimeMrtr= 535.19.10132` | 更新回退小区的 BCH 边界。 |
| `0x8403` | 218 | 1 | `PhyMode Config :g_zPHY_SID.uPhyRatMode= (1)[0:Tdd, 1:FDD], dwId = 21,dwMode = 1,tCsrMainState = 0` | PHY 制式配置（0:TDD 1:FDD）。 |
| `0x8676` | 206 | 1 | `IntraRSSI: RSSI:[-60.9],TempRssi[-15124,0]` | 同频 RSSI：左为补偿后 dBm，右为未补偿内部对数值。 |
| `0x8583` | 200 | 22 | `Add CellToDB(01):Pos:1,Earfcn= 1900,CellId=226,Finger= 995,FB= 0.11.14528,FbAge= 0,Throld=[ 80, 4, 0, 0, 80 4],MeasAge=2560,NewFlag=0,CellNum= 2,DelayCfg=[800,800.` | 新小区入库：Finger 峰值、FB 帧边界、老化计数与门限。 |
| `0x5924` | 184 | 1 | `<FreqScan>[MP report]BW = 1979,FreqPoint = [34015 35994] Earfcn = [41605 43584] FreqBandNo = 42,dwMaxValMean = 68557!}` | 频扫 MP 阶段宽带能量报告（频点区间与均值）。 |
| `0x5927` | 184 | 1 | `<FreqScan>MpInfo::BW = 28,LeftFreqPoint = 21567,RightFreqPoint = 21595,FreqBandNo = 1, MaxPeakVal = 6396` | 频扫 MP 详细信息（左右频点与峰值）。 |
| `0x8589` | 134 | 1 | `L1l_SchedCsrc_CtrlUpdateBoundary:AdjustTs = 0.` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8469` | 130 | 1 | `<SetSearchPhase[Earfcn:3201],bRet=0,wIndex = 1,wSearchState = 2 [1:PSS 2:SSS],wSearchEnable=0,wMeasEnable=0` | 小区频扫/PSS/SSS 搜索与同步。 |
| `0x841F` | 103 | 1 | `SchedCsrc_T1To2DistTs: Time1=[903. 0. 0], Time2=[903. 8. 1462], Distance=124342.` | 两个时间点之间的 Ts 距离。 |
| `0x590B` | 92 | 1 | `<FreqScan>>@@@@@@@@@@@:Delete_[PointValid]:FreqPoint=8645,MaxPeakValue=54,FreqBandNo=26 dwSlope = 620 @@@@@@@@@@@:!` | 删除无效频点（含斜率判据）。 |
| `0x8404` | 91 | 1 | `L1l_SchedCsrc_TpuEventMark: dwEventId = 0x460,Event = [1,0,0],dwDelayTs = 212087,DelaySubFrame = 6` | 小区搜索控制/候选数据库/载波管理。 |
| `0x852C` | 87 | 6 | `ProWriteBch2CsrDb>PBCH:Cell_1Info = [ 2452,125, 535.19.11744,1],Cell_2Info = [ 8842,125, 535.19.11744,1]` | 把 PBCH 结果写回小区库。 |
| `0x866E` | 85 | 1 | `zPHY_ecsrc_CsrSNR>>Snr[0]=39.1,Snr[1]=39.6,Snr[2]=39.8,Snr[3]=36.1` | 四天线 CRS SNR，单位 dB（一位小数）。 |
| `0x8663` | 84 | 2 | `#INTRA#MEAS:PCC Intra-Meas:Cell INFO >Earfcn = 1275, Cell[0],ID= 30,Rsrp= 75,Rsrq= 34,MeasAge= 2,SearchAge=114 MobileCxt= 1` | PCC 同频测量对外结果：Rsrp/Rsrq 为上报域，MeasAge/SearchAge 是新鲜度。 |
| `0x8648` | 76 | 1 | `L1l_SchedCsrm_CtrlWriteSccMeasResult > Earfcn= 1900,wNeiNum=11,Cid = [226],RSRP=[0],RSRQ=[0].` | 写 SCC 测量结果（邻区数与各 PCI 的 RSRP/RSRQ）。 |
| `0x849B` | 68 | 1 | `CSRRfDlDlyCfgSave: wCCcIdx=0, wIndex=0, wCCBw100KHz=14, wCABw100KHz=0` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8483` | 65 | 1 | `PssConvertFinger: NoisePower=19913,NoiseCount=2907,MaxPower=26.` | 小区频扫/PSS/SSS 搜索与同步。 |
| `0x5911` | 64 | 1 | `<FreqScan>[500K]Band=1,LeftFreqPoint=21105,RightFreqPoint=21694 Earfcn = 0 599.` | 频扫 500 kHz 栅格阶段的频段边界。 |
| `0x843E` | 59 | 1 | `CSRC > Recv ZPHY_EMC_ECSR_CELL_SEARCH_REQ CsrMainState = 0,dwEarfcn = 3201,wSearchType = 2[1:Strongst 2:List 3:Appointed],CellNum = 0,wIniSearchTime = 1` | 收到小区搜索请求；wSearchType=1:最强 2:按列表 3:指定（自带枚举）。 |
| `0x8491` | 53 | 1 | `CsrRfReqRes: Earfcn,=5, Bandwidth=6` | 小区搜索控制/候选数据库/载波管理。 |
| `0x849A` | 53 | 1 | `CSRWaitRfReqCnf: RxReqRst=1,Para1=11360` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8528` | 48 | 1 | `<IniCellSearch> TPU Adjust End->Measure Start, dwEarfcn = 1275 wIniMeasTime = 45(ms).` | 小区频扫/PSS/SSS 搜索与同步。 |
| `0x848F` | 45 | 4 | `EarfcnTable:[10][14759 14958 4750 4949 14279 14478 22750 22949 1 25];` | 小区搜索控制/候选数据库/载波管理。 |
| `0x5926` | 41 | 1 | `<FreqScan>******ptFSSSSResult[0]. INFO[1500 18350 3] wGain = 95,swRsrpOffsetAnt0 = -21,wGainPort = 1******.` | 频扫 SSS 结果与增益/RSRP 偏置。 |
| `0x8477` | 33 | 1 | `Csrc_NeedWork: bRet=0, wWorkOnDrxNum=0, wN=1, wT=24, bSearch=1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x851B` | 30 | 1 | `>>>>>>>>>> Search Finished:Result:-1,SearchInFreqscanState=0,CellEarfcn=3201,TaskInfo: [E_PHY_PLMN]=0` | 搜索结束与结果：Result=0 成功、-1 失败（画时间线的锚点）。 |
| `0x5934` | 26 | 1 | `<FreqScan>Pss100KResult FreqOffsetNum = 0,FsMethod = 0` | 频扫 100 kHz 后的 PSS 结果。 |
| `0x5912` | 24 | 1 | `<FreqScan>[100K]FreqPoint = 18350,LeftFreqPoint = 18345,RightFreqPoint = 18355,FreqBandNo = 3, MaxPeakVal = 1749. GAIN = 105 95` | 频扫 100 kHz 栅格结果（峰值与增益）。 |
| `0x592D` | 24 | 1 | `<FreqScan>[AGC]wPssWorkTime=32,NotSyncAGCDone=1 wCsrsWorkFlag=0` | 频扫 AGC 完成标志与 PSS 工作时长。 |
| `0x592F` | 24 | 1 | `<FreqScan>[AGC Start]Freq = [18350 3] FsState = 4` | 频扫 AGC 阶段开始。 |
| `0x8400` | 24 | 1 | `CtrlMemCellProc > CSRC Send Mib Request To PBCH-IC:dwEarfcn = 1275,CellId = 30,CPMode = 0 Info[-1 -1 -1 -1 36992]` | 无线链路监测；看有效输入、同步/失步状态与连续计数。 |
| `0x846D` | 24 | 1 | `L1l_SchedCsrs_CtrlICPTpuAdjust > Send to TPU Adjust Frame.Slot.Ts[0.2.6272].` | SRS 配置、周期/触发、冲突处理或发射。 |
| `0x84B2` | 24 | 1 | `ZD\|Flag = 0,PcellBand = 5,ScellBand = 5,predwBoundry=36992,sdwAdjustTs = 0, CompsdwAdjustTs = 0,dwBoundry = 36992` | 载波/SCell 配置或活动状态；区分配置、激活与承载。 |
| `0x8506` | 24 | 1 | `CSRC > Recv PBCH_SUCC_IND/PBCH_FAIL_IND = 1 wHoOnflag = 0.` | PBCH 成功/失败指示（1/0）。 |
| `0x851C` | 24 | 1 | `>>>>>>>>>> Measure Finished: Result: 0,TaskInfo: [E_PHY_PLMN]=0` | 测量结束与结果。 |
| `0x8526` | 24 | 1 | `Cell Rank:wTempIndex = 0,CellID = 30,dwEarfcn = 1275,RSRP = -73,wInitSearchNewCellFlag = 1` | 候选小区排序；此处 RSRP 单位是 dBm。 |
| `0x5908` | 23 | 1 | `<FreqScan>******FreqScan Report to ps. dwEarfcn = 1275,BandNo = 3,wFreqPoint = 18125,Vaule = 7081,Gain= 49,FingerValueComp= 65536,FingerValueAfterComp= 1812736` | 向协议栈上报频扫候选频点：Value/Gain 与增益补偿后的值。 |
| `0x592C` | 23 | 1 | `<FreqScan>[GAIN]LeftFreqPoint = 18350,RightFreqPoint = 18350,FreqBandNo = 3 pssFlag = 0` | 频扫增益标定阶段。 |
| `0x8479` | 22 | 1 | `GetIntraMeasTime:wEarfcn = 0,eCsrDDMode=1,wCellNum=0,wAddTimeIntra = 0,workTime=10` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x844E` | 18 | 1 | `RCV PI_START_REQ,PlmnFlag =0 InLowSnr=0 RapcState=0 drxCount=0 SearchT=24 Delay=0 EndTIme=0 EventReg=0 wIntSta=0 MobileCxt= 1 IntraMeasType = 0 [30 1095 1] [-1 -1 -1] [-1 -1 -1]` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x8471` | 18 | 1 | `Csrc_FreqIndexAcc: freqIndex=1, freqNum=0, NonHighDoneNum=0` | 小区搜索控制/候选数据库/载波管理。 |
| `0x850B` | 18 | 1 | `CtrlICPReportResult >RIGHT:Earfcn 1275,CellNum 1,searchType=2 [0]ID=30,Rsrp=68,Rsrq=28 [1]ID=0,Rsrp=0,Rsrq=0 [2]ID=0,Rsrp=0,Rsrq=0` | 初搜结果上报：最多 3 个小区，Rsrp/Rsrq 为 36.133 上报域。 |
| `0x5910` | 15 | 1 | `<FreqScan>[500K]BandNum=20,Band38Index=13,Band41Index=16,Band38Freq=[25700 ~ 26199],Band41Freq=[24960 ~ 26899],wFsMethod = 1` | 频扫 500 kHz 阶段的频段表（含 band38/41 范围）。 |
| `0x845A` | 15 | 1 | `CSRC > Recv ZPHY_EMC_ECSR_FREQ_SCAN_REQ CsrMainState = 0, IcsFlg = 0, BackATIcsFlg = 0` | 收到频扫请求。 |
| `0x8584` | 12 | 1 | `RefreshDB:Flag = 2 0,Earfcn = 1900,CellId = 226,Age = [667,645],Throld = [512,100],State = [0xff,0xff,0xff]` | 小区库刷新与老化。 |
| `0x865A` | 11 | 1 | `Counts&Period L1l_SchedCsrc_ProInitial: wTimer1Cnt=0,wScReportPeriod=4` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8675` | 10 | 1 | `CSRC:SNR_dB[ -12; -13; -13; -12]` | 另一处 SNR 打印（整数 dB），多见于频扫/空频点，勿与 0x866E 混用。 |
| `0x8473` | 8 | 1 | `Csrc_AdjustBndPowerOff: bNeedAdjustBndPowerOff=0, wLteUseSoftPowerStatus=0, wCirProtectInd=0, wMeasBitMask=0` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x8497` | 8 | 1 | `SRCellInfo: MobileCxt= 1 [30 1095 1] [-1 -1 -1] [-1 -1 flag:0]` | 小区搜索控制/候选数据库/载波管理。 |
| `0x843D` | 7 | 1 | `CSRC > Recv RESET_REQ` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x8640` | 7 | 1 | `WriteServingCellResult >wCarIndex = 1, SCellInfo-ID[1900-47]RSRP=[76][24] NeighCellNum = 0` | SCell 测量结果写入。 |
| `0x8656` | 7 | 1 | `SccIntraMeasFilter >:wCellNum =1 ,wZeroCnt = 11,` | SCC 同频测量过滤。 |
| `0x5A08` | 4 | 3 | `<RecoverPeakList[0][Earfcn:1275 ]swCirAdjust=0 ,Pss_Sw:>Id1: 0(0 ), 0(0 ), 0(0 ),ID2: 0(0 ), 0(0 ), 0(0 ),ID3: 0(0 ), 0(0 )` | 小区频扫/PSS/SSS 搜索与同步。 |
| `0x845E` | 4 | 1 | `zPHY_ecsrc_EventTaskEntry > Recv ZPHY_ECSR_STOP_INTER_SEARCH_MEAS_REQ,InterFlag=0ScheduleInfoCnt=0,CsrMainState=1[0:E_CSR_IDLE].` | 收到停止异频搜索/测量请求。 |
| `0x5909` | 3 | 1 | `<FreqScan>End:******FreqScan Report to ps Earfcn Num = 0` | 频扫结束；Earfcn Num=0 表示本轮没有任何候选。 |
| `0x8449` | 3 | 1 | `zPHY_ecsrc_ThreadEntry > Recv E_L1L_MC_CSR_COMMON_CONFIG_REQ wCampOn = 1[0:EPHY NO 1 :ephy] PiPeriod=128.` | 收到公共配置请求；wCampOn=1 表示 PHY 认为已驻留，PiPeriod 为寻呼周期帧数。 |
| `0x8459` | 3 | 1 | `zPHY_ecsrc_ThreadEntry > Recv ZPHY_EMC_ECSR_REL_REQ or SETMODE_REQ.` | 收到释放或 SetMode 请求。 |
| `0x8496` | 3 | 1 | `SRCellInfo: MobileCxt= 0 [30 1216 1] [-1 -1 -1] [-1 -1 flag:1]` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8645` | 3 | 1 | `SearchMeasAgeThrold:Intra = [3712,128,512,],Inter = [512,50,100,],Filter [0,0] Info = [0,0,0,],Inter=[0,3200],ScInfo=[0,0,0]` | 同频/异频搜索与测量的老化门限。 |
| `0x846A` | 2 | 1 | `Csrc_CalcInitDrxNum: HighNum=0, NonHighNum=0, FreqNumPerDrx=8, DrxNum=1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x846C` | 2 | 1 | `Csrc_CalcWorkDrxNum: OverNonHighNum=0, NonHighNumPerRpt=0, NeedMoreDrxNum=0, WorkOnDrxNum=1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8492` | 2 | 1 | `CounterRfReq: RxRfDb State = 7, search proc delay to next half frame` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8641` | 2 | 1 | `MeasSetProcess>>wMeasBitMask=0,workstate = 4 PccFastSync=0,PccNeiReport=0,Pccworkflag = 0 IntraMeasType = 0` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x8677` | 2 | 1 | `cat6:wFastSync_flag0=0,flag1=0` | 小区搜索控制/候选数据库/载波管理。 |
| `0x592B` | 1 | 1 | `<FreqScan>[Redo 500K]LeftFreqPoint = 8730,RightFreqPoint = 8750,FreqBandNo = 5` | 频扫 500 kHz 重做。 |
| `0x5930` | 1 | 1 | `<FreqScan>[GAIN]LeftFreqPoint = 8740,RightFreqPoint = 8740,FreqBandNo = 5 pssFlag = 1` | 频扫增益标定阶段（另一频点）。 |
| `0x8445` | 1 | 1 | `L1l_SchedCsrc_AddCarrier > eCgIdx=0, eCoreID=3, wCurCcIdx=1, wValidCc=3, dwSpCellFlg=0, dwServCellIdx=1.` | 增加载波（CC 索引与服务小区索引）。 |
| `0x8448` | 1 | 1 | `zPHY_ecsrc_ThreadEntry > Recv ZPHY_EMC_ECSR_ABORT_CELL_SEARCH_REQ.` | 收到中止小区搜索请求。 |
| `0x844B` | 1 | 1 | `Recv ZPHY_EMC_ECSR_MEAS_CONFIG_REQ.,Freq = [1275,0x10000],[-1,0xffffffff],[-1,0xffffffff],[-1,0xffffffff],[-1,0xffffffff],[-1,0xffffffff],[-1,0xffffffff],[-1,0xffffffff]` | 收到测量配置请求（频点列表与掩码）。 |
| `0x844D` | 1 | 1 | `zPHY_ecsrc_ThreadEntry > Recv ZPHY_EMC_ECSR_ABORT_MEAS_REQ.` | 收到中止测量请求。 |
| `0x8463` | 1 | 1 | `Update ScReport:wReportT=20 ScRepPeriod =1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8586` | 1 | 1 | `AddScellInfoToDatabase>>>SCELL Info:wAddModNum =1,wReleaseNum = 0 wServCellNum = 2` | SCell 入库：本次新增/释放数量与服务小区总数。 |
| `0x858A` | 1 | 1 | `AddScellInfoToDatabase>>>SCELL Info:wii =1,Earfcn = 1900,Pci = 47,ScellState[1 0]` | SCell 条目：频点、PCI 与 ScellState（配置态，不等于激活）。 |
| `0x858B` | 1 | 1 | `SccMeasStateCfg>>> CaIdex=1,bScellExist=0,bScellActive=0,wFastFlag=1,wAddSccNum=0,wFastCaIdxFirst=1,wFastCaIdxSecond=0,wSccCellId=47` | SCC 测量状态配置：是否存在/是否激活/快速同步标志。 |
| `0x8642` | 1 | 1 | `CtrlMeasConfigProcess>FreqNum =1 Serv[ 1275 75]Inter1[ 0 0] 2[ 0 0] 3[ 0 0] 4[ 0 0] 5[ 0 0] 6[ 0 0] 7[ 0 0] HighPrio=0x0 Period[0]` | 无线链路监测；看有效输入、同步/失步状态与连续计数。 |
| `0x8659` | 1 | 1 | `Counts&Period: wTimer1Cnt=1,wScReportPeriod=1,FirstReportSCC_flag=1,wCaIndex=1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x865B` | 1 | 1 | `Counts&Period L1l_SchedCsrc_UpdateScReportT: wTimer1Cnt=0,wScReportPeriod=1` | 小区搜索控制/候选数据库/载波管理。 |
| `0x8665` | 1 | 1 | `#INTRA#MEAS:SCC Intra-Meas:Cell INFO > CsrMainState = 2,CsrWorkState = 9,SccRsltFlg = 1,CellNum = 2,SCellEarfcn= 1900,SCellID= 47,Rsrp= 77,Rsrq= 27,Cell_FB= 0. 2.4640 ,MeasAge= 2,MeasBand= 50,SearchAge= 7` | SCC 同频测量结果（SCell 频点/PCI/上报值）。 |
| `0x8666` | 1 | 1 | `#INTRA#MEAS:SCC Intra-Meas:Cell INFO >Earfcn=1900, wii=0, Cell[226],Rsrp= 64,Rsrq= 1,Cell_FB= 0.11.14528,BCH_FB=1023.31.16383,MeasAge= 1,MeasBand= 50,SearchAge= 18` | SCC 同频邻区测量结果。 |

### A.4 `CSRM`（21个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x7E88` | 26284 | 1 | `CsrmMeasSeek > bCsrmfFlag=1, bGapConfigState=0, bGapConfigCsrRecive = 0` | 测量前的 gap 与配置状态检查。 |
| `0x7E8E` | 26284 | 1 | `CsrmValidCcIdx > wValidCc: 1, wCCcIdx: 0` | 测量有效载波索引。 |
| `0x688C` | 17745 | 1 | `MeasNvInfo:wFreq= 3201,OffsetAnt0=[0,0,0,0,0,0],OffsetAnt1=[0,0,0,0,0,0],wStandardPoint = [141,141,141,141,141,141]` | 测量 NV 校准：各天线频段偏置与 wStandardPoint=141（RSRP 上报基点）。 |
| `0x68BC` | 432 | 1 | `MeasHwInfo\|RSRP00=[107;18],RSRP01=[131;16],RSRP10=[63;10],RSRP11=[24;5],RSSI=[559;225]` | 硬件 RSRP/RSSI 累加原始值；RSSI=[0;0] 时该次 RSRQ 不可信。 |
| `0x68C3` | 432 | 1 | `CalRsrp:wMaxAntIndex=1,TA=2, RA=2, RsN=24,SingleEn=0,AgcAnt=[85,76,380,9],RSRP=[-75,-73,-386,-386],RSRQ*2=[-13,-12,-768,-768],swTempRssiLog=[2334,2000,0,0],swRssiLog1=[-5084,-4992,0,0],sawOffset=[-25,-21,0,0]` | 把内部量换成 dBm：AgcAnt 增益、sawOffset 校准、各天线 RSRP/RSRQ/RSSI。 |
| `0x68C8` | 432 | 1 | `#CSRM:adwMod=[1756964,2150852,1034992,397904],swTempRsrpLog=[1800,1530,-32768,-32768],wRsCollideFlag=0,wAbsflag=0,wRsNumLog=1174,tResult.swRsrq=-470` | 线性模与对数域中间量；-32768 是无效哨兵。 |
| `0x6889` | 351 | 1 | `#CSRM:RX:RSRPoffsetCal:wEarfcn=1275, CellID = 30, Rsrp = -71,Rsrq = 74, Rssi = [-22639,-22594,-19825,-19861], RSRP=[-73,-71,-62,-62]` | RSRP 偏置计算：同时打印 dBm 与上报域，便于交叉验证。 |
| `0x6895` | 351 | 1 | `ServCellRxMeas\|CellInfo[0] Bandwidth dwEarfcn wCellId=[75,1275,30],CellInfo[1] Bandwidth dwEarfcn wCellId=[-1,-1,-1],CellInfo[2] Bandwidth dwEarfcn wCellId=[-1,-1,-1].` | 服务小区带宽/频点/PCI 三元组；-1 表示该位未配置。 |
| `0x68C7` | 324 | 1 | `#CSRM:wMeasResultNum=1,aswRsrq=[-6124,0,0,0,0,0],swValLog2=-6124,swTempValLog2=0,swTemp1ValLog2=0,swTemp2ValLog2=0,swTemp3ValLog2=0` | 逐量上报的对数值（依次为 RSRP、RSRQ、RSSI）。 |
| `0x7E86` | 218 | 1 | `CsrmCfgRfcData > bRfOn= 1, wRfOpenSfNum=1, wOpenSf=5, dwTsOfOpenRfc = -1, dwTsOfEndRfc = -1` | 为测量配置 RF 开启子帧。 |
| `0x7E82` | 206 | 1 | `MeasGetMeasPDInfo:wCaIndex = 0, pdwMeasPdIdx=4, pdwAntGroupIdx=0` | 取测量的电源域与天线组索引。 |
| `0x7E8C` | 130 | 1 | `MeasCaSwitch: wCoreId = 3, wCCcIndex = 0, wSwitch = 1` | 载波聚合下的测量切换。 |
| `0x6704` | 82 | 1 | `WriteTDDRfcEventTab >wCaIndex=0, Earfcn=1275, CellID=30,ServingCellID=132,wStSfPos = 5, dwMeasTabOffset is 36992 -- dwMainSycTabOffset is 36992 -- wMeasToMainTabRelation=2,MeasMode=0,currentGap=0,Cpmode=0]` | 写测量事件表（含表偏置与测量模式）。 |
| `0x7E83` | 82 | 1 | `MeasCfg(wHwIndex=3)>CellNum=1;CellInfo=(0x1e,0,0,0,0,0,0,0);MeasEn=1;Mode=0(0:Normal,1:SingleSym);BandWidth=5;Quantity=7df; Gclk_bpReg=2;RamBypass=0;AntEnable=0;SpecialMode=0;ClkSwitch=0;MeasStart=1.` | 测量硬件配置：CellInfo 是要测的 PCI 列表，Mode=0:Normal 1:SingleSym。 |
| `0x7E85` | 82 | 1 | `CsrmCalcRfOpenTime > caIdx=0, RfcStart = [903.10.0], MeasCellBoundy = [0.2.6272], CaRfcOpenTs=0, dwTpuIdx=1` | 计算 RF 需要提前多久打开以完成测量。 |
| `0x7E87` | 82 | 1 | `MeasCfgMT(wHwIndex=3)>CellNum=1;CellInfo=(0x1e,0,0,0,0,0,0,0);MeasEn=1;Mode=0(0:Normal,1:SingleSym);BandWidth=5;Quantity=7df; Gclk_bpReg=2;RamBypass=0;AntEnable=0;SpecialMode=0;ClkSwitch=0;MeasStart=1.` | 测量硬件配置寄存器镜像。 |
| `0x7E8B` | 82 | 1 | `ModifyRfOpenTime > Earfcn=1275, TpuOffset.RxOffsetMin.OffsetRxToTpu = [314535808.36992.0] TpuIdx=1, MeasSingleSymModeFlag=0` | 修改 RF 开启时间与 TPU 偏置。 |
| `0x68C6` | 81 | 1 | `#CSRM:wIndex =0,dwEarfcn = 1275,CellId = 30,AveLogRsrp = -73,AveLogRsrq = -6,swAveRsrp = [-6124,0,0,0,0,0],swAveRsrq = [-470,0,0,0,0,0],swAveRssi = [-15124,-14848],` | 平均后的最终测量：AveLogRsrp(dBm)、AveLogRsrq(dB) 与对应内部值。 |
| `0x7E84` | 81 | 1 | `MeasHwInfo:HwIndex=3, dwDoneFlag=1,dwSymInfo.Slot.Sym = [10,0],SymCnt=4,dwSymInfo=1104,RspCnt=4` | 测量硬件完成标志与符号计数；dwDoneFlag=1 才算测完。 |
| `0x6706` | 35 | 1 | `MeasProcStart:eCoreId=3,wCCcIdx=0,eDdMode=1,tMeasState=1,IntraFreq=1275,InterFreq=1275,FreqWorkEn=0,weICICState=0,wNextSchTime=2,wPatternCycle=0` | 测量流程开始：同频/异频频点与下次调度时间。 |
| `0x7EB9` | 10 | 1 | `MeasStartSubFrame = 0` | 测量起始子帧。 |

### A.5 `CSRS`（54个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x5802` | 57051 | 1 | `CsRfcCfgComm>>wCaIndex=0,SFData[0]:bRfcCsrFlag=1,tRfcState=2,FP=26651,BN=7,BW=5,RxOffset=0,MeasOffset=-1,TsOfOpenRfc=-1,TsOfCloseRfc=-1,DFE[eState=1,Antnum=4,agcmode=0,SynState=0,Freq=26651,BwRb=6,Serach=1,Syn…` | 每子帧的搜索 RF/DFE 状态（数量极大，只看状态跳变）。 |
| `0x590F` | 35818 | 4 | `FreqscanRfcCfg >> Freq = 21105, RfcSFData[9]:SfStart=9,BW=5,SyncMode=0,Rssi[State=0,Len=0,FreeMrtr= 0. 0. 0,LocalMrtr= 984.15. 6624,cfgTime=0]` | 频扫用 RF 配置（每子帧）。 |
| `0x7D91` | 27074 | 1 | `L1l_DrvCsr_CheckCsBufferHw: HwIndex=0, BufState=0x0, BufOutState=0x0, BufProcState=0x0, ProcValid=0x0` | 搜索缓冲硬件状态检查（数量大，只看异常）。 |
| `0x588F` | 6272 | 1 | `CsrSrcClkPdCtrl: dwPdStatus=1, dwClkStatus=1, wHwCcIdx=0, wCCcIdx=0, wCCcIdx=0, wIndex=0, wModuleID=0, wPreClkBitmap=0` | 搜索时钟/电源域控制。 |
| `0x580F` | 4764 | 1 | `GetPssStartTime> PssHwCfgMode=2, PssStartTimeFlag =1(1:TRUE),PssStartMrtr= 765,0,0(LocalMrtr:765,0,0)` | PSS 起始时间获取（硬件配置模式与起始 Mrtr）。 |
| `0x7D83` | 4699 | 44 | `PssCfg(00)>CurrMrtr= 97.10.10572;HFB= 97.14.10752( 58.10. 2016);Config=0x10d;NumHF= 2;ThrePower=0x4100008;ThreNoise=0x0;MaskW=2;IdEn=7;ReportAhead=0;SearchEn=0;F0Start=1.` | PSS 硬件配置：半帧边界、门限功率/噪声、检测使能与半帧数。 |
| `0x7D84` | 4699 | 11 | `PssCfg(00)>TopClk{Ram0/1=[0,0];Gating=0;CsrT61/122=[0,0]};Pss{F0CfgTimes=2;F0HalfFrameCnt=1;FreqInd=0;Busy=1;FilterMrtr= 785.15.14816;LastHFB= 785.15.14816;Done=0};ErrFlag=0.` | PSS 配置状态：时钟、半帧计数、Busy/Done 与 ErrFlag。 |
| `0x7D89` | 4699 | 1 | `PssCfgHwMT(0)>Top=CcSourceEn=0x00030003,DataSource=0x00100000,SoftClk=0x0000f020,Config=0x0000010b,FrameBoundry=0x17e80000,ClkGating=0x00000000,wFreqInd=0x00000000,NumHalfFrame=0x0000000a,Threholdpower=0x04100…` | PSS 硬件寄存器镜像（需寄存器手册）。 |
| `0x5801` | 2325 | 1 | `CsrRfcConfig\| bRet=1,wCurCarrierOption=0,ConfigRfNum=6,RfOpenSubFrame=0,wState=2.` | 搜索 RF 配置结果与状态。 |
| `0x580C` | 1406 | 1 | `PssProc,ResultFlg[00]=0,SwStatus=1,Config=0,Read=0,HwCfgtime=1,GapConfilt=[0,0],Frame_Num[0,0],Done=[0,0],RfOpen=(0,6),Filter2=[0,6],GapDistan=-1,PssReadTime=0,wFreqInd=0,PssBusy=1.` | PSS 过程状态：结果标志、读写状态、RF 窗口与 Busy。 |
| `0x7D8C` | 1389 | 1 | `L1l_DrvCsr_TopHwReset: dwResetIdx=0x23, dwHwType=0` | 搜索硬件复位。 |
| `0x5811` | 749 | 2 | `SssProc,ResultFlg[00]=0,SwStatus=1,Config=0,Read=0,HwCfgtime=1,FreqChange=1,ProcStatus=[000000ff,00000000],GapConfilt=[0,0],RfOpen=(5,6),Filter2=[5,6],GapDistan=-1,wFreqInd=0.` | SSS 过程状态（含频点变更标志）。 |
| `0x5883` | 670 | 1 | `CfoResultMerge >> Done = 1,wIndex = 1,NcpIQ-EcpIQ = [1938,28,5461,291]` | CFO 原始 IQ 相关累加结果合并。 |
| `0x7D85` | 670 | 15 | `CfoCfg>HFBNcp= 771. 5.12800,HFBEcp= 771. 5.12800,SymMapNcp=0x22448f1,SymMapEcp=0x249279,SearchWinLen=0x1,SearchEn=0x0,Accnum=0x0(0);CurrMrtr= 771. 1.14429;ConfigFlag=1.` | CFO 配置：NCP/ECP 半帧边界、符号映射与搜索窗。 |
| `0x7D96` | 670 | 1 | `CfoCfg[MT]>HFBNcp=0x18197200,HFBEcp=0x18197200,SymMapNcp=0x22448f1,SymMapEcp=0x249279,SearchWinLen=0x0,SearchEn=0x1;dwAccuNum=0x0.` | CFO 硬件寄存器镜像。 |
| `0x7D86` | 630 | 6 | `SssCfg(00)>Id2=0x5824;En=0xff;WorkSta=0x70a2;DLen=0x3c0000;RdWrSta=0000;ChEsti=8e26;Thre1~3=0x8ccc3db2,80003800,80002d70;NorThre=0x800;Cp=ffff;GtEn=0x0;PowerOn= 776.16. 6400,Gating=0,Count=[00000000,00000000],…` | SSS 硬件配置：检测使能、信道估计与三档门限。 |
| `0x7D87` | 630 | 66 | `SssCfg( )>NoiseKill=0x120080;HfIndict=0xff;WinPos0~7=0xab286e0,ab29840,ab29ac0,ab29be0,ab29be0,ab29c20,ab29e00,ab4cc40;BufferTime= 342.10. 1760( 303. 0. 5376);CurrMrtr= 342. 5.13953;ConfigFlag= 1.` | SSS 配置：噪声抑制、半帧指示与 8 个窗口位置。 |
| `0x7D8D` | 630 | 1 | `SssCfgHwMT[0]>tProcessId2=0x00005824;tProcessHfIndict=0x00000000;tProcessEnable=0x000000ff;tWorkStatus=0x000070a2;tBufWinPosTdd=0x1843f200;tBufDataLen=0x003c0000;auBufWinPos=0x18441920,0x18441b00,0x18441b80,0x…` | SSS 硬件寄存器镜像。 |
| `0x7D8E` | 630 | 1 | `SssCfgHwMT>SpecCellidDet=0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000;CpMode=0x0000ffff;ClkGatEn=0x00000000;dwResv2=0xffffffff;tMemPowerOn=0xffffffff;PinSelect=0x0000…` | SSS 指定 PCI 检测与 CP/时钟寄存器镜像。 |
| `0x7D99` | 630 | 1 | `SssCfgHwMT>Top=CcSourceEn=0x00030003,DataSource=0x00100000,SoftClk=0x0000f080,PssStartPos=0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000` | SSS/PSS 起始位置寄存器镜像。 |
| `0x7D95` | 520 | 1 | `LogSssTddStartInfoAllProc: dwEcpFingerOffset=0,dwNcpFingerOffset=0,dwNcpPssStartBuff=0` | 搜索/同步执行侧私有打印。 |
| `0x5820` | 358 | 1 | `L1l_LogCsrs_TaskEntry >eCoreId=3,SubFrameOnOff=0[0:Off; 1:On].` | CSRS 任务入口与子帧开关。 |
| `0x5B00` | 260 | 1 | `SssStartFinger(0)>CellId=0,PeakValue=23,OffsetTime=104384,HwOffsetTime=3262,FingerOffset=99776 (1)>1,20,103680,3240,99072` | 每个 finger 的 SSS 结果：CellId、峰值与时间偏移。 |
| `0x5B06` | 173 | 11 | `Sss(3)(0)(01)[00]:Earfcn= 1900,CellId=226,FB=0.11.14528,HfIndic=1,CPMode=1(0:NCP,1:ECP),Total:10ms,Mode=1,Finger=995,Threshold= 621,CellFBTsBefAdj=183488,dwFilter=0,HandoverPading[2]=0` | 小区搜索结论行：PCI、帧边界、HfIndic 半帧、CP 类型、积累时长、Finger 与门限。 |
| `0x7D9E` | 131 | 1 | `LogSetSrcFlag: bIntraSrcOn=0,bInterSrcOn=0,bInterSrcBuff=0` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x5B05` | 130 | 1 | `SssFingerReorder:tWindowTime=776.15.12800, HwOffsetTime:[3262,3240,3280,3253,3283,3244,3225,2326],dwSssDelayMrtrTs:[10176,9472,10752,9888,10848,9600,8992,133824],` | 8 个 finger 按时延重排。 |
| `0x7DA0` | 121 | 1 | `SearchProcReset: eCoreId=4,wCCcIdx=3,wValidCc=0,wCurCCcIndex=0` | 搜索流程复位。 |
| `0x5B04` | 119 | 2 | `GetSssReadFlagInfor(00):SssReadFlag=0[0:FALSE,1:TRUE];ProcCount=0x9aaaaaaa;RdWrState=0xfe00;ProcEnable=0xff,SssCfgTimes=10,SssReadTimes=1,SssReadTimesBuff=0.IcResult:WorkState=0x 0,PssEnd0.1=[0x 1bae0.0x 26b0]…` | SSS 读取标志与配置/读取次数。 |
| `0x5810` | 67 | 1 | `newcsr CsCfoProcEnd.` | CFO 处理结束。 |
| `0x5881` | 67 | 1 | `CSRS_ICS: >CFO Result >==========Frequency offset=-126 Hz,CFOCordicValue = 0 Hz CPMode=0[0/1:Ncp/Ecp],TimeShift=0.IniFreq=0.=========<` | CFO 结果：频偏 Hz（可直接读），并给出 CP 模式。 |
| `0x5813` | 66 | 1 | `L1l_DevCsrs_GetHwConfigMode >ECsrSearchState=0;Intra/Inter_PssHwCfgMode=(2,0);Intra/Inter_SssHwCfgMode=(2,0);CfoHwCfgMode=2` | 取搜索硬件配置模式（PSS/SSS/CFO 各自的模式）。 |
| `0x5A00` | 65 | 42 | `Pss(3)(00):Earfcn=3201,Id0=[ 23(3262), 19(3253), 18(3225)];Id1=[ 20(3240), 18(3283), 16(2326)];Id2=[ 26(3280), 24(3244)];wFingerCnt=8` | 整理后的 PSS 三候选：每项为 峰值(位置Ts)，wFingerCnt 为保留的时间候选数。 |
| `0x5A04` | 65 | 1 | `GetUrfcn&&NumHalfFrame>>>Uarfcn:3201,dwNumHalfFrame:10,wDataInd=0` | 本次搜索的频点与半帧数。 |
| `0x5B0A` | 65 | 1 | `SssGetThreshold:MaxFinger=0,MaxFingerInHW=0,AccuNum=10,CommThreshold=0,Proc0~7[CellId/Finger]={(0/0),(0/0),(0/0),(0/0),(0/0),(0/0),(0/0),(0/0)}.` | SSS 判决门限与各处理通道的 CellId/Finger。 |
| `0x7D88` | 65 | 13 | `LogPssHwOriValue[0]:Id0 = [ 23,3198],Id1 = [ 20,3176],Id2 = [ 26,3216],dwHalfNum = 10,dwDoneFlag=1` | PSS 硬件原始峰值：Id0/Id1/Id2 对应 N_ID2=0/1/2。 |
| `0x581A` | 64 | 1 | `GetSearchHwIdxIdle wHwNum=3,eCoreId=3,cCCcIndex=0 wSrcStartDone=0,wHwIdx=0` | 取空闲的搜索硬件实例。 |
| `0x5806` | 62 | 7 | `Search Req>coreid=3;CurCCcIdx=0;State=0[0:Ini,1:Intra,2:Inter,3:Fast];PCC/SCC/Inter=[Earfcn:( 3201, 0, 0);Freq:(26651, 0, 0);Flag:(1,0,0);Band:( 7, 0, 0);DDMode:( 1, 0)];CurIntra= 0;CsrMainstate=0;WorkMode=0;S…` | 搜索请求：State=0:Ini 1:Intra 2:Inter 3:Fast，并列出 PCC/SCC/Inter 的频点与 band。 |
| `0x5809` | 59 | 31 | `PSS_TPU_ADJUST(Macro): TpuAdjTimeSub=0,PssAdjustBoundry= 0. 9.11872(150112Ts),OldTpuOffset= 488.10. 3520;TpuAdjMacTime= 535. 9.11872.` | 把 PSS 得到的帧边界写给 TPU（Ts 三元组）。 |
| `0x581C` | 59 | 3 | `zPHY_TPU2csrs_TaskEntry >Recv ZPHY_ETPU_ECSRS_PSS_UPDATE_COUNTER_CNF,NewTpuOffset= 984.17. 9104.` | TPU 回确认，给出新的 TpuOffset。 |
| `0x5A01` | 59 | 1 | `L1l_LogCsrs_TaskEntry ----------- SEND RFC OFFSET, dwMrtrTs = 89600, ICSPssFrameTs = 235404800` | 向 RFC 发送 PSS 时间偏置。 |
| `0x5A07` | 40 | 1 | `AdjustPssStartTime:tPreTime = 196.2.6560,tNewTime = 196.12.6560,wSubSlot=0` | 调整 PSS 起始时间。 |
| `0x5B0C` | 40 | 1 | `GetRfcEnableInfo[0]:Finger = [5376,61440],Rfc = [wRfcStart=5,wEnd=2,wLength=3]` | 本次搜索占用的 RF 窗口。 |
| `0x5918` | 34 | 1 | `"<FreqScan>[FreqScanAddSearch][0].FreqPoin =18125,dwEarfcn=1275,BandNo=3,GAIN=[49 1]Value=[7081 65536 1812736],wFlag=2"` | 频扫新增搜索频点及其增益/峰值。 |
| `0x5943` | 22 | 1 | `Scan-Procinfo-flag=[buf=0,coar=0,fine=0,pss=0,sss=0,meas=0],Done=[buf=0,coar=0,fine=0,pss=0,sss=0,measone=0,measall=0,cs=0],accnum=0,freqpoint=[0,0,0],freqnum=[0,0],cFlag=57.` | 频扫各阶段完成标志汇总。 |
| `0x5903` | 18 | 1 | `"<FreqScan>[0]FreqScan add CellSearch ok!!!dwEarfcn=2452,SSSValue=713,wGain=101,dwFingerValueAfterComp=0"` | 频扫命中后加入小区搜索（SSS 峰值与增益）。 |
| `0x7D90` | 16 | 1 | `CSRAddr:Top=6462BD88,pss0=6462F2B8,cfo0=64631C18,sss0=6462F2A8,scan0=6462C880,ic0=646334E8,buf0=646334D8,afc0=64631C08,cache0=00000000,pss1=6462F2C0,cfo1=64631C20,sss1=6462F2B0,scan1=6462C888,ic1=646334F0,buf1…` | 各硬件块基址表（调试用）。 |
| `0x9478` | 13 | 1 | `zPHY_emulm_TaskEntry >SubFrameOnOff=0[0:Off; 1:On],SubIntType=2[0:NOT_SYN_SUB_FRAME_INT,1:ECSRM_PERIOD_INT_EVENT,2:ALL].` | 搜索/同步执行侧私有打印。 |
| `0x5904` | 11 | 1 | `"<FreqScan>[Fail]FreqScan add CellSearch Fail!!!***dwEarfcn=1275,dwMaxPeakValue=0"` | 频扫加入小区搜索失败（峰值为 0）。 |
| `0x588B` | 9 | 1 | `GetNextSrcCCcIndex: NextCCcIdx=0, wValidCc=3, wCCcIdx=2, wSearchStartDone=0, wSearchMeasWork=0` | 取下一个搜索载波索引。 |
| `0x588D` | 4 | 1 | `SetServEarfcnInfo: wValidCc=3, wCCcIdx=1, wCaIdx=1, dwDlEarfcn=1900` | 设置服务小区频点信息。 |
| `0x580B` | 3 | 1 | `CaFreqChange wCurCCcIndex=1;wCaIdx=1;dwEarfcn=1900;SearchMeasWorkFlag=1` | 载波聚合下的频点切换。 |
| `0x5906` | 3 | 1 | `<FreqScan>******Function L1l_FS_CalcSssAgcGainCompen ERROR Min AGC gain is -1******.` | 频扫 AGC 增益计算错误。 |
| `0x5826` | 1 | 1 | `SCC\|GetSccCommonInfo coreid=3;CurCCcIdx=1;wCaIdx=1;Earfcn: 1900;Freq:18750;Band: 3;DDMode: 1;Fast=3.` | SCC 公共信息（频点/band/双工/快速搜索标志）。 |
| `0x588C` | 1 | 1 | `SccProcStart: eCoreId=3, wCaIndex=1, bStartNow=0` | SCC 搜索开始。 |

### A.6 `DFE`（58个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x5C80` | 190238 | 1 | `DFE\|AgcMeanpwr: PdIdx=4,AntNum=4,AntChIdx=0,SymIdx=[0,6],MeanPwr0=[3247,3192,3121,3139,3405,3162,3214], MeanPwr1=[3349,3565,3534,3541,3620,3565,3582]` | AGC 各符号平均功率（私有刻度）。 |
| `0x5CA0` | 115860 | 1 | `DFE\|TotalGainInfo Core[3] CC[0] SwPd[4]: NextIdx=5, AgcGain=[114,104,113,116],RaDagcGainLin=[102,102,102,102],CsrmDagcGainLin=[128,128,128,128],CsrsDagcGainLin=[128,128,128,128]` | 总增益：模拟 AgcGain(dB) 与随机接入/测量/搜索三套数字增益（128=1×）。 |
| `0x5CA9` | 113394 | 1 | `DFE\|AsyncIntMrtr: LocalMrtr=761.0.8511, LocalSfMrtr=8511, FreeMrtr=761.0.8501, FreeSfMrtr=8501` | 异步中断时间戳。 |
| `0x5C29` | 107765 | 1 | `DfeDebug!ID=[319000]: Para=[1 2 241 3208, 0 241 112880 0, 0 0 0 0, 0 0 0 0, 0 0 0]` | DFE 调试快照（ID+参数数组）。 |
| `0x5C0A` | 85522 | 1 | `DFE\|Config Cordic Offset CC[0]: En=1, CordicOfsHz=13, SpRate=1920000, fcPhaseInit=0, fcRotVal=-29080, fcPhaseInitEn=1, fcRotValEn=1` | 数字频偏旋转补偿配置。 |
| `0x5C17` | 80123 | 1 | `DFE\|AgcSeqCfg Cc[0]: SeqScene[2](1:Open,2:RfOpt),CfgTs=1226472, AgcGain=[114,104,113,116], Ret=178257921, band=3, Freq100Hz=18500000, ResCc=[0,0,0]` | AGC/增益/接收功率环；看饱和、分支差异与突变。 |
| `0x5C8E` | 47723 | 1 | `DFE\|SyncAgcInfo CC[0]: AgcWorkState=13,AgcLen=0x4,FreqPoint=18500,BandWidth=6,Lf=0x400,Target=3280,MeanPwr[3247,3349,3248,3246],Gain=[114,104,113,116],AgcdoneFlag=[1,0],Offset=0,Gap=[0,0], PosIdx=5, MeanpwrVa…` | 同步态 AGC：BandWidth 是 RB 数（6=同步/PBCH 的中心 6 RB），MeanPwr 应收敛到 Target。 |
| `0x5C0F` | 47719 | 1 | `DFE\|DcEstiRet: PdIdx=4 AntNum=4 SPattern=15 EstiI=[161,290,-106,-308] EstiQ=[-186,-36,34,-122]` | 各天线直流估计；某天线明显离群说明该通道有直流泄漏。 |
| `0x5CA5` | 42573 | 1 | `DFE\|RxDagcHandle SwPd[4]: AntNum=4 SfMode=0 AgcSib1Done=1 TransMode =0 DagcPwrNormal=[572080,767832,783096,641568] RxDagcLin=[102,102,102,102] RxDagcLog2=[685,685,685,685, 98304,98304,98304,98304]` | 接收数字 AGC 处理与归一化功率。 |
| `0x5C0D` | 42572 | 1 | `DFE\|FftPWinOffset: PdIdx=4 FftCfg=[28,72] AntNum=4 ReCpBitMap=0x3fff WinOfs=[0,0,0,0] t0=[0,0,0,0] t1=[0,0,0,0]` | FFT 窗位置；WinOfs 非 0 表示在补偿多径/定时误差。 |
| `0x5C86` | 41331 | 1 | `DFE\|AgcAsyncInt: PdIdx=4,SubFrame=2,UpdateFlag=1,CurMeanpwr=[2444,2374,2731,2544],Count=1,TotalPwr=[2444,2374,2731,2544]` | AGC 异步中断的功率累加。 |
| `0x5C94` | 37798 | 1 | `DFE\|AsyncAgcPwr CC[0]: Freq=26651 AgcWorkState=3 Meanpwr0=[2438,2244,2254] Meanpwr0=[2592,2318,2325] Meanpwr2=[2797,2669,2690] Meanpwr3=[2621,2503,2524] TotalPwr=[0,0,0,0] AvePwr=[0,0,0,0],cont=0` | 非同步 AGC 各天线均值功率。 |
| `0x5C95` | 37798 | 1 | `DFE\|AsyncAgcGain CC[0]: Freq=26651 AgcWorkState=3 AgcGain0=[95,95,95] AgcGain1=[95,95,95] AgcGain2=[95,95,95] AgcGain3=[95,95,95] FSNewStartFlag=0 FSFixedAGCFlag=0` | 非同步 AGC 各天线增益。 |
| `0x5C87` | 29868 | 1 | `DFE\|FastAgc: PdIdx=4,Frame.Sub=452.0,swMeanPower=[2579,2587,2989,2829],wCommonAgcGain=[71,71,69,69]` | 快速 AGC：按子帧收敛的均值功率与公共增益。 |
| `0x5C96` | 21880 | 1 | `DFE\|InitSubPwrDB CC[0]: Flag=1(1:all, 0:some) SfNum=0 AllInfo: Count=0 TotalPwr=[0,0,0,0]` | 子帧功率数据库初始化。 |
| `0x5C93` | 20840 | 3 | `DFE\|TotalSubFramePwr: PdIdx=4 AntNum=4 wSfNum=9 AntIdx=0 Cnt=3 SumPwr=9183 MeanPwr=[3352,2945,2886, 0, 0, 0,3512,3355, 3292, 0, 0, 0, 3342,3398]` | 整子帧功率统计。 |
| `0x5C02` | 9142 | 1 | `DFE\|DfeReq CC[0]: eUid[0](0:DL,1:CSR), eState[0](1:Open,0:Close), ReqSfNum=1, SfPattern=[0,15], dwPPIdx=1, SyncState=2` | DFE 开关请求：eUid=0:DL 1:CSR，eState=1:Open 0:Close。 |
| `0x5C9E` | 8827 | 1 | `DFE\|FindSaveAgcInfo Core[3] CC[0] SwPd[4]: FN=243.0 Idx=1 Path=0 AgcGain=[96,92,96,96] RxDagcGain=[685,685,685,685],CsrmDagcGain=[0,0,0,0]` | 按频点/带宽缓存增益，便于回切时快速恢复。 |
| `0x5C1A` | 6783 | 1 | `DFE\|Lpm: PdIdx=4 TaskId[61] PwrType[0] UidEn=0` | DFE 低功耗控制。 |
| `0x5C16` | 4755 | 1 | `DFE\|AfcReq Cc[0]: Mode=1 Type=1 Freq100Hz=18500000 CfgTs=245760 OffsetHz=[3776,14] TmpPpm=[2090,8] CalcPpm=[-156764,-612] CordPpm=[4004,15] CordHz=[7233,28] TotalOffsetHz=-1106.` | AFC 请求：所有字段为 [定点值, 取整值]，Q8；CalcPpm 为 ppm×1024 的 Q8。 |
| `0x5C27` | 4755 | 1 | `DFE\|AfcInComingPara Core[3] Cc[0]: SwPd=4, dwAfcMode=1, AfcType=1, Earfcn=1650, FreqOffset=[14,3776], Source=1` | AFC 输入参数（模式/类型/频点/频偏来源）。 |
| `0x5C06` | 4440 | 1 | `DFE\|CmnCfg: ccSpRate=0(0:1.92M, 1:3.84M, 2:7.68M, 3:15.36M, 4:30.72M, 5:61.44M, 6:122.88M), ccAntNum=3(0-3), workmode=1(0:NR,1:LTE), scs=0, dcCmpEn=0, iqCmpEn=0, fcEn=1, cfgDown=0` | 前端公共配置：采样率/天线数/制式/子载波间隔（自带枚举）。 |
| `0x5C08` | 4229 | 1 | `DFE\|CsrReq CC[0]: CsrAntNum=4,Freq100Khz=26651,eBwRb=6,AgcMode=0,AgcSymMap=0,bCellSearch=1,SyncMode=0,SyncCcIdx=[1,0x10],eCellMeas=0,MeasBwRb=0,MeasMode=0,MeasReCpMap=0x0,MeasChIdx=[0,0],Rssi=0,Prs=[0,0],AgcD…` | 搜索用前端请求（天线数、6 RB 带宽、AGC 模式、是否小区搜索）。 |
| `0x5C0E` | 4085 | 1 | `DFE\|ConnectCc2LteaSync: PdIdx=4 SyncCcIdx=1 Ant=[0,1] CurCcCfg=0x44 LeftShiftNum=7 ConnCfg=0x2254 RegVal=0x2254` | 把 CC 连到 LTEA 同步通道。 |
| `0x5C2B` | 4085 | 1 | `DFE\|SyncBranchCfg Core[3] CC[0] SwPd[4]: SpRate=[0,0], HbfBitMap=[0x3f, 0x3f] SyncMode=0, RssiState=0x0` | 同步支路配置（采样率与滤波位图）。 |
| `0x5C98` | 3392 | 1 | `DFE\|NextSfAgcInfo SwPd[4]: Freq=[-1(Para),-1(GainSD)],Bw=[100(Para),100(GainSD)],Flag=3,Idx=[4,15],AgcGain=[60,60,60,60],AgcDiffAvgAnt=[0,0,0,0]` | 下一子帧的 AGC 参数。 |
| `0x5C99` | 3330 | 1 | `DFE\|AgcReload Core[3] CC[0] SwPd[4]: Freq=[-1(current),26651(next)],Bw=[100(current),6(next)],Idx=4,AgcGain=[60,60,60,60],AgcDiffAvgAnt=[0,0,0,0]` | 增益重载（当前/下一频点与带宽）。 |
| `0x5C97` | 2365 | 1 | `DFE\|CSRSetAGCGain CC[0]: SyncState=0 Freq=7985 AntNum=4 Gain01=[110,110] FSNewStartFlag=1 FSFixedAGCFlag=0` | 为搜索通道设置 AGC 增益。 |
| `0x5C9F` | 2115 | 1 | `DFE\|SetFSNewState Core[3] CC[0] SwPd[4]: CSRFlag=0 FSNewStartFlag=0 LocalFSNewStartFlag=0` | 频扫新起始状态标志。 |
| `0x5C22` | 2085 | 1 | `DFE\|DcOffset_Clear CC[0]: PdIdx=4, FreqPoint:[7985(cur),21105(next)], DC_I:[0,0,0,0], DC_Q:[0,0,0,0]` | 换频点时清直流。 |
| `0x5C19` | 1783 | 1 | `DFE\|AsyncAgc CC[0]: =======Pss SoftMode: Flag=[0,1] Frame.SubFrame=0.0===========` | PSS 软模式下的非同步 AGC。 |
| `0x5C91` | 1405 | 1 | `DFE\|CSRS DAGC CC[0]: PdIdx=4,Target=0,Cellid=0,Intra\|Inter_Dagc_Done=[0,0],Agcworkstate=4,SynState=1,MasterState=2,PssPwr=[1051136,948992,795264,627648],Dagc=[128,128,128,128],DagcLog2=[0,0,0,0]` | 搜索通道 DAGC 计算（含 PSS 功率）。 |
| `0x5C8D` | 1371 | 1 | `DFE\|AgcCalInitPwr: PdIdx=4, AntNum=4, StartSym=4, SymNum=4, Pwr0=[2962,2922,2925,2869], Pwr1=[2807,2843,2872,2757], Pwr2=[3067,3066,3095,3102], Pwr3=[2927,2961,2989,2974]` | AGC 初始功率计算（各符号）。 |
| `0x5C0C` | 824 | 1 | `DFE\|MeasRst: RstEn=1 MeasPdIdx=4 AntChIdx=0 MeasRstType=0 sdwRet=0 Addr = 0xe570b100.` | 测量结果搬运使能与地址。 |
| `0x5C2D` | 590 | 1 | `RFC\|TRx[0] Group[0] Interval Num[0]:[0-0][0-0]` | 收发分组间隔（RFC 归属打印）。 |
| `0x5CA7` | 589 | 1 | `DFE\|IntraCaAgcCollect CaGropup[0]: IntraCcNum=1, CcActBitMap=0x0, AgcInitBitMap=0x0, Sym=0, CcActive=[0,0,0], CcGain=[CC0:0,0,0,0, CC1:0,0,0,0, CC2:0,0,0,0], Uid=[0x0, 0x0, 0x0], MaxGain=[0,0,0,0], MinGain=[2…` | 载波聚合内多 CC 的 AGC 采集。 |
| `0x5CA8` | 589 | 1 | `DFE\|IntraCaAgcUpdate CaGropup[0]: IntraCcNum=1, AgcGain=[60,255,255,255], MasterCc=1, MasterAgc=[60,255,255,255], Sym=0` | 载波聚合内多 CC 的 AGC 更新（MasterCc 决定主控）。 |
| `0x5C07` | 325 | 1 | `DFE\|RxReq CC[0]: RxAntNum=4,Freq100Khz=18500,eBwRb=6,AgcMode=16383,AgcSymMap=0x0,bRxReceive=-1,RxChe=[0x200,0x201]` | 接收前端请求（天线数、频点、带宽、信道估计器）。 |
| `0x5C2E` | 324 | 1 | `DFE\|ConnectCc2Che CC[0]: PdIdx=4 AntNum=4 CheCfg=[0x200,0x201] PreCheCfg=[0x200,0x201]` | 接收链与信道估计器绑定。 |
| `0x5C8A` | 313 | 1 | `DFE\|SemiStaticAgc: Calculate AgcGain for CSR: [PdIdx,AntIdx]=[4,0], MaxAgcMeanPwr=2441, AgcdBGain=102` | 半静态 AGC 正常计算（按最大均值算增益）。 |
| `0x5CAC` | 292 | 1 | `DFE\|IntraCaAgcCfg: IntraCcNum=2, Cc0=[65,58,46,52], Cc1=[60,60,60,60], Cc2=[0,0,0,0], Cc3=[0,0,0,0], Cc4=[0,0,0,0]` | 载波聚合各 CC 的 AGC 配置。 |
| `0x5C8B` | 229 | 1 | `DFE\|SemiStaticAgc:Add Once AGC Calculate for check Cover of PSS state: PdIdx=4, MaxAgcMeanPwr01=[2684,2719,2945,2813], AgcdBGain=[105,105,104,105], PssNotSyncAgcCoverFlag=[0,0,0,0]` | PSS 阶段追加一次 AGC 计算并检查覆盖。 |
| `0x5C28` | 122 | 1 | `DFE\|IntCfg Core[3] IntType[0](0-Agc,1-Dc): SwPd=4, IntMask=1, IntLine=4, BitIdx=2` | 中断配置（0-Agc,1-Dc）。 |
| `0x5C90` | 94 | 1 | `DFE\|CSRS DAGC CalStep1: PdIdx=1, PssDagcMeanpwr=[691072,922624,4350336,2118784], PssLog2Pwr=[3171,3279,3852,3587], PssLinGain=[128,128,128,128]` | 搜索通道 DAGC 第一步计算。 |
| `0x5C18` | 83 | 1 | `DFE\|AsyncAgc CC[0]: =======Csr Search: AGC Enable, Clear NotSyncAGCDone, Frame.SubFrame = 760.9===========` | 搜索期间使能 AGC 并清除未同步完成标志。 |
| `0x5C10` | 82 | 1 | `DFE\|MeasFftConfig: PdIdx=4 AntNum=2 SfPattern=0 MeasHwIdx=[4,0] SpRate=0 MeasRbNum=6 FftCfg=[28,72]` | 测量专用 FFT 配置（测量只用 6 RB）。 |
| `0x5C9D` | 81 | 1 | `DFE\|GetTotalAGCGain Core[3] CC[0] SwPd[4]: Gain=[85,76], wTotalAgcDagcRsrpOffset=[64,69,7246,6479], wTotalAgcGain0Log2=[0,0,7246,6479]` | 总增益与 RSRP 偏置——把内部功率换成 dBm 的关键量。 |
| `0x5C9A` | 76 | 1 | `DFE\|ChangeBandwithCalAGCGain SwPd[4]: Freq= 18500,Bw=[6(cur),100(next)],CurGain=[114,104,113,116],NextGain=[96,92,96,96],Entrace=2!!!` | 带宽切换时重算增益：CurGain→NextGain（带宽增大增益要降）。 |
| `0x5C8C` | 63 | 1 | `DFE\|SemiStaticAgc SwPd[4]:******************The End of AGC for PSS state!********************` | PSS 阶段 AGC 结束。 |
| `0x5C9B` | 62 | 1 | `DFE\|ChangeNotSyncToSyncState Core[3] CC[0] SwPd[4]: Flag=3 Idx=4 Freq=[-1(Para),23498(Gain)] Bw=[100(Para),6(Gain)] SyncGain=[60,60,60,60] NoSyncGain=[60,60,60,60]` | 从非同步态切到同步态的增益。 |
| `0x5C9C` | 60 | 1 | `DFE\|SyncToNotSyncSetAgc Core[3] CC[0] SwPd[4]: AsyncAgcdB=[120,120,120,120] AyncLog2Gain=[2560,2560,2560,2560] AgcGainRpt=[60,60,60,60]` | 从同步态切到非同步态的增益。 |
| `0x5C82` | 16 | 1 | `DFE\|AgcFirstCal: PdIdx=4,AntNum=4,AntIdx=0,Frame.Sub=771.0,PhyState=1,IDLE_Inter=0,Gap=0` | 首次 AGC 计算（含 PHY 状态与 gap）。 |
| `0x5C88` | 14 | 1 | `DFE\|SemiStaticAgc: Greater than Max Value: [PdIdx,AntIdx]=[4,1], MaxAgcMeanPwr=3853, AgcdBGain=65` | 半静态 AGC 输入过强，增益压到下限——仪表功率可能超量程。 |
| `0x5C89` | 5 | 1 | `DFE\|SemiStaticAgc: Less than Min Value: [PdIdx,AntIdx]=[4,0], MaxAgcMeanPwr=2242, AgcdBGain=105` | 半静态 AGC 输入过弱，增益顶到上限。 |
| `0x5C21` | 4 | 1 | `DFE\|Enter/Leave FS New Section CC[0]: PdIdx=4, Flag =1, localFlag=0` | 进入/离开频扫新区段。 |
| `0x5C2F` | 4 | 1 | `DFE\|Dfe Reset! Pd=4` | DFE 复位。 |
| `0x5C2C` | 3 | 1 | `DFE\|EvtTabOffsetInit! Core[3] CC[0] SwPd[4]` | 事件表偏置初始化。 |
| `0x5C2A` | 2 | 1 | `RFC\|Resource is empty! Cc:[0] Band:[3] CellId:[0], ResFlg:[0], Para:[0](0-Next,1-Cur)` | 数字前端配置/状态；结合 AGC、CFO、DC、路径与时序。 |

### A.7 `DLA`（6个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x5E1A` | 42024 | 1 | `DLA\|INFO Che Reg: CcIdx=0, HwIdx=1, wTagId=1, SubFrm=0, adwCfoChSel=(0xeffff,0x0),YBufAddr=(0xbb6000,0xc8000), adwCellInfo=(0x1350f,0x60000000), StartReg=(0x66c311,0x5067000), dwServCellInfo=0xe613b05, dwForg…` | 下行硬件配置/报告搬运。 |
| `0x5E2A` | 41989 | 4 | `DLA\|CDTR_Reg DciRpt > HW[1]-CC[0]-TagId[0]-PdcchRpt\|HiRpt=Tag0[0x 0\|0x 0,0x 0],Tag1[0x 0\|0x 0,0x 0],Tag2[0x 0\|0x 1418,0x 0],Tag3[0x 0\|0x 0,0x 0]` | PDCCH/DCI 检测或硬件报告；与仪表 DCI 格式、RNTI、CCE 对齐。 |
| `0x7F8B` | 41989 | 1 | `DLA\|INFO > CInt >>> Cdtr Rpt: CcIdx=0, HwIdx=1, CcPpIdx=0, Cfi=0, DciValid=0x0[bit0:0,3:1a1cSib,5:1a1cPch,7:1a1cRa,9:33a,13:Other,15:4], RntiEnInd=0xffffff[bit0:C,1:Tc,2:Sps,3:Si,4:P,5:Ra,6:Cch,7:Sch,8:M], Dl…` | PDCCH/DCI 检测或硬件报告；与仪表 DCI 格式、RNTI、CCE 对齐。 |
| `0x5E06` | 38 | 1 | `DLA\|INFO > ProDbgStateSwitchPrint > State Switch 0x80 => 0x80, CurState=0x80[80:Search,81:Campon], Msg Comm=0, Dedi=0, Ho=0, Cp=0[0:Ncp,1:Ecp], CellId=503, Freq=1650, Bw=0, Ant=2, DlTM=2, FrmType=-1, SpecPat=…` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |
| `0x5E08` | 10 | 1 | `DLA\|STATIC > ProDlCtrlChStatInfoMonitor > CcIdx=0, CfiNum(0/1/2/Total)=(0, 0, 1322, 1540), HiAckNum(2CC2TB)=(15, 0, 0, 0), HiNackNum(2CC2TB)=(0, 0, 0, 0), DciNum=(01= 14,02= 0,1A= 19,2= 1,2A= 0,2B= 0,2C= 0,4=…` | 下行硬件配置/报告搬运。 |
| `0x5E02` | 3 | 1 | `DLA\|INFO > ProDbgMsgRecvCommMsg > ZPHY_EMC_EDLA_COMM_REQ(0x2b0e): State=0x81[80:Search,81:Campon], Cp=0[0:Ncp,1:Ecp], CellId=30, Freq=1275, Bw=75, Ant=4, DlTM=2, FrmType=1, SpecPat=-1, PhichNg=2, PhichDur=1[0…` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |

### A.8 `DLS`（14个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x6221` | 164 | 18 | `DLS\|DDTR_Reg BdRpt+CcTop >[Hw1-Tag1],CcTbRlt=0x3FFF0080,ErrorInfo=0x 0,CcHbTime=0x 3,Deseg=0x 0\|PdschEn=0x 3FE1DC4,CcTag=0x 98D,Cw0Cinit=0x801E4E84,Cw1Cinit=0x801E6E84,TB1Addr=0x 0,TB2Addr=0x6001D800,RptTagi…` | PDSCH/TB 结果；看 TBS、MCS、HARQ、NDI/RV 与 CRC。 |
| `0x6222` | 164 | 10 | `DLS\|DDTR_Reg DtchTb >[Hw1-Tag1],[CK,RE,QmNl,K0,K1,Ncb]-TB1-TB2:[0x 500101,0x E800E8,0x48000000,0x22000006,0x CE00B6,0x 120]-[0x18001018,0x 2160215,0x48000007,0x22000182,0x 94722D2,0x54503190],HarqPara = [0x 2…` | PDSCH/TB 结果；看 TBS、MCS、HARQ、NDI/RV 与 CRC。 |
| `0x628D` | 113 | 9 | `DLS\|CommDecCfg;Idx=3;RNTI=0xFFFF, Frame= 890,SubfNum=5;Sys=1,Id=125,Tx=2,Bw= 50;Rb= 5,RbS= 0,REs= 660,MCS= 6,RV=2,TBS= 256,TPCl=3;DCI=0x00000000,0x04320C12.` | 下行调度/译码状态。 |
| `0x6382` | 110 | 9 | `DLS\|SIBDecode:Cfg: 1,Int: 1,Ack: 1,Nack: 0,Data:[0x23805168,0x11407040,0x51236861,0x10520204,0x 5878a20]` | SIB/SI 调度与译码；看 SI-RNTI 窗口与 CRC。 |
| `0x638D` | 54 | 1 | `StdLoglInfoDl:CC[0],AvgPHYTb.kbps=[ 0, 0],1TB/2TB=[ 3, 0],TB1Dec=[ 3315, 65401],TB2Dec=[ 3233, 65533],aveRbNum= 2,MCS=[ 20, 0],CQI= 0,RI= 0,Rv_Num=[ 3, 0, 0, 0],hq_fail= 0,subFrame=256.` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x6407` | 51 | 1 | `INF >DLS_Task:wCcIndex:0,Ccb=0x120 ,dwNirDivC=0x37c20, dwKPIplus=0x120,wNumCB=0x1,wHarqIDmin=0x8,wKmimo=0x1` | 下行调度/译码状态。 |
| `0x6297` | 50 | 1 | `DLS\|DLDdrBaseRep:wHarqId=0,apbMacPduDataTB1=0x61a5a2d8,apbMacPduDataTB2=0x0,g_wHarqDdrRep=1` | 下行调度/译码状态。 |
| `0x6391` | 50 | 1 | `DLS\|MacPdu g_aw_MacPdu=[0x551c,0x9b07,0x6f78,0x86,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0],len=7!` | 下行调度/译码状态。 |
| `0x6386` | 17 | 1 | `DLS\|DecStat[CC0];PDSCH:ACK0= 0,NACK0= 0;ACK1= 0,NACK1= 0;HI:ACK= 0,NACK= 0.DCI[F0= 0,F1= 0,F1A= 1,F2= 0,F2A= 0,F2B= 0];CFI=[ 0, 0, 48, 264];DtchInt= 2;` | PDSCH/TB 结果；看 TBS、MCS、HARQ、NDI/RV 与 CRC。 |
| `0x6387` | 17 | 1 | `DLS\|DecStat;TB1Nack[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];TB2Nack[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | 下行调度/译码状态。 |
| `0x6206` | 15 | 1 | `DLS\|UeCategoryInfo;UECat=4;Cat=12,CatV1020=0;Kc*6=6,Nsoft= 1827072;wSysMode=1.wHarqIDmin=8,wKmimo=1,wAltCqiTableR12=0,wTransMode=2,PhyNirDivC=[ 228384, 6920]` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x6202` | 3 | 1 | `DLS\|Recv Common Msg; UDC=-1,SSC=-1,CellId= 30,BW= 75,Ant=4,Pb=0. wSysMode=1` | 下行调度/译码状态。 |
| `0x6384` | 3 | 1 | `DLS\|RarDecode:Cfg:1,Int:1,Ack:1,Nack:0` | 下行调度/译码状态。 |
| `0x6385` | 3 | 1 | `DLS\|Feedback[0];Ack0=[1,0,0,0],Valid0=[1,0,0,0],Ack1=[0,0,0,0],Valid1=[0,0,0,0].VALID[0x00010000,0x00000000].` | 下行调度/译码状态。 |

### A.9 `LPC`（2个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x6602` | 14 | 1 | `LPM \| LTE Dvfs Req MasterType 2 PosId 17 BeforeAdjFreq 1600 Input 1600 Type 0, result: CpuFreq 1600 Axi 800 Ddr 2 psFreq 1600 l2 5` | 低功耗/DVFS 资源状态；非 3GPP 空口值。 |
| `0x8F97` | 4 | 1 | `DBG >SleepSchd: Set dwAwakeTimer[E_ACCESS_T300_TIMER] = 0,dwStayAwakeTime[E_ACCESS_T300_TIMER]=4020!` | 低功耗/DVFS 打印。 |

### A.10 `MC`（41个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x9052` | 410795 | 1 | `TPU \| LTEA Lpm Ctrl: wCellIndex = 0 ,dwDAPSID = 0 ,eScenario= 48 ,bIsNeedOpen= 0 ,DfeIdx= 4 ,CsrIdx= 1 ,PbchIdx= 0 ,DlIdx= 1 ,UlIdx= 1 ,SUlIdx= 0 ,TpuIdx= 1 MeasDfeIdx=4` | 低功耗场景切换（eScenario 为厂商场景号）；看 bIsNeedOpen 与各硬件实例 Idx。 |
| `0x904E` | 162960 | 1 | `PRINT\|Core= 3 dwMsgId = 1058,0-4[ 0 9 10 0 0] 5-9[ 1 2 8 9 10] 10-14[ 0 0 1 2 8] 15-19[ 9 10 0 0 0]` | 每子帧 20 时隙任务表快照；只在怀疑某时隙没排上任务时看。 |
| `0x9054` | 8395 | 1 | `PRINT\|Id=0x 2b67 0-4[ 100 0 0 104 1] 5-9[ 0 0 0 0 0] 10-14[ 0 0 0 0 0] 15-19[ 0 0 0 0 0]` | 内部消息参数快照（Id 为消息号）；需版本接口表。 |
| `0x9C01` | 2676 | 7 | `RLMState:0; N[Cnt:0, 310311:1-1]; CalcRst:0; Drx[State0, Cyc: 0, CalcFlg:0, Act:0, QC:998428672- 0]; SlpCnt:0, Win[Flg:1, Con&DrxCnt:40, 0]; FI&THFilter:0, 0; Final[TH:34000, QC:24960716]; CurSNR:100000.` | RLM 状态：TH 为质量门限、CurSNR=100000 是无效哨兵；本文件 RLMState 恒 0，未进入实质评估。 |
| `0x9055` | 718 | 1 | `L1l_RcvUnknownMsg > ThreadId = 4,MsgId = 1069,FileNo =93,Line = 9638` | 收到未实现的内部消息（ThreadId/MsgId/FileNo/Line）；MsgId 固定时通常无害。 |
| `0x9E10` | 374 | 1 | `SIR\|INFO > RxRcvCtrlMonitor > CurTime: Frm = 242, SubFrm = 7, RcvSubFrm = 2444, RcvInd = 1 [1:Close 2:Open]` | 接收控制监视；RcvInd=1:Close 2:Open。 |
| `0x9E12` | 335 | 2 | `SIR\|INFO > StartWinMonitor > ZPHY_ETPU_EMC_SI0_WIN_START: CurTime = 2445, RntiEn = 1, CurSiNxtPos:246-10-0, CurSiPeriod = 2, WinLen = 1, SchedNum = 1, PreDec[Idx:0x0,Pos:0,Ok:0], CurDec[Idx:0x42d,Pos:2445,Ok:…` | SI 窗口开始，RntiEn=1 打开 SI-RNTI 搜索；CurSiPeriod 单位为无线帧（2=SIB1 的 20 ms）。 |
| `0x9E13` | 333 | 1 | `SIR\|INFO > EndWinMonitor > ZPHY_ETPU_EMC_SI0_WIN_END: CurTime = 2446, RntiEn = 0, CurSiNxtPos:246-10-0, CurSiPeriod = 2, WinLen = 1, SchedNum = 1, PreDec[Idx:0x0,Pos:0,Ok:0], CurDec[Idx:0x42d,Pos:2445,Ok:0]` | SI 窗口结束，RntiEn=0。 |
| `0x904F` | 235 | 1 | `L1l_SchedRF_SendRfResReq > eScene = 3,dwReqType = 0x100c1,dwEarfcn =1650,Bandwidth = 6,cRxNum = 255,eRxType = 0,dwPara = 0,bCnfMsg = 1,wMsgId = 11333` | 向 RF 调度请求资源（eScene/dwReqType/Earfcn/Bandwidth）。 |
| `0x904A` | 186 | 1 | `TPU \| LTEA Macro Adj Cnf Int TpuHwIdx = 1 ,Cur[Frame,Hsf] = 242 ,0` | TPU 宏调整完成中断；与 0x904C 请求应一一对应。 |
| `0x904C` | 186 | 1 | `TPU \| LTEA Macro Req TpuHwIdx = 1 ,[frame,hsf,tc:15khz] OldMrtrOffset = [488,14,3968] ,NewMrtrOffset = [488,0,3936] tAdjTime = [242,0,12904]` | TPU 宏调整请求：新旧 MrtrOffset 之差即本次挪动量。 |
| `0x9E0D` | 84 | 9 | `SIR\|INFO > CellMonitor > Earfcn = 1650, CellId = 503, CurFreq = 18500, NxtFreq = 18500, Tpu = 303264, MibMark = 1, Bch = 535.19.11424, Search = 0.19.11424` | 小区监视：Earfcn/CellId/CurFreq/MibMark（-1 无效，1 有效）与 BCH、搜索边界。 |
| `0x904D` | 80 | 1 | `TPU \| LTEA -->Micro Req :TpuHwIdx = 1 ,AdjTc= 3 ,[IsAdd = 0 , IsSub = 1] ,OldMrtrOffset = [488,0,5616] ,NewMrtrOffset = [488,0,5613] [frame,hsf,tc:15khz]` | TPU 微调（AdjTc、IsAdd/IsSub）；持续单向微调说明有残余频差。 |
| `0x9610` | 80 | 1 | `PreSync[0xFF];Sys=1,Udc=-1,T= 6;PO=9;Rf=0,Agc=0,Fss=1,Cfo=1;Subf=3,4,5;Idx=1,2,2.Gota:T= 0,PO=0;Subf=1965594,0,0.` | 预同步进度：Rf/Agc/Fss/Cfo 四个子步骤状态与安排的子帧。 |
| `0x9E0F` | 56 | 1 | `SIR\|INFO > MibCnfMonitor > ZPHY_ECSR_EMC_BCH_DECODE_CNF: MsgId = 0x2b9c, Earfcn = 1650, CellId = 453, Crc = 0, Mark = 0, Band = 0, Ant = 0, Phich = 0, BchSfn = 0.0.0` | MIB 解码确认：Crc=1 有效；Band 是下行 RB 数、Ant 是 CRS 端口数、Phich 是打包的 phich-Config。 |
| `0x960F` | 54 | 1 | `PreSync[SleepSched]; SetRxCirPreSyncFlag=0!!!` | 睡眠调度改写预同步标志。 |
| `0x9E0B` | 46 | 1 | `SIR\|ERROR > ErrMonitor > PrintId = 0x11, ErrCode = 3, SysMode = 1 [0:Tdd 1:Fdd], MainState = 1[1:Sib1 2:Si 3:Abort], StepState = 2[1:Msg 2:Mib 3:Rfc 4:Tpu 5:AdjOver 6:Sched], MsgId = -1` | SIB 读取错误监视；MainState=1:Sib1 2:Si 3:Abort，StepState=1:Msg 2:Mib 3:Rfc 4:Tpu 5:AdjOver 6:Sched。 |
| `0x9E0E` | 46 | 1 | `SIR\|INFO > SibParaMonitor > UpSibPara: SirNeiBor = 1[1:Neibor 2:Back 3:Serve], RatMode = 1 [0:TDD 1:FDD], Earfcn = 1650, CellId = 503, FrmTye = -1, SpecPat = -1, MiBld = 0, SchedNum = 1, WinLen = 1` | 更新 SIB 读取参数；SirNeiBor=1:邻区 2:回退 3:服务小区。 |
| `0x9E06` | 44 | 1 | `SIR\|INFO > Sib1MsgMonitor > ZPS_LTE_P_READ_SIB1_REQ_EV: MsgId = 53506, Earfcn = 1650, CellId = 503, ReqForHo = 0, ProcId = 23` | 收到协议栈的 READ_SIB1_REQ（Earfcn/CellId/ProcId）。 |
| `0x9E11` | 41 | 1 | `SIR\|INFO > SchedParaMonitor > SiIdx = 0, CurTime = 2427, WinValid = 1, SchedPos:244-10-0, Period = 2, SchedIdx = 0, WinLen = 1, SchedNum = 1` | SI 调度参数：SchedPos/Period/WinLen（时间编码为 SFN×10+子帧）。 |
| `0x9E05` | 40 | 1 | `SIR\|INFO > DelyTpuAdjust > DelayEvtRegister: MsgId=0x443, CurFrm=242, CurSunFrm=1, DelayTime:Frm=0,Slot=12,Ts=0` | 延迟 TPU 调整事件注册。 |
| `0x9E0C` | 40 | 1 | `SIR\|INFO > RfcMonitor > Freq = 18500, Bw = 0[0:20M,1:15M,2:10M,3:5M,4:3M,5:1.4M,6:NoChange,Other:Ca], UDMode = 0, SpecMode = 0, CpMode = 0, Tpu = 301584, NextFreq = 18500, AbortSiState = 0` | 为读 SI 配置 RF；Bw=0:20M 1:15M 2:10M 3:5M 4:3M 5:1.4M 6:不变。 |
| `0x9E03` | 38 | 1 | `SIR\|INFO > SndMibReq > E_L1L_MC_PBCH_MIB_READ_REQ: Freq = 1650, CellId = 503, MsgId = 11001, AntCnt = -1, Boundry = 303264, Cp = 0, FrmType = -1, SpecFrmPat = -1, StartBchTime = 0` | 向 PBCH 发起 MIB 读取请求（Boundry 为帧边界 Ts）。 |
| `0x960A` | 32 | 1 | `zPHY_emc_CalPagingParam(NbValue=2;T=128;nB=128;N=128;UeID= 277;Ns=1;SubframIndex=0;RatMode=1; PagingState=1, wActivedPageCycle =128, PO= 917:9, tIdlePiStartTime= 916:9,PoGapCflict=0,wModDstSfn=[21,917],[0,0]` | 寻呼 PF/PO 计算：T/nB/N/Ns/UeID；可用 36.304 公式直接复算。 |
| `0x960D` | 29 | 1 | `DBG > 917- 9, -- > Cur PO INT Event in MC,ActivedPageCycle: 128, PoAndGapConflict: 0` | PO 中断真正到来。 |
| `0x9E09` | 26 | 1 | `SIR\|INFO > SibReportMonitor > E_L1L_DLS_MC_SI_CRC_OK_MSG: SibPos = 2445, SibTbs = 22, ReptNum = 1, SchedNum = 1` | DLS 上报 SI CRC OK：SibPos/SibTbs/ReptNum。 |
| `0x960B` | 18 | 1 | `&&&&&&&&&&&&&&&&&&& PI Task START &&&&&&&&&&&&&&&&&&& RX===>CIR Adjust Total Value in One DRX for Ant0, Ant1, Ant2, Ant3: [ -167, -46, -62, -62]` | 一个 DRX 周期内各天线累计 CIR 定时调整量（Ts）。 |
| `0x9612` | 18 | 1 | `&&&&&&&&&&&&&&&&&&& DBG > zPHY_emc_tRxCirPreSyncStart, RxCirPreSyncState=0! CsrNoUseRxMeasFlag=1(Value=0) wIdlePiCnt = 0.` | RxCirPreSync 启动。 |
| `0x9618` | 18 | 1 | `DBG > -- > Cannot Colse RF at RAPC, SIB1 or SI PLMN PROC` | 不能关 RF 的原因（RAPC/SIB1/SI PLMN 过程进行中）。 |
| `0x9614` | 16 | 1 | `PreSyncAccNum:1,AbsSumCirAdjVal:[0,0],AbsSumCfoAdjVal;[0],pre[0,0,0,0,0,0,0,0],[0,0,0,0]Flag:0,M=0,[0,0]` | 预同步累计：CIR/CFO 调整绝对值之和。 |
| `0x9615` | 16 | 1 | `PreSync:0,CurCnt:0,PreSyncExtendFlag:0,PchCycleExtendFlag:0,TrailingFlag:0,CurAdjTs:[0,0],SingleAntInd:0,CurIdx=1` | 预同步状态与本次调整 Ts。 |
| `0x9E01` | 16 | 1 | `SIR\|INFO > MainCtrlFlow > Recv ZPS_LTE_P_ABORT_SI_READ_REQ_EV!` | SIR 主控流程：收到 ABORT_SI_READ_REQ。 |
| `0x9E1D` | 15 | 1 | `*SIR\|INFO > AbortSi > *******SIB Fail!!!********/` | SIB 读取整体放弃（SIB Fail）；与 MainState=3 的 ErrMonitor 成对出现。 |
| `0x9049` | 13 | 1 | `TPU \| LTEA Tpu --->Reset TpuHwIdx= 0` | TPU 复位。 |
| `0x9048` | 7 | 1 | `TPU \| LTEA Tpu ModeSet TpuHwIdx= 1 ,eCpMode= 0[0:Ncp 1:Ecp] ,eWorkMode= 1[0:NR 1:LTE] eScsMu= 0[0:15Khz 1:30Khz 2:60Khz 3:120Khz]` | TPU 模式设置：CP 类型、工作制式、子载波间隔（自带枚举）。 |
| `0x9605` | 3 | 1 | `DBG >zPHY_emc_sys_info_update PO Para: CurSfn = 881,CurSubSfn = 6,PoStartFrame = 917,PoStartSubFrame = 9, wActivedPageCycle = 128` | 寻呼参数更新（PoStartFrame/PoStartSubFrame/PageCycle）。 |
| `0x9E08` | 3 | 1 | `SIR\|INFO > SiMsgMonitor > ZPS_LTE_P_SCHED_SI_REQ_EV: SiIdx = 1, WindowLen = 1, Period = 20` | 收到 SI 调度请求（SiIdx/WindowLen/Period）。 |
| `0x9611` | 2 | 1 | `DBG > -- > no need to do RxCirPreSync !!!` | 判定本次不需要 RxCirPreSync。 |
| `0x9E07` | 2 | 1 | `SIR\|INFO > SiMsgMonitor > ZPS_LTE_P_SCHED_SI_REQ_EV: MsgId = 53507, Earfcn = 1275, CellID = 30, SubFrmAssign = -1, SpecAssign = -1, SchedNum = 2` | 收到 SI 调度请求（Earfcn/CellID/SchedNum）。 |
| `0x8A1E` | 1 | 1 | `zPHY_emc_ProDrxSchedFlow --- DlSet DL abDLHarqPdcchFlag:wCaIdx = 0,wHarqId = 6,HarqPdcchFlag = 1,rtttimelen = 8,Cursfn = 902,Cursubsfn = 6` | DRX 下 DL HARQ 与 PDCCH 标志设置。 |
| `0x8F07` | 1 | 1 | `DBG > DEDICATED First Config at Frame 895, SubFrame 6, E_ZPHY_STATE_CONN` | DEDICATED 首次配置＝真正进入连接态的时刻。 |

### A.11 `MULM`（4个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x848C` | 63140 | 1 | `MulmSlaveMeasureFlow:>>=PlmnSearchMeasFlag/cnt=(0xffff ,300),GapRfState=0,SlaveSyn/fun_State=(0,0)SearchOn =-1 0[0 -1 0]Cnt[-1]AbortGapFlag = 0 Lpc = 0 0 -1 CfoCnt= 0 40msGapCnt = 0` | 多频测量从流程周期状态；有价值的是 GapRfState 与各计数的差分。 |
| `0x9401` | 13 | 1 | `Csr Slave State Change to:0[0=SLAVE_IDLE;1=SLAVE_SEAR_MEAS_IDLE;2=SLAVE_ASYN_SEAR_MEAS;3=SLAVE_SYN_SEAR_MEAS;4=SLAVE_FREQSCAN;5=SLAVE_PBCH].` | 从流程状态跳转：0=IDLE 1=SEAR_MEAS_IDLE 2=ASYN 3=SYN 4=FREQSCAN 5=PBCH（自带枚举）。 |
| `0x9402` | 13 | 1 | `Csr Slave SYN State Change to:0[0=E_CSR_SLAVE_ASYN;1=E_CSR_SLAVE_SYNING;2=E_CSR_SLAVE_SYN` | 从流程同步状态：ASYN/SYNING/SYN（自带枚举）。 |
| `0x9412` | 4 | 1 | `zPHY_ecsrc_TaskEntry > REV ZPHY_EMULM_EMC_ECSR_SET_MODE_REQ IratMode = 0` | 收到异系统模式设置请求（IratMode）。 |

### A.12 `PBCH`（17个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x6A10` | 2584 | 1 | `PBCH\|INFO > IntRptMonitor > PBCH RESULT: Frame.SubFrm=241.1, MIB_CRC=0x1b9d0400, CrcOk=0, BchMib=0x1b9d04, dwAntNum=3, dwBranch=2, SFN=926, wPbchIntCnt=73` | PBCH/MIB 译码与 CRC；看 CrcOk、天线端口、带宽与 SFN。 |
| `0x7EC1` | 2583 | 1 | `PBCH\|DRV > ConfigPbchReg > Cdtr: HwIdx=0, CpMode=0, NewDataInt=0, Branch=3, AntNum=3, CellId=453, MaxIts=3, Update=1, ScGenEn=1, PbchEn=0x10000, State=0[0:Idle,1:WaitLlr,2:WaitViterbi,3:ViterbiWork]` | PDCCH/DCI 检测或硬件报告；与仪表 DCI 格式、RNTI、CCE 对齐。 |
| `0x7F8D` | 423 | 1 | `PBCH\|Drv > Update Rx Regs > CellInfo = 0x00010b57, MimoScheme = 0x00003110, CheStartParm = 0x000ec311.` | PBCH/MIB 处理。 |
| `0x6A0D` | 273 | 1 | `PBCH\|INFO > RxRsrpMoniter > RSRP00[I,Q]:[0,-2], RSRP01[I,Q]:[1,3], RSRP10[I,Q]:[3,1], RSRP11[I,Q]:[2,6], RSSI[Rx0,Rx1]:[92,328], RSPRx0:[20,24,4], RSPRx1:[69,67,38]` | 测量量；区分 dBm 域与 36.133 上报域。 |
| `0x6A04` | 151 | 1 | `PBCH\|INFO > UpRxState > RxCurrState = 3, RxPreState = 2[0:Idle,1:Meas,2:Pbch,3:Nomal,4:Sync]` | 同/异频测量、gap 或 L3 过滤流程。 |
| `0x6A0E` | 62 | 1 | `PBCH\|INFO > MibReqMonitor > MsgId=11001, Earfcn=1650, CellId=503, Boundry=303264, CpMode=0, Ant=-1, Bw=-1, FrmTyp[-1,-1]` | PBCH/MIB 处理。 |
| `0x6A08` | 61 | 1 | `PBCH\|INFO > AdjTpuTime > AdjFrame = 535, OffsetSlot = 5, OffsetTs = 11392, AdjustBoundry = 19.11424` | PBCH/MIB 处理。 |
| `0x6A0F` | 61 | 1 | `PBCH\|INFO > RfcTpuMonitor > ZPHY_ETPU_EPBCH_UPDATE_COUNTER_CNF: dwMsgId = 1064, TpuOffset = 307200, AFC = 0` | 频偏估计或补偿；与 PSS/SSS/PBCH 稳定性关联。 |
| `0x7EC2` | 61 | 1 | `PBCH\|DRV > ScGeneration > wHwIdx=0, CellId = 503, ScGenEn = 1` | PBCH/MIB 处理。 |
| `0x6A14` | 39 | 1 | `PBCH\|INFO > MibInfoCheck > PsPhyATNvcom: g_zPsPhyATNvcom.mtnetTestFlag=0, g_bHandoverCsrCnf=0, g_tHandoverReq.tPhyCellInfo.wDlBandWidth=0, awBwIdxMapToNrbdl[(ptrMibRlt->uMibCrc.tReg.dwBchResult & 0xe00000) >>…` | PBCH/MIB 译码与 CRC；看 CrcOk、天线端口、带宽与 SFN。 |
| `0x6A11` | 38 | 1 | `PBCH\|INFO > CrcRltMonitor > =========================================== MIB OK!! ===========================================` | PBCH/MIB 处理。 |
| `0x6A12` | 38 | 1 | `PBCH\|INFO > CrcRltMonitor > PBCH CRC OK: Earcfn=1650, CellID=503, dwCrcOk=1, dwBchMib=0xa8f000, RBNum=100, dwAntNum=4, BchPhich=0x2, dwBranch=2, PbchIntNum=3, FrmIntNum=1, NowMrtrTime=777.15.12576` | PBCH/MIB 译码与 CRC；看 CrcOk、天线端口、带宽与 SFN。 |
| `0x6A03` | 25 | 1 | `PBCH\|INFO > PbchTaskEntry > Recv CSR Reset Msg, MsgId=11007` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x6A09` | 24 | 1 | `PBCH\|INFO > DecideNxtDecode > Extra Max Branch,Report Fail: Earfcn = 1650, CellID = 453, PbchIntNum = 84, FrmIntNum = 7` | PBCH/MIB 处理。 |
| `0x6A01` | 7 | 1 | `PBCH\|INFO > PbchTaskEntry > Recv MC Reset/SetMode Msg, MsgId=11008` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x6A02` | 3 | 1 | `PBCH\|INFO > PbchTaskEntry > Recv Release Msg, MsgId=11010` | PBCH/MIB 处理。 |
| `0x6A0C` | 1 | 1 | `PBCH\|ERROR > ErrorMoniter > ################ Error Print, ErrId = 3[1:MibSucc_ReqWrong,2:MibSucc_ReqWrong,3:EventDelFail,4:StartMibFail]################` | PBCH/MIB 处理。 |

### A.13 `PS_PHY接口消息`（2个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x8F83` | 251 | 3 | `INTF >Send MSG Phy--->Ps, ID=0x D40B, Size= 4, at:Frame 241,Subbframe 1.{68(53565,2048)}.` | PHY→协议栈消息包络：消息 ID、长度与发生的帧/子帧。 |
| `0x8F82` | 185 | 1 | `INTF >Recv MSG Ps--->Phy, ID=0x D306, MsgIdx=13, at:Frame 241,Subbframe 4! C-RNTI_En=0.` | 协议栈→PHY 消息包络：确认上层到底有没有下发。 |

### A.14 `RAPC`（56个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x9910` | 108 | 1 | `UL\| FuncIdx=0 [L1l_erapc_PdcchOrderTimeCalc() ,zPHY_erapc_RaResourceSelectFDD()]-> Preamble select TimeStart: tTimeStart.wFrame=882, tTimeStart.wSubFrame=9,` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9803` | 54 | 1 | `UL\| L1l_Rapc_TpuRarEnableEv()-> RAR Detect start at Frame 883, Subframe 4!,RfTxOffset=0.` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9804` | 54 | 1 | `UL\| L1l_Rapc_TpuRarEnableEv()-> Enable RA-RNTI flag at frame:883,subframe:4!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9908` | 54 | 1 | `UL\| zPHY_erapc_RaResourceSelect()-> Resource Select Result: wFra=0, wFrameDes=883, wSubFrameDes=1,` | 随机接入过程控制。 |
| `0x9909` | 54 | 1 | `UL\| zPHY_erapc_RaResourceSelect()-> RA-RNTI Value=2` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9911` | 54 | 1 | `UL\| zPHY_erapc_RaResourceSelectFDD()-> wCfgAvailNum = 1,awCfgAvailSubFrame[0]=1,[1]=255,[2]=255,[3]=255,[4]=255` | 随机接入过程控制。 |
| `0x9915` | 54 | 1 | `UL\|Err > zPHY_erapc_PreambleGroupSelect()-> Invalid input of L1l_Rapc_PreambleGroupSelect(VOID) function!` | PRACH/preamble 选择、功率与发射时序。 |
| `0x991A` | 54 | 1 | `UL\| zPHY_erapc_PreamCycShiftCalc()->Preamble LogicU=22` | PRACH/preamble 选择、功率与发射时序。 |
| `0x991F` | 54 | 1 | `UL\| zPHY_erapc_PreambleTransPower()->Preamble trans Frame = 883, Subframe = 1, Pathloss = 91, PrachPower = -13.0!` | PRACH/preamble 选择、功率与发射时序。 |
| `0x993B` | 54 | 1 | `UL\| zPHY_erapc_RandomNumGenerate()->Random number=4 between 0~52` | 随机接入过程控制。 |
| `0x9943` | 54 | 1 | `UL\| zPHY_erapc_ConfigSAD()->CurDesFrame = 1` | 随机接入过程控制。 |
| `0x9946` | 54 | 1 | `UL\| zPHY_erapc_ConfigSAD()->wPreambleLength=1` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9948` | 54 | 1 | `UL\| zPHY_erapc_ConfigSAD()->SubFrame=1 PValue=1, PreaFormat=0, Cv=104, KValue=1955594240, PreamPower=-13, tCfgType=3, tTxChannelType=1` | 随机接入过程控制。 |
| `0x994A` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->Preamble Group=0` | PRACH/preamble 选择、功率与发射时序。 |
| `0x994B` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->PreambleID=4` | PRACH/preamble 选择、功率与发射时序。 |
| `0x994D` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->NPrbRa Value=4` | 随机接入过程控制。 |
| `0x994E` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->PValue=1, Cv=104, KValue=1955594240 Preamble Trans Power=-13` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9951` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->Add RAR detect start TPU event success! TPU event time: wFrameDes=883,wSubFrameDes=4, dwCurSSubframe=0,dwDelayTss=184320` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9952` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->Add RAR detect stop TPU event success! TPU event time: wAbsSfn=8847,` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9954` | 54 | 1 | `UL\| zPHY_erapc_RaRetransProc()->wPreamTransCounter=1` | 随机接入过程控制。 |
| `0x9965` | 54 | 1 | `UL\| zPHY_erapc_PreamTransPro()->RAPC_RAR_DETECT_ENABLE_EV:wCurSubFrame=8,wCurFrame=882,wFrame=883,wSubFrame=4` | 随机接入过程控制。 |
| `0x9805` | 50 | 1 | `UL\| L1l_Rapc_TpuRarDetectDisableEv()-> RAR Detect stop at frame:222,subframe:7, failed to recieve RAR PDU. RfTxOffset=0.` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9806` | 50 | 1 | `UL\| L1l_Rapc_TpuRarDetectDisableEv()->Disable RA-RNTI flag at frame:222,subframe:7!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9904` | 50 | 1 | `UL\| zPHY_erapc_BiProc()-> BI Delay time=0ms` | 随机接入过程控制。 |
| `0x9930` | 37 | 1 | `UL\| zPHY_erapc_SetRapcState()->RAPC state = E_RAPC_IDLE !!` | 随机接入过程控制。 |
| `0x991D` | 28 | 1 | `UL\| zPHY_erapc_PreambleTransPower()->Current Frame = 882, Subframe = 8, PathLoss = 91, swRsrpFilterValue = -73!` | PRACH/preamble 选择、功率与发射时序。 |
| `0x991E` | 28 | 1 | `UL\| zPHY_erapc_PreambleTransPower()->Preamble trans Frame = 883, Subframe = 1, PrachPower = -13!` | PRACH/preamble 选择、功率与发射时序。 |
| `0x992E` | 13 | 1 | `UL\| zPHY_erapc_RntiDelete()->Disable C-RNTI flag at frame:268,subframe:7!` | 随机接入过程控制。 |
| `0x980A` | 10 | 1 | `UL\| L1l_Rapc_McResetOrSetmodeReq()-> Recv ZPHY_EMC_ERAPC_RESET_REQ.` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x9802` | 4 | 1 | `UL\| L1l_Rapc_TpuPreamProcEv()-> Recv ZPHY_ETPU_ERAPC_PREAMBLE_PROCESS_EV.` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9812` | 4 | 1 | `UL\| L1l_Rapc_McAccessReq()-> Recv ZPHY_EMC_ERAPC_ACCESS_REQ at sf: 8816, CCCH-SDU Content RA Start.` | 随机接入过程控制。 |
| `0x9814` | 4 | 1 | `UL\| L1l_Rapc_McAccessReq()->Preamble Format=0` | PRACH/preamble 选择、功率与发射时序。 |
| `0x981D` | 4 | 1 | `UL\| zPHY_erapc_Entry()-> Recv ZPHY_EMC_ERAPC_MAC_RESET_REQ at sf: 4510.` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x990C` | 4 | 1 | `UL\| zPHY_erapc_RaResourceSelect()-> ZPHY_ETPU_ERAPC_PREAMBLE_PROCESS_EV DelayTime:wDelayTime=12` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9931` | 4 | 1 | `UL\| zPHY_erapc_SetRapcState()->RAPC state = E_RAPC_STEP1_2 !!` | 随机接入过程控制。 |
| `0x9807` | 3 | 1 | `UL\| L1l_Rapc_TpuContensResoluStartEv()-> Contesion Resolution Window start at frame:884,subframe:6 !` | 随机接入过程控制。 |
| `0x9808` | 3 | 1 | `UL\| L1l_Rapc_TpuContensResoluStartEv()->Enable T-C-RNTI flag at frame:884,subframe:6!` | 随机接入过程控制。 |
| `0x980B` | 3 | 1 | `UL\| L1l_Rapc_McCommConfigOrHandoverReq()-> Recv ZPHY_EMC_ERAPC_COMMON_CONFIG_REQ.` | 随机接入过程控制。 |
| `0x9815` | 3 | 1 | `UL\| L1l_Rapc_McRarMacpduEv()->Recieve RAR PDU from DLS at frame:884,subframe:0!!!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9818` | 3 | 1 | `UL\| L1l_Rapc_McMsg4PduEv()-> Recv ZPHY_EDLS_ERAPC_MSG4_PDU_DETECT_EV at frame:888,subframe:4.` | 随机接入过程控制。 |
| `0x981A` | 3 | 1 | `UL\| L1l_Rapc_McMsg4PduEv()->Need to response Ack, Set Msg4 Ack Flag to TRUE! wSubFrame=0` | 随机接入过程控制。 |
| `0x981C` | 3 | 1 | `UL\| L1l_Rapc_McMsg4PduEv()-> Recv ZPHY_EMC_ERAPC_ABORT_ACCESS_REQ at frame:888,subframe:6, Random Access Success*******` | 随机接入过程控制。 |
| `0x981E` | 3 | 1 | `UL\| zPHY_erapc_Entry()-> Recv ZPHY_EMC_ERAPC_REL_REQ at sf: 1861.` | 随机接入过程控制。 |
| `0x990F` | 3 | 1 | `UL\| L1l_erapc_PdcchOrderTimeCalc()-> RF is not opened yet, delay to transmit preamble for 10ms !` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9922` | 3 | 1 | `UL\| zPHY_erapc_RarMacPduDecode()->*** RA_PID match! *** PreambleId=4, RA_PID=4` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9924` | 3 | 1 | `UL\| zPHY_erapc_RarMacPduDecode()->RAR content: TA command=0, Temporary C-RNTI=121, RarUlGrant=284` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9932` | 3 | 1 | `UL\| zPHY_erapc_SetRapcState()->RAPC state = E_RAPC_STEP3_4 !!` | 随机接入过程控制。 |
| `0x9940` | 3 | 1 | `UL\| zPHY_erapc_SendRaCnfMsg()->Random Access result : Content RA Success!` | 随机接入过程控制。 |
| `0x9958` | 3 | 1 | `UL\| zPHY_erapc_RarDetectedProc()->Disable RA-RNTI flag at frame:884,subframe:0!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9959` | 3 | 1 | `UL\| zPHY_erapc_RarDetectedProc()->Delete RAR detect stop TPU event success!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x995C` | 3 | 1 | `UL\| zPHY_erapc_RarDetectedProc()->RAPC RAR GRANT Content: ConResoTimer = 48, cMaxHarqMsg3Tx = 4, wSubFrame = 8839, wPowerRampup = 0, dwRarUlGrant = 284, bContensionFlag = 1, wDataValid = 1!!!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x995E` | 3 | 1 | `UL\| zPHY_erapc_CRntiMsg4Proc()->Delete contension stop TPU event success!` | 随机接入过程控制。 |
| `0x995F` | 3 | 1 | `UL\| zPHY_erapc_CRntiMsg4Proc()->Disable T-C-RNTI flag at frame:888,subframe:4!` | 随机接入过程控制。 |
| `0x992A` | 1 | 1 | `UL\| zPHY_erapc_TpuEventDelete()->Delete RAR detect stop TPU event success!` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x9942` | 1 | 1 | `UL\| zPHY_erapc_SendRaCnfMsg()->Random Access result : Preamble Trans Counter Exceed Maximum!` | PRACH/preamble 选择、功率与发射时序。 |
| `0x9955` | 1 | 1 | `UL\| zPHY_erapc_RaRetransProc()->wPreamTransCounter reach max......... !!` | 随机接入过程控制。 |

### A.15 `RFC`（72个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x6F48` | 198153 | 1 | `drvRffe_evt 17-1\|0()[(clk[23:20],DbbOrV4[19:16],ch[15:8],usid[7:4],type[3:0])0x152-0x5-0x1],[0x52-0x2-0x5],[0x152-0x2-0x5],[0x153-0x12-0x100fe],[0x53-0x12-0x100fe],[0xa2-0x1c-0x7],[0x1a2-0x1c-0x7],[0x52-0x1c-…` | RFFE 总线事件（地址-命令对）：只用于确认在写 RF，逐条解读需寄存器手册。 |
| `0x6F0F` | 177967 | 1 | `drvRfic_evt 13-0\|8()[0x2000000-0x5883f200],[0x2000004-0x3483f200],[0x2000008-0x5683f200],[0x200000c-0x4083f200],[0x2000508-0xf],[0x200051c-0xf],[0x2000c70-0x0],[0x20003b8-0x0],[0x20003b8-0x11],[0x20003bc-0x0]` | RFIC 寄存器事件（地址-数据对）：同上。 |
| `0x5C03` | 163359 | 1 | `DFE\|UidCfg CC[0]: preSlot=[DL=0, CSR=0, UL=0], curSlot=[DL=1, CSR=0, UL=0], nextSlot=[DL=0, CSR=0, UL=0], CsrSync=[0,0], CsrMeas=[0,0], CsrRssi=[0,0], CsrPrs=[0,0] SyncState=1 TxState=[0(pre),0(Cur),0(nex) Pw…` | 每时隙的前端用户占用（DL/CSR/UL）与同步状态。 |
| `0x5C04` | 163359 | 11 | `DFE\|OffsetCfg CC[0]: RxOfs=[ 0, 0, 0, 0], MinRxOfs= 0, TxDbbOfs= 0, TxEnFix= 0, MeasOfs= 0, TxEnOfs= 0, ReCpOfs=[ 0, 0, 0, 0], TxOfs=[ 0, 0, 0, 0, 0], BwDly= 0` | 各类收发偏置汇总。 |
| `0x5C31` | 163359 | 1 | `DFE\|SsIntMrtr CC[0]: CurLocalMrtr=241.0.509, CurSfMrtr=509, CurSym=0` | 时隙中断时间戳。 |
| `0x6F43` | 163359 | 1 | `Sched-RfcTxOffset\|mdm=2, slot=1, RfcTxOffset=[1,1]` | 每时隙的发射偏置。 |
| `0x6F53` | 163359 | 1 | `UserNVInfo\|platform=MC86001 version=.0.6.6 2 timestamp=0260820_112058` | 平台与固件版本（每子帧打印，用于确认日志来源版本）。 |
| `0x700F` | 163359 | 1 | `RFC\|TxDbbCfg: TxIdx=1 CurSf=0 TimeOfs=[117977,121820] TxEvt=0 TxEvtEnBitMap=[0x19,0x0]` | 发射 DBB 配置与事件位图。 |
| `0x7041` | 163359 | 1 | `RFC\|RFSDMerge: State[9] Rx Cur:[18500,5], Next:[18500,5] Tx Cur:[25451,0], Next:[25451,0], Ant Cur:[4], Next:[4]` | 收发频点与天线的当前/下一状态合并。 |
| `0x6F44` | 162861 | 3 | `DBB\|Ppi[5f7df e3c7ff e3c7ff e3c7ff 7df7df 7df7df ffffff 7e1be2 7df7df 7df7df 7df7df 7df7df 8e2860 7e1be0 7df7df 7df7df 7df7df 7df7df 7df7df 7df7df 7df7df 7df7df 7df7df 7df7df] Int[391400]` | DBB 并行接口状态与中断计数。 |
| `0x6F45` | 162861 | 4494 | `DBB\|RxDlcClc[b7b7 3131 7c7c d8d80000 0 b3b3 0 0] TxDlcClc[b2b20000 ebeb7f7f 1717 5a5a7474 3030000 800] RxLaErr[0 0 0 0] B810B[0 0 0 0] RXFlagReg[0 0 0 0] TXFlagReg[0 0] TXJitter[0 0] ClkDiffCnt[1]` | DBB 收发链路计数与错误标志：看 RxLaErr/FlagReg 是否由 0 变非 0。 |
| `0x6F46` | 162861 | 52 | `DBB\|RxFifo[0 0 0 0 0 0 0 0][a02 2 802 2 2 2 1902 2] TxFifo[1 1 1 1][1 1 1 1] DlcCrc[0 0 0] ClcCrc[0 0 0]` | DBB 收发 FIFO 与 CRC 计数：DlcCrc/ClcCrc 非 0 即链路误码。 |
| `0x6F59` | 162861 | 14 | `DBB\|FrameNum[3100c4 b30000] FrameLenErr[0 0] FrameLen[3c0012 3c0000] DiffId[0 0] HdId[0 0] uInt[0 0 0 0 4000 0] Riclc[9db] Raslc[0] Rbclc0[0] Rbclc1[0] Rnclc[80000edd] Raclc[0] Rhlclc0[0] Rhlclc1[0] Rfclc[0]` | DBB 帧号与帧长错误计数。 |
| `0x6F5F` | 162861 | 1 | `DBBV4\|Rx:La[2] Dr[0] LaR[1] Frame[12] LaA[3] Clk[0] DrR[7 7 7 7] LaEn[3 3 0 0] ClkG[3 3 3 3] Tx:La[1] Dr[0] Clk[11] M1[1010] M[1490057] La[0] Clc[36435] DrR[7 7] LaE[1 0] ClkEn[3 3]` | DBB V4 收发通道/时钟状态。 |
| `0x6F4F` | 162837 | 3 | `RficMcuStat\|McuHang, WARNING! Rfic may hang, AliveCnt has not changed! Before:552063996, Current:552068876, InitErrBit:4` | RFIC 存活告警；只有 Before==Current 才是真挂起，本文件 99.99% 为误报。 |
| `0x6F13` | 94397 | 1 | `RfTu\|(Addr Cmd RfCnt)(352768-114364)([0xe507a218, 0x38105888, 459275] [0xe507a220, 0x38238889, 459889] [0xe507a228, 0x3896c08a, 463576] [0xe507a230, 0x389f908e, 463858] [0xe507a238, 0x38a1808f, 463920] [0xe50…` | RF 时序单元写入队列（地址/命令/计数）。 |
| `0x7005` | 89966 | 16 | `RFC\|DbbStateDl CC[0] SwPd[4]: RxReCp=[ 0, 0, 0, 0] MeasReCp=[ 0, 0, 0, 0] RxSlot= 0 MeasSlot= 0 DfeCfg= 1 DlState=0x1d3 SpRate=0 PrsStr=0 MeasRst=0x1ac688 AgcEsti=1 DcEsti=1 IqEsti=0 DcCfg=0 RxDagc=0 SyncDagc…` | 下行数字基带状态与测量结果指针。 |
| `0x700C` | 89898 | 1 | `RFC\|RxDbbCfg: PdIdx=4 Sf=0 TimeOfs=118912 TimeCmp=0 TuTs=475648 EvtEnBitMap=[0xf0000000, 0xf0000000, 0x7], EnBitMap0=[0xf0000000, 0x7], EnBitMap1=[0xf0000000, 0x7], EnBitMap2=[0xf0000000, 0x7], EnBitMap3=[0xf…` | 接收数字基带配置与事件使能位图。 |
| `0x700B` | 88353 | 1 | `RFC\|DbbStateUl CC[0] Tx[1]: TxSlotCnt=4 SpRate=0 TxFltStatus=0x0` | 上行数字基带状态与发射时隙计数。 |
| `0x7028` | 85725 | 1 | `RFC\|Cc [0],[241, 0] Uid 0:CSR,1:RX,2:TX[1]: Time[0--30720] Status[2] Freq[-1]100k` | 载波各 Uid（CSR/RX/TX）的时间窗与状态。 |
| `0x6F0A` | 84719 | 1 | `RfNV-GetRxLnaRffeCw\|AswState=1 CwNum[14 0] TriggerCwNum[8 0] RxPath=[43 47 56 52 0 0 0 0] LnaCfgIdx=[10 24 60 46 0 0 0 0] LnaLvl=[0 0 0 0 0 0 0 0]` | 从 NV 取接收 LNA 的 RFFE 命令字与路径。 |
| `0x6F1D` | 80416 | 1 | `RfRxPwr\|Band[3] Freq[18500000] RfTime[241 1226472] LnaLvl[0 0 0 0] DgaIdx[26 16 27 24] PgaIdx[19 19 19 19] AgcLvl[12 12 12 12] AgcGain[114 104 113 116]` | 接收功率链：LnaLvl/DgaIdx/PgaIdx/AgcLvl 与四通道总增益 AgcGain(dB)。 |
| `0x6F58` | 80123 | 1 | `RficRssiInfo \|FrmRssiPre[3400],FrmRssiPost[3400],AdcRssiPre[3400],AdcRssiPost[3400],AcsFlag[0],BlockFlag[0]` | RFIC RSSI 与阻塞标志：BlockFlag=1 表示检测到强干扰。 |
| `0x7011` | 47499 | 1 | `RFC\|AsyncInfo CC[0]: SwPdIdx=4 Sf.Ts=[9.1] SyncState=0 FsNewFlag=0 FSFixedAGCFlag=0 FastAgcFlag=0 CsrsWorkFlag=1 NotSyncAGCDoneFlag=0 CntFirstDc=0` | 同步/异步状态快照：判断当前处于同步还是异步最快的一行。 |
| `0x700E` | 16100 | 1 | `RFC\|DbbTxReq CC[0]: ReqSfNum=8, TxEn=0, EvtSelect=0 CurFn=881.6` | 请求 DBB 发射（子帧号与使能）。 |
| `0x7029` | 9517 | 1 | `RFC\|Cc [0], Opt [2], Time[241,1,0] Freq[18500000] Bw[14] TimeOffSet[352768]` | 载波级 RF 操作（频点、带宽、时间偏置）。 |
| `0x6F01` | 9369 | 3 | `RfReq\|uid=0 opt=2 time[241 122880] Band[0],freq[3],bw[18500000],AntOrder[e],dlp[64ab2e92],(Trx:scs:dplex)[0],Sence[1],pll[0],anaidx[70 43],port[47 56 52 0 0 0 0 0],bandGp[0],ccPath[0],antPath[0 0 0 0 0 0 0 0]` | RF 请求：场景、频点、带宽、天线顺序与 PLL 选择。 |
| `0x702C` | 9078 | 1 | `RFC\|RxCfg Cc [0]: Opt[2],Ret=178257920,Band=3,Bw=14(100k),Freq=18500000(100Hz),Duplex=1(0:TDD,1:FDD),TxRxInd=[0,0],Mdm=2,IntraCcNum=1,RxAntNum=4,DbbOrder=[1,2,3,4],Scene=0` | 接收通道配置：Band/Bw(100kHz)/Freq(100Hz)/Duplex(0:TDD,1:FDD)/天线数/DBB 顺序。 |
| `0x702D` | 9078 | 1 | `RFC\|RxCfg Cc [0]: Opt[2], RxCcPathId=70,RxAntPathId=[43,47,56,52],RxPort=[5,6,7,8],Pdidx=4,ChCcOrd=[0,1,2,3],Dlp=[0,0],DbbPpi=[0,1,2,3],Dlc=[0,0]` | 接收通道路径：CcPathId/AntPathId/RxPort/电源域索引。 |
| `0x6F08` | 9076 | 1 | `RfNV-GetRxAswRffeCw\|CwNum[14 3] TriggerCwNum[4 1] Band=3 AntNum=4 AswState=0 BandGroupId=8 RxPath=70[43 47 56 52 0 0 0 0] Sw=[95 88 126 119 52 265 247 284 171 222 204 240 181 0 0 0]` | 从 NV 取接收天线开关命令字。 |
| `0x6F56` | 6584 | 1 | `RfNV-GetPubSwRffeCw\|BandGroupId=8 CwNum[0 3] TriggerCwNum[0 1] OptType=4 PubSwIdx=[43 45 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]` | 从 NV 取公共开关命令字。 |
| `0x6F10` | 4755 | 1 | `Drv-Afc_Cfg\|Adj100Ppt=-613, Cmp100Ppt=[39246,1515], InitCw=[50,1268], Rate=[18,19,20], CalcCfgCw=[53,957], CfgTs=223296` | AFC 驱动配置：初始控制字与本次计算值。 |
| `0x6F4E` | 4755 | 1 | `LTE\|AFC\|CalcAfcCw, swPpmx1024[902], pdwCoarseCw[53], pdwFineCw[957], pswCordPpm1024[15], numSeg[13], idx[9]` | 由 ppm 计算 AFC 控制字（粗调/细调）。 |
| `0x7018` | 4755 | 1 | `RFC\|PccAfcCwChgInfo: Connect=0, AfcCw=[0x3bd(pre), 0x3bd(Cur)], CordPpmQ8=[1914(pre), 4004(Cur)], CordPpmAdj1024Q8=0, AfcAdjPpm1024Q8=5120, AfcCwChgFlag=0` | AFC 控制字变化：AfcCwChgFlag=1 表示本次真的改了控制字。 |
| `0x7001` | 4441 | 1 | `RFC\|AlterSpRate CC[0]:PdIdx=4 RxHbfByPass=0x30 TxHbfByPass=0x30 SpRate=[0,0] sdwRet=0` | 采样率切换与滤波旁路。 |
| `0x6F34` | 3437 | 1 | `RfV4ReCfgDlcRx\|IntraCcIdx[0], DlcId[0], Num[10 6], Dlc[0 0], IqNum[0x4f 0x4f] RenSel[0x7 0x3], AntNum[3 3], Afifo[361 361], RdEn[0 0], Sub6gOrV2x[0 0], ClkSel[0x1 0x31], ClkEn[0x3 0x3], DlcData[0x18 0x18], Dl…` | V4 接收 DLC 重配。 |
| `0x6F30` | 3436 | 1 | `RficInfoRx\|RealAnt[4], ReqAnt[4], BitMap[0xf], Scene[0], Payload[240 0 0 0 0]` | 接收 RFIC 天线信息（实际/请求天线数）。 |
| `0x6F31` | 3436 | 1 | `NULL` | 内容为空（NULL）的占位打印，本文件 3,436 条全部相同；无有效负载，分析时可直接过滤。 |
| `0x6F20` | 1172 | 1 | `RfQueryCcInfoRx\|QueryType=1, PccBit=0x1, CcNum=1, Mdm[2 0 0 0 0], Band[3 0 0 0 0], AntNum[255 0 0 0 0], IsmmW[0 0 0 0 0], IntraNum[1 0 0 0 0]` | 查询接收载波资源（band/天线/同频 CC 数）。 |
| `0x6F27` | 933 | 1 | `RfMultiCcResInfoRx\|GroupId[0],Band[3 8 70 4 0],Scene[5 6 7 8 0],AntNum[0 0 0 43 47],CcPathId[56 52 0 0 0]` | 多载波接收资源分配。 |
| `0x7012` | 364 | 1 | `RFC\|SetAgcState Core[3] CC[0]: SwPdIdx=4 AgcWorkState=12, SyncState=1` | 设置 AGC 工作状态与同步状态。 |
| `0x6F05` | 315 | 3 | `Sched-RxTimeOfs\|mdm=2, slot=8, mode=0, rxOfs=[303264-0-0-0]` | 接收时间偏置调度。 |
| `0x6F09` | 295 | 1 | `RfNV-GetTxAswRffeCw\|CwNum[1 0] TriggerCwNum[0 0] Band=3 AntNum=1 AswState=0 optSrsBitMap=0 BandGroupId=8 TxPath=12[13 0] sw=[189 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]` | 从 NV 取发射天线开关命令字。 |
| `0x6F0B` | 295 | 1 | `RfNV-GetTxPaRffeCw\|AswState=0 CwNum[0 5] TriggerCwNum[0 1] TxPath=[13 0] PaCfgIdx=[15 0] PaLvl=[0 0]` | 从 NV 取 PA 命令字与档位。 |
| `0x702A` | 291 | 1 | `RFC\|TxCfg Cc [0]: Opt[6],Ret=196083712,Band=3,Bw=150(100k),Freq=17175000(100Hz),Duplex=1(0:TDD,1:FDD),TxRxInd=[1,1],Srs=[1,2],Mdm=2,IntraCcNum=1,TxAntNum=1,Scene=[0,6]` | 发射通道配置：Band/Bw(100kHz)/上行 Freq(100Hz)/SRS 标志/发射天线数。 |
| `0x702B` | 291 | 1 | `RFC\|TxCfg Cc [0]: Opt[6], TxCcPathId=12,TxAntPathId=[13,0],TxPort=[10,0],AnaIndex=[1,0],Dlp=[4,0],DbbPpi=[2,0],Dlc=[2,15],RficV4CcIdx=[2,15],DbbV4CcIdx=[1,0]` | 发射通道路径：TxCcPathId/AntPathId/TxPort/模拟索引。 |
| `0x7013` | 288 | 1 | `RFC\|TxEnEvt Tx[1]: WrEnHwIdx=2 EvtEn=1` | 发射使能事件写入。 |
| `0x6F54` | 245 | 1 | `RxLaneInfo\|LaneNum[0x2] Datar[0x0] BitMap[0x1] Serdes(Datar[0x3] Clk[0xf] Pre[0x0] Sleep[0x0] Syn[0x0 0x0]) TotalR[524851] CurDaraR[0x0] CurLaneNum[0x2] V4ReCfg[0] LPM_Rx[0]` | 接收 SerDes lane 信息与速率。 |
| `0x6F55` | 245 | 1 | `TxLaneInfo\|LaneNum[0x1] Datar[0x0] BitMap[0x1] Serdes(Datar[0x6] Clk[0x1] Pre[0x28] Sleep[0xd1] Syn[0x10 0x10]) TotalR[0] CurDaraR[0x0] CurLaneNum[0x1] V4ReCfg[0] LPM_Tx[0]` | 发射 SerDes lane 信息与速率。 |
| `0x7006` | 191 | 1 | `RFC\|MeasDbbCfg: PdIdx=4 Sf=4 TimeOfs=190592, EvtIdx=[12,23,25,31] EvtCode=[0x156,0x1156,0x176,0x1176] TuIdx=0 RamEn=0xc3c05cff TimeCmp=0 TuTs=762368 EvtEn=0xc3c05cff Ram=3 SlotOfs=0 MeasSlotIdx=10 MeasMode=0 …` | 测量数字基带配置与事件表。 |
| `0x6F0C` | 148 | 1 | `RfNV-GetTxAptRffeCw\|DbbOrV4=2 AswState=1 CwNum=2 TriggerCwNum=1 TxPath=[13 0] AptCfgIdx=[2 0] PaLvl=[176 0]` | 从 NV 取 APT 命令字与电压索引。 |
| `0x6F36` | 148 | 1 | `RfTxPwr\|Band[3],TxPwr[-13.0,-13.0][-11.25,0.0],AntOrd[1 2],AntPath[13 0],Pa[2 0],Apt[176 0],Apc[13036 0],Lol[-6 5][0 0],AswSta[1],Bit[0x8],RssiCfg[883 122880 0 0 0],RfTime[0 0]` | 发射功率：TxPwr 单位 dBm 可直接读，23.0 dBm 即 Power Class 3 顶格；Pa 为功放档位。 |
| `0x6F50` | 148 | 1 | `TxTempComp_AntPath\|FreqComp[6 0] TempComp[1 0] TempCompIdx[14 0] TempAdc[36909 35014][6822 7171] Band[0]0 Freq[0] AntNum[0] optSrsBitMap[0] CcPath[0] AntPath([0 0]->[0 0]->[0 0]) find[0] SrsInfoTabIdx[0] SubI…` | 发射温度/频率补偿（含温度 ADC 读数）。 |
| `0x6F52` | 148 | 1 | `GetSrsTxAntPath_SubBandAntPath\|Band=3 Freq=17175000 AntNum=1 optSrsBitMap=1 CcPath=12 AntPath([13 0]->[0 0]->[13 0]) find=0 SrsInfoTabIdx=0 SubIdx=[0 0]` | SRS 发射天线路径选择。 |
| `0x6F4D` | 129 | 1 | `LTE\|DCXOTemp, sdTmpDegree[38073], sdInitAfcppm[-1326], sdTemp0[25600], sdCoef0[4418], sdCoef1[-3517101], sdCoef2[0], sdCoef3[429496]` | DCXO 温度与温补曲线系数（长时间挂测的温漂看这里）。 |
| `0x6F14` | 106 | 1 | `Sched-MeasTimeOfs\|mdm=2, slot=8, mode=0, rxOfs=[36992-0-0-0-0-0-0-0]` | 测量时间偏置调度。 |
| `0x7040` | 99 | 1 | `RFC\|RxMicroOffsetCfg CC[0]: SfNum=5, Offset:[4,1,0,0]` | 接收微偏置配置。 |
| `0x6F07` | 83 | 1 | `Sched-TxTimeOfs\|mdm=2, slot=5, TxFixOfs=9` | 发射时间偏置调度。 |
| `0x702E` | 77 | 1 | `RFC_UL\|L1l_Rfc_LTXTxTaConfig >cResult=1,LastHWChannelType=0,CurrentHWChannelType=0,swTxTa=-2,wSmallDelay=0,wSmallDelayFlag=0,wStaRegUpFlag=1,dwCurTxHwIdx=1` | 把 TA 落到发射硬件（swTxTa 微调量）。 |
| `0x703E` | 77 | 1 | `RFC\|RxoffsetAcumulatorInfo CC[0] SwPd[4]: Config TA to LTX! MainAcumulator=5 TQ=[1,2] Acumulator=[3,-3,-6,-6] MainAnt=0` | 各天线接收定时累计偏差，用主天线驱动发射定时。 |
| `0x6F1F` | 35 | 1 | `RfQueryCcInfoTx\|QueryType=1, PccBit=0x1, CcNum=1, Mdm[2 0 0], Band=[3 0 0], AntNum[255 0 0], IntraNum[1 0 0], DbbV4CcIdx([1 0][0 0][0 0]), Bw([150 0][0 0][0 0]), Scs([0 0][0 0][0 0])` | 查询发射载波资源。 |
| `0x6F1E` | 31 | 1 | `RF ERR\|code=-2212,parm=[26651000,7,0,0][0,0,0,0]\|*****\|->_<-\|->_<-\|->_<-\|->_<-\|->_<-\|->_<-\|*****\|` | RF 错误：第一个参数是频率(100Hz)、第二个是 band；每条都必须解释。 |
| `0x6F25` | 28 | 1 | `RfMultiCcResInfoTx\|GroupId[8],Band[0 0 3],Scene[0 0 1],AntNum[0 0 12],CcPathId[0 0 13],txpll[0 0 0]` | 多载波发射资源分配。 |
| `0x6F26` | 28 | 1 | `RfSingleCcResInfoRx\|CcIdx=0, Band=0, GroupId=0, CcPathId=6, AntNum=0, Scene=0, Rxport[10 0 0 0 0 0 1 0], AntPathId[0 0 0 0 0 0 0 0]` | 单载波接收资源与端口。 |
| `0x960C` | 18 | 1 | `INF >RF RxOffset:TxTab=0, TxOffset=0, MeasTab=0, TpuMrtrOffset=35288,RXFrePoint=0,TXFrePoint=0,RfBand=0` | RF 收发与测量表的时间偏置。 |
| `0x9613` | 18 | 1 | `INF >RF RxOffset:TxTab=0, TxOffset=0;MeasTab=0, TpuMrtr=35288,RXFrePoint=0,TXFrePoint=0,RfBand=0` | RF 收发与测量表的时间偏置（另一处打印）。 |
| `0x6F33` | 5 | 1 | `RfV4ReCfgDlcTx\|IntraCcIdx[0], DlcId[2], Num[10 9], Dlc[0 0], DataSp[0x3303 0x3303], DlcSel[0 0], ClkSel[0x1 0x1], ClkEn[0x3 0x3], IqNum[0x2f 0x2f], DlcRenSel[0x1 0x1], Digrfen[569 569], DlcClkSel[0x9 0x9], Dl…` | V4 发射 DLC 重配。 |
| `0x6F2F` | 4 | 1 | `RficInfoTx\|RealAnt[1], ReqAnt[1], BitMap[0x1], DBBppi([0 6][2 0]), chIdx[0 0], dlc[10 0], DbbV4ccIndex[1 0], RficV4ccIndex[4 0]` | 发射 RFIC 天线信息。 |
| `0x7010` | 4 | 1 | `RFC\|TxDbbInit Core[3] CC[0]: Frame.Sf=881.6, TxIdx=1` | 发射 DBB 初始化。 |
| `0x7016` | 3 | 1 | `zPHY_erfc_TaskEntry: receive MC Rest MSG!` | RFC 收到复位消息。 |
| `0x703F` | 3 | 1 | `RFC\|TAoffsetCfg CC[0]: SfNum=3 TAOffset=0` | 发射 TA 偏置配置。 |
| `0x5C30` | 2 | 1 | `DFE\|EsIntDelay! CC[0]: ExpectSlot=0, CurSlot=7, CurSfMrtr=0` | 前端中断延迟告警（ExpectSlot 与 CurSlot 不符）。 |

### A.16 `RXP`（10个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x7300` | 36069 | 1 | `RX\|CRsPwr CC_HW_CH[0010];RSRP00[6;9],RSRP01[8;12],RSRP10[50;72],RSRP11[50;61];RSP0[74;93;44],RSP1[664;730;544];RSSI[578,5209].` | 接收信号质量；先确认单位与内部编码。 |
| `0x7302` | 36069 | 35 | `RX\|SNRInfo CC_HW_CH[0011];RxN0[0x0006;0x0006];SNR[ 40.7, 40.0, 40.0, 40.0],SINR[255;14770899,18831147,11828480,10962756];LowSINRInd: 0;9506581,RxAntAdapt: 0,IrcCfg:0x0001ff06` | 接收信号质量；先确认单位与内部编码。 |
| `0x7381` | 8395 | 1 | `RX\|CFO Info:Cc=0,Flag = 0, AbsCfo = 295, NewCfo = 3406, OldCfo = 9668, Out= 966800, Coeff=2, K=0, Extend=2, AdjCfo=14,AdjInd=1]` | 频偏估计或补偿；与 PSS/SSS/PBCH 稳定性关联。 |
| `0x7400` | 1596 | 16 | `RX\|CIRADJ[0];PeakValue/Pos,WinStart/End:Rx0Tx0[00048ad1,-8,-8,-6],Rx0Tx1[00044aad,-8,0,0]Rx0Tx2[0004457c,-8,0,0],Rx0Tx3[00044aa7,-8,0,0]` | 接收处理、信号质量或 CIR。 |
| `0x7401` | 1596 | 16 | `RX\|CIRADJ[0];PeakValue/Pos,WinStart/End:Rx1Tx0[0002597e,-5,0,0],Rx1Tx1[0002654c,-4,0,0]Rx1Tx2[000431c4,24,0,0],Rx1Tx3[0005273a,22,9,25]` | 接收处理、信号质量或 CIR。 |
| `0x7402` | 1271 | 16 | `RX\|CIRADJ[0];PeakValue/Pos,WinStart/End:Rx2Tx0[00037eec,0,0,0],Rx2Tx1[000483fc,0,0,1]Rx2Tx2[00043f74,0,0,0],Rx2Tx3[000441f7,0,0,0]` | 接收处理、信号质量或 CIR。 |
| `0x7403` | 1271 | 16 | `RX\|CIRADJ[0];PeakValue/Pos,WinStart/End:Rx3Tx0[00037f5f,0,0,0],Rx3Tx1[000482f0,0,0,1]Rx3Tx2[00043fb7,0,0,0],Rx3Tx3[0004417a,0,0,0]` | 接收处理、信号质量或 CIR。 |
| `0x7404` | 600 | 6 | `RX\|CIRADJ0[0];CIRAdjustValue:[ 4, 1, 0, 0],[ 70, 28, 0, 0],Remain:[6,12, 0, 0];PreSyncState=0;Bw:100,ShiftNum:0.` | 接收处理、信号质量或 CIR。 |
| `0x7405` | 600 | 2 | `RX\|CIRADJ1[0];MaxDelay00~33[0,0,0,0,0,0,240,240];ModeType=17;Fgt=[ 51, 51];32KCALIHw=0000.0000;RefSen:0x00000000,FixCoef=1` | 接收处理、信号质量或 CIR。 |
| `0x7382` | 16 | 1 | `CfoFilterCoeffAdapt:UpdateFlag:0,Old/NewTemp:[0,37],K=32` | 频偏估计或补偿；与 PSS/SSS/PBCH 稳定性关联。 |

### A.17 `ULA`（48个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x7727` | 15055 | 1 | `UL\| zPHY_eula_GetHarqProcessId(): GetHarqID in ULA,HarqID=3.` | 上行硬件配置/资源映射/功控。 |
| `0x781C` | 15054 | 1 | `UL\| zPHY_eula_LtxConfigure(): CurCcNum:0,wCurTxHwIdx:[1,1,1],wConfigLutr:0,Subfrm 1, ChanType 2,wULScaleDownDbNum=0,tTxChannelType is 1,eTXPuschType is 0,dwAckLen 0, dwOAckInfo 0,dwScrambleIndex 0, pucchforma…` | PUSCH 选择、映射、发射或功控。 |
| `0x791F` | 15001 | 1 | `UL\| zPHY_eula_CommSrsProc():LastSymbol is reserved for SRS Configuration at Subframe[884 3] Cell[0]!ScheduleLocation=1[0:Pusch 1:other]` | PUSCH 选择、映射、发射或功控。 |
| `0x7822` | 14972 | 1 | `UL\| zPHY_eula_GetPucchHarqAckInfo() > *pbHasHarqAck 1` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7738` | 14912 | 1 | `UL\| zPHY_eula_TPU_INT2_process:Hwidx = 1,wCurCcIdx = 0,eChannelType = 0,Frame = 881,Hsf = 17,Ts = 9957,LtxStatus = 0,SubframeCnt = 1,SymCnt1sf = 12,RegCfgErr = 0,CddValueErr = 0,LtxSymDataDone = 0` | 上行硬件发射配置/功率；配置成功不等于仪表已检出。 |
| `0x781A` | 150 | 1 | `UL\| zPHY_eula_RfcConfigure(): the 0 Ccnum eHWChannelType = 1,tRfcFirstPower.swUlPowerInt=2,tRfcSecondPower.swUlPowerInt=-13,bUlPowerFixFlg=-63,bCloseLoopPCFlg=0!!!` | 上行硬件配置/资源映射/功控。 |
| `0x6C0F` | 99 | 1 | `UL\| zPHY_eulpc_PcmaxCalc()-> :Pcmax:23,wCcNum:0, wMpr:0,wAmpr:0,Pemax:23,Powerclass:23!ModulationMode：0` | 上行硬件配置/资源映射/功控。 |
| `0x6C0B` | 97 | 1 | `UL\| zPHY_eulpc_NoCaAMprDeterm()-> ptPcmaxInputInfo->wAdditionalSpectrumEmission=1,ptPcmaxInputInfo->wEUtraBandNo=3,wAmpr=0` | 上行硬件配置/资源映射/功控。 |
| `0x6C28` | 97 | 1 | `UL\| zPHY_eulpc_NoCaAMprDeterm()->:NS_1,Band:3,BandWidth:75,Rbsize:1,Ampr:0` | 上行硬件配置/资源映射/功控。 |
| `0x7923` | 71 | 1 | `UL\| Set shorten format for PUCCH!PucchType=9,format=2,subframe=0,AckNackSrsSimulTrans=3,PcellSrsCellSpecSfFlg=1` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x6D01` | 68 | 1 | `UL\| zPHY_eulpc_PucchPowCtrl()-> Current has Pucch Trans:Pcmax:23,PucchPow[-18,0],wHNcqiNharqNsr:-18,ePucchFormat=0,wDeltaFPucchIdx=0,swDeltaFPucch:1,swDeltaTxD:0,swPopucch:0,swPathLoss:0,swCloseLoopGi=-117,bP…` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7811` | 68 | 1 | `UL\| zPHY_eula_LtxParas_PucchFormat1Spec():wN1Pucch:4, wNcsAn:6, wDeltaPucchShift:2,dwWDrsNocSlot0/1:[1,0]` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7812` | 68 | 1 | `UL\| zPHY_eula_LtxParasResMappingPucch(): dwPucchRbStartSlot0:2, dwPucchRbStartSlot1:72` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x782E` | 68 | 1 | `UL\| zPHY_eula_FDD_PucchAckParasCalc()-> CellNum 0, dwPucchHarqAckLen 1, dwPucchHarqAckValue 1!!` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7838` | 68 | 1 | `UL\| zPHY_eula_LtxParas_PucchRSeqLength(): PUCCH format: 1, dwMpucch4RB: 1, PucchRLength: 12` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x783B` | 68 | 1 | `UL\| L1l_Dev_eula_SetPucchScale():dwPucchScale1:0x00001400,dwPucchScale2:0x0000029d` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7732` | 59 | 2 | `UL\| Ul Debug[1]:para = [495235007,536837085,495235007,536837085],para = [ 0, 0, 0, 0]` | 上行硬件配置/资源映射/功控。 |
| `0x7823` | 50 | 1 | `UL\| zPHY_eula_GetPucchHarqAckLen() > wAckToSendLen:1 bPCellOnly:1 ,eCurCellConfig -1` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7833` | 32 | 1 | `UL\| L1l_Dev_Ula_SemiStaticReg():wCurCCNum:0,wHwIdx:1` | 上行硬件配置/资源映射/功控。 |
| `0x6C09` | 29 | 1 | `UL\| zPHY_eulpc_SingleCarrierMprDeterm()-> One cell:0, Mpr: 0,UlBandWidth:75,ModulationMode:0,RB:1!` | 上行硬件配置/资源映射/功控。 |
| `0x6D86` | 29 | 1 | `UL\| zPHY_eulpc_PuschPowCalcProc()->: CellNum=0,Pcmax=23,LogMpusch=0,DeltaTF=0,PathLoss=91,Alpha=128,Popusch=-96,Fi=8,PuschPowVal=[3,0],reachMaxPow=3,reachMinPow=0,RsrpFilterVal=0,MPR=0,AMPR=-73,Pemax=-1,PHR1=…` | PUSCH 选择、映射、发射或功控。 |
| `0x780B` | 29 | 1 | `UL\| zPHY_eula_SchdPhichRecInSad(): wCurCCNum =0,wCurTBNum=0,wValidNum=[1,0,0,0],awIPrbLowestIdx=[0,0,0,0],awNDmrs=[0,0,0,0],awIPhich=[0,0,0,0],wABSPhichSubFrmNo=8849,ePuschSendType=[1,0]` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |
| `0x6C20` | 28 | 1 | `UL\| ULPC > zPHY_eulpc_Type1PhrCalc(): wCcNum:0,ePhrCalcScene:0,swType1PhrVal:43!` | 上行硬件配置/资源映射/功控。 |
| `0x7806` | 27 | 1 | `UL\| L1l_Dev_eula_UlDataSendCtrlInfoProcess_PCCAndTm2(): Get PS DATA pDataSrc is 0x50364a, DataSendSize is 15` | 上行硬件配置/资源映射/功控。 |
| `0x6D8B` | 26 | 1 | `UL\| zPHY_eulpc_PuschPowAdjustProc()->: After adjust:CellIndex:0,PuschPow[14,0],PuschPowLinearV=0,PucchPowLinearV=0!!!` | PUSCH 选择、映射、发射或功控。 |
| `0x7846` | 26 | 1 | `UL\|Err > zPHY_eula_FDD_PucchAckProcess(): Tpc:-1` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x6C17` | 25 | 1 | `UL\| zPHY_eulpc_PuschTpcProc() -> wDCI0Tpc:1 wPuschDCI3or3aTpc:-1,abHasTpcflag:1!` | PUSCH 选择、映射、发射或功控。 |
| `0x6C13` | 18 | 1 | `UL\| zPHY_eulpc_PucchTpcProc()-> Receive Dl Tpc or DCI3/3A Tpc at 9846, Tpc Valid at Subframe 0,DlTpcIdx = 3, DCI3or3aTpcIdx = -1, PucchTpcValue = 3` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x6C1E` | 18 | 1 | `UL\| zPHY_eulpc_CloseLoopPowCtrlProc()->:wCcNum:0, bHasPucchFlag:0,Gi:11,bPowReachMax 0,bPowReachMin 0,PucchTpcPositive 3,PucchTpcNegative 0!!` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x780D` | 18 | 1 | `UL\| zPHY_eula_LTXParasCalc(): SR Send at sf=9033! SR send counter=1, RxOffset=0,TxOffset=0,TAOffset=0,sr_index=30` | 上行硬件配置/资源映射/功控。 |
| `0x782D` | 15 | 2 | `UL\| zPHY_eula_FDD_PucchAckParasCalc()-> awn1pucch 0,0,0,0,0,0,acAckValue0,0,0,0,acAckValid 0,0,0,0!!` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x6C1D` | 9 | 1 | `UL\| zPHY_eulpc_CloseLoopPowCtrlProc()->: wCcNum:0,PuschTpc=-1,Fi_Pusch:7,AccumulationEnabled:1,bPuschPowReachMax:0,bPuschPowReachMin:0` | PUSCH 选择、映射、发射或功控。 |
| `0x7717` | 7 | 1 | `UL\| zPHY_eula_ProInitial_Release():<-------- ZPHY_EMC_EULA_RESET_REQ. at sf=2959` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x7902` | 7 | 1 | `UL\| zPHY_eula_UpdataSrsBGParas_Cell[0](): Active Subframe:[1 1 1 1 1 1 1 1 1 1] RatMode=1,SrsSubframeConfig=0` | SRS 配置、周期/触发、冲突处理或发射。 |
| `0x7903` | 7 | 1 | `UL\| zPHY_eula_UpdataSrsBGParas_Cell[0](): RbStart=7 RbEnd=66 Msrsb[60 20 4 4] Nb[1 3 5 1] BandWidth=75 Csrs=2` | SRS 配置、周期/触发、冲突处理或发射。 |
| `0x791B` | 7 | 1 | `UL\| zPHY_eula_UpdataSrsBGParas():SRS Para Calculating in Cell 0 Start!` | SRS 配置、周期/触发、冲突处理或发射。 |
| `0x6D05` | 6 | 1 | `UL\| zPHY_eulpc_NharqParaCalc()->:wNharq=1!` | 上行硬件配置/资源映射/功控。 |
| `0x771E` | 6 | 1 | `UL\| zPHY_eula_ComCfgReqPro()-->:[wNs,awNprs] = [0,92],[1,204],[2,4],[3,179],[4,46],[5,105],[6,241],[7,70],[8,22],[9,156]` | 上行硬件配置/资源映射/功控。 |
| `0x771C` | 4 | 1 | `UL\| zPHY_eula_MACReset(): Rec ZPHY_EMC_EULA_MAC_RESET_REQ,b_RESET_Hardware_RF_Flag is 1,AckAbsSubFrame is -1,AbsCurTime is 4510,AbsTimeSub is 4511` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x7723` | 4 | 1 | `UL\| L1l_Dev_eula_DediRelatedParasNns2ScellCalc():wCurCcIdx=0,wn2Pucch = 0,wPucch2ResNum = 6,wNcsAn = 48,wNns1 = 0,wNns2 = 11` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x6C03` | 3 | 1 | `UL\| zPHY_eulpc_DeltaTcEUtraBandNoDeterm()-> DeltaTcEUtraBandNoDeterm:wEUtraBandNo=3,dwUlEarfcn=17175,wDeltaTc=0!` | 上行硬件配置/资源映射/功控。 |
| `0x6C1B` | 3 | 1 | `UL\| zPHY_eulpc_RarTpcProc() -> RAR Tpc Index = 7, Tpc Value = 8,wPowerRampUp=0,initialFi/Gi=8` | RAR 检测或解析；连接 Msg1 与 Msg3。 |
| `0x7718` | 3 | 1 | `UL\| zPHY_eula_ProInitial_Release():Rec ZPHY_EMC_EULA_REL_REQ,sf=2046.` | 上行硬件配置/资源映射/功控。 |
| `0x7922` | 3 | 1 | `UL\| zPHY_eula_DetermineSrsCellSpecStateInPusch(): SRS CellState is Cancel because The is no resource conflicted!Cell=0,RaType=0 SrsRb[7 66] PuschRB[0 0][0 0]` | PUSCH 选择、映射、发射或功控。 |
| `0x6D02` | 2 | 1 | `UL\| zPHY_eulpc_PucchPowCtrl()->:Current no Pucch Trans:bHasPuschFlg:0,swPopucch:-117,swPathLoss:86,Pcmax=23,swCloseLoopGi=16,PucchPow[-15,0],PowReachMax:0,PowReachMin:0,RsrpFilterV:[-68,-66]!` | PUSCH 选择、映射、发射或功控。 |
| `0x7716` | 1 | 1 | `UL\| zPHY_eula_HandoverReqPro():ULA DEDI ScellINFO: IsScellExist=1, wSCellPci=47, wUlConfigCtrlFlag=1!` | 载波/SCell 配置或活动状态；区分配置、激活与承载。 |
| `0x7729` | 1 | 1 | `UL\|Err > zPHY_eula_HarqNewTransNoData(): ERR >Has no PS data!The Current wCurCCNum = 0, TransType = 1, DataValid = 0, DataSendSize = 6120, DataSrc = 0x50a0c8,bHarqBufferNotEmpty is 1!` | 上行硬件配置/资源映射/功控。 |
| `0x7730` | 1 | 1 | `UL\| L1l_Dev_Ula_TPU_INT2_Process() :Current no use Channal and Srs,return` | SRS 配置、周期/触发、冲突处理或发射。 |

### A.18 `ULS`（32个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x7A0D` | 1099 | 1 | `UL\|Err > L1l_Dev_euls_TPU_INT4_Step1_process() --> TA timer is not running.` | 定时/CIR/TA 状态；用于失步、DTX 与多径定位。 |
| `0x7B63` | 68 | 1 | `UL\| zPHY_euls_DeterminePucchFmt(): eTXPucchType=2,ePucchFormat = 1 wAckToSendLen=1, bPCellOnly=1,wUDai[0]=0,wUDai[1]=0,acVDlDai[0]=0,acNSps[0]=0,bDlAssignMissed=0,wPucchFormatFlag=-1!` | PUCCH/UCI 格式、ACK/SR/CSI 资源或功率。 |
| `0x7A12` | 55 | 1 | `UL\| L1l_Dev_euls_TPU_INT1_Step1_process() --> wUlHarqId = 1` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B07` | 55 | 1 | `UL\| zPHY_euls_HARQEntity()-->RntiType=2,NDIState=2,NewTranCondition=100,UlGtType4CurTrans=2,HarqTransType=1,LastNDI=0,Msg3ID=1,MaxMsg3Tx=4,HarqProceIDInfo.bValid=0,HarqProceIDInfo.cHarqID=0` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B6B` | 53 | 1 | `StdLoglInfoUl:CC[0],AvgPHY.Kbps=77,avgTB.byte=831,trans:[newTx:3,AdReTx:0,NadReTx:0,HqFai:0],aveRbNum*10=720,MCS*10=[80,0],UlBLER*10:0,SR[trig: 0;reTx:0;Fail:0],RACH:[trig: 1; succ: 1; msg12Fail: 0; msg34Fail:…` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7A11` | 29 | 1 | `UL\| L1l_Dev_euls_TPU_INT4_Step2_process() -->TransType=1(0:NONE, 1:NEW, 2:Adapt_RE, 3:NonAdapt_RE),harqid=1,transCnt0=0,schd_pusch=5,schd_phich=8849,wAbsSubFrm=8845.` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |
| `0x7B20` | 29 | 1 | `UL\| zPHY_euls_ReportUlGrantParas() -> harqid:1 TBS:15` | 上行 DCI0/grant；看 CC、HARQ、RB、MCS、TPC、DMRS。 |
| `0x7B62` | 29 | 1 | `UL\| zPHY_euls_DeterminePuschTransType() :wCurCCNum = 0,bSimultaneousPucchPuschFlag = 0, eTXPuschType=1, eTXPuschTypeTB1=0,eTXPucchType =0,eCurCsiType=0,bHasHarqAck=0,ePuschSend=1!` | PUSCH 选择、映射、发射或功控。 |
| `0x7B69` | 29 | 1 | `UL\| zPHY_euls_GetPuschHarqAckInfo()-> *pbHasHarqAck 0` | PUSCH 选择、映射、发射或功控。 |
| `0x7B09` | 28 | 1 | `UL\| zPHY_euls_HARQProcess()-->Rec PHICH:value = 1,wCurCcNum = 0,AbsSF=8849,Harq Id = 1,TotalPuschNumTB = 0,TotalPuschNackTB(adapt+nonadapt) = 268356` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |
| `0x7B21` | 28 | 1 | `UL\| zPHY_euls_ReportUlGrantParas()--> GrantType=2(0:None, 1:Rar_content, 2:Rar_nonConten, 3:SPS, 4:Dyn, 5:Configed), bUlSchedType=2, harqid=1, datasize=15(BYTES), TransTpye=1, DataSrc=5256512` | 上行 DCI0/grant；看 CC、HARQ、RB、MCS、TPC、DMRS。 |
| `0x7B22` | 28 | 1 | `UL\| L1l_Dev_euls_ReportUlGrantToPS_PCell() ->GrantTpye=2,bUlSchedType=2,harqid=1,datasize=15,TransType0=1(0:NONE, 1:NEW, 2:Adapt_RE, 3:NonAdapt_RE),TransType1=255,,DataSrc=5256512,wABSPhichSubFrmNo=8849,trans…` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |
| `0x7B38` | 28 | 1 | `UL\| zPHY_euls_DecodeModuleCodeSchem() ->mcs:0,Qm:8,TBS:2,CurIRV:120,NextRiv:0,PuschSC:1` | PUSCH 选择、映射、发射或功控。 |
| `0x7B0D` | 26 | 1 | `UL\| zPHY_euls_HARQProcess() --> DO not Trans.UlHarqTransCnt = 1` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B1C` | 25 | 1 | `UL\| zPHY_euls_DecodeDci0() -->DCI0 Decoded:wCurCCNum = 0,wDci0Length=27,HarqID=2,CIF=0,HopFlag=0,HopInfo=0,RBstart0=0,Lcrb0=72,MCS=8,NDI=1,TPC=1,DMRS=0,cUlIndex=0,cDAI=0,CSI=0,SrsReq=0,Ratype=0,aldwDci0:0x002…` | 上行 DCI0/grant；看 CC、HARQ、RB、MCS、TPC、DMRS。 |
| `0x7A05` | 15 | 1 | `UL\| zPHY_euls_TaskEntry() --> Recv unknown msgid=11303` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B53` | 10 | 1 | `UL\| zPHY_euls_Release() -> ULS Release!` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7A04` | 7 | 1 | `UL\| zPHY_euls_TaskEntry() --> ULS RESET OK!sf=2697.` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x7B51` | 6 | 1 | `UL\| zPHY_euls_TATimerStop() --> TA stop! wTagId=0.` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B52` | 4 | 1 | `UL\| zPHY_euls_MACReset() --> MAC Reset!.` | 复位/释放流程；看结果位图是否达到全就绪值。 |
| `0x7A02` | 3 | 1 | `UL\| zPHY_euls_TaskEntry() --> ULS SAVE COMMON MSG OK!` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7A07` | 3 | 1 | `UL\| zPHY_euls_TPU_INT1_RARGrantProcess()-->: ABS Msg3 PUSCH SubFrame = 8845.` | PUSCH 选择、映射、发射或功控。 |
| `0x7B11` | 3 | 1 | `UL\| zPHY_euls_InitUlHarqIDInHarqDB() -> HARQ ID Init Ok.wCurCcIdx = 0,harqid:[0,1,2,3,4,5,6,7,0,1]` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B1D` | 3 | 1 | `UL\| zPHY_euls_DecodeRARGrant() -->RARINFO:HopFlag=0,RIV=0,MCS=8,TPC=7,UlDelay=0,CQIFlag=0,sf=8839.` | CSI 计算或反馈打包；警惕无效哨兵与零长度 UCI。 |
| `0x7B42` | 3 | 1 | `UL\| zPHY_euls_AddMsg4DetectStartEvent() -> MSG3 Contension Start SubFrm: 8846 ,Delay SubFrm: 5` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B44` | 3 | 1 | `UL\| UL\| zPHY_euls_AddMsg4DetectStopEvent() -> MSG3 Contension Stop SubFrm: 8897 ,Delay SubFrm: 56` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B47` | 3 | 1 | `UL\| zPHY_euls_AddMsg3LtxDealEvent()--> MSG3 LTX Start Frame: 884 SubFrm:3, Ts = 0.` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B60` | 2 | 1 | `UL\| zPHY_euls_CalcDciCsiReqFlag()-->Aperiodic CQI request bit configured at sf=7, PdcchSf = 6, Kvalue = 4, PuschSf = 0,wPrintIdx = 1!` | PDCCH/DCI 检测或硬件报告；与仪表 DCI 格式、RNTI、CCE 对齐。 |
| `0x7B06` | 1 | 1 | `UL\|Err > zPHY_euls_HARQEntity()-->Invalid, Not toggled, but over the harq length, NDI_STATE=0, AbsPhich=9315, SysFrm=158, SysSubFrm=0, RNTY=1` | PHICH 资源与 ACK/NACK；用后续是否重传验证内部编码。 |
| `0x7B0C` | 1 | 1 | `UL\| zPHY_euls_HARQProcess()--> Non-AdaptReTrans --> HarqId = 4,HarqTrans=1, MAX=5, SF=8644!` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B39` | 1 | 1 | `UL\| zPHY_euls_DecodeRIV_Ratype1() --> invalid DCI0 Grant:RB!!RIV=500,RB=32,RbStart=28.` | 上行调度、HARQ 或 PHICH 状态机。 |
| `0x7B3B` | 1 | 1 | `UL\| zPHY_euls_DecodeRIV_Ratype1() --> rb_start:20,lcrb=20.` | 上行调度、HARQ 或 PHICH 状态机。 |

### A.19 `USER_DEF_VD_12`（8个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x9E91` | 40 | 1 | `L1mcrRecvMsgInfo:dwMsg_id=53569, L1mcrState=0` | 自定义层收到消息（内部消息号）。 |
| `0x9E85` | 22 | 1 | `L1mcrSendPhy/PSMsgInfo:dwMsg_id=53605, L1mcrState=0` | 自定义层发送消息（内部消息号）。 |
| `0x9E84` | 18 | 1 | `RecvLteIntraMsg:LteCurrentTime=2780,bSuspendPsMsg=0,bIsIdleGapReprot =0` | 收到 LTE 内部消息（含当前时间与挂起标志）。 |
| `0x9E86` | 18 | 1 | `IdleServMeasReport[Freq(0)]:FreqNum=1, Narfcn=1275,wCellNum=1` | 空闲态服务频点测量上报（频点数与小区数）。 |
| `0x9E87` | 18 | 1 | `IdleServMeasReport[Freq(0)]:cell[0]:id:30,rsrp:78,rsrq:28,sinr:60,beamIdx:0` | 空闲态各小区测量：rsrp/rsrq 为 36.133 上报域（rsrp-141=dBm）。 |
| `0x9E88` | 18 | 1 | `IdleServCellInfo:Narfcn=1275, wCellPhyId=30,wScs=0,rsrp=78, rsrq=28,sinr=60,ScellSrxlev=78, ScellSqual=28,scellRvalue=78` | 空闲态服务小区信息：含 Srxlev/Squal（36.304 S 准则）与重选 R 值。 |
| `0x9E99` | 1 | 1 | `LteMaterSendMaskMsgStratFreqInfo:Narfcn=1275,wReselPrio=50,wNewMask=1` | 频点掩码与重选优先级下发。 |
| `0x9EAB` | 1 | 1 | `OOSCause:6(0:10s expired 1:oosthresh 2:islandthresh 3:phythresh 4:barscell 5:nomeas 6:10s or bar 7:10s or nomeas 8:10s or nomeas or bar locked cell)` | 脱网原因码（自带完整枚举 0-8），是判定 out of service 的直接证据。 |

### A.20 `USER_DEF_VD_2`（1个ID）

| 消息ID | 次数 | 结构数 | 典型LOG | 是什么/主要作用 |
|---|---:|---:|---|---|
| `0x628E` | 54 | 4 | `DLS\|DCIInfo[0];TM=4,RbNum= 68,RbS= 0,REs= 7888,Tpmi=0x01010001,TbCrcHw=0x00020000,HarqId=3,MCS1= 4,NDI1=0x01012001,RV1=0x24330000,TBS1= 9528,MCS2= 0,NDI2=0x00010001,RV2=0x12B60001,TBS2= 0;Cfg=20,Int=20.CbCRC=…` | `DLS\|DCIInfo` 的详细副本：TM/RB/MCS/TBS/HARQ/层数/调制阶数，解释同下行组。 |

附录合计：**611** 个 `模块+消息ID`。

## 附录 B：全量结构与字段取值域覆盖证明

本附录回答一个问题：**586 万行里，有没有哪一行是没被看过的。**

### B.1 覆盖口径

"看过"的定义分三层，逐层收紧：

1. **行被解析**：每行拆出 12 列，模块、消息 ID、正文都取到。
2. **行落到一个已枚举的结构**：把正文里的数值/十六进制替换成占位符后得到"结构"，每一行都必须落进某个已列出的结构。
3. **结构被解释**：该结构所属的消息 ID 在附录 A 有人工核定的说明；结构内每个字段的取值范围在全文件范围内被统计过。

### B.2 覆盖结果（`tools/build_full_coverage.py` 重新独立跑出）

| 项目 | 结果 |
|---|---:|
| 读入行数 | 5,865,467 |
| 解析失败行数 | **0** |
| 落到已枚举结构的行数 | **5,865,467（100%）** |
| 唯一 `模块+消息ID` | 611 |
| 唯一结构（数值归一化后） | 5,656 |
| 有取值域统计的 `消息ID×字段` 组合 | 2,337 |
| 附录 A 中人工核定的 ID | **611 / 611** |

也就是说：**没有任何一行落在字典之外**。

### B.3 5,656 个结构变体的真实构成

611 个 ID 里，558 个只有一种结构（覆盖 78.2% 的行）。剩下 53 个 ID 有多种结构，但绝大多数不是真的有分支：

| 类别 | ID 数 | 说明 |
|---|---:|---|
| 只因数组元素个数不同 | 43 | 例如 `0x7D83 PssCfg`（44 种）、`0x5A00 Pss`（42 种）、`0x7302 SNRInfo`（35 种）——打印的是变长数组，语义完全相同 |
| 只因归一化假象 | 5 | 见下 |
| **真正有语义分支** | **5** | 见 B.4，全部已逐条确认 |

**归一化假象**是指：掩码规则要求十六进制串同时含字母和数字，因此像 `ebebbbbb`、`aeae` 这种纯字母的十六进制值没被掩掉，被当成了"不同的文字"。受影响的是 `RFC 0x6F45`（4,494 种结构、1,537 个假分支）、`RFC 0x6F46`、`RFC 0x6F01`、`CSRS 0x5809`（`89600Ts` 的末位数字粘在 `Ts` 上）、`DLS 0x628D`。用更严格的掩码重算后，这 5 个 ID 各自只有 1 种语义结构。

**结论：5,656 个结构里，需要人读的语义分支只有 5 处。**

### B.4 5 处真正的语义分支（全部已确认）

| ID | 分支 | 行数 | 含义 |
|---|---|---:|---|
| `MC 0x9E12` | `SI0_WIN_START` / **`SI1_WIN_START`** | 333 / **2** | 两条不同的 SI 消息窗口。SI0 在本文件里承载 SIB1（20 ms 重复）；SI1 只出现 2 次，对应 `WinLen=20`、`CurSiPeriod=16/32` 的真正 SI 消息窗口 |
| `ULA 0x782D` | `acAckValue0` / `acAckValue1` | 9 / 6 | 两个下行码字各自的 PUCCH ACK 数组；双码字调度时才出现 `acAckValue1` |
| `DLS 0x6221` | `Tag0` / `Tag1` | 34 / 130 | 两套 PDSCH 硬件 tag 交替使用（流水线），不是两种结果 |
| `DLS 0x6222` | `Tag0` / `Tag1` | 34 / 130 | 同上，解速率匹配参数按 tag 分组 |
| `DLS 0x6221/0x6222` | `TB1Addr`/`TB2Addr` 为 0 或为地址 | — | 地址为 0 表示该 TB 本次没有数据，是判断单码字/双码字的辅助证据 |

### B.5 字段取值域全量索引

`generated_lte_log_index/field_domain_all.tsv` 给出全部 2,337 个 `消息ID × 字段` 组合在**整个文件**上的：出现次数、最小值、最大值、不同取值个数（上限 60）、以及出现最多的 8 个取值。

这就是回答"这个值正常吗"的依据。用法：

```bash
# 某个字段在全文件的取值分布
awk -F'\t' '$2=="0x5C8E" && $3=="AgcWorkState"' generated_lte_log_index/field_domain_all.tsv

# 只出现过一种取值的字段（说明本次会话没有覆盖到它的其他分支）
awk -F'\t' '$7==1' generated_lte_log_index/field_domain_all.tsv | wc -l

# 某个 ID 的全部结构变体与各自占比
awk -F'\t' '$2=="0x6221"' generated_lte_log_index/template_index_full.tsv
```

**重要提醒**：某字段在本文件里只出现过一种取值，只说明**这次会话没走到它的其他分支**，不代表它只有这一种合法值。字典里凡是标"自带枚举"的，才是固件自己给出的完整值域。

### B.6 配套文件与复现

| 文件 | 内容 |
|---|---|
| `tools/build_lte_log_inventory.py` | 逐行扫描，建 ID 与结构索引 |
| `tools/build_full_coverage.py` | 全量结构 + 全量字段取值域 + 覆盖率证明 |
| `tools/render_lte_message_id_appendix.py` | 渲染附录 A（含 611 条人工核定描述） |
| `generated_lte_log_index/coverage_report.json` | 本附录 B.2 的原始输出 |
| `generated_lte_log_index/template_index_full.tsv` | 5,656 个结构：次数、占比、首末行、时间与典型原文 |
| `generated_lte_log_index/field_domain_all.tsv` | 2,337 个字段的全文件取值域 |
| `generated_lte_log_index/message_id_index.tsv` | 611 个 ID 的首末位置与典型原文 |

```bash
python3 tools/build_lte_log_inventory.py loglte_phich.txt --out-dir generated_lte_log_index
python3 tools/build_full_coverage.py loglte_phich.txt generated_lte_log_index
python3 tools/render_lte_message_id_appendix.py \
    generated_lte_log_index/message_id_index.tsv generated_lte_log_index/message_id_appendix.md
```
