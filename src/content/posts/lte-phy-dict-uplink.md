---
title: "LTE PHY LOG 逐行字典（三）：上行模块逐项字典"
published: 2026-09-03
updated: 2026-09-03
description: "ULA / ULS / RAPC / LPC 四个上行打印模块共 138 个消息 ID 的逐项字典，含字段取值域与典型原文。"
image: ''
tags: [LTE, PHY, MT8000A, Log分析, 3GPP, 消息ID, zCAT]
category: 协议笔记
draft: false
lang: zh-CN
password: cpetest
---

> **本文是 LTE PHY LOG 逐行字典系列的一篇。**  
> - [第一篇 · 排障手册与协议值字典](/posts/lte-phy-log-line-dictionary/)
> - [第二篇 · 下行模块逐项字典](/posts/lte-phy-dict-downlink/)
> - **第三篇 · 上行模块逐项字典**（本篇）
> - [第四篇 · 搜索、同步、射频与测量逐项字典](/posts/lte-phy-dict-rf-search-meas/)
> - [第五篇 · 611 个消息 ID 全量字典 A.1–A.6](/posts/lte-phy-dict-msgid-a-1/)
> - [第六篇 · 611 个消息 ID 全量字典 A.7–A.20 与覆盖证明](/posts/lte-phy-dict-msgid-a-2/)

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

