---
title: LTE PHY / MT8000A 自动化挂测分析手册
published: 2026-09-03
updated: 2026-09-03
description: 结合 MT8000A 仪表 Trace/Measure、PS-PRIMARY 与 zCAT LTE PHY STD-LOG 的排障手册：15 分钟快速流程，以及 PDCCH/PDSCH/PUCCH/PUSCH/PHICH/SRS/CSI、测量、SCell、重配、切换、性能各场景的判据与端到端定位链路。
image: ''
tags: [LTE, PHY, MT8000A, 自动化测试, Log分析, 排障]
category: 协议笔记
draft: false
lang: zh-CN
password: cpetest
---

> 适用对象：需要结合 MT8000A 仪表 Trace/Measure、PS-PRIMARY 和 zCAT LTE PHY STD-LOG 定位自动化挂测问题的测试与开发人员。  
> 样例日志：`loglte_phich.txt`，约 586 万行、1.3 GB。  
> 版本提醒：文中的 zCAT 打印名来自当前样例固件；不同平台、分支和 LOG 开关可能改名或不打印。判定必须同时依靠仪表、协议栈和 PHY 三条证据链。

快速导航：

- [15 分钟快速排障](#3-15-分钟快速排障流程)
- [小区搜索、SIB、随机接入](#4-小区搜索mibsib-与随机接入)
- [PDCCH](#5-pdcch--dci-分析) / [PDSCH](#6-pdsch-分析) / [PUCCH](#7-pucch-分析) / [PUSCH](#8-pusch-分析) / [PHICH](#9-phich-分析)
- [SRS](#10-srs-分析) / [CSI](#11-csicqipmiri分析) / [测量](#12-测量分析)
- [SCell](#13-sccscell-增加删除与激活) / [重配置](#14-重配置问题) / [切换](#15-切换分析)
- [性能与峰值](#16-性能pdschpusch-峰值与误码)
- [当前日志结论](#18-当前-loglte_phichtxt-实例结论) / [命令速查](#19-大日志检索命令速查) / [报告模板](#20-自动化挂测问题报告模板)
- [规范、Context7 与官方资料](#21-规范context7-与官方资料)

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
