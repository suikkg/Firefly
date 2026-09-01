---
title: LTE 物理层完整学习与 Log 分析指南
published: 2026-09-01
updated: 2026-09-01
description: 系统梳理 TX/RX、TM、Rank、Layer、TB、TBS、Codeword、MCS、HARQ、CSI、PDCCH/PDSCH、PUCCH/PUSCH、随机接入、CA/SCell、测量之间的关系及异常定位流程。
image: ''
tags: [LTE, PHY, MIMO, TM, PDSCH, PUSCH, HARQ, CSI, CA, SCell, Log分析]
category: 协议笔记
draft: false
lang: zh-CN
password: cpetest
---

> 面向 LTE CPE/UE + 仪表测试。`Tpmi`、`TbCrcHw`、`DtchCfg`、`DtchInt` 等为当前测试平台私有 Log 字段，并非 3GPP 标准字段名；换平台/版本必须重新核对位域定义。

## 1. 完整物理层模型

```text
                         ┌──── CSI反馈：CQI/PMI/RI ────┐
UE信道估计 ──────────────┘                            ▼
                                               eNB Scheduler
                                                    │ RB/MCS/Rank/PMI/TBS
                                                    ▼
MAC SDU / MAC CE → MAC PDU → TB0 / TB1
                              │ CRC / Turbo / Rate Matching
                              ▼
                         CW0 / CW1
                              │ Scrambling / Modulation
                              ▼
                         Layer Mapping
                              │ Precoding
                              ▼
                    Antenna Port / TX
                              │ 空中信道
                              ▼
                       UE RX0/RX1/...
                              │ 信道估计/MIMO检测
                              ▼
                     解调/解扰/Turbo译码
                              │
                           TB CRC
                       ┌──────┴──────┐
                      ACK           NACK
```

必须区分三个层级：**TB/CW 是编码数据组织；Layer 是空间数据流；TX/RX 是无线收发资源。** 它们有关联，但不能直接互相等同。

## 2. 核心名词总表

| 名词 | 全称 | 所属 | 作用 | Log重点 |
|---|---|---|---|---|
| TX | Transmit | RF/天线 | 发射无线信号 | 小区TX/端口配置 |
| RX | Receive | UE RF | 接收支路 | RX0~RX3、SNR/RSSI |
| TM | Transmission Mode | RRC/PDSCH | 规定下行MIMO机制 | Dedicated/Reconfiguration |
| Rank | Transmission Rank | MIMO | 实际空间维数 | 实际Layer总数 |
| RI | Rank Indicator | CSI | UE建议Rank | CSI上报 |
| PMI | Precoding Matrix Indicator | CSI | 推荐预编码 | CSI/预编码 |
| CQI | Channel Quality Indicator | CSI | 链路质量建议 | MCS自适应 |
| Layer | Transmission Layer | PHY/MIMO | 空间数据流 | `Tpmi`/实际调度 |
| TB | Transport Block | MAC↔PHY | HARQ数据块 | TBS/CRC/HARQ |
| TBS | Transport Block Size | 调度 | TB比特数 | `TBS1/TBS2` |
| CW | Codeword | PHY | TB编码后的码字流 | 与TB一一对应 |
| MCS | Modulation and Coding Scheme | PHY | 调制/编码效率 | MCS1/MCS2 |
| RV | Redundancy Version | HARQ | 重传冗余版本 | 0/2/3/1 |
| HARQ ID | HARQ Process ID | MAC/PHY | 区分并行HARQ | 新传/重传对齐 |
| DCI | Downlink Control Information | PDCCH | 下发调度参数 | DCI漏检 |
| CE | MAC Control Element | MAC | MAC控制命令 | SCell激活/TA等 |

## 3. TX、RX、Layer、Rank、CW、TB 的关系

### TX 与 RX

TX 表示发射侧资源。当前指导资料中 `dwAntNum=4` 表示小区侧 TX 天线数/相关天线配置。但严格来说：

```text
物理TX天线 ≠ Antenna Port ≠ Layer
```

RX0/RX1/RX2/RX3 是 UE 接收链。4RX UE 可以当前只收 Rank1，也可以在条件和能力允许时收更高 Rank，所以：

```text
4RX ≠ 永远4Layer
2Layer ≠ 2RX
```

RX 数应从 RF/PHY RX-chain 开关及 RX0~RX3 的 SNR/RSSI/RSRP/AGC 判断，不能从 `Tpmi` 推导。

### Layer 与 Rank

对一次实际 PDSCH 空间传输：

```text
Rank1 = 1 Layer
Rank2 = 2 Layer
Rank3 = 3 Layer
Rank4 = 4 Layer
```

RI 是 UE 推荐值；最终实际 Rank/Layer 是 eNB Scheduler 的调度结果。因此看某个子帧实际几层，应看 PDSCH 调度 Log，而不是只看 RI。

### TB、TBS 与 CW

```text
MAC PDU → TB → TB CRC/分段/Turbo编码/Rate Matching → Codeword
```

TB 和 CW 不是同一个概念，但 LTE PDSCH 中数量一一对应：

```text
1 TB → 1 CW
2 TB → 2 CW
```

工程判断通常是：

```text
TBS1 != 0, TBS2 = 0 → 1 TB / 1 CW
TBS1 != 0, TBS2 != 0 → 2 TB / 2 CW
```

### CW 与 Layer

CW 数不等于 Layer 数。多个 Layer 可以承载一个或两个 CW。当前平台指导资料定义：

```text
Tpmi = 0x0p0q00ab
```

`p/q` 分别表示第一个、第二个 TB/CW 对应 Layer 数，`ab` 为 PMI index。

例如：

```text
Tpmi=0x02020422
TBS1 != 0
TBS2 != 0
```

解析为：

```text
TB0/CW0 → 2 Layer
TB1/CW1 → 2 Layer
总Layer/实际Rank = 4
```

## 4. TM 控制什么

TM 是 RRC 配置给 UE 的 PDSCH Transmission Mode，规定 UE 使用哪套 MIMO、参考信号和预编码机制。

| TM | 工程用途 | 重点 |
|---|---|---|
| TM1 | 单天线端口 | 基础单流 |
| TM2 | 发射分集 | 可靠性 |
| TM3 | 开环空间复用 | Rank/Layer |
| TM4 | 闭环空间复用 | RI/PMI/CQI、Rank、CW、Layer |
| TM5 | MU-MIMO | 多用户MIMO |
| TM6 | 单层闭环预编码 | Rank1闭环 |
| TM7~10 | UE-specific RS/高级传输 | 高版本MIMO/波束机制 |

TM4 的实际链路：

```text
UE测量信道
→ RI + PMI + CQI
→ PUCCH/PUSCH上报CSI
→ eNB Scheduler
→ 选择实际Rank/PMI/MCS/RB/TBS
→ 形成1或2 TB
→ 形成1或2 CW
→ Layer Mapping
→ Precoding
→ PDSCH
```

所以不能写成 `TM4=4TX`、`TM4=4RX`、`TM4=4Layer` 或 `TM4=2CW`；这些必须分别确认。

## 5. TBS、MCS、Qm、RE、Layer 与码率

一次 PDSCH 的数据量由资源和链路效率共同决定：

```text
RB/可用RE + MCS + 调制阶数Qm + Layer + TBS表 → TBS
```

调制阶数：

```text
QPSK=2
16QAM=4
64QAM=6
256QAM=8 bit/symbol
```

当前指导资料的 zCAT 估算：

```text
近似码率 = TBS / REs / Layer数 / Qm
```

例如 `195816/13600/2/8≈0.900`。仪表侧指导使用 `LTE_PHY_DATA_REQ/IND` 中 `A/UINT`，并以 0.93 作为当前测试配置检查阈值；这属于测试规则，不是所有 LTE 场景统一的协议硬限制。

高 MCS 通常意味着更高调制/编码效率，也意味着更高 SINR/SNR 要求。因此 BLER 异常要同时看：

```text
SNR + MCS + Qm + TBS + RE + Layer
```

## 6. CSI：CQI、PMI、RI 如何影响下行

```text
参考信号
→ UE信道估计
→ CQI/PMI/RI
→ PUCCH/PUSCH
→ eNB Scheduler
→ MCS/Rank/PMI
→ PDSCH
→ CRC/ACK/NACK
→ 下一轮链路自适应
```

- **CQI**：UE 对可支持下行传输效率的建议，不等于 MCS，也不等于 SNR。
- **RI**：UE 推荐空间 Rank；实际 Rank 仍由调度决定。
- **PMI**：在当前 Rank 下推荐预编码矩阵，不表示天线数量。

周期 CSI 常走 PUCCH，有 PUSCH 时可按配置复用；非周期 CSI 通常由网络触发并通过 PUSCH 上报。

CSI 异常固定查：

```text
RRC CSI配置
→ UE是否生成CSI
→ 应走PUCCH还是PUSCH
→ 上行资源/TA/TX是否正常
→ 仪表是否收到CSI
→ CQI/RI/PMI是否与信道条件一致
→ Scheduler实际Rank/MCS
→ 最终PDSCH BLER
```

## 7. 从搜网到 CA 的完整主流程

```text
频点扫描
→ PSS
→ SSS/PCI
→ PBCH/MIB
→ PDCCH/SI-RNTI DCI
→ PDSCH/SIB
→ Msg1 PRACH
→ Msg2 RAR
→ Msg3 PUSCH
→ Msg4 Contention Resolution
→ RRC Connection
→ NAS Attach/Service
→ PDCCH/PDSCH/PUSCH业务
→ CSI/HARQ闭环
→ RRC Reconfiguration添加SCell
→ Reconfiguration Complete
→ SCell Activation MAC CE
→ SCell Active
→ PCC+SCC调度
```

异常定位原则：从异常点向前找它依赖的上一环，找到**最后一个确定成功点**和**第一个失败点**。

## 8. 小区搜索与同步

常见：

```text
Earcfn=1300
CellID=1
RBNum=100
dwAntNum=4
```

带宽：`1.4M=6RB, 3M=15RB, 5M=25RB, 10M=50RB, 15M=75RB, 20M=100RB`。

PSS 使用 `N_ID_2 = PCI mod 3`。PCI=1 时重点观察 Id1 PSS 峰值。

搜不到小区：

```text
EARFCN
→ RF功率/衰减
→ PSS峰值
→ SSS
→ PCI
→ PBCH/MIB CRC
→ 频偏/时钟/同步
```

## 9. PBCH、MIB、SIB、PDCCH、PDSCH

```text
PSS/SSS成功
→ PBCH/MIB成功
→ PDCCH盲检
→ 找到对应DCI
→ 按DCI指示解PDSCH
→ SIB CRC OK
```

无 SIB 至少分三类：PBCH/MIB 未解对、SIB DCI 漏检、DCI 正常但 PDSCH/SIB CRC FAIL。

当前平台 DCI 位图指导：`bit0 DCI0, bit3 SIB, bit5 Paging, bit7 RA, bit9 DCI3/3a, bit13 Other DCI, bit15 DCI4`。这是平台 Log 位图，不是 3GPP DCI format 的统一编码。

## 10. 随机接入 Msg1~Msg4

```text
Msg1：UE发PRACH Preamble
Msg2：eNB通过PDCCH+PDSCH发送RAR
Msg3：UE按RAR中的UL Grant通过PUSCH发送
Msg4：竞争解决
```

排查：

```text
无Msg1 → PRACH配置/occasion/UE是否进入RA/TX
有Msg1无Msg2 → 仪表是否收到Preamble/是否发RAR/RA DCI/RAR PDSCH CRC
RAR OK无Msg3 → RAR解析/UL Grant/TA/PUSCH TX
Msg3后失败 → Msg4下行/竞争定时器/上行Msg3解码
```

当前指导资料 `wResult`: 1非竞争成功，2竞争成功，3竞争定时器超时，4 Preamble达到最大次数。

## 11. 下行 PDCCH→PDSCH→HARQ

```text
Scheduler生成DCI
→ PDCCH
→ UE Blind Decode
→ 获得RB/MCS/HARQ ID/NDI/RV/MIMO信息
→ UE解PDSCH
→ TB0/TB1 CRC
→ ACK/NACK
→ PUCCH/PUSCH反馈
```

先看 DCI。当前指导资料中 `DtchCfg` 为发送 DCI 总数，`DtchInt` 为实际硬件中断数；两者明显不匹配优先怀疑 DCI 漏检。

再看 TB：`TBS2=0` 通常单 TB/CW，`TBS2!=0` 双 TB/CW。

再看 Layer：读取 `Tpmi` 中 p/q，并只统计实际存在的 TB。

再看 CRC：当前平台格式 `TbCrcHw=0x000m000m`。若当前版本定义 `2=CRC OK, 1=CRC FAIL`，则 `0x00020002` 两 TB 均成功，`0x00010002` 一个成功一个失败。持续如此，按 TB 计 BLER 会接近 50%。

HARQ 必须按：

```text
同CC + 同HarqId + 同TB/CW + NDI/RV
```

对齐。FDD 下行通常 8 个 HARQ process；指导资料典型 RV 为 0→2→3→1。
