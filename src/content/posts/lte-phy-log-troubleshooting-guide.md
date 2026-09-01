---
title: LTE 物理层 Log 分析与异常排查指南
published: 2026-09-01
description: 面向 LTE CPE/UE 仪表测试，梳理小区搜索、随机接入、PDCCH/PDSCH、TB/CW/Layer、HARQ、PUCCH/PUSCH、CSI、CA/SCell、测量及上下行异常的 Log 定位方法。
image: ''
tags: [LTE, PHY, PDSCH, PUSCH, HARQ, CSI, CA, SCell, Log分析]
category: 协议笔记
draft: false
lang: zh-CN
slug: lte-phy-log-troubleshooting-guide
---

> 本文以 LTE CPE/UE + 仪表测试为主要场景。文中的 `Tpmi`、`TbCrcHw`、`DtchCfg`、`DtchInt` 等字段属于具体平台/工具的 Log 表达，不是 3GPP 统一字段名。位域含义必须以对应平台版本的 Log 解析定义为准，不能把某个平台的十六进制布局直接套到其他芯片或版本。

## 1. 一张图理解 LTE 数据链路

下行：

```text
eNB MAC
  │  MAC PDU（业务数据 + MAC CE）
  ▼
Transport Block (TB)
  │ TB CRC → Code Block 分段 → Turbo 编码 → Rate Matching
  ▼
Codeword (CW)
  │ Scrambling → Modulation
  ▼
Layer Mapping
  │ Precoding
  ▼
Antenna Port / TX
  │ 无线信道
  ▼
UE RX → 均衡/解调/译码 → TB CRC
  │
  ├─ CRC OK   → ACK
  └─ CRC FAIL → NACK（或无法形成有效反馈）
```

上行方向基本相反：UE MAC 形成 UL-SCH TB，经 PHY 编码、调制后通过 PUSCH 发给 eNB；eNB 解码后通过 PHICH 对相应 PUSCH HARQ 传输给出 ACK/NACK。

控制面与数据面的核心关系：

```text
PDCCH/DCI ──告诉 UE──> 去哪里、用什么参数解 PDSCH
PDSCH     ──承载────> DL-SCH / TB / RRC / MAC CE / 用户数据
PUCCH     ──常承载──> HARQ-ACK、SR、周期 CSI
PUSCH     ──承载────> UL-SCH，也可复用 UCI/CSI
PHICH     ──反馈────> eNB 对 UE PUSCH 的 ACK/NACK
```

---

## 2. 高频名词

### 2.1 TB：Transport Block

TB 是 MAC 交给 PHY 的一块传输数据，是 HARQ 操作的重要数据单位。

- `TBS`：Transport Block Size，TB 大小。
- 下行空间复用时一个调度最多可有两个 TB。
- Log 常见 `TBS1/TBS2`。
- `TBS2 = 0` 通常表示当前调度只有一个 TB；`TBS2 != 0` 表示两个 TB，但最终应结合平台字段定义确认。

### 2.2 CW：Codeword

CW 是 PHY 编码链路中的码字。工程分析时可以按下面的链路理解：

```text
TB → CRC/分段/信道编码/速率匹配 → CW → 调制 → Layer Mapping
```

LTE PDSCH 空间复用中，TB 与 CW 数量是一一对应的，因此工程 Log 中经常通过 TB 数量判断 CW 数量：

```text
1 TB → 1 CW
2 TB → 2 CW
```

但 TB 和 CW 不是同一个协议概念。

### 2.3 Layer：传输层

Layer 是 MIMO 空间复用的数据流层，不等同于物理 TX/RX 天线。

```text
Codeword → Layer Mapping → Precoding → Antenna Ports
```

因此：

- 2 Layer 不等于 UE 有 2 RX。
- 4 Layer 不等于当前一定存在 4 根物理发射天线。
- Layer 数描述当前 PDSCH 空间传输 Rank。

对于本文所用平台 Log，指导资料给出的 `Tpmi` 格式为：

```text
Tpmi = 0x0p0q00ab
             ││  └─ PMI index（平台定义）
             │└──── 第二个 TB/CW 对应 Layer 数
             └───── 第一个 TB/CW 对应 Layer 数
```

例如：

```text
Tpmi = 0x02020422
```

按该平台定义解析为：

```text
TB1/CW1 → 2 Layer
TB2/CW2 → 2 Layer
总 Rank/Layer = 4
```

前提是当前确实存在两个 TB，例如 `TBS2 != 0`。

### 2.4 Rank / RI

Rank 表示当前 MIMO 信道能够支持、并被选择使用的空间层数。RI（Rank Indicator）是 UE CSI 反馈的一部分，用于向 eNB 报告建议 Rank。

```text
RI = 1 → 建议单 Layer
RI = 2 → 建议双 Layer
...
```

注意：UE 上报 RI 是建议/信道状态反馈，最终实际 PDSCH Layer 由 eNB 调度决定。分析某个具体子帧实际用了几层，应优先看该子帧实际 PDSCH/PHY 调度 Log，而不是只看历史 RI。

### 2.5 PMI

PMI（Precoding Matrix Indicator）是 CSI 的一部分，UE 根据下行信道估计向 eNB 推荐预编码矩阵。

PMI 不直接表示“几根天线”。它与 Rank、天线端口配置、TM 等共同决定可用预编码方式。

### 2.6 CQI

CQI（Channel Quality Indicator）表示 UE 对下行信道质量/可支持传输效率的量化反馈。eNB 可据此选择 MCS。

典型关系：

```text
信道变差 → CQI 下降 → eNB 应倾向降低 MCS
信道变好 → CQI 上升 → eNB 可提高 MCS
```

CQI 不是 SNR 的直接等价值，也不是 MCS 命令。

### 2.7 CSI

CSI（Channel State Information）在 LTE 中主要涉及：

- CQI：建议的信道质量/传输效率。
- PMI：建议的预编码矩阵。
- RI：建议的空间 Rank。

根据配置，CSI 可周期性通过 PUCCH 上报，也可在 PUSCH 上承载；非周期 CSI 通常由网络触发并通过 PUSCH 上报。

### 2.8 CE：Control Element

CE 通常指 MAC Control Element，是 MAC 层控制信息。

典型例子：

- SCell Activation/Deactivation MAC CE
- Timing Advance Command MAC CE
- DRX Command MAC CE
- BSR
- PHR

SCell 激活链路可以理解为：

```text
eNB 构造 SCell Activation MAC CE
        ↓
放入 MAC PDU
        ↓
形成 DL-SCH Transport Block
        ↓
PHY 编码为 CW
        ↓
PDSCH 发送
        ↓
UE PDSCH 解码 + TB CRC
        ↓ CRC OK
MAC 解复用并识别 CE
        ↓
执行 SCell Activation
```

所以 `dedicated` 中已经添加 SCell，只说明 **SCell 已配置**，并不等于 **SCell 已激活**。

### 2.9 HARQ / HarqId / RV

HARQ = Hybrid ARQ。

下行典型过程：

```text
eNB 新传 TB
   ↓
UE CRC OK  → ACK → 结束
UE CRC FAIL→ NACK
   ↓
eNB 重传（相同 HARQ process）
   ↓
UE 软合并后再次译码
```

FDD LTE 下行通常有 8 个 HARQ process，因此分析某个 TB 的新传/重传时必须对齐 `HarqId`。

指导资料中的典型 RV 顺序：

```text
新传     RV=0
第一次重传 RV=2
第二次重传 RV=3
第三次重传 RV=1
```

实际是否发生全部重传取决于 ACK/NACK、网络配置和调度。

### 2.10 ACK / NACK / DTX

- ACK：接收端确认对应 HARQ TB 正确。
- NACK：接收端完成接收但译码/CRC 未通过，请求重传。
- DTX：期望存在反馈/传输，但没有检测到有效信号或有效反馈。

因此：

```text
大量 NACK     → 优先检查数据解码质量
大量 DTX      → 优先检查控制信道漏检、时序、上行反馈链路
NACK/DTX 混合 → 两条链路都需要对齐排查
```

---

## 3. TM、CW、Layer、TX/RX 的关系

TM（Transmission Mode）规定 LTE 下行 PDSCH 的传输模式及相关参考信号/预编码行为。

例如 TM4 用于闭环空间复用，可支持多 Layer 传输，并结合 RI/PMI/CQI 进行链路自适应。

但不能使用：

```text
TM4 → 直接判断当前一定 4 Layer
```

正确逻辑是：

```text
TM → 定义允许的传输机制
CSI/RI/PMI → UE 给网络的信道建议
Scheduler → 决定当前子帧实际调度
PDSCH Log → 观察当前实际 TB/CW/Layer/MCS
```

### TX 与 RX 怎么看

`Tpmi` 不能直接确定 UE RX chain 数。

本文指导资料中：

```text
dwAntNum = 4
```

用于表示小区侧配置的 TX 天线数/天线端口相关配置。

UE 实际 RX 路数应查看 RF/PHY RX chain Log，例如 RX0/RX1/RX2/RX3 是否启用及各链路 SNR/RSSI/RSRP。不能从 `2 Layer` 推导为 `2RX`。

---

## 4. LTE 从开机到业务传输的 PHY 排查主线

建议固定按以下顺序排查：

```text
1. 小区搜索/同步
   ↓
2. PBCH/MIB + 系统消息
   ↓
3. 随机接入 Msg1~Msg4
   ↓
4. RRC 建链
   ↓
5. PDCCH/DCI
   ↓
6. PDSCH 下行解码
   ↓
7. PUCCH HARQ/CSI/SR
   ↓
8. PUSCH 上行数据
   ↓
9. RRC Reconfiguration
   ↓
10. CA SCell 配置/激活
   ↓
11. CSI/测量/链路自适应
```

不要在随机接入尚未完成时直接从 SCell、吞吐量或 CSI 结果倒推。

---

# 5. 小区搜索与同步

常见 Log：

```text
Earcfn=1300
CellID=1
RBNum=100
dwAntNum=4
```

含义：

| 字段 | 含义 |
|---|---|
| EARFCN | LTE 频点编号 |
| CellID/PCI | Physical Cell ID |
| RBNum | 下行带宽对应 PRB 数 |
| dwAntNum | 平台记录的小区 TX/天线端口配置 |

常见带宽：

| LTE 带宽 | PRB |
|---|---:|
| 1.4 MHz | 6 |
| 3 MHz | 15 |
| 5 MHz | 25 |
| 10 MHz | 50 |
| 15 MHz | 75 |
| 20 MHz | 100 |

### PSS 排查

LTE PCI：

```text
PCI = 3 × N_ID_1 + N_ID_2
N_ID_2 = PCI mod 3
```

所以 `PCI mod 3` 可用于确定对应 PSS 序列索引。

如果：

- 目标频点能量存在；
- 对应 PSS 峰值仍很低；

优先检查频点、功率、衰减、射频通路、频偏/时钟、仪表小区是否真正 ON AIR。

如果 PSS 正常但搜不到小区，则继续查 SSS、PBCH/MIB 解码和同步状态。

---

# 6. PBCH、MIB、SIB 与 PDCCH

UE 完成 PSS/SSS 同步后，需要解 PBCH 获取 MIB，然后继续获取系统信息。

关键依赖关系：

```text
PSS/SSS
  ↓
PBCH/MIB
  ↓
PDCCH/DCI
  ↓
PDSCH
  ↓
SIB
```

因此“没有 SIB”不一定是 SIB 内容问题，也可能是：

- PBCH/MIB 未正确解码；
- PDCCH 未检出对应 DCI；
- PDSCH CRC FAIL。

指导资料中的 DCI 位图示例：

```text
bit0  : DCI0
bit3  : DCI1A/1C SIB
bit5  : DCI1A/1C Paging
bit7  : DCI1A/1C RA
bit9  : DCI3/3A
bit13 : Other DCI
bit15 : DCI4
```

例如：

```text
0x8    → bit3  → SIB 相关 DCI
0x80   → bit7  → RA 相关 DCI
0x2000 → bit13 → Other DCI
```

这同样属于该平台 Log 位图定义，不能视为 3GPP DCI format 的统一十六进制编码。

---

# 7. 随机接入：Msg1 ~ Msg4

## Msg1：PRACH Preamble

UE 发送随机接入前导。

异常：完全没有 Msg1。

优先检查：

- PRACH 配置是否正确；
- UE 是否已经完成小区驻留；
- 上行频率/功率/射频链路；
- MAC 是否触发 RA。

## Msg2：RAR

网络通过下行发送 Random Access Response。

RAR 中包含 UE 后续 Msg3 所需的重要信息，例如 UL Grant、Timing Advance、Temporary C-RNTI 等。

如果仪表发 Msg2、UE 无 RAR：

```text
先查 RA-RNTI 对应 PDCCH/DCI
        ↓
再查 RAR 所在 PDSCH CRC
```

不要直接判定为 MAC/RRC 问题。

## Msg3

UE 根据 RAR 的 UL Grant 发送 Msg3，走 PUSCH。

异常时检查：

- UL Grant 是否正确收到；
- TA 是否正确应用；
- PUSCH 发射功率；
- 仪表是否正确解 PUSCH；
- UL HARQ 状态。

## Msg4

竞争随机接入中，网络完成 contention resolution。

指导资料中的 `wResult`：

```text
1 = 非竞争接入成功
2 = 竞争接入成功
3 = 竞争定时器超时失败
4 = Preamble 发送达到最大次数
```

---

# 8. 下行核心：PDCCH → DCI → PDSCH

下行问题应拆成两个阶段：

```text
第一阶段：UE 有没有发现“这里有一包数据” → PDCCH/DCI
第二阶段：发现以后能不能把数据解出来 → PDSCH/TB CRC
```

这两个问题不能混在一起。

## 8.1 DCI 漏检

指导资料中的统计：

```text
DtchCfg = 配置/发送的 DCI 数
DtchInt = 实际产生的硬件中断数，包括 CRC OK + CRC FAIL
```

若仪表确实调度 DCI，而 UE `DtchInt` 明显缺失，应优先怀疑：

- PDCCH 解码问题；
- CCE/Aggregation Level；
- RNTI；
- 控制信道 SNR；
- 搜索空间；
- 时频同步；
- DCI format/配置不匹配。

## 8.2 PDSCH 解码

确认 DCI 存在后，再看：

- `TBS1/TBS2`
- `MCS1/MCS2`
- `HarqId`
- `RV`
- `Tpmi`
- `TbCrcHw`
- SNR

### 判断几个 TB/CW

```text
TBS1 != 0, TBS2 = 0 → 通常 1 TB / 1 CW
TBS1 != 0, TBS2 != 0 → 通常 2 TB / 2 CW
```

### 判断 Layer

本文平台按：

```text
Tpmi = 0x0p0q00ab
```

解析每个 TB/CW 的 Layer 数。

例如：

```text
Tpmi = 0x02020422
```

若 `TBS2 != 0`，则按指导资料为：

```text
CW1 = 2 Layer
CW2 = 2 Layer
总计 = 4 Layer
```

### 判断 TB CRC

指导资料中：

```text
TbCrcHw = 0x000m000m
```

两个 `m` 分别表示两个 TB 的硬件 CRC/译码状态。

例如当前平台定义若确认：

```text
2 = CRC OK
1 = CRC FAIL
```

则：

```text
TbCrcHw=0x00020002 → TB1 OK + TB2 OK
TbCrcHw=0x00010002 → 一个 TB FAIL + 一个 TB OK
```

如果连续大量双 TB 调度均固定为 `0x00010002`，并且自动化按每个 TB 分别统计，则理论表现就是约：

```text
1 FAIL / 2 TB = 50% TB BLER
```

这类固定 50% 不像随机弱信号造成的均匀误码，应重点检查是否存在固定 CW/Layer/天线链路、MCS/TBS、预编码或平台译码路径异常。

> 必须先核对当前软件版本 `TbCrcHw` 的状态枚举；如果 `1/2` 的定义不同，上述结论随之改变。

---

# 9. 下行 BLER 高的标准排查流程

```text
Measure 出现 NACK
      ↓
确认 PDCCH/DCI 是否收到
      ↓ YES
确认 PDSCH 对应 TB CRC
      ↓ FAIL
区分 TB1 / TB2 是否固定失败
      ↓
检查 HarqId + RV 重传链
      ↓
检查 SNR / 各 RX chain
      ↓
检查 MCS / TBS / RE / Code Rate
      ↓
检查 Rank / Layer / PMI / TM
      ↓
检查是否只在某 CC / CW / Layer 失败
```

### 9.1 SNR

指导资料中每个载波可能有两行 SNR：

```text
第一行：RX0、RX1
第二行：RX2、RX3
```

并进一步区分 TX→RX 组合，例如：

```text
SNR00 = TX0 → RX0
SNR01 = TX1 → RX0
SNR10 = TX0 → RX1
SNR11 = TX1 → RX1
```

分析重点不是只看平均 SNR，而是检查是否存在某个 RX/TX 空间路径明显异常。

如果一个 CW/Layer 长期失败，而某一路空间信道 SNR 同时显著异常，二者具有较强关联性。

### 9.2 码率

指导资料给出两种方法。

仪表精确检查：

```text
LTE_PHY_DATA_REQ → 找 A、UINT
Code Rate ≈ A / UINT
```

资料给出的经验门限：

```text
A / UINT <= 0.93
```

zCAT 估算：

```text
Code Rate ≈ TBS / REs / Layer数 / 调制阶数
```

例如：

```text
195816 / 13600 / 2 / 8 ≈ 0.900
```

其中调制阶数：

```text
QPSK   → 2 bit/symbol
16QAM  → 4
64QAM  → 6
256QAM → 8
```

估算只能用于快速定位，精确判断应以实际 RE 分配、编码参数及仪表 PHY 数据为准。

---

# 10. 为什么固定 50% BLER 要特别关注 CW/Layer

假设当前持续：

```text
TM4
TBS1 != 0
TBS2 != 0
Tpmi = 0x02020422
TbCrcHw = 0x00010002
```

若平台定义确认 `1=FAIL, 2=OK`，则意味着：

```text
2 TB / 2 CW
CW1 → 2 Layer
CW2 → 2 Layer
其中一个 TB/CW 长期 FAIL
```

自动化按 TB 统计时：

```text
每次调度 2 TB
每次固定失败 1 TB
BLER ≈ 50%
```

排查优先级：

1. 确认到底固定 TB1 还是 TB2 FAIL。
2. 对齐 `MCS1/MCS2`、`TBS1/TBS2`。
3. 对齐 Rank/Layer/PMI。
4. 检查 RX0~RX3 SNR 是否存在固定坏路。
5. 降 Rank，例如从 4 Layer 降到 2 Layer，观察 BLER 是否消失。
6. 降 MCS，观察失败 CW 是否恢复。
7. 改预编码/PMI 或仪表 MIMO 配置进行交叉验证。
8. 检查仪表端口、RF cable、衰减器及 UE RF chain。

这比单纯把问题归为“信号差”更有定位价值。

---

# 11. HARQ 重传怎么跟

必须使用相同 `HarqId` 跟踪同一个 HARQ process。

例如：

```text
SF x     HarqId=2 RV=0 CRC FAIL
SF x+8   HarqId=2 RV=2 CRC FAIL
SF x+16  HarqId=2 RV=3 CRC OK
```

说明：

```text
新传失败
第一次重传失败
第二次重传成功
```

不能仅看到第一次 `CRC FAIL` 就统计为最终业务丢包；HARQ 的意义就是允许 PHY/MAC 通过重传和软合并恢复数据。

但测试仪表的 BLER 指标可能按初传 BLER、每次传输 BLER 或最终 HARQ 失败分别统计，因此必须确认自动化指标口径。

---

# 12. PDSCH CRC 与 HARQ ACK/NACK 为什么可能对不上

理论链路：

```text
PDSCH TB CRC OK   → UE 生成 ACK
PDSCH TB CRC FAIL → UE 生成 NACK
```

但仪表最终看到的反馈还经过：

```text
TB CRC
 ↓
HARQ feedback generation
 ↓
PUCCH/PUSCH UCI
 ↓
UE RF TX
 ↓
无线信道
 ↓
eNB/仪表解调
```

因此可能出现：

```text
UE 本地 CRC OK
但仪表解成 NACK/DTX
```

这时不能再把根因归到 PDSCH，应转查上行 HARQ feedback。

重点检查：

- PUCCH resource；
- ACK/NACK bundling/multiplexing；
- FDD/TDD feedback timing；
- TA；
- UL power；
- PUCCH SNR；
- CA 场景下 HARQ feedback 配置。

---

# 13. 上行：PUSCH 与 PHICH

上行主链路：

```text
UL Grant（DCI 0）
   ↓
UE 形成 UL-SCH TB
   ↓
PUSCH
   ↓
eNB 解码
   ↓
PHICH ACK/NACK
```

指导资料中 PHY Log 的 PHICH `value=1` 表示 ACK，则意味着对应 PUSCH 被仪表/eNB 正确解码。

上行误码高时按顺序检查：

```text
是否收到正确 UL Grant
      ↓
PUSCH 是否按正确 RB/MCS 发射
      ↓
TA 是否正常
      ↓
UE TX Power 是否正常
      ↓
仪表 PUSCH SNR/EVM
      ↓
仪表 UL HARQ ACK/NACK/DTX
      ↓
同 HarqId 的重传情况
```

仪表可进一步检查：

```text
Measure_BTSx_PHYMAC(UL_HARQ).csv
```

观察 ACK/NACK/DTX。

上行码率可按指导资料通过：

```text
LTE_PHY_DATA_IND → A / UINT
```

进行检查。

---

# 14. PUCCH：HARQ、SR、CSI 的关键出口

PUCCH 是很多“看起来像下行问题、实际是上行控制问题”的根源。

常见承载：

- HARQ ACK/NACK
- SR（Scheduling Request）
- CQI/PMI/RI 等 UCI

因此出现以下现象时要查 PUCCH：

```text
UE 本地 PDSCH CRC OK，但仪表看到 DTX
CSI 长期不上报
SR 发出但网络侧检测不到
CA 激活后 HARQ feedback 异常
```

重点参数：

- PUCCH resource index；
- PUCCH format；
- UCI bit 数；
- ACK/NACK 与 CSI 是否碰撞；
- SR/CSI 周期；
- TA；
- UL power control；
- 仪表是否正确检测 PUCCH。

---

# 15. CSI 异常排查

CSI 问题不要只看 CQI 一个字段。

完整链路：

```text
UE 接收参考信号
   ↓
信道估计
   ↓
计算 CQI / PMI / RI
   ↓
按 RRC 配置的周期/触发条件生成 CSI
   ↓
PUCCH 或 PUSCH 上报
   ↓
eNB/仪表接收
   ↓
Scheduler 根据 CSI 调整 MCS/Rank/PMI
```

## 15.1 没有 CSI

按顺序检查：

1. RRC `dedicated` 是否配置 CQI/CSI reporting。
2. 周期、offset、resource 是否有效。
3. 到上报子帧时 UE 是否生成 CSI。
4. CSI 是走 PUCCH 还是 PUSCH。
5. UE 本地已经生成，但仪表是否收到。
6. 是否因 UCI 冲突、DRX、measurement gap、上行失步等被影响。

## 15.2 CQI 明显异常

检查：

- 下行 SNR/SINR；
- CRS 信道估计；
- RX chain 是否缺路；
- 干扰；
- CQI 配置是 wideband 还是 subband；
- 256QAM 能力/配置；
- CQI 与实际 MCS 是否长期严重背离。

## 15.3 RI 异常

例如仪表配置 4x4 MIMO，但 UE 长期只报 RI=1：

优先检查：

- 4 条 RX/空间信道是否真的有效；
- MIMO channel correlation；
- 某 RX chain SNR 是否异常；
- TM 是否支持目标空间复用；
- UE capability；
- RI reporting 配置。

RI 低本身不代表故障；在高相关信道环境下 Rank=1 可能是合理结果。

## 15.4 PMI 异常

如果 PMI 固定、跳变异常或与仪表预期不一致，检查：

- Rank 是否变化；
- CSI codebook 配置；
- 天线端口配置；
- 信道矩阵；
- UE 是否正确完成 channel estimation；
- 仪表 MIMO fading/channel model。

---

# 16. CA：PCC、SCC 与 SCell

常见 Log 中：

```text
ccIdx=0 → PCC/PCell
ccIdx=1 → 第一个 SCC/SCell
ccIdx=2 → 第二个 SCC/SCell
```

具体编号仍以平台定义为准。

CA 建立不是一步完成：

```text
PCell 已连接
   ↓
RRC Connection Reconfiguration
   ↓
SCell dedicated configuration
   ↓
UE Reconfiguration Complete
   ↓
SCell 已配置，但未必激活
   ↓
SCell Activation MAC CE
   ↓
UE MAC 正确收到并处理
   ↓
SCell Active
   ↓
SCell 上开始实际调度
```

---

# 17. SCell 配置成功但不激活

指导资料中的 UE 事件：

```text
LTE_P_ACT_DEACT_SCELL_CTRL_ELEMNT_IND_EV
```

仪表已发送 Activation CE，但 UE 没有该事件时，不应直接判定“UE MAC 不支持”。

完整排查链：

```text
仪表确认发送 SCell Activation MAC CE
             ↓
找到 CE 所在 MAC PDU
             ↓
找到承载该 MAC PDU 的 DL-SCH TB
             ↓
记录 SFN.Subframe + HarqId + TB/CW index
             ↓
UE 查对应 DCI/PDSCH
             ↓
对应 TB CRC 是否 OK？
       ├─ NO → 跟 HARQ 重传
       │        ↓
       │      最终仍 FAIL → PHY 下行问题
       │
       └─ YES → MAC 是否解复用出 Activation CE？
                    ↓
                 无事件 → MAC CE 解析/处理问题
```

## 17.1 为什么 `TbCrcHw=0x00010002` 与 SCell 不激活可能有关

如果当前是两个 TB，并且固定一个 TB FAIL：

```text
TbCrcHw = 0x00010002
```

那么 Activation CE 如果恰好被放入持续失败的那个 TB，UE 就无法拿到该 MAC PDU，自然不会产生 SCell Activation 的 MAC 处理事件。

但是，仅凭 `0x00010002` **不能证明 Activation CE 就在失败 TB**。

必须从仪表 Trace 确定：

```text
Activation CE
  → 哪个 MAC PDU
  → 哪个 TB
  → 哪个 HARQ process
  → 哪个 SFN/Subframe
```

再与 UE Log 对齐。

这是确认因果关系的关键步骤。

## 17.2 ActDeactSCellInfo

指导资料示例：

```text
ActDeactSCellInfo = 6
6 = 0b00000110
```

对应 bit 置位的 SCell 被激活，因此示例表示激活 SCC1、SCC2。

Ci：

```text
Ci = 1 → 激活对应 sCellIndex
Ci = 0 → 去激活对应 sCellIndex
```

---

# 18. 上行 CA

LTE 上行 CA 能力、Band Combination 和网络配置需要同时满足。

指导资料的平台 Log 可在 SCC `dedicated/common` 配置中检查：

```text
UlConfigCtrlFlag = 1
```

用于判断该 SCC 是否配置上行。

如果只有 UL 1CC，通常由 PCell 承载；配置 UL CA 后，再检查对应 SCell 是否真正获得 UL Grant 并发送 PUSCH。

---

# 19. RRC Reconfiguration 排查

仪表：

```text
RRC CONNECTION RECONFIGURATION
```

UE：

```text
LTE_P_DEDICATED_CONFIG_REQ_EV
```

完成：

```text
T_zEurrc_RRCConnectionReconfigurationComplete
```

仪表最终应收到：

```text
RRC CONNECTION RECONFIGURATION COMPLETE
```

定位逻辑：

```text
仪表发 Reconfiguration
      ↓
UE 没 LTE_P_DEDICATED_CONFIG_REQ_EV
      → 优先下行：DCI/PDSCH/RLC/RRC 接收链

UE 收到 dedicated
但没有 ReconfigurationComplete
      → 配置处理/能力/参数问题

UE 已生成 ReconfigurationComplete
仪表没收到
      → 优先上行：UL Grant/PUSCH/RLC/RRC 发送链
```

同时排除：

```text
EL2_EURRC_RADIOLINK_FAIL_IND_EV
```

避免把 RLF 后果误判为单纯重配失败。

---

# 20. 测量：RSRP 与同频/异频

指导资料中的 PHY 搜索关键字：

```text
#INTRA#MEAS → 同频测量
#INTER#MEAS → 异频测量
```

PS-PRIMARY：

```text
LTE_P_INTRA_MEAS_IND_EV
LTE_P_INTER_MEAS_IND_EV
```

可以检查：

- PCell；
- SCell；
- 同频邻区；
- 异频邻区；
- PCI；
- RSRP/RSRQ 等。

测量异常按链路定位：

```text
目标频点是否配置
  ↓
measurement object 是否存在
  ↓
measurement gap 是否需要/是否配置
  ↓
PHY 是否实际执行测量
  ↓
是否搜到目标 PCI
  ↓
L1 measurement
  ↓
L3 filtering
  ↓
RRC event A1/A2/A3/A4/A5/Bx 条件
  ↓
Measurement Report
```

---

# 21. NACK、DTX、NACK/DTX 的快速分流

## 大量 NACK

优先：

```text
PDSCH/PUSCH 实际解码失败
→ SNR
→ MCS/TBS/Code Rate
→ MIMO Layer/PMI
→ RF chain
→ HARQ 重传
```

## 大量 DTX

下行 HARQ feedback 场景优先：

```text
UE 是否收到 DCI/PDSCH
→ UE 是否生成 ACK/NACK
→ PUCCH/PUSCH UCI
→ TA
→ UL Power
→ 仪表是否检测到反馈
```

控制信道场景：

```text
DCI 是否漏检
→ PDCCH SNR
→ RNTI/Search Space/CCE
```

## NACK/DTX

先分别证明：

```text
PDSCH CRC 是否失败？
DCI 是否收到？
UE 是否生成 HARQ feedback？
仪表是否正确收到 feedback？
```

不要仅凭 Measure 一列 `NACK/DTX` 定义根因。

---

# 22. 典型问题：自动化 BLER 50%，SCell 不激活

假设现场：

```text
TM4
SCell dedicated 已下发并处理完成
仪表确认发送 SCell Activation CE
UE 无 LTE_P_ACT_DEACT_SCELL_CTRL_ELEMNT_IND_EV
Tpmi=0x02020422
TbCrcHw 后续持续 0x00010002
自动化 DL BLER≈50%
```

推荐排查：

### Step 1：确认 TB 数

检查同一批子帧：

```text
TBS1 != 0 ?
TBS2 != 0 ?
```

如果两个均有效，确认是 2 TB / 2 CW。

### Step 2：确认固定失败的是哪个 TB

核对当前版本 `TbCrcHw` 位域定义，确定：

```text
TB1 = FAIL / TB2 = OK
```

还是反过来。

### Step 3：确认是否确实为 4 Layer

按当前平台 `Tpmi` 解析，并与实际 Rank/调度信息交叉验证。

### Step 4：检查失败是否与 CW 固定绑定

连续统计：

```text
TB1 BLER
TB2 BLER
```

而不是只看总 BLER。

如果结果接近：

```text
TB1 ≈ 100%
TB2 ≈ 0%
```

则总 BLER 自然约为 50%。

### Step 5：做 MIMO 降阶交叉验证

仪表将 Rank/Layer 降低，例如 4 Layer → 2 Layer。

若问题立即消失，重点转向：

- 4x4 MIMO 空间链路；
- RX chain；
- PMI/precoding；
- 特定 CW/Layer 解调；
- 仪表 MIMO 端口连接。

### Step 6：找到 Activation CE 对应 TB

仪表 Trace：

```text
SCell Activation CE
→ MAC PDU
→ LTE_PHY_DATA_REQ / DL-SCH
→ TB index
→ HarqId
→ SFN.Subframe
```

### Step 7：UE 对齐

在相同：

```text
SFN.Subframe
HarqId
TB index
```

查看 `TbCrcHw`。

### Step 8：跟完重传

如果初传 FAIL，继续跟同一 HarqId 的 RV=2/3/1。

只有最终都无法正确解码，才能形成：

```text
Activation CE 所在 TB 最终解码失败
        ↓
UE MAC 没收到 CE
        ↓
没有 ACT_DEACT_SCELL event
        ↓
SCell 未激活
```

如果最终 CRC OK 但仍无 MAC CE event，则问题从 PHY 转移到 MAC CE 解复用/处理。

---

# 23. 建议固定使用的 Log 对齐键

跨仪表、PHY、MAC、RRC 分析时，不要只按打印时间肉眼对应。

优先记录：

| 层级 | 对齐字段 |
|---|---|
| Radio Frame | SFN + Subframe |
| CA | CC index / PCell / SCell index |
| HARQ | HarqId |
| 重传 | RV / NDI |
| TB | TB index + TBS |
| CW | CW index |
| MIMO | Rank/Layer/PMI |
| 调制编码 | MCS + modulation |
| PHY 结果 | TB CRC |
| 控制信道 | DCI format/RNTI |
| RRC | transaction identifier / message sequence |

最有效的故障证据通常是一条完整时间链，而不是单个字段截图。

---

# 24. 分层定位原则

最终建议把 LTE 问题固定拆成以下边界：

```text
RF
│  功率、频偏、链路、RX/TX chain
▼
PHY Sync
│  PSS/SSS/PBCH
▼
PHY Control
│  PDCCH/DCI/PHICH
▼
PHY Data
│  PDSCH/PUSCH/TB CRC/HARQ
▼
MAC
│  HARQ process / MAC CE / scheduling
▼
RLC
│  segmentation/reassembly/retransmission
▼
RRC
│  configuration/measurement/connection
▼
NAS/IP/Application
```

定位时必须先证明故障发生在哪一层边界：

- **没有 DCI**：不要先分析 PDSCH TB。
- **DCI 有、TB CRC FAIL**：优先 PHY 数据链路。
- **TB CRC OK、没有 MAC CE event**：优先 MAC 解复用/处理。
- **UE 已发 RRC Complete、仪表没收到**：优先上行链路。
- **UE 本地 ACK、仪表 DTX**：优先 PUCCH/PUSCH feedback，而不是重新查 PDSCH。

这套方法可以把“小区没起来、吞吐低、50% BLER、SCell 不激活、CSI 异常、重配失败”等表象统一到同一条可验证的协议链路上。