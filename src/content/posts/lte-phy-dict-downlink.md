---
title: "LTE PHY LOG 逐行字典（二）：下行模块逐项字典"
published: 2026-09-03
updated: 2026-09-03
description: "PBCH / DLA / DLS / CSI / RXP 五个下行打印模块共 56 个消息 ID 的逐项字典，含字段取值域与典型原文。"
image: ''
tags: [LTE, PHY, MT8000A, Log分析, 3GPP, 消息ID, zCAT]
category: 协议笔记
draft: false
lang: zh-CN
password: cpetest
---

> **本文是 LTE PHY LOG 逐行字典系列的一篇。**  
> - [第一篇 · 排障手册与协议值字典](/posts/lte-phy-log-line-dictionary/)
> - **第二篇 · 下行模块逐项字典**（本篇）
> - [第三篇 · 上行模块逐项字典](/posts/lte-phy-dict-uplink/)
> - [第四篇 · 搜索、同步、射频与测量逐项字典](/posts/lte-phy-dict-rf-search-meas/)
> - [第五篇 · 611 个消息 ID 全量字典 A.1–A.6](/posts/lte-phy-dict-msgid-a-1/)
> - [第六篇 · 611 个消息 ID 全量字典 A.7–A.20 与覆盖证明](/posts/lte-phy-dict-msgid-a-2/)

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

