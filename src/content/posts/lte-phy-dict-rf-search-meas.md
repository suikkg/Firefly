---
title: "LTE PHY LOG 逐行字典（四）：搜索、同步、射频与测量逐项字典"
published: 2026-09-03
updated: 2026-09-03
description: "CSRC / CSRS / CSRM / MULM / MC / CMN / RFC / DFE 及接口与自定义模块共 417 个消息 ID 的逐项字典。"
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
> - [第三篇 · 上行模块逐项字典](/posts/lte-phy-dict-uplink/)
> - **第四篇 · 搜索、同步、射频与测量逐项字典**（本篇）
> - [第五篇 · 611 个消息 ID 全量字典 A.1–A.6](/posts/lte-phy-dict-msgid-a-1/)
> - [第六篇 · 611 个消息 ID 全量字典 A.7–A.20 与覆盖证明](/posts/lte-phy-dict-msgid-a-2/)

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

