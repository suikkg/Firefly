---
title: "LTE PHY LOG 逐行字典（六）：611 个消息 ID 全量字典 A.7–A.20 与覆盖证明"
published: 2026-09-03
updated: 2026-09-03
description: "全量消息 ID 字典下半：DLA 起至自定义模块共 313 个消息 ID，附全量结构与字段取值域覆盖证明。"
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
> - [第四篇 · 搜索、同步、射频与测量逐项字典](/posts/lte-phy-dict-rf-search-meas/)
> - [第五篇 · 611 个消息 ID 全量字典 A.1–A.6](/posts/lte-phy-dict-msgid-a-1/)
> - **第六篇 · 611 个消息 ID 全量字典 A.7–A.20 与覆盖证明**（本篇）

## 附录 A（续）：611 个消息 ID 全量字典 A.7–A.20

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
