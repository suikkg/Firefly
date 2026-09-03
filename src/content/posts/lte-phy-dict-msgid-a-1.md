---
title: "LTE PHY LOG 逐行字典（五）：611 个消息 ID 全量字典 A.1–A.6"
published: 2026-09-03
updated: 2026-09-03
description: "全量消息 ID 字典上半：CMN、CSI、CSRC、CSRM、CSRS、DFE 六个模块，共 298 个消息 ID。"
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
> - **第五篇 · 611 个消息 ID 全量字典 A.1–A.6**（本篇）
> - [第六篇 · 611 个消息 ID 全量字典 A.7–A.20 与覆盖证明](/posts/lte-phy-dict-msgid-a-2/)

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

