import type { AnalyticsConfig } from "../types/analyticsConfig";

export const analyticsConfig: AnalyticsConfig = {
	// Google Analytics ID
	// 新版主题使用 gtag.js，需要 GA4 的 G- 开头 ID（旧 GTM- 格式不兼容）
	// 获取：https://analytics.google.com 创建资源后获得，格式如 G-XXXXXXXXXX
	googleAnalyticsId: "G-2T47VXCDHK",
	// Microsoft Clarity ID
	// 获取：https://clarity.microsoft.com 创建项目后获得，格式如 tx9equrgr6
	// 若不需要可留空禁用
	microsoftClarityId: "xvwfugsoho",
	// Umami 统计配置
	umamiAnalytics: {
		// Umami Website ID
		websiteId: "",
		// Umami JS地址，支持使用自建
		scriptUrl: "https://cloud.umami.is/script.js",
		// Umami 会话回放脚本地址，支持使用自建
		replaysScriptUrl: "https://cloud.umami.is/recorder.js",
		// 是否追踪出站链接
		trackOutboundLinks: true,
		// 是否收集浏览器性能指标
		collectWebVitals: false,
		// 会话回放配置
		replays: {
			// 是否启用会话回放
			enabled: false,
			// 录制会话采样率，范围 0-1，例如 0.15 表示记录 15% 的会话
			sampleRate: 0.15,
			// 隐私遮罩级别："moderate" 会遮罩所有输入框；"strict" 额外遮罩页面全部文本
			maskLevel: "moderate",
			// 单次录制最大时长（毫秒）
			maxDuration: 300000,
			// 需要排除录制的元素 CSS 选择器，例如 ".sensitive-widget"
			blockSelector: "",
		},
	},
	// 51la 统计配置
	la51Analytics: {
		// 51la 统计 ID
		Id: "",
		// 自定义 SDK JS 地址，防止 DNS 污染，留空使用默认地址
		sdkUrl: "",
		// 多个统计 ID 的数据分离标识，留空则使用 Id
		ck: "",
		// 是否开启事件分析功能
		autoTrack: false,
		//  Hash路由模式, 项目使用History API路由, 所以不必开启默认false
		hashMode: false,
		// 是否开启网站录屏功能
		screenRecord: true,
	},
};
