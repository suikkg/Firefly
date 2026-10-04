/** Reading order is editorial, separate from homepage pinning and publication date. */
export const phyDictionarySeries: string[] = [
	"lte-phy-log-line-dictionary",
	"lte-phy-dict-downlink",
	"lte-phy-dict-uplink",
	"lte-phy-dict-rf-search-meas",
	"lte-phy-dict-msgid-a-1",
	"lte-phy-dict-msgid-a-2",
];

export type PostSeries = {
	id: string;
	title: string;
	href: string;
	postIds: string[];
};

/** Add published series here to include them in the shared topic index. */
export const postSeries: PostSeries[] = [
	{
		id: "lte-phy",
		title: "LTE PHY 日志字典",
		href: "/series/lte-phy/",
		postIds: phyDictionarySeries,
	},
];
