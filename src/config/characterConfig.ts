export interface CharacterScene {
	id: string;
	title: string;
	character: string;
	game: string;
	desktop: string;
	mobile: string;
	thumbnail: string;
	position: string;
	mobilePosition: string;
	sourceNote: string;
	sourceUrl?: string;
}

const base = "/images/characters/firefly";
const restored =
	"原站收藏，已从 Git 历史恢复；原作者与使用条件尚未追溯，保留原图署名。";
const pvSource =
	"HoYoLAB kaoskey 的流萤壁纸图集，图集标注为官方 PV 画面；使用条件以原发布方为准。";
const scene = (
	id: string,
	title: string,
	desktop: string,
	mobile: string,
	sourceNote = restored,
): CharacterScene => ({
	id,
	title,
	character: "流萤",
	game: "崩坏：星穹铁道",
	desktop: `${base}/${desktop}`,
	mobile: `${base}/${mobile}`,
	thumbnail: `${base}/thumb-${id}.avif`,
	position: "50% 0%",
	mobilePosition: "50% 25%",
	sourceNote,
});

/** Add other characters here; writing topics remain independent of this collection. */
export const characterScenes: CharacterScene[] = [
	scene("sea", "海风", "d1.avif", "m1.avif"),
	scene("blossom", "花间", "d2.avif", "m3.avif"),
	scene("night", "夜色", "d3.avif", "m2.avif"),
	scene("petals", "落樱", "d4.avif", "m5.avif"),
	scene("afternoon", "午后", "d5.avif", "m4.avif"),
	scene("window", "窗边", "d6.avif", "m6.avif"),
	...[
		scene("meteors", "流星与城市", "pv1.webp", "pv1.webp", pvSource),
		scene("rooftop", "天台星空", "pv2.webp", "pv2.webp", pvSource),
		scene("smile", "夕阳微笑", "pv3.webp", "pv3.webp", pvSource),
	].map((item) => ({
		...item,
		sourceUrl: "https://www.hoyolab.com/article/28855308",
		position:
			item.id === "rooftop"
				? "50% 90%"
				: item.id === "smile"
					? "50% 35%"
					: "50% 0%",
		mobilePosition: "50% 50%",
	})),
];

export const defaultCharacterScene: CharacterScene = characterScenes[0];
