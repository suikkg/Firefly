/**
 * 首页 / 文章页共用的小工具。
 *
 * 单独成模块的原因：Astro 的 getStaticPaths 在独立作用域里执行，
 * 除了 import 进来的东西，拿不到同文件 frontmatter 里定义的任何变量或函数。
 */

/** 由字符串派生出稳定色相，给没有封面的条目一个各自的 identity */
export function hueFromString(value: string): number {
	let hash = 0;
	for (let i = 0; i < value.length; i++) {
		hash = (hash << 5) - hash + value.charCodeAt(i);
		hash |= 0;
	}
	return Math.abs(hash) % 360;
}

/** 只认可以直接使用的绝对地址；相对路径与 "api" 占位一律返回空串 */
export function usableImage(src: string): string {
	if (!src) return "";
	if (src.startsWith("http://") || src.startsWith("https://")) return src;
	if (src.startsWith("/")) return src;
	return "";
}

import { backgroundWallpaper } from "@/config";

/**
 * 取壁纸地址。配置里 src 可能是字符串、数组，或
 * { desktop, mobile, playerUrl } 三种形态。
 */
export function pickWallpaper(kind: "desktop" | "mobile"): string {
	const src = backgroundWallpaper.src;
	if (!src) return "";
	const first = (v: string | string[] | undefined): string =>
		Array.isArray(v) ? (v[0] ?? "") : (v ?? "");
	if (typeof src === "string") return src;
	if (Array.isArray(src)) return first(src);
	return first(src[kind]) || first(src.desktop);
}
