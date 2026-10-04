import type { CollectionEntry } from "astro:content";
import { readdirSync } from "node:fs";

/** Match the recursive Markdown loader without treating an empty site as an error. */
export function hasLocalDynamicContent(
	directory = "src/content/dynamic",
): boolean {
	try {
		return readdirSync(directory, {
			recursive: true,
			withFileTypes: true,
		}).some((entry) => entry.isFile() && entry.name.endsWith(".md"));
	} catch {
		return false;
	}
}

export const sortDynamics = (
	entries: CollectionEntry<"dynamic">[],
): CollectionEntry<"dynamic">[] =>
	entries.sort((a, b) => {
		// 置顶优先，然后按发布时间降序
		if (a.data.pinned && !b.data.pinned) return -1;
		if (!a.data.pinned && b.data.pinned) return 1;
		return b.data.published.getTime() - a.data.published.getTime();
	});

export const dynamicSlug = (id: string): string =>
	id.replace(/\.(md|mdx)$/i, "");

export const dynamicAnchor = (id: string): string =>
	`dynamic-${id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;

export const dynamicPlainText = (entry: CollectionEntry<"dynamic">): string =>
	(entry.body || "")
		.replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
		.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
		.replace(/<[^>]+>/g, " ")
		.replace(/[#>*_`~[\]()-]/g, " ")
		.replace(/\s+/g, " ")
		.trim();

export const dynamicSearchText = (entry: CollectionEntry<"dynamic">): string =>
	[dynamicPlainText(entry), entry.data.location]
		.filter(Boolean)
		.join(" ")
		.toLocaleLowerCase();
