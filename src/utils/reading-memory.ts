/** Local reading preferences contain public metadata and position only. */
export interface ReadingPost {
	id: string;
	title: string;
}

export interface ReadingMemory {
	version: 1;
	path: string;
	title: string;
	progress: number;
	anchor: string | null;
	updatedAt: number;
}

export type ReadingSize = "standard" | "comfortable";

export const READING_MEMORY_KEY: string = "kk:reading-memory:v1";
export const READING_SIZE_KEY: string = "kk:reading-size:v1";

function hasControlCharacters(value: string): boolean {
	for (const character of value) {
		const code = character.charCodeAt(0);
		if (code <= 31 || code === 127) return true;
	}
	return false;
}

export function readingPostPath(id: string): string | null {
	if (
		!id ||
		id.length > 400 ||
		/[\\%?#:\s]/u.test(id) ||
		hasControlCharacters(id) ||
		id.split("/").some((part) => !part || part === "." || part === "..")
	) {
		return null;
	}
	return `/posts/${id}/`;
}

function localStore(): Storage | null {
	try {
		return typeof window === "undefined" ? null : window.localStorage;
	} catch {
		return null;
	}
}

function validAnchor(anchor: unknown): anchor is string | null {
	return (
		anchor === null ||
		(typeof anchor === "string" &&
			anchor.length > 0 &&
			anchor.length <= 400 &&
			!hasControlCharacters(anchor))
	);
}

/** Match the path to the current published catalog; never trust stored titles. */
export function readReadingMemory(posts: ReadingPost[]): ReadingMemory | null {
	try {
		const raw = localStore()?.getItem(READING_MEMORY_KEY);
		if (!raw || raw.length > 3000) return null;
		const value: unknown = JSON.parse(raw);
		if (!value || typeof value !== "object") return null;
		const record = value as Record<string, unknown>;
		if (
			record.version !== 1 ||
			typeof record.path !== "string" ||
			typeof record.progress !== "number" ||
			!Number.isFinite(record.progress) ||
			record.progress < 0 ||
			record.progress > 1 ||
			typeof record.updatedAt !== "number" ||
			!Number.isFinite(record.updatedAt) ||
			record.updatedAt <= 0 ||
			record.updatedAt > Date.now() + 300000 ||
			!validAnchor(record.anchor)
		) {
			return null;
		}
		const post = posts.find((item) => readingPostPath(item.id) === record.path);
		if (!post || typeof post.title !== "string" || !post.title.trim())
			return null;
		return {
			version: 1,
			path: record.path,
			title: post.title,
			progress: record.progress,
			anchor: record.anchor,
			updatedAt: record.updatedAt,
		};
	} catch {
		return null;
	}
}

export function saveReadingMemory(
	post: ReadingPost,
	progress: number,
	anchor: string | null,
): boolean {
	const path = readingPostPath(post.id);
	if (
		!path ||
		!post.title.trim() ||
		!Number.isFinite(progress) ||
		!validAnchor(anchor)
	) {
		return false;
	}
	try {
		const storage = localStore();
		if (!storage) return false;
		const record: ReadingMemory = {
			version: 1,
			path,
			title: post.title,
			progress: Math.min(1, Math.max(0, progress)),
			anchor,
			updatedAt: Date.now(),
		};
		storage.setItem(READING_MEMORY_KEY, JSON.stringify(record));
		return true;
	} catch {
		return false;
	}
}

export function readReadingSize(): ReadingSize {
	try {
		return localStore()?.getItem(READING_SIZE_KEY) === "comfortable"
			? "comfortable"
			: "standard";
	} catch {
		return "standard";
	}
}

export function saveReadingSize(size: ReadingSize): boolean {
	if (size !== "standard" && size !== "comfortable") return false;
	try {
		const storage = localStore();
		if (!storage) return false;
		storage.setItem(READING_SIZE_KEY, size);
		return true;
	} catch {
		return false;
	}
}
