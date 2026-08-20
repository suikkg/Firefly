/**
 * 生成 src/constants/icons-data.json
 *
 * Svelte 组件通过 @iconify/svelte/offline 渲染图标，需要在本地内联图标数据。
 * 本脚本扫描所有 .svelte 文件里出现的 `前缀:名称` 图标标识，
 * 从 @iconify-json/* 中抽取对应条目，生成精简的离线图标集。
 *
 * .astro 文件由 astro-icon 在构建期处理，不需要出现在这份数据里。
 *
 * 用法：pnpm icons
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { glob } from "glob";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const OUT = resolve(ROOT, "src/constants/icons-data.json");

/** 允许出现在 Svelte 组件里的图标集前缀 */
const ALLOWED_PREFIXES = ["lucide", "simple-icons"] as const;

interface IconifyIcon {
	body: string;
	width?: number;
	height?: number;
	[key: string]: unknown;
}

interface IconifyCollection {
	prefix: string;
	icons: Record<string, IconifyIcon>;
	aliases?: Record<string, { parent: string; [key: string]: unknown }>;
	width?: number;
	height?: number;
}

type OutputCollection = Pick<IconifyCollection, "prefix" | "icons"> & {
	width: number;
	height: number;
};

function loadCollection(prefix: string): IconifyCollection {
	const path = resolve(ROOT, `node_modules/@iconify-json/${prefix}/icons.json`);
	return JSON.parse(readFileSync(path, "utf8")) as IconifyCollection;
}

/** 顺着 aliases 链解析出真正的图标数据 */
function resolveIcon(
	collection: IconifyCollection,
	name: string,
): IconifyIcon | undefined {
	const seen = new Set<string>();
	let current = name;
	while (!seen.has(current)) {
		seen.add(current);
		const icon = collection.icons[current];
		if (icon) return icon;
		const alias = collection.aliases?.[current];
		if (!alias) return undefined;
		current = alias.parent;
	}
	return undefined;
}

async function main(): Promise<void> {
	const files = await glob("src/**/*.svelte", { cwd: ROOT, absolute: true });

	const pattern = new RegExp(
		`\\b(${ALLOWED_PREFIXES.join("|")}):([a-z0-9]+(?:-[a-z0-9]+)*)`,
		"g",
	);

	/** prefix -> 图标名集合 */
	const used = new Map<string, Set<string>>();
	for (const file of files) {
		const source = readFileSync(file, "utf8");
		for (const match of source.matchAll(pattern)) {
			const [, prefix, name] = match;
			if (!used.has(prefix)) used.set(prefix, new Set());
			used.get(prefix)?.add(name);
		}
	}

	const output: Record<string, OutputCollection> = {};
	const missing: string[] = [];
	let count = 0;

	for (const prefix of [...used.keys()].sort()) {
		const collection = loadCollection(prefix);
		const icons: Record<string, IconifyIcon> = {};
		for (const name of [...(used.get(prefix) ?? [])].sort()) {
			const icon = resolveIcon(collection, name);
			if (!icon) {
				missing.push(`${prefix}:${name}`);
				continue;
			}
			icons[name] = icon;
			count++;
		}
		output[prefix] = {
			prefix,
			icons,
			width: collection.width ?? 24,
			height: collection.height ?? 24,
		};
	}

	if (missing.length > 0) {
		console.error(`✗ 以下图标在图标集中不存在：\n  ${missing.join("\n  ")}`);
		process.exit(1);
	}

	writeFileSync(OUT, `${JSON.stringify(output, null, 2)}\n`);
	const kb = (readFileSync(OUT).byteLength / 1024).toFixed(1);
	console.log(
		`✓ 已写入 ${count} 个图标（${[...used.keys()].join(", ")}）→ src/constants/icons-data.json（${kb} KB）`,
	);
}

main();
