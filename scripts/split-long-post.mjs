/**
 * 把一篇超长文章按 h2/h3 边界拆成一个系列
 *
 * 起因：LTE PHY 逐行字典单篇 25 万字，解锁后一次性注入 18000+ 个 DOM 节点、
 * 页面高 17 万 px，iOS Safari 会因内存直接杀掉标签页。拆成多篇是治本办法。
 *
 * 这份字典由 zcat 那边的工具生成，正文会随日志重新生成，所以拆分做成可重跑的。
 *
 * 用法：
 *   node scripts/split-long-post.mjs            # 按 PARTS 拆分并写回
 *   node scripts/split-long-post.mjs --dry-run  # 只打印每篇的体积和锚点改写，不落盘
 *
 * 注意：第一篇会覆盖源文件本身，从而保住原有 URL；其余篇是新增文件。
 * 也就是说拆完之后 SOURCE 已经只剩第一篇了，直接重跑会在找不到 "## 24." 时报错退出
 * （不会写坏东西）。重新生成字典后要再拆一次，先把完整版整篇覆盖回 SOURCE 再跑。
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const POSTS = join(ROOT, "src/content/posts");

/** 源文档（同时也是拆分后的第一篇，URL 保持不变） */
const SOURCE = "lte-phy-log-line-dictionary.md";

/**
 * 拆分计划。startsAt 是该篇的起始标题行前缀（原文里唯一匹配即可）。
 * 第一篇 startsAt 为 null，表示从正文开头起。
 * heading 不为空时，会给该篇补一个 h2 标题——用于从 h3 中间切开的情况。
 */
const PARTS = [
	{
		file: SOURCE,
		startsAt: null,
		title: "LTE PHY LOG 逐行字典（一）：排障手册与协议值字典",
		description:
			"25 分钟内定位一个 LTE PHY 问题：证据优先级、15 分钟快速排障流程、按信道与场景的判据、端到端会话流程，以及 RSRP/RSRQ 映射、EARFCN 换算、寻呼 PF/PO、Ts 三元组与 Q8 定点等协议值字典。",
	},
	{
		file: "lte-phy-dict-downlink.md",
		startsAt: "## 24.",
		title: "LTE PHY LOG 逐行字典（二）：下行模块逐项字典",
		description:
			"PBCH / DLA / DLS / CSI / RXP 五个下行打印模块共 56 个消息 ID 的逐项字典，含字段取值域与典型原文。",
	},
	{
		file: "lte-phy-dict-uplink.md",
		startsAt: "## 25.",
		title: "LTE PHY LOG 逐行字典（三）：上行模块逐项字典",
		description:
			"ULA / ULS / RAPC / LPC 四个上行打印模块共 138 个消息 ID 的逐项字典，含字段取值域与典型原文。",
	},
	{
		file: "lte-phy-dict-rf-search-meas.md",
		startsAt: "## 26.",
		title: "LTE PHY LOG 逐行字典（四）：搜索、同步、射频与测量逐项字典",
		description:
			"CSRC / CSRS / CSRM / MULM / MC / CMN / RFC / DFE 及接口与自定义模块共 417 个消息 ID 的逐项字典。",
	},
	{
		file: "lte-phy-dict-msgid-a-1.md",
		startsAt: "## 附录 A",
		endsBefore: "### A.7",
		title: "LTE PHY LOG 逐行字典（五）：611 个消息 ID 全量字典 A.1–A.6",
		description:
			"全量消息 ID 字典上半：CMN、CSI、CSRC、CSRM、CSRS、DFE 六个模块，共 298 个消息 ID。",
	},
	{
		file: "lte-phy-dict-msgid-a-2.md",
		startsAt: "### A.7",
		heading: "## 附录 A（续）：611 个消息 ID 全量字典 A.7–A.20",
		title:
			"LTE PHY LOG 逐行字典（六）：611 个消息 ID 全量字典 A.7–A.20 与覆盖证明",
		description:
			"全量消息 ID 字典下半：DLA 起至自定义模块共 313 个消息 ID，附全量结构与字段取值域覆盖证明。",
	},
];

const dryRun = process.argv.includes("--dry-run");

/** github-slugger 口径：转小写 → 去掉标点 → 空白转连字符 */
function slugify(text) {
	return text
		.trim()
		.toLowerCase()
		.replace(/[^\p{L}\p{N}_\s-]/gu, "")
		.replace(/\s/g, "-");
}

function splitFrontmatter(raw) {
	const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
	if (!m) throw new Error(`${SOURCE} 没有 frontmatter`);
	return { frontmatter: m[1], body: m[2] };
}

/** 只替换 frontmatter 里的一个标量字段，其余原样保留 */
function setField(frontmatter, key, value) {
	const line = `${key}: ${JSON.stringify(value)}`;
	const re = new RegExp(`^${key}:.*$`, "m");
	return re.test(frontmatter)
		? frontmatter.replace(re, line)
		: `${frontmatter}\n${line}`;
}

function findIndex(body, marker, label) {
	const i = body.indexOf(`\n${marker}`);
	if (i === -1) throw new Error(`找不到${label}标记：${marker}`);
	const next = body.indexOf(`\n${marker}`, i + 1);
	if (next !== -1) throw new Error(`${label}标记不唯一：${marker}`);
	return i + 1;
}

const raw = readFileSync(join(POSTS, SOURCE), "utf8");
const { frontmatter, body } = splitFrontmatter(raw);

// ── 1. 切段 ────────────────────────────────────────────────────
const bounds = PARTS.map((p, i) => {
	const start =
		p.startsAt === null ? 0 : findIndex(body, p.startsAt, `第 ${i + 1} 篇起始`);
	return { ...p, start };
});

for (let i = 0; i < bounds.length; i++) {
	const explicitEnd = bounds[i].endsBefore
		? findIndex(body, bounds[i].endsBefore, `第 ${i + 1} 篇结束`)
		: null;
	const nextStart = i + 1 < bounds.length ? bounds[i + 1].start : body.length;
	bounds[i].end = explicitEnd ?? nextStart;
	if (bounds[i].end <= bounds[i].start) {
		throw new Error(
			`第 ${i + 1} 篇（${bounds[i].file}）切出来是空的，检查 startsAt/endsBefore 顺序`,
		);
	}
	bounds[i].text = body.slice(bounds[i].start, bounds[i].end);
}

// endsBefore 会在两篇之间留空档，这里兜底检查一遍，别把内容切丢了
const covered = bounds.reduce((n, b) => n + b.text.length, 0);
if (covered !== body.length) {
	throw new Error(
		`切分后总长 ${covered} 与原文 ${body.length} 不符，有内容被丢弃`,
	);
}

// ── 2. 建立锚点 → 篇目 的映射 ──────────────────────────────────
const anchorOwner = new Map();
for (const b of bounds) {
	for (const m of b.text.matchAll(/^#{2,4}\s+(.+?)\s*$/gm)) {
		const slug = slugify(m[1]);
		if (!anchorOwner.has(slug)) anchorOwner.set(slug, b.file);
	}
}
if (bounds[bounds.length - 1].heading) {
	// 补出来的标题也要能被链接到
	const h = bounds[bounds.length - 1].heading.replace(/^#+\s+/, "");
	anchorOwner.set(slugify(h), bounds[bounds.length - 1].file);
}

const slugOf = (file) => file.replace(/\.md$/, "");
const unresolved = new Set();

/** 把跨篇的 ](#anchor) 改写成 ](/posts/<slug>/#anchor)，同篇内的保持原样 */
function rewriteAnchors(text, selfFile) {
	return text.replace(/\]\(#([^)]+)\)/g, (whole, anchor) => {
		const decoded = decodeURIComponent(anchor);
		const owner = anchorOwner.get(decoded) ?? anchorOwner.get(anchor);
		if (!owner) {
			unresolved.add(decoded);
			return whole;
		}
		return owner === selfFile ? whole : `](/posts/${slugOf(owner)}/#${anchor})`;
	});
}

// ── 3. 系列导航 ────────────────────────────────────────────────
function navBlock(selfFile) {
	const items = PARTS.map((p, i) => {
		const label = `第${"一二三四五六七八九十"[i] ?? i + 1}篇 · ${p.title.replace(/^.*?：/, "")}`;
		return p.file === selfFile
			? `- **${label}**（本篇）`
			: `- [${label}](/posts/${slugOf(p.file)}/)`;
	});
	return [
		"> **本文是 LTE PHY LOG 逐行字典系列的一篇。**  ",
		...items.map((s) => `> ${s}`),
		"",
	].join("\n");
}

// ── 4. 落盘 ────────────────────────────────────────────────────
const results = [];
for (const b of bounds) {
	let text = rewriteAnchors(b.text, b.file);
	if (b.heading) text = `${b.heading}\n\n${text.replace(/^\n+/, "")}`;

	let fm = setField(frontmatter, "title", b.title);
	fm = setField(fm, "description", b.description);

	const out = `---\n${fm}\n---\n\n${navBlock(b.file)}\n${text.replace(/^\n+/, "")}`;
	results.push({
		file: b.file,
		chars: text.length,
		bytes: Buffer.byteLength(out),
	});

	if (!dryRun) writeFileSync(join(POSTS, b.file), out);
}

if (unresolved.size) {
	console.error("\n以下锚点在任何一篇里都找不到对应标题，链接会失效：");
	for (const a of unresolved) console.error("  #" + a);
	process.exitCode = 1;
}

console.log(dryRun ? "\n[dry-run] 拆分结果：" : "\n已写入：");
for (const r of results) {
	const exists = existsSync(join(POSTS, r.file));
	console.log(
		`  ${r.file.padEnd(34)} ${String(r.chars).padStart(8)} 字符  ${(r.bytes / 1024).toFixed(0).padStart(4)} KB` +
			(r.file === SOURCE
				? "  （覆盖源文件，URL 不变）"
				: exists || !dryRun
					? ""
					: "  （新增）"),
	);
}
console.log(
	`  合计 ${results.reduce((n, r) => n + r.chars, 0)} 字符（原文正文 ${body.length}）`,
);
