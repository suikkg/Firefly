// Verify production artifacts: public routes, encryption boundary, and link integrity.
import assert from "node:assert/strict";
import { createDecipheriv, pbkdf2Sync } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";

const root = path.resolve("dist");
async function walk(dir) {
	const entries = await fs.readdir(dir, { withFileTypes: true });
	return (
		await Promise.all(
			entries.map((entry) =>
				entry.isDirectory()
					? walk(path.join(dir, entry.name))
					: [path.join(dir, entry.name)],
			),
		)
	).flat();
}
const files = await walk(root);
const htmlFiles = files.filter((file) => file.endsWith(".html"));
let indexed = 0;
let encrypted = 0;
const issues = [];
for (const file of htmlFiles) {
	const html = await fs.readFile(file, "utf8");
	const route = `/${path
		.relative(root, file)
		.replaceAll(path.sep, "/")
		.replace(/index\.html$/, "")}`;
	if (html.includes("data-pagefind-body")) indexed++;
	if (
		!route.startsWith("/classic/") &&
		!route.startsWith("/muscle/") &&
		!route.startsWith("/sponsor/") &&
		!route.startsWith("/dynamic/comments/")
	) {
		assert.equal(
			(html.match(/<h1\b/g) || []).length,
			1,
			`Expected one page heading at ${route}`,
		);
	}
	for (const [, raw] of html.matchAll(/(?:href|src|content)="([^"\n]+)"/g)) {
		if (!raw.startsWith("/") || raw.startsWith("//")) continue;
		const target = new URL(raw.replaceAll("&amp;", "&"), "https://local.test");
		const name = decodeURIComponent(target.pathname);
		const candidate = path.join(root, name);
		try {
			const stat = await fs.stat(candidate);
			if (stat.isDirectory())
				await fs.access(path.join(candidate, "index.html"));
		} catch {
			issues.push(`${route}: ${name}`);
		}
	}
}
assert.deepEqual(issues, [], "Broken production links/assets");
const rssOutput = await fs.readFile(path.join(root, "rss.xml"), "utf8");
const dynamicOutput = await fs.readFile(
	path.join(root, "api/dynamic.json"),
	"utf8",
);
const posts = (await walk(path.resolve("src/content/posts"))).filter((file) =>
	/\.(md|mdx)$/.test(file),
);
let expectedIndexed = 0;
let expectedEncrypted = 0;
for (const file of posts) {
	const { data, content } = matter(await fs.readFile(file, "utf8"));
	if (data.draft) continue;
	expectedIndexed++;
	if (!data.password) continue;
	expectedEncrypted++;
	const slug = path
		.relative("src/content/posts", file)
		.replaceAll(path.sep, "/")
		.replace(/\.(md|mdx)$/, "")
		.toLowerCase();
	const html = await fs.readFile(
		path.join(root, "posts", slug, "index.html"),
		"utf8",
	);
	assert.match(
		html,
		/data-pagefind-ignore="all"/,
		`Encrypted body excluded: ${slug}`,
	);
	const passwordInput = html.match(
		/<input\b[^>]*\bid="password-input"[^>]*>/,
	)?.[0];
	assert.ok(
		passwordInput && !/\bname=/.test(passwordInput),
		`Password must not enter native form submission: ${slug}`,
	);
	assert.match(
		html,
		/<button\b[^>]*type="submit"[^>]*disabled/,
		`Unlock stays disabled until local handler is ready: ${slug}`,
	);
	const match = html.match(/data-encrypted="([A-Za-z0-9+/=]+)"/);
	assert.ok(match, `Encrypted payload missing: ${slug}`);
	const bytes = Buffer.from(match[1], "base64");
	const key = pbkdf2Sync(
		String(data.password),
		bytes.subarray(0, 16),
		100000,
		32,
		"sha256",
	);
	const decipher = createDecipheriv("aes-256-gcm", key, bytes.subarray(16, 28));
	decipher.setAuthTag(bytes.subarray(28, 44));
	const decrypted = Buffer.concat([
		decipher.update(bytes.subarray(44)),
		decipher.final(),
	]).toString("utf8");
	assert.ok(decrypted.length > 100, `Empty decrypted article: ${slug}`);
	const protectedHeading = content
		.split("\n")
		.find((line) => /^##\s/.test(line) && line.length > 18)
		?.replace(/^##\s+/, "");
	if (protectedHeading) {
		assert.ok(
			!html.includes(protectedHeading),
			`Protected heading leaked: ${slug}`,
		);
		assert.ok(
			!rssOutput.includes(protectedHeading) &&
				!dynamicOutput.includes(protectedHeading),
			`Protected heading leaked through feed/API: ${slug}`,
		);
	}
	encrypted++;
}
assert.equal(indexed, expectedIndexed, "Only published notes may be indexed");
assert.equal(
	encrypted,
	expectedEncrypted,
	"All protected notes must remain encrypted",
);
assert.ok(
	!files.some((file) => file.startsWith(path.join(root, "md") + path.sep)),
	"Local editor must not be publicly deployed",
);
assert.ok(
	!files.some((file) =>
		/(?:Live2D|l2d-widget|Swup|MusicPlayer)/.test(path.basename(file)),
	),
	"Legacy bundles must not be emitted",
);
const sitemap = (
	await Promise.all(
		files
			.filter((file) => /sitemap-\d+\.xml$/.test(file))
			.map((file) => fs.readFile(file, "utf8")),
	)
).join("");
assert.ok(
	!/classic\/|dynamic\/comments\/|encrypted-test\//.test(sitemap),
	"Non-public routes in sitemap",
);
console.log(
	`Verified ${htmlFiles.length} HTML pages, ${indexed} indexed notes, ${encrypted} encrypted notes; no broken internal links/assets.`,
);
