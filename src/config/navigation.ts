/** Shared navigation for the public site. Legacy category/tag URLs remain available. */
export const siteNavigation: { name: string; href: string; match: string[] }[] =
	[
		{
			name: "文章",
			href: "/archive/",
			match: ["/archive/", "/posts/", "/categories/", "/tags/", "/series/"],
		},
		{
			name: "生活",
			href: "/life/",
			match: [
				"/life/",
				"/gallery/",
				"/music/",
				"/anime/",
				"/bangumi/",
				"/dynamic/",
			],
		},
		{
			name: "关于",
			href: "/about/",
			match: ["/about/", "/friends/", "/guestbook/", "/sponsor/"],
		},
	];
