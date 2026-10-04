# 正式站配置

- `siteConfig.ts`：身份、页面开关、列表与文章选项。
- `navigation.ts`：正式站的主导航，页头共用。
- `profileConfig.ts`：作者与联系渠道。
- `src/utils/post-series.ts`：真实专题与篇章顺序，`/series/` 自动生成索引。
- `personalConfig.ts`：简短首页介绍与问候；配套类型位于 `src/types/personalConfig.ts`。文章数量与日期仍从内容生成。
- `backgroundWallpaper.ts`：保留的旧壁纸配置；当前个人站使用 `characterConfig.ts` 管理角色图片。
- `commentConfig.ts`：唯一 Waline 评论配置。
- `analyticsConfig.ts`：正式站使用 Google Analytics，localhost 不加载。
- `friendsConfig.ts`、`galleryConfig.ts`、`dynamicConfig.ts`：次级页面的数据与开关。
- `expressiveCodeConfig.ts`、`mermaidConfig.ts`、`plantumlConfig.ts`：正文渲染能力。
- `licenseConfig.ts`：文章许可的默认值，文章字段可覆盖。
- `fontConfig.ts`：保留字体工具能力，当前使用系统字体，不下载额外字体。

已退役的侧栏、模型、播放器、读者调参、旧导航和打赏设置保存于 `tests/fixtures/legacy-config`，不再是应用入口。打赏渠道在 `src/content/spec/about.md` 中维护。

- `characterConfig.ts`：首页与角色图库共用的图片注册表；可扩展角色、游戏、出处与横竖图焦点。
