# 正式站组件

`site/` 提供统一页头、页脚、SEO/主题初始化与文章列表。普通页面使用 `SitePage`；文章页使用同一套页头/页脚与专用阅读排版。`EmbedLayout` 仅用于动态内嵌评论。

`CharacterHero` 提供角色图片、缩略图选择和可收起的展示，`/gallery/characters/` 通过同一注册表扩展角色；首页使用按时间排列的文章流，标签、专题与归档提供次级查找入口。`ReadingTools` 提供字号、链接复制和阅读位置恢复；`ResumeReading` 通过已发布文章的公开清单校验继续阅读入口。阅读偏好仅存于当前浏览器，受保护文章完成解锁后才记录数值位置，不保存正文或密码。个人文案在 `src/config/personalConfig.ts` 中维护。

`features/` 保留加密、代码组、数学样式与按需图片灯箱。`comment/` 仅保留 Waline，接近视口后加载。`pages/` 保留收藏/动态的实际交互。

退役的旧版外壳不再参与构建，`/classic/*` 由兼容页面跳转到正式路径。Markdown 插件的数学和图表能力仍在 `src/plugins` 中维护。
