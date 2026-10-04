# 本地写作工具

保留编辑、预览和导出源码，从公共博客部署中移出。运行：

```sh
python3 -m http.server 4330 --bind 127.0.0.1 --directory tools/md
```

打开 http://127.0.0.1:4330/ 。GitHub 同步引用的 `/api/github` 不由本博客提供，优先本地导出。
