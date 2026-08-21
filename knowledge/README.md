# 知识库状态与使用边界

本目录同时包含两类知识：

- 既有 POC 资产：原有 3 类事件的政策、规则和案例；
- `PROJECT_DEMO` 扩展资产：2026-08-20 为另外 6 类事件新增的项目演示知识。

`PROJECT_DEMO` 表示内容根据公开法规、行业自律资料和客服质检常见做法整理，适合用于求职项目的检索、评测和自动判定演示；它不是某家公司的内部制度复刻。

因此，新增六条规则均设置为：

```json
{
  "reviewStatus": "PROJECT_DEMO",
  "automationStatus": "AUTO_ELIGIBLE",
  "penalty": "项目示例扣分"
}
```

后续接入真实知识库时，再替换来源、有效期和扣分标准即可。

公开来源及使用限制见 `sources_manifest.json`，覆盖状态见 `coverage_matrix.json`。
