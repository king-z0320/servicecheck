# 本地模型资产

`model_store/` 保存需要长期复用的模型快照，不属于可清理的运行时缓存。

- ModelScope 快照位于 `modelscope/models/`；
- Sentence Transformers 快照位于 `sentence-transformers/`；
- `models.lock.json` 记录当前验收使用的模型 ID、来源、revision、主权重大小和 SHA-256；
- `.runtime/` 可以清理，但不得把 `model_store/` 当作临时目录删除；
- 更新模型时应先完成真实模型回归，再更新 lock 中的 revision 和哈希。

2026-08-19 阶段 2 验收新增了 `iic/emotion2vec_plus_large`。其主权重约
1.95 GB，首次下载耗时约 7 分 24 秒；后续真实 Runner 验收已确认复用本地
快照，不再重新传输主权重。
