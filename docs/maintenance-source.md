# 维护包来源与适配

维护规则和脚本取自本机 `agent-maintenance`，固定提交 `1ebeb6f3fd564ab72e1625c15dc10f1956838de4`，许可副本见 [MIT License](agent-maintenance-LICENSE)。

采用三个维护技能及共享 references、templates、Notes 规则；未复制维护包已有决策记录或预览资源。采用文档与 Notes 校验、调度器、index 检查、两阶段 hook 安装器，以及文档、调度和 push 环境相关行为测试。

目标适配包括：

- `check_docs.py` 排除 `sources/` 原始证据。
- `staged_checks.py` 加入本仓资料完整性检查。
- Ruff 只启用必要 Python 规则，格式化排除外部快照。
- 本仓依赖锁、检查集合、hooks 和 CI 独立维护。
- hook 的 PATH 包含本仓 `.tools/bin`，可运行项目本地 Gitleaks。
- `sources/** -text` 禁止 Git 换行转换，保持原始证据字节。
- 凭据扫描仅豁免两个指定快照中准确匹配的 Unity 注册表键读取行；该字段是角色数量配置名，不是凭据。

Markdown 检查直接使用 `markdownlint` 0.41.1 API，规则归 `.markdownlint.json`。`scripts/lint-markdown.mjs` 枚举 Git 跟踪与未忽略新文件，并将文件名原样传入检查器；不使用 glob 展开，也不依赖引入 braces 的 Markdownlint CLI 包。保留 KaTeX 0.19.0 覆盖；已无 smol-toml 消费者。2026-10-21 的干净安装与 npm audit 报告零告警，后续升级仍需重新审计。

MIT 许可仅覆盖相应维护包内容，不能替代其他仓库快照的许可。研究来源归 `sources/source-manifest.json` 和主报告参考文献。
