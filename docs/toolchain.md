# 开发工具链

使用 Python 3.10+、Node.js 22+ 和 Git。依赖属于本仓，不依赖维护包的运行目录。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
npm ci
python scripts/setup.py
python scripts/run_checks.py --config scripts/checks.json --mode ci
```

setup 安装 pre-commit／pre-push hook 和被忽略的 `.claude/skills` 软链。已有外来 hooks 或 hooksPath 时拒绝覆盖。不要再次运行普通 `pre-commit install`。

日常格式化与检查：

```bash
npm run format
ruff format .
python scripts/run_checks.py --config scripts/checks.json --mode ci
```

Prettier 负责自有 Markdown、JSON 和 YAML；Ruff 负责自有 Python；Markdownlint、技能结构、本地文档链接、Notes 和资料哈希检查共同运行。`sources/` 是逐字保存的外部证据，退出所有格式化与自有文档检查，但仍受 `check_research.py` 哈希校验。

Markdownlint 通过 `scripts/lint-markdown.mjs` 直接调用同版本 API，配置为 `.markdownlint.json`。不传参数时检查 Git 跟踪与未忽略的新 Markdown；也可传入具体文件名，不接受 glob 表达式。源码快照、冻结归档、依赖目录和软链退出扫描；空集合、缺失文件和违规正文会返回非零。`npm run test:markdown` 验证这些边界，CI 和 hooks 使用同一入口。Prettier 同时格式化本仓 JavaScript 检查脚本。

暂存格式化使用维护包的两阶段 hook：先由 git-format-staged 格式化 index，再由 pre-commit 隐藏未暂存内容并运行检查。资料完整性也在导出的 index 快照上运行。

当前不采用可选 MPE、AutoCorrect、Shell／TOML 格式化和游戏构建检查；仓库没有对应自有产品实现。系统级凭据扫描使用 Gitleaks 8.30.1，需在 PATH 中提供 `gitleaks`；它在提交和 CI 扫描，输出隐藏匹配内容。本机可将受忽略的 `.tools/bin` 加入 PATH。

CI 与本地运行同一检查入口，并在干净 npm 安装后运行 `npm audit`；本地也可手动执行该联网审计。远端 Actions 是否成功与是否设置分支保护分别核实，工作流文件本身不是保护已启用的证据。
