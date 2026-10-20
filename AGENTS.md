# GOI-RL 研究仓库

本仓保存《掘地求升》物理机制、环境接口与速通算法研究资料。已有研究结论不等于运行过游戏；区分源码事实、作者报告、推导和实测结果。

## 内容归属

- [README](README.md) 为阅读入口；根目录日期命名主报告拥有研究结论。
- `sources/` 保存来源索引、外部原始响应及固定提交的源码文本快照。不要格式化、重写或执行这些快照；更新来源时同步 `source-manifest.json`，保留出处。
- `scripts/` 拥有资料完整性与维护检查；[脚本规则](scripts/AGENTS.md)。
- [docs](docs/AGENTS.md) 拥有仓库开发指南；[文档入口](docs/README.md)。
- `.agents/skills` 是本仓技能事实源；`.claude/skills` 由 setup 创建。

## 开发与验证

文档维护采用 [maintain-docs](.agents/skills/maintain-docs/SKILL.md) 和 [maintain-prose](.agents/skills/maintain-prose/SKILL.md)。只有需要长期理由时才写 [Agent Notes](.agents/notes/README.md)，不要把逐次运行进度当成研究结果。

按 [工具链指南](docs/toolchain.md) 安装项目依赖和两阶段 hook。提交前运行 `python scripts/run_checks.py --config scripts/checks.json --mode ci`。研究资料完整性使用 `python scripts/check_research.py`；这些检查均不验证游戏物理、算法成绩或外部链接。

提交信息使用英文 Conventional Commits。保留原始源码和研究事实；格式改动不得改变观测字段、公式、实验条件或证据确定性。不得用普通 `pre-commit install` 覆盖两阶段入口。

本机执行 Git SSH 操作前，在同一 shell 执行 `source /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/.ssh/activate.bash`。
