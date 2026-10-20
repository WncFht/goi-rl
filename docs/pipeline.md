# 检查流程

`scripts/checks.json` 调度自有文档、技能、Notes、研究完整性、维护工具测试、Python、Markdown 与配置格式检查。

提交前分为两阶段：

1. `install_hooks.py` 对暂存文件执行声明的 manual formatter。
2. pre-commit 运行静态检查、凭据扫描和 `staged_checks.py`。后者将 index 导出到临时目录，检查文档、Notes、研究文件及已有归档的历史。

`sources/` 不进入格式化；对应 manifest 哈希仍必须通过。push hook 执行完整 CI 检查。CI 另进行 Gitleaks 扫描；没有部署或自动发布。

检查只能证明各自覆盖的文件、格式和约束，不能证明模拟器保真、训练复现或速通成绩。工具修改需要反例；检查失败应修复并重新运行，不跳过 hook。
