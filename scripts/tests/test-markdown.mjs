import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import {
    mkdtempSync,
    mkdirSync,
    writeFileSync,
    rmSync,
    symlinkSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { selectMarkdown } from "../lint-markdown.mjs";

const runner = fileURLToPath(new URL("../lint-markdown.mjs", import.meta.url));

function fixture(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "goi-markdown-"));
    t.after(() => rmSync(root, { recursive: true, force: true }));
    execFileSync("git", ["init", "-q", root]);
    writeFileSync(path.join(root, ".markdownlint.json"), "{}\n");
    return root;
}

function run(root, ...files) {
    return spawnSync(process.execPath, [runner, ...files], {
        cwd: root,
        encoding: "utf8",
    });
}

test("valid documents pass; violations fail even before a valid file", (t) => {
    const root = fixture(t);
    writeFileSync(path.join(root, "valid.md"), "# Valid\n");
    assert.equal(run(root).status, 0);
    writeFileSync(path.join(root, "bad.md"), "#Bad\n");
    const result = run(root, "bad.md", "valid.md");
    assert.equal(result.status, 1);
    assert.match(result.stderr, /MD018/);
});

test("literal Unicode, spaces and brace names work without glob expansion", (t) => {
    const root = fixture(t);
    const names = ["中文 文档.md", "{a,b}[x]*.md", ".hidden.md"];
    for (const name of names) writeFileSync(path.join(root, name), "# Valid\n");
    assert.deepEqual(
        selectMarkdown(root)
            .map((name) => path.basename(name))
            .sort(),
        names.sort(),
    );
    assert.equal(run(root, ...names).status, 0);
});

test("snapshots, ignored files and symlinks are excluded", (t) => {
    const root = fixture(t);
    for (const dir of [
        "sources",
        ".agents/notes/archived",
        "node_modules",
        "ignored",
    ]) {
        mkdirSync(path.join(root, dir), { recursive: true });
        writeFileSync(path.join(root, dir, "bad.md"), "#Bad\n");
    }
    writeFileSync(path.join(root, ".gitignore"), "ignored/\nnode_modules/\n");
    writeFileSync(path.join(root, "valid.md"), "# Valid\n");
    symlinkSync("valid.md", path.join(root, "linked.md"));
    assert.deepEqual(selectMarkdown(root), [path.join(root, "valid.md")]);
    assert.equal(run(root).status, 0);
    assert.equal(run(root, "linked.md").status, 1);
});

test("empty scans, missing files and outside paths fail", (t) => {
    const root = fixture(t);
    assert.equal(run(root).status, 1);
    assert.equal(run(root, "missing.md").status, 1);
    assert.equal(run(root, "../outside.md").status, 1);
});
