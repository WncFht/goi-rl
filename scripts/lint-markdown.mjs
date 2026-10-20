import { execFileSync } from "node:child_process";
import { lstatSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { lint } from "markdownlint/promise";

// These are literal directory prefixes, not glob expressions.
const excluded = [
    "sources/",
    ".agents/notes/archived/",
    "node_modules/",
    ".venv/",
    ".tools/",
    ".claude/",
    ".crossnote/",
];

export function selectMarkdown(root, explicit = []) {
    const names = explicit.length
        ? explicit
        : execFileSync(
              "git",
              ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
              { cwd: root, encoding: "utf8" },
          ).split("\0");
    const files = new Set();
    for (const name of names) {
        if (!name) continue;
        const absolute = path.resolve(root, name);
        const relative = path
            .relative(root, absolute)
            .split(path.sep)
            .join("/");
        if (
            relative === ".." ||
            relative.startsWith("../") ||
            path.isAbsolute(relative)
        ) {
            throw new Error(`Path outside repository: ${name}`);
        }
        if (excluded.some((prefix) => relative.startsWith(prefix))) continue;
        if (!relative.endsWith(".md")) {
            if (explicit.length)
                throw new Error(`Not a Markdown file: ${name}`);
            continue;
        }
        let info;
        try {
            info = lstatSync(absolute);
        } catch (error) {
            if (!explicit.length && error.code === "ENOENT") continue;
            throw error;
        }
        if (!info.isFile() || info.isSymbolicLink()) {
            if (explicit.length) throw new Error(`Not a regular file: ${name}`);
            continue;
        }
        files.add(absolute);
    }
    if (!files.size) throw new Error("No owned Markdown files selected");
    return [...files].sort();
}

export async function checkMarkdown(root, explicit = []) {
    const files = selectMarkdown(root, explicit);
    const config = JSON.parse(
        readFileSync(path.join(root, ".markdownlint.json"), "utf8"),
    );
    const results = await lint({ files, config });
    const diagnostics = Object.entries(results).flatMap(([file, errors]) =>
        errors.map(
            (error) =>
                `${path.relative(root, file)}:${error.lineNumber}: ${error.ruleNames[0]} ${error.errorDescription}`,
        ),
    );
    if (diagnostics.length) console.error(diagnostics.join("\n"));
    console.log(`Markdown: ${files.length} files checked`);
    return diagnostics.length ? 1 : 0;
}

if (
    process.argv[1] &&
    path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
    try {
        const root = execFileSync("git", ["rev-parse", "--show-toplevel"], {
            encoding: "utf8",
        }).trim();
        process.exitCode = await checkMarkdown(root, process.argv.slice(2));
    } catch (error) {
        console.error(error.message);
        process.exitCode = 1;
    }
}
