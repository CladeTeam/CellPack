import {
  copyFileSync,
  mkdirSync,
  readdirSync,
  renameSync,
  rmSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const output =
  process.env.CELLPACK_DOCS_OUTPUT || path.join(root, "_build/html");
const stage = path.join(root, `_build/publish-${process.pid}-${Date.now()}`);

function files(directory, prefix = "") {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const relative = path.join(prefix, entry.name);
    return entry.isDirectory()
      ? files(path.join(directory, entry.name), relative)
      : [relative];
  });
}

try {
  const result = spawnSync("npm", ["run", "build"], {
    cwd: root,
    stdio: "inherit",
    env: { ...process.env, CELLPACK_DOCS_OUTPUT: stage },
  });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw new Error(
      `Website build failed (${result.status}); the published site was preserved.`,
    );
  const paths = files(stage).sort((a, b) => {
    const priority = (file) =>
      file === "hashmap.json" ? 2 : file.endsWith(".html") ? 1 : 0;
    return priority(a) - priority(b);
  });
  for (const relative of paths) {
    const destination = path.join(output, relative);
    mkdirSync(path.dirname(destination), { recursive: true });
    const temporary = `${destination}.${process.pid}.tmp`;
    copyFileSync(path.join(stage, relative), temporary);
    renameSync(temporary, destination);
  }
  // Open tabs retain their app's hashed chunks and stylesheet until a full reload.
  console.log(
    `Published ${paths.length} files to ${output}; previous version assets retained.`,
  );
} finally {
  rmSync(stage, { recursive: true, force: true });
}
