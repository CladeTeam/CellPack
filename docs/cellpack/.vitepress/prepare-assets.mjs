import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { prepareLogo } from "./prepare-logo.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const rustDownloads = path.join(root, "public/downloads/rust");
mkdirSync(rustDownloads, { recursive: true });
for (const file of readdirSync(path.join(root, "examples"))) {
  if (file.endsWith(".rs")) {
    cpSync(path.join(root, "examples", file), path.join(rustDownloads, file));
  }
}
cpSync(
  path.join(root, "../../assets/cellpack-animation-first-frame.svg"),
  path.join(root, "public/cellpack-animation-first-frame.svg"),
);
prepareLogo(root);
const target = path.join(root, "public/pdfjs");
mkdirSync(target, { recursive: true });
for (const directory of ["cmaps", "standard_fonts", "wasm"]) {
  cpSync(
    path.join(root, "node_modules/pdfjs-dist", directory),
    path.join(target, directory),
    { recursive: true },
  );
}
cpSync(
  path.join(root, "node_modules/pdfjs-dist/LICENSE"),
  path.join(target, "LICENSE"),
);
