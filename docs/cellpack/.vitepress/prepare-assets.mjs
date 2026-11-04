import { cpSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { prepareLogo } from "./prepare-logo.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
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
