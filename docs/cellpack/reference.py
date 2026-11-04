"""Render the public rustdoc surface as searchable VitePress API pages."""

from __future__ import annotations

import json
import re
import html
from functools import cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter
from api_content import EXAMPLES, ITEMS

DOCS = Path(__file__).resolve().parent
API = DOCS / "_build" / "rustdoc"
CATALOG = DOCS / "_catalog"
CODE = chr(96)
FENCE = CODE * 3
SYMBOLS = {}
MODULES = {
    "": ("root", "reference-errors"),
    "store": ("store", "reference-store"),
    "scans": ("scans", "reference-scans"),
    "format": ("format", "reference-format"),
    "format/cpk": ("format-cpk", "reference-format"),
    "loader": ("loader", "reference-loader"),
    "opt": ("opt", "reference-opt"),
    "opt/defaults": ("opt-defaults", "reference-opt"),
    "prof": ("prof", "reference-prof"),
    "runtime": ("runtime", "reference-runtime"),
}
ITEM = re.compile(r"(struct|enum|trait|fn|type|constant)\.([^.]+)\.html$")


def member_anchor(target: str, fragment: str) -> str:
    return (name_for(target) + "-" + fragment).lower().replace(".", "-").replace("_", "-")


@cache
def documented_anchors(target: str) -> set[str]:
    soup = BeautifulSoup((API / target).read_text(), "html.parser")
    return {node["id"] for node in soup.select(
        "#main-content > .structfield, #main-content > .variants > .variant, "
        "#implementations-list section.method, #implementations-list section.associatedconstant, "
        "#main-content > .methods section.method"
    )}


def documentation_link(target: str, fragment: str = "") -> str | None:
    if not target.startswith("cellpack/"):
        return None
    path = Path(target)
    module = "/".join(path.parts[1:-1])
    if path.name == "index.html":
        return "/api" if not module else "/" + MODULES.get(module, (None, "api"))[1]
    if not ITEM.fullmatch(path.name):
        return None
    link = "/" + name_for(target)
    if fragment and fragment in documented_anchors(target):
        link += "#" + member_anchor(target, fragment)
    return link


class RustMarkdown(MarkdownConverter):
    def __init__(self, target: str):
        super().__init__(heading_style="ATX", bullets="-")
        self.target = target

    def convert_a(self, el, text, **kwargs):
        href = el.get("href", "")
        if href and not urlsplit(href).scheme and not href.startswith("//"):
            url = urlsplit(urljoin("https://reference.invalid/" + self.target, href))
            target = url.path.lstrip("/")
            relative = documentation_link(target, url.fragment)
            label = text.strip() or "API"
            if relative is None:
                return label
            return "[" + label + "](" + relative + ")"
        return super().convert_a(el, text, **kwargs)


def markdown(node: Tag | None, target: str) -> str:
    if node is None:
        return ""
    fragment = BeautifulSoup(str(node), "html.parser")
    for extra in fragment.select(".anchor, .doc-anchor, .copy-path, .src, .example-wrap .buttons"):
        extra.decompose()
    for heading in fragment.select("h1, h2, h3, h4, h5, h6"):
        heading.name = "h4"
    for pre in fragment.select("pre"):
        text = pre.get_text()
        pre.clear()
        pre.string = text
    return RustMarkdown(target).convert(str(fragment)).strip()


def doc_for_member(member: Tag) -> Tag | None:
    parent = member.parent
    if parent.name == "summary":
        return parent.parent.find("div", class_="docblock", recursive=False)
    sibling = member.find_next_sibling()
    if sibling and "docblock" in sibling.get("class", []):
        return sibling
    return None


def signature_text(node: Tag) -> str:
    return node.get_text().replace("§", "").replace("ⓘ", "").strip()


def signature_html(node: Tag, target: str) -> str:
    """Keep compiler-resolved type links instead of guessing names in code text."""
    fragment = BeautifulSoup(str(node), "html.parser")
    for extra in fragment.select(".src, .anchor, .doc-anchor"):
        extra.decompose()
    for link in fragment.select("a"):
        href = link.get("href", "")
        resolved = urlsplit(urljoin("https://reference.invalid/" + target, href))
        local = documentation_link(resolved.path.lstrip("/"), resolved.fragment)
        if local:
            page, separator, anchor = local.partition("#")
            href = page + ".html" + (separator + anchor if separator else "")
            link.attrs = {":href": "withBase(" + json.dumps(href) + ")", "class": "cp-type-link"}
        elif resolved.scheme == "https" and resolved.netloc != "reference.invalid":
            link.attrs = {"href": resolved.geturl(), "target": "_blank", "rel": "noopener"}
        else:
            link.unwrap()
    code = fragment.select_one("code") or fragment.select_one(".code-header") or fragment
    for tag in list(code.find_all(True)):
        if tag.name not in {"a", "span", "br", "wbr"}:
            tag.unwrap()
        elif tag.name != "a":
            tag.attrs = {}
    signature = code.decode_contents().replace("§", "").replace("\n", "&#10;")
    return '<div class="cp-signature vp-raw"><pre><code>' + signature + "</code></pre></div>\n"


def source_info(node: Tag | None, target: str) -> dict | None:
    link = node.select_one("a.src") if node else None
    if not link:
        return None
    url = urlsplit(urljoin("https://reference.invalid/" + target, link["href"]))
    prefix = "src/cellpack/"
    if not url.path.lstrip("/").startswith(prefix):
        return None
    relative = url.path.lstrip("/")[len(prefix):].removesuffix(".html")
    path = (DOCS.parents[1] / "crates/cellpack/src" / relative).resolve()
    if not path.is_relative_to(DOCS.parents[1] / "crates/cellpack/src") or not path.is_file():
        return None
    first = int(url.fragment.split("-")[0]) if url.fragment else 1
    last = int(url.fragment.split("-")[-1]) if url.fragment else first
    lines = path.read_text().splitlines()
    start = max(1, first - 3)
    end = min(len(lines), max(last, first + 8), start + 100)
    return {"path": "crates/cellpack/src/" + relative, "line": first,
            "start": start, "code": "\n".join(lines[start - 1:end])}


def register_symbol(key: str, name: str, target: str, node: Tag, doc: Tag | None, signature: str, kind: str, anchor: str = "") -> None:
    paragraph = doc.find("p") if doc else None
    SYMBOLS[key] = {
        "key": key, "name": name, "kind": kind,
        "module": "/".join(Path(target).parts[1:-1]).replace("/", "::") or "root",
        "href": "/" + name_for(target) + ("#" + anchor if anchor else ""),
        "signature": signature,
        "summary": paragraph.get_text(" ", strip=True) if paragraph else "",
        "source": source_info(node, target),
    }


def split_parameters(text: str) -> list[str]:
    result, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in "(<[{":
            depth += 1
        elif char in ")>]}":
            if not (char == ">" and i > 0 and text[i - 1] == "-"):
                depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    result.append(text[start:].strip())
    return [part for part in result if part]


def prose(text: str) -> str:
    """Keep inline code intact while treating Rust type brackets as prose."""
    return "".join(part if part.startswith("`") else part.replace("<", "&lt;").replace(">", "&gt;")
                   for part in re.split(r"(`+[^`]*`+)", text))


def signature_details(signature: str, level: int, explanation: dict | None = None) -> str:
    match = re.search(r"\bfn\s+\w+", signature)
    if not match:
        return ""
    depth, opening = 0, None
    for i in range(match.end(), len(signature)):
        char = signature[i]
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif char == "(" and not depth:
            opening = i
            break
    if opening is None:
        return ""
    depth, closing = 1, opening
    while depth and closing + 1 < len(signature):
        closing += 1
        if signature[closing] == "(":
            depth += 1
        elif signature[closing] == ")":
            depth -= 1
    rows = []
    explanation = explanation or {}
    meanings = explanation.get("parameters", {})
    parameter_names = set()
    for parameter in split_parameters(signature[opening + 1:closing]):
        if ":" in parameter:
            name, kind = parameter.split(":", 1)
        else:
            name, kind = "self", parameter.replace("self", "Self")
        name, kind = name.strip(), " ".join(kind.split())
        parameter_names.add(name)
        kind = kind.replace("|", r"\|")
        if name != "self" and name in meanings:
            rows.append(f"| {CODE}{name}{CODE} | {prose(meanings[name])} |")
    if meanings.keys() - parameter_names:
        raise ValueError(f"Parameter explanations do not match {signature}: {meanings.keys() - parameter_names}")
    returns = signature[closing + 1:].strip()
    returns = re.split(r"\bwhere\b", returns, maxsplit=1)[0].strip()
    returns = returns.removeprefix("->").strip().rstrip(";") or "()"
    heading = "#" * level
    result = ""
    if rows:
        result += f"\n{heading} Parameters\n\n| Parameter | Meaning and constraints |\n|---|---|\n" + "\n".join(rows) + "\n"
    if explanation.get("returns"):
        result += f"\n{heading} Result\n\n" + prose(explanation["returns"]) + "\n"
    return result


def name_for(target: str) -> str:
    path = Path(target)
    kind, name = ITEM.fullmatch(path.name).groups()
    module = "-".join(path.parts[1:-1]) or "root"
    return f"generated-{module}-{kind}-{name}"


def render_item(target: str, reference: str) -> tuple[str, int, int]:
    soup = BeautifulSoup((API / target).read_text(), "html.parser")
    main = soup.select_one("#main-content")
    declaration = main.select_one(":scope > .item-decl")
    if declaration is None:
        raise ValueError(f"Missing rustdoc declaration: {target}")
    name = ITEM.fullmatch(Path(target).name)[2]
    module = "::".join(Path(target).parts[:-1])
    key = module.removeprefix("cellpack::") + "::" + name if module != "cellpack" else name
    content = ITEMS.get(key, {})
    contract = reference + ".md"
    intro = main.select_one(":scope > details.top-doc > .docblock")
    parts = [
        f'---\napi: true\n---\n\n<script setup>\nimport {{ withBase as siteBase }} from "vitepress";\nimport {{ canonicalLink }} from "./.vitepress/routes.mjs";\nconst withBase = value => siteBase(canonicalLink(value));\n</script>\n\n# {module}::{name}\n',
        f'<ApiToolbar symbol="{key}" />\n\n[Module contract]({contract}) · [All interfaces](api.md#browse-interfaces)\n',
        f"## At a glance\n\n" + signature_html(declaration, target),
        markdown(intro, target),
        "\n" + prose(content["when"]) + "\n" if content.get("when") else "",
    ]
    register_symbol(key, name, target, main.select_one(".main-heading"), intro,
                    signature_text(declaration), ITEM.fullmatch(Path(target).name)[1])
    for example_id, example in EXAMPLES.items():
        if key in example["symbols"]:
            parts.append(f'\n## Try it\n\n<ApiExample example="{example_id}" />\n')
    if Path(target).name.startswith("fn."):
        parts.append(signature_details(signature_text(declaration), 2, content))
    fields = main.select(":scope > .structfield, :scope > .variants > .variant")
    if fields:
        parts.append("\n## Fields and variants\n")
    for field in fields:
        title = field.get("id", "").split(".", 1)[-1]
        header = field.select_one(".code-header") or field
        anchor = member_anchor(target, field["id"])
        parts.extend([
            f'\n<a id="{anchor}"></a>\n\n### {title}\n\n' + signature_html(header, target),
            markdown(doc_for_member(field), target),
        ])
    members = main.select(
        "#implementations-list section.method, "
        "#implementations-list section.associatedconstant, "
        ":scope > .methods section.method"
    )
    if members:
        index = []
        for member in members:
            title = member["id"].split(".", 1)[-1]
            anchor = member_anchor(target, member["id"])
            doc = doc_for_member(member)
            paragraph = doc.find("p") if doc else None
            summary = " ".join(paragraph.get_text(" ", strip=True).split()) if paragraph else "See the exact signature and contract below."
            index.append({"name": title, "href": "#" + anchor, "summary": summary})
        payload = html.escape(json.dumps(index), quote=True)
        parts.append(f'\n## Member index\n\n<ApiMemberIndex :members="{payload}" />\n')
        parts.append("\n## Methods and associated constants\n")
    for member in members:
        title = member["id"].split(".", 1)[-1]
        header = member.select_one(".code-header")
        if header is None:
            raise ValueError(f"Missing member signature: {target}#{member['id']}")
        signature = signature_text(header)
        anchor = member_anchor(target, member["id"])
        parts.extend([
            f'\n<a id="{anchor}"></a>\n\n### {title}\n\n' + signature_html(header, target),
            markdown(doc_for_member(member), target),
            signature_details(signature, 4, content.get("methods", {}).get(title)),
        ])
        register_symbol(key + "::" + title, title, target, member,
                        doc_for_member(member), signature, "method", anchor)
    traits = main.select("#trait-implementations-list > details > summary > section.impl > .code-header")
    if traits:
        parts.append("\n## Implemented traits\n\n")
        parts.extend(signature_html(trait, target) for trait in traits)
    parts.append(
        "\n## Contract reference\n\n"
        f"See [the module reference]({contract}) for semantic input/output rules, "
        "units, allocation, mutation, failure behavior, and lifetime requirements. "
        "Declarations and documented members are synchronized from the crate source.\n"
    )
    (DOCS / (name_for(target) + ".md")).write_text("\n".join(parts).strip() + "\n")
    return name, len(members), len(fields)


def generate() -> None:
    SYMBOLS.clear()
    CATALOG.mkdir(exist_ok=True)
    for old in DOCS.glob("generated-*.md"):
        old.unlink()
    inventory, references = {}, {}
    for module, (catalog, reference) in MODULES.items():
        target = "cellpack/" + (module + "/" if module else "") + "index.html"
        soup = BeautifulSoup((API / target).read_text(), "html.parser")
        groups = {}
        for table in soup.select("#main-content > dl.item-table"):
            category = table.find_previous_sibling("h2").get_text().replace("§", "").strip()
            rows = []
            for term in table.find_all("dt", recursive=False):
                anchor = next((a for a in term.select("a[href]") if ITEM.search(urlsplit(a["href"]).path)), None)
                if anchor is None:
                    continue
                resolved = urlsplit(urljoin("https://reference.invalid/" + target, anchor["href"])).path.lstrip("/")
                if not resolved.startswith("cellpack/") or not (API / resolved).is_file():
                    continue
                name = ITEM.fullmatch(Path(resolved).name)[2]
                description = term.find_next_sibling("dd")
                summary = " ".join(description.get_text(" ", strip=True).split()) if description else ""
                if not summary:
                    item = BeautifulSoup((API / resolved).read_text(), "html.parser")
                    intro = item.select_one("#main-content > details.top-doc .docblock p")
                    if intro:
                        summary = " ".join(intro.get_text(" ", strip=True).split())
                summary = summary.replace("|", r"\|").replace("\n", " ")
                rows.append((name, resolved, summary or "See declaration, fields, and member contracts."))
                owner = "/".join(Path(resolved).parts[1:-1])
                references[resolved] = MODULES.get(owner, ("", reference))[1]
            if rows:
                groups[category] = rows
        inventory[catalog] = groups
    stats = {"items": 0, "members": 0, "fields": 0}
    for target, reference in sorted(references.items()):
        _, methods, fields = render_item(target, reference)
        stats["items"] += 1
        stats["members"] += methods
        stats["fields"] += fields
    navigation = {}
    titles = {
        "root": "Crate exports", "store": "Matrix storage", "scans": "Codecs and mapping",
        "format": "Chunk metadata", "format-cpk": "Stream format", "loader": "Data loading",
        "opt": "Plans and costs", "opt-defaults": "Cost defaults", "prof": "Profiling",
        "runtime": "Process settings",
    }
    for catalog, groups in inventory.items():
        parts, leaves = [], []
        for category, rows in groups.items():
            parts.append(f"### {category}\n\n| Interface | Responsibility |\n|---|---|")
            for name, target, summary in rows:
                document = name_for(target)
                parts.append("| [" + CODE + name + CODE + "](/" + document + ") | " + summary + " |")
                owner = "/".join(Path(target).parts[1:-1])
                if MODULES.get(owner, (None, None))[0] == catalog:
                    leaves.append({"text": name, "link": "/" + document})
            parts.append("")
        (CATALOG / (catalog + ".md")).write_text("\n".join(parts) + "\n")
        reference = next(ref for _, (name, ref) in MODULES.items() if name == catalog)
        navigation[catalog] = {"title": titles[catalog], "reference": reference, "items": leaves}
    (CATALOG / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    (CATALOG / "navigation.json").write_text(json.dumps(navigation, indent=2) + "\n")
    public_api = DOCS / "public/api"
    public_api.mkdir(parents=True, exist_ok=True)
    sources = {}
    for key, symbol in SYMBOLS.items():
        if symbol.get("source"):
            sources[key] = symbol["source"].pop("code")
    (public_api / "symbols.json").write_text(json.dumps(SYMBOLS, ensure_ascii=False) + "\n")
    (public_api / "sources.json").write_text(json.dumps(sources, ensure_ascii=False) + "\n")
    examples = {}
    for example_id, example in EXAMPLES.items():
        examples[example_id] = {**example, "code": (DOCS / "api-examples" / (example_id + ".rs")).read_text(),
                                "common": (DOCS / "examples/common.rs").read_text()}
    (public_api / "examples.json").write_text(json.dumps(examples, ensure_ascii=False) + "\n")
    print(f"API reference: {stats['items']} public items, {stats['members']} methods/constants, {stats['fields']} fields/variants")


if __name__ == "__main__":
    generate()
