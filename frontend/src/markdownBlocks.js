export function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line);
}

export function classifyLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return "blank";
  if (trimmed.startsWith("```")) return "code";
  if (/^(#{1,6})\s+/.test(trimmed)) return "heading";
  if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) return "hr";
  if (/^\s*[-*+]\s+/.test(trimmed)) return "ul";
  if (/^\s*\d+[.)]\s+/.test(trimmed)) return "ol";
  if (/^>/.test(trimmed)) return "quote";
  return "text";
}

export function splitBlocks(text) {
  const lines = String(text ?? "").split("\n");
  const blocks = [];
  let current = [];
  let currentKind = null;
  let inCode = false;

  const flush = () => {
    if (current.length) blocks.push(current.join("\n"));
    current = [];
    currentKind = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      if (inCode) {
        current.push(line);
        flush();
        inCode = false;
      } else {
        flush();
        current.push(line);
        inCode = true;
        currentKind = "code";
      }
      continue;
    }

    if (inCode) {
      current.push(line);
      continue;
    }

    if (!trimmed) {
      flush();
      continue;
    }

    if (current.length === 1 && current[0].includes("|") && isTableSeparator(line)) {
      current.push(line);
      currentKind = "table";
      continue;
    }
    if (currentKind === "table" && line.includes("|")) {
      current.push(line);
      continue;
    }

    const kind = classifyLine(line);

    if (kind === "heading" || kind === "hr") {
      flush();
      blocks.push(line);
      continue;
    }

    if (kind === "ul" || kind === "ol" || kind === "quote") {
      if (currentKind !== kind) flush();
      current.push(line);
      currentKind = kind;
      continue;
    }

    if (kind === "text") {
      if (currentKind && currentKind !== "text") flush();
      current.push(line);
      currentKind = "text";
      continue;
    }

    current.push(line);
  }

  if (inCode) {
    console.warn("MarkdownView: unclosed code block, auto-closed at end of input");
  }
  flush();
  return blocks;
}
