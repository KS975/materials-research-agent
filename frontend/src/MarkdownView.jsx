import {useState} from "react";

const INLINE_PATTERN = /(`[^`]+`)|(\[[^\]]+\]\([^)\s]+\))|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g;

function renderInline(text, keyPrefix) {
  const nodes = [];
  let lastIndex = 0;
  let match;
  INLINE_PATTERN.lastIndex = 0;
  while ((match = INLINE_PATTERN.exec(text))) {
    if (match.index > lastIndex) nodes.push(text.slice(lastIndex, match.index));
    const token = match[0];
    const key = `${keyPrefix}-${match.index}`;
    if (token.startsWith("`")) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("[")) {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      if (linkMatch) {
        nodes.push(
          <a key={key} href={linkMatch[2]} target="_blank" rel="noreferrer noopener">
            {linkMatch[1]}
          </a>
        );
      } else {
        nodes.push(token);
      }
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

function Heading({level, children}) {
  const Tag = `h${Math.min(6, Math.max(1, level))}`;
  return <Tag>{children}</Tag>;
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|")
    .map(cell => cell.trim());
}

function MarkdownBlock({block, index}) {
  const lines = block.split("\n");
  const first = lines[0].trim();

  if (!block.trim()) return null;

  if (first.startsWith("```")) {
    const language = first.slice(3).trim();
    const code = lines.slice(1, first.endsWith("```") ? -1 : undefined).join("\n");
    return (
      <div className="mdCodeBlock">
        <div><span>{language || "code"}</span></div>
        <pre><code>{code}</code></pre>
      </div>
    );
  }

  if (lines.length > 1 && first.includes("|") && isTableSeparator(lines[1])) {
    const headers = tableCells(lines[0]);
    const rows = lines.slice(2).filter(line => line.trim()).map(tableCells);
    return (
      <div className="mdTableBlock">
        <div className="mdTableScroll">
          <table>
            <thead><tr>{headers.map((cell, cellIndex) => <th key={cellIndex}>{renderInline(cell, `th-${index}-${cellIndex}`)}</th>)}</tr></thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>{renderInline(cell, `td-${index}-${rowIndex}-${cellIndex}`)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  if (/^(-{3,}|\*{3,}|_{3,})$/.test(first)) return <hr/>;

  const heading = first.match(/^(#{1,6})\s+(.*)$/);
  if (heading) {
    return <Heading level={heading[1].length}>{renderInline(heading[2], `h-${index}`)}</Heading>;
  }

  if (lines.every(line => /^\s*[-*+]\s+/.test(line))) {
    return (
      <ul>
        {lines.map((line, itemIndex) => (
          <li key={itemIndex}>
            <span className="mdListItemContent">
              {renderInline(line.replace(/^\s*[-*+]\s+/, ""), `ul-${index}-${itemIndex}`)}
            </span>
          </li>
        ))}
      </ul>
    );
  }

  if (lines.every(line => /^\s*\d+[.)]\s+/.test(line))) {
    return (
      <ol>
        {lines.map((line, itemIndex) => (
          <li key={itemIndex}>
            <span className="mdListItemContent">
              {renderInline(line.replace(/^\s*\d+[.)]\s+/, ""), `ol-${index}-${itemIndex}`)}
            </span>
          </li>
        ))}
      </ol>
    );
  }

  if (lines.every(line => line.startsWith(">"))) {
    return (
      <blockquote>
        {renderInline(lines.map(line => line.replace(/^>\s?/, "")).join("\n"), `quote-${index}`)}
      </blockquote>
    );
  }

  return <p>{renderInline(lines.join("\n"), `p-${index}`)}</p>;
}

export default function MarkdownView({content}) {
  const text = String(content ?? "");
  const blocks = [];
  let current = [];
  let inCode = false;

  for (const line of text.split("\n")) {
    if (line.trim().startsWith("```")) inCode = !inCode;
    if (!inCode && !line.trim()) {
      if (current.length) blocks.push(current.join("\n"));
      current = [];
    } else {
      current.push(line);
    }
  }
  if (current.length) blocks.push(current.join("\n"));

  return (
    <div className="markdownView">
      {blocks.map((block, index) => <MarkdownBlock key={index} block={block} index={index}/>)}
    </div>
  );
}

export function CopyControl({value, label="复制"}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(String(value ?? ""));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button className="mdCopyControl" type="button" onClick={copy}>
      {copied ? "已复制" : label}
    </button>
  );
}
