import { useEffect, useState } from "react";

const RECOMMENDATION_CLASS = {
  "首选复用": "good",
  "高性能备选": "neutral",
  "谨慎使用": "warn",
  "不推荐": "bad",
};

const MATCH_MODE_LABELS = {
  formula_name_jaccard: "按原料名称近似匹配",
  same_project_recent: "同项目近期样品参考",
  none: "没有可用降级匹配",
};

function uniqueText(values) {
  return [...new Set(
    (Array.isArray(values) ? values : [])
      .map(value => String(value || "").trim())
      .filter(Boolean)
  )];
}

function formatValue(value, precision) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number" && Number.isFinite(value)) {
    if (Number.isInteger(value)) return String(value);
    const digits = precision === undefined ? 4 : precision;
    return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "string") {
    const numeric = Number(value.trim());
    if (Number.isFinite(numeric) && value.trim() !== "") {
      if (Number.isInteger(numeric)) return String(numeric);
      const digits = precision === undefined ? 4 : precision;
      return numeric.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
    }
  }
  return String(value);
}

function FieldRows({ title, fields }) {
  const rows = Array.isArray(fields) ? fields.filter(Boolean) : [];
  if (!rows.length) return null;
  const precision = title === "性能" || title === "服役性能" ? 3 : 4;
  return <div className="dataCardSection">
    <small>{title}</small>
    <div className="dataCardFields">
      {rows.map((field, index) => <div className="dataCardField" key={index}>
        <span>{field.name || "-"}</span>
        <b>{formatValue(field.value, precision)}{field.unit ? ` ${field.unit}` : ""}</b>
      </div>)}
    </div>
  </div>;
}

function RecommendationBadge({ value }) {
  if (!value) return null;
  const cls = RECOMMENDATION_CLASS[value] || "neutral";
  return <span className={`dataCardBadge ${cls}`}>{value}</span>;
}

function BusinessNotices({ card }) {
  const notices = uniqueText([
    ...(card.dimension_notices || []),
    card.unit_notice,
    card.similarity_notice,
  ]);
  if (!notices.length) return null;
  return <div className="dataCardNotices">
    {notices.map((notice, index) => <p key={index}>{notice}</p>)}
  </div>;
}

function LoadMoreButton({ remaining, onClick }) {
  if (remaining <= 0) return null;
  return <button className="dataCardLoadMore" type="button" onClick={onClick}>
    加载更多<span>剩余 {remaining} 条</span>
  </button>;
}

function SampleListItem({ item }) {
  return <article className="dataCardItem">
    <header>
      <div>
        <b>{item.sample_name || item.sample_id || "-"}</b>
        {item.similarity !== null && item.similarity !== undefined
          ? <span>相似度 {formatValue(item.similarity)}%</span>
          : null}
      </div>
      <RecommendationBadge value={item.recommendation} />
    </header>
    <FieldRows title="配方" fields={item.formula} />
    <FieldRows title="工艺" fields={item.process} />
    <FieldRows title="性能" fields={item.performance} />
  </article>;
}

function SampleListCard({ card }) {
  const items = Array.isArray(card.items) ? card.items : [];
  const [visibleCount, setVisibleCount] = useState(5);
  useEffect(() => setVisibleCount(5), [card]);
  const visibleItems = items.slice(0, visibleCount);
  return <section className="dataCard">
    <header className="dataCardHeader">
      <b>{card.title || "候选样品"}</b>
      {card.matched_count ? <span>共 {card.matched_count} 条，展示前 {Math.min(items.length, 10)} 条{card.sort_label ? `，${card.sort_label}` : ""}</span> : null}
      {card.degrade_level ? <span className="dataCardNote">匹配方式：{MATCH_MODE_LABELS[card.degrade_level] || "近似匹配"}</span> : null}
    </header>
    <BusinessNotices card={card} />
    {card.reference && (card.reference.name || card.reference.id) && <article className="dataCardItem reference">
      <header>
        <div><b>参照：{card.reference.name || card.reference.id}</b></div>
      </header>
      <FieldRows title="配方" fields={card.reference.formula} />
      <FieldRows title="工艺" fields={card.reference.process} />
      <FieldRows title="性能" fields={card.reference.performance} />
    </article>}
    {visibleItems.map((item, index) => <SampleListItem item={item} key={index} />)}
    {!items.length && <p className="dataCardEmpty">未找到匹配样品，建议放宽条件或检查数据范围。</p>}
    <LoadMoreButton
      remaining={Math.max(0, items.length - visibleCount)}
      onClick={() => setVisibleCount(count => Math.min(items.length, count + 5))}
    />
  </section>;
}

function SampleDetailCard({ card }) {
  return <section className="dataCard">
    <header className="dataCardHeader"><b>{card.title || "样品画像"}</b></header>
    <BusinessNotices card={card} />
    <FieldRows title="配方" fields={card.formula} />
    <FieldRows title="工艺" fields={card.process} />
    <FieldRows title="性能" fields={card.performance} />
    <FieldRows title="服役性能" fields={card.service_performance} />
  </section>;
}

function MetricTableCard({ card }) {
  const columns = card.columns || [];
  const rows = card.rows || [];
  const [visibleCount, setVisibleCount] = useState(8);
  useEffect(() => setVisibleCount(8), [card]);
  const visibleRows = rows.slice(0, visibleCount);
  return <section className="dataCard">
    <header className="dataCardHeader">
      <b>{card.title || "数据表"}</b>
      {card.sample_count ? <span>样本 {card.sample_count}</span> : null}
    </header>
    <BusinessNotices card={card} />
    <div className="dataCardTableWrap">
      <table>
        <thead><tr>{columns.map((column, index) => <th key={index}>{column}</th>)}</tr></thead>
        <tbody>
          {visibleRows.map((row, rowIndex) => <tr key={rowIndex}>
            {row.map((cell, cellIndex) => <td
              className={typeof cell === "number" ? "numeric" : "text"}
              key={cellIndex}
            >{formatValue(cell)}</td>)}
          </tr>)}
        </tbody>
      </table>
      {!rows.length && <p className="dataCardEmpty">当前没有可展示的数据行。</p>}
    </div>
    <LoadMoreButton
      remaining={Math.max(0, rows.length - visibleCount)}
      onClick={() => setVisibleCount(count => Math.min(rows.length, count + 8))}
    />
  </section>;
}

function WarningPanel({ warnings }) {
  const rows = uniqueText(warnings);
  if (!rows.length) return null;
  return <details className="dataCardWarnings dataCardWarningsTop" open={rows.length <= 2}>
    <summary>数据使用提示 {rows.length} 项</summary>
    <div>{rows.map((item, index) => <span key={index}>{item}</span>)}</div>
  </details>;
}

export default function DataCards({ cards, warnings }) {
  const list = Array.isArray(cards) ? cards.filter(Boolean) : [];
  const warningRows = uniqueText(warnings);
  if (!list.length && !warningRows.length) return null;
  return <div className="dataCards">
    {list.map((card, index) => {
      if (card.card_type === "metric_table") return <MetricTableCard card={card} key={index} />;
      if (card.card_type === "sample_detail") return <SampleDetailCard card={card} key={index} />;
      return <SampleListCard card={card} key={index} />;
    })}
    <WarningPanel warnings={warningRows} />
  </div>;
}
