const RECOMMENDATION_CLASS = {
  "首选复用": "good",
  "高性能备选": "neutral",
  "谨慎使用": "warn",
  "不推荐": "bad",
};

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
  return <section className="dataCard">
    <header className="dataCardHeader">
      <b>{card.title || "候选样品"}</b>
      {card.matched_count ? <span>共 {card.matched_count} 条</span> : null}
      {card.degrade_level ? <span className="dataCardNote">降级：{card.degrade_level}</span> : null}
    </header>
    {card.reference && (card.reference.name || card.reference.id) && <article className="dataCardItem reference">
      <header>
        <div><b>参照：{card.reference.name || card.reference.id}</b></div>
      </header>
      <FieldRows title="配方" fields={card.reference.formula} />
      <FieldRows title="工艺" fields={card.reference.process} />
      <FieldRows title="性能" fields={card.reference.performance} />
    </article>}
    {(card.items || []).map((item, index) => <SampleListItem item={item} key={index} />)}
    {!(card.items || []).length && <p className="dataCardEmpty">当前没有可直接展示的候选样品。</p>}
    {!!(card.warnings || []).length && <div className="dataCardWarnings">
      {card.warnings.map((item, index) => <span key={index}>{String(item)}</span>)}
    </div>}
  </section>;
}

function SampleDetailCard({ card }) {
  return <section className="dataCard">
    <header className="dataCardHeader"><b>{card.title || "样品画像"}</b></header>
    <FieldRows title="配方" fields={card.formula} />
    <FieldRows title="工艺" fields={card.process} />
    <FieldRows title="性能" fields={card.performance} />
    <FieldRows title="服役性能" fields={card.service_performance} />
    {!!(card.warnings || []).length && <div className="dataCardWarnings">
      {card.warnings.map((item, index) => <span key={index}>{String(item)}</span>)}
    </div>}
  </section>;
}

function MetricTableCard({ card }) {
  const columns = card.columns || [];
  const rows = card.rows || [];
  return <section className="dataCard">
    <header className="dataCardHeader">
      <b>{card.title || "数据表"}</b>
      {card.sample_count ? <span>样本 {card.sample_count}</span> : null}
    </header>
    <div className="dataCardTableWrap">
      <table>
        <thead><tr>{columns.map((column, index) => <th key={index}>{column}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, rowIndex) => <tr key={rowIndex}>
            {row.map((cell, cellIndex) => <td key={cellIndex}>{formatValue(cell)}</td>)}
          </tr>)}
        </tbody>
      </table>
      {!rows.length && <p className="dataCardEmpty">当前没有可直接展示的数据行。</p>}
    </div>
    {!!(card.warnings || []).length && <div className="dataCardWarnings">
      {card.warnings.map((item, index) => <span key={index}>{String(item)}</span>)}
    </div>}
  </section>;
}

export default function DataCards({ cards }) {
  const list = Array.isArray(cards) ? cards.filter(Boolean) : [];
  if (!list.length) return null;
  return <div className="dataCards">
    {list.map((card, index) => {
      if (card.card_type === "metric_table") return <MetricTableCard card={card} key={index} />;
      if (card.card_type === "sample_detail") return <SampleDetailCard card={card} key={index} />;
      return <SampleListCard card={card} key={index} />;
    })}
  </div>;
}
