import { jsPDF } from "jspdf";

/** Step order / labels mirrored from App.jsx for standalone PDF generation. */
const STEP_ORDER = [
  "variant_check",
  "aso_check",
  "inheritance_pattern",
  "pathomechanism",
  "splicing_effects",
  "exon_skipping",
  "knockdown",
  "wt_upregulation",
];

const STEP_LABELS = {
  variant_check: "Variant validation",
  aso_check: "Existing ASO therapy check",
  inheritance_pattern: "Inheritance pattern",
  pathomechanism: "Pathomechanism",
  splicing_effects: "Splicing effects",
  exon_skipping: "Exon skipping",
  knockdown: "Transcript knockdown",
  wt_upregulation: "WT upregulation",
};

const STRATEGY_LABELS = {
  splice_correction: "Splice correction",
  exon_skipping: "Exon skipping",
  transcript_knockdown: "Transcript knockdown",
  wt_upregulation: "WT upregulation",
};

const CLS_LABELS = {
  eligible: "eligible",
  likely_eligible: "likely eligible",
  not_eligible: "not eligible",
  unable_to_assess: "unable to assess",
  not_applicable: "not applicable",
  unlikely_eligible: "unlikely eligible",
  applicable: "applicable",
  valid: "valid",
  invalid: "invalid",
};

const MARGIN = 54;
const PAGE_W = 612;
const PAGE_H = 792;
const CONTENT_W = PAGE_W - MARGIN * 2;
const FOOTER_Y = PAGE_H - 28;

/**
 * jsPDF's standard/core fonts (helvetica, courier, etc.) only support
 * single-byte WinAnsiEncoding (roughly Latin-1). If a string contains ANY
 * character outside that set, jsPDF silently re-encodes the WHOLE string
 * as UTF-16BE bytes, which the single-byte font then renders one byte at
 * a time -- producing a "letters with big gaps" artifact. LLM-generated
 * text commonly contains smart quotes, primes, en/em dashes, ellipses,
 * non-breaking spaces, bullets, etc. that trigger this. Normalize those to
 * ASCII equivalents before anything reaches jsPDF.
 */
function sanitizeForPdf(text) {
  if (text == null) return text;
  return String(text)
    .replace(/[\u2018\u2019\u201A\u201B]/g, "'") // curly single quotes
    .replace(/[\u201C\u201D\u201E\u201F]/g, '"') // curly double quotes
    .replace(/[\u2032\u2035]/g, "'") // prime / reversed prime (e.g. 2'-OMe)
    .replace(/[\u2033\u2036]/g, '"') // double prime
    .replace(/[\u2012\u2013]/g, "-") // figure dash, en dash
    .replace(/\u2014/g, "--") // em dash
    .replace(/\u2026/g, "...") // ellipsis
    .replace(/[\u00A0\u2000-\u200A\u202F\u205F]/g, " ") // nbsp + various spaces
    .replace(/[\u2022\u2023\u25E6]/g, "-") // bullets
    .replace(/[\u2192\u2794\u21D2]/g, "->") // right arrows
    .replace(/[\u2190\u21D0]/g, "<-") // left arrows
    .replace(/[^\x00-\xFF]/g, "?"); // anything else outside Latin-1: don't let it through
}

function formatClassification(cls) {
  if (cls == null || cls === "") return "—";
  const key = String(cls).toLowerCase().replace(/\s+/g, "_");
  return CLS_LABELS[key] || String(cls).replace(/_/g, " ");
}

function formatReasoningDisplay(reasoning) {
  if (reasoning == null || reasoning === "") return "";
  if (typeof reasoning === "string") {
    try {
      const parsed = JSON.parse(reasoning);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed.reason || reasoning;
      }
    } catch {
      /* use full string */
    }
    return reasoning;
  }
  return JSON.stringify(reasoning, null, 2);
}

function humanizeKey(key) {
  return String(key).replace(/_/g, " ");
}

function formatPaper(paper, index) {
  if (!paper || typeof paper !== "object") return null;
  const parts = [];
  if (paper.title) parts.push(paper.title);
  const meta = [];
  if (paper.journal) meta.push(paper.journal);
  if (paper.pubdate || paper.year) meta.push(paper.pubdate || paper.year);
  if (paper.pmid) meta.push(`PMID: ${paper.pmid}`);
  if (paper.doi) meta.push(`DOI: ${paper.doi}`);
  if (meta.length) parts.push(`(${meta.join("; ")})`);
  if (!parts.length) return null;
  return `${index != null ? `${index}. ` : ""}${parts.join(" ")}`;
}

function formatPapers(papers, limit = 25) {
  if (!Array.isArray(papers) || papers.length === 0) return [];
  const lines = [];
  papers.slice(0, limit).forEach((p, i) => {
    const line = formatPaper(p, i + 1);
    if (line) lines.push(line);
  });
  if (papers.length > limit) lines.push(`… and ${papers.length - limit} more`);
  return lines;
}

class PdfWriter {
  constructor() {
    this.doc = new jsPDF({ unit: "pt", format: "letter" });
    this.y = MARGIN;
    this.pageNum = 1;
  }

  ensureSpace(needed = 16) {
    if (this.y + needed > FOOTER_Y - 12) {
      this.newPage();
    }
  }

  newPage() {
    this.drawFooter();
    this.doc.addPage();
    this.pageNum += 1;
    this.y = MARGIN;
  }

  drawFooter() {
    this.doc.setFont("helvetica", "normal");
    this.doc.setFontSize(8);
    this.doc.setTextColor(120);
    this.doc.text(`Page ${this.pageNum}`, PAGE_W / 2, FOOTER_Y, { align: "center" });
    this.doc.setTextColor(0);
  }

  addTitle(text) {
    text = sanitizeForPdf(text);
    this.ensureSpace(28);
    this.doc.setFont("helvetica", "bold");
    this.doc.setFontSize(16);
    this.doc.setTextColor(20);
    const lines = this.doc.splitTextToSize(text, CONTENT_W);
    this.doc.text(lines, MARGIN, this.y);
    this.y += lines.length * 20 + 4;
  }

  addSubtitle(text) {
    text = sanitizeForPdf(text);
    this.ensureSpace(18);
    this.doc.setFont("helvetica", "normal");
    this.doc.setFontSize(11);
    this.doc.setTextColor(60);
    const lines = this.doc.splitTextToSize(text, CONTENT_W);
    this.doc.text(lines, MARGIN, this.y);
    this.y += lines.length * 14 + 6;
  }

  addMetaLine(text) {
    text = sanitizeForPdf(text);
    this.ensureSpace(14);
    this.doc.setFont("helvetica", "normal");
    this.doc.setFontSize(9);
    this.doc.setTextColor(100);
    this.doc.text(text, MARGIN, this.y);
    this.y += 12;
  }

  addSectionHeading(text) {
    text = sanitizeForPdf(text);
    this.y += 10;
    this.ensureSpace(24);
    this.doc.setFont("helvetica", "bold");
    this.doc.setFontSize(12);
    this.doc.setTextColor(20);
    this.doc.text(text, MARGIN, this.y);
    this.y += 6;
    this.doc.setDrawColor(180);
    this.doc.setLineWidth(0.5);
    this.doc.line(MARGIN, this.y, MARGIN + CONTENT_W, this.y);
    this.y += 14;
  }

  addSubheading(text) {
    text = sanitizeForPdf(text);
    this.ensureSpace(18);
    this.doc.setFont("helvetica", "bold");
    this.doc.setFontSize(10);
    this.doc.setTextColor(30);
    const lines = this.doc.splitTextToSize(text, CONTENT_W);
    this.doc.text(lines, MARGIN, this.y);
    this.y += lines.length * 13 + 4;
  }

  addLabel(text) {
    text = sanitizeForPdf(text);
    this.ensureSpace(14);
    this.doc.setFont("helvetica", "bold");
    this.doc.setFontSize(9);
    this.doc.setTextColor(90);
    this.doc.text(text, MARGIN, this.y);
    this.y += 12;
  }

  addBody(text, { indent = 0, mono = false } = {}) {
    if (text == null || text === "") return;
    const str = sanitizeForPdf(text);
    const width = CONTENT_W - indent;
    this.doc.setFont(mono ? "courier" : "helvetica", "normal");
    this.doc.setFontSize(mono ? 8 : 9.5);
    this.doc.setTextColor(40);
    const lineH = mono ? 10 : 13;
    const lines = this.doc.splitTextToSize(str, width);
    for (const line of lines) {
      this.ensureSpace(lineH + 2);
      this.doc.text(line, MARGIN + indent, this.y);
      this.y += lineH;
    }
    this.y += 4;
  }

  addBullet(text) {
    this.ensureSpace(14);
    this.doc.setFont("helvetica", "normal");
    this.doc.setFontSize(9.5);
    this.doc.setTextColor(40);
    const lines = this.doc.splitTextToSize(sanitizeForPdf(text), CONTENT_W - 14);
    this.doc.text("•", MARGIN, this.y);
    this.doc.text(lines[0], MARGIN + 12, this.y);
    this.y += 13;
    for (let i = 1; i < lines.length; i++) {
      this.ensureSpace(14);
      this.doc.text(lines[i], MARGIN + 12, this.y);
      this.y += 13;
    }
    this.y += 2;
  }

  addKeyValue(label, value) {
    if (value == null || value === "") return;
    this.ensureSpace(14);
    this.doc.setFont("helvetica", "bold");
    this.doc.setFontSize(9.5);
    this.doc.setTextColor(50);
    const labelText = `${sanitizeForPdf(label)}: `;
    const labelW = this.doc.getTextWidth(labelText);
    this.doc.text(labelText, MARGIN, this.y);
    this.doc.setFont("helvetica", "normal");
    this.doc.setTextColor(40);
    const valueLines = this.doc.splitTextToSize(sanitizeForPdf(value), CONTENT_W - labelW);
    this.doc.text(valueLines[0], MARGIN + labelW, this.y);
    this.y += 13;
    for (let i = 1; i < valueLines.length; i++) {
      this.ensureSpace(14);
      this.doc.text(valueLines[i], MARGIN + labelW, this.y);
      this.y += 13;
    }
  }

  finish() {
    this.drawFooter();
  }

  save(filename) {
    this.finish();
    this.doc.save(filename);
  }
}

/**
 * Build the jsPDF document for a report, without saving/downloading it.
 * Factored out of downloadReportPdf() so callers (e.g. an inline preview
 * pane) can get the finished doc/blob without triggering a file download.
 */
export function buildReportPdfDoc(report) {
  if (!report) return null;

  const w = new PdfWriter();
  const summaryRaw = report.summary;
  const s =
    summaryRaw !== null && typeof summaryRaw === "object" && !Array.isArray(summaryRaw)
      ? summaryRaw
      : null;
  const summaryText = typeof summaryRaw === "string" ? summaryRaw : null;
  const steps = report.step_results ?? report.steps ?? {};

  const strategiesFromSummary =
    s?.strategy_assessments !== null && typeof s.strategy_assessments === "object"
      ? s.strategy_assessments
      : {};
  const classifications = report.classifications ?? {};
  const strategies =
    Object.keys(strategiesFromSummary).length > 0
      ? strategiesFromSummary
      : Object.fromEntries(
          Object.entries(classifications).map(([key, cls]) => [
            key,
            { classification: cls, key_evidence: null, caveats: null },
          ]),
        );

  w.addTitle("N1C Variant ASO Assessment Report");
  w.addSubtitle(report.hgvs ?? "—");
  if (s?.variant_description) w.addBody(s.variant_description);
  w.addMetaLine(
    [
      `Gene: ${report.gene_id || "—"}`,
      `Date: ${report.date || "—"}`,
      `Model: ${report.model_name || "—"}`,
    ].join("  ·  "),
  );

  const inheritance =
    steps.inheritance_pattern?.metadata?.inheritance_pattern?.replace(/_/g, " ") || "—";
  const pathomech =
    steps.pathomechanism?.metadata?.pathomechanism?.replace(/_/g, " ") || "—";
  const haplo = steps.pathomechanism?.metadata?.is_haploinsufficient ? "Yes" : "No";
  w.addKeyValue("Inheritance", inheritance);
  w.addKeyValue("Pathomechanism", pathomech);
  w.addKeyValue("Haploinsufficient", haplo);

  w.addSectionHeading("Clinical summary");
  w.addBody(summaryText ?? s?.overall_summary ?? "—");
  if (s?.inheritance_summary) {
    w.addLabel("Inheritance note");
    w.addBody(s.inheritance_summary);
  }
  if (s?.pathomechanism_summary) {
    w.addLabel("Pathomechanism note");
    w.addBody(s.pathomechanism_summary);
  }
  if (s?.splicing_summary) {
    w.addLabel("Splicing note");
    w.addBody(s.splicing_summary);
  }

  if (Object.keys(strategies).length > 0) {
    w.addSectionHeading("Therapy strategy assessment");
    for (const [key, strat] of Object.entries(strategies)) {
      w.addSubheading(STRATEGY_LABELS[key] || humanizeKey(key));
      w.addKeyValue("Classification", formatClassification(strat?.classification));
      if (strat?.key_evidence) {
        w.addLabel("Key evidence");
        w.addBody(strat.key_evidence);
      }
      if (strat?.caveats && strat.caveats !== "None") {
        w.addLabel("Caveats");
        w.addBody(strat.caveats);
      }
    }
  }

  w.addSectionHeading("Pipeline steps");
  for (const stepKey of STEP_ORDER) {
    const result = steps[stepKey];
    if (!result) continue;

    w.addSubheading(STEP_LABELS[stepKey] || humanizeKey(stepKey));
    w.addKeyValue("Classification", formatClassification(result.classification));

    if (result.summary) {
      w.addLabel("Summary");
      w.addBody(result.summary);
    }

    const reasoningText = formatReasoningDisplay(result.reasoning);
    const summaryStr = result.summary == null ? "" : String(result.summary);
    if (reasoningText && reasoningText !== summaryStr) {
      w.addLabel("Reasoning");
      w.addBody(reasoningText);
    }

    if (result.metadata?.warnings?.length) {
      w.addLabel("Warnings");
      result.metadata.warnings.forEach((warn) => w.addBullet(warn));
    }

    if (result.metadata?.evidence_snippets?.length) {
      w.addLabel("Evidence snippets");
      result.metadata.evidence_snippets.forEach((snip) => w.addBullet(snip));
    }
  }

  if (s?.recommended_next_steps?.length) {
    w.addSectionHeading("Recommended next steps");
    s.recommended_next_steps.forEach((step, i) => {
      w.addBody(`${i + 1}. ${step}`);
    });
  }

  if (s?.important_caveats?.length) {
    w.addSectionHeading("Important caveats");
    s.important_caveats.forEach((c) => w.addBullet(c));
  }

  w.finish();
  return w.doc;
}

/**
 * Build and download a high-level PDF summary of an ASO assessment report.
 * Omits tool-call traces and user revision history.
 */
export function downloadReportPdf(report, filename) {
  const doc = buildReportPdfDoc(report);
  if (!doc) return;
  const safeName =
    filename ||
    `aso-report-${String(report.hgvs || "report").replace(/[^a-zA-Z0-9]/g, "_")}.pdf`;
  doc.save(safeName.endsWith(".pdf") ? safeName : `${safeName}.pdf`);
}

/** Build a report PDF and return an object: URL you can preview in an <iframe>. */
export function getReportPdfBlobUrl(report) {
  const doc = buildReportPdfDoc(report);
  if (!doc) return null;
  return doc.output("bloburl");
}