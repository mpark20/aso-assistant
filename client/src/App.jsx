import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

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
  aso_check: "Existing ASO / therapy check",
  inheritance_pattern: "Inheritance pattern",
  pathomechanism: "Pathomechanism",
  splicing_effects: "Splicing effects",
  exon_skipping: "Exon skipping",
  knockdown: "Transcript knockdown",
};

const STRATEGY_LABELS = {
  splice_correction: "Splice correction",
  exon_skipping: "Exon skipping",
  transcript_knockdown: "Transcript knockdown",
  wt_upregulation: "WT upregulation",
};

const CLS_CONFIG = {
  eligible: { label: "eligible", bg: "#E1F5EE", color: "#0F6E56", dot: "#0F6E56" },
  likely_eligible: { label: "likely eligible", bg: "#EAF3DE", color: "#3B6D11", dot: "#3B6D11" },
  not_eligible: { label: "not eligible", bg: "#FCEBEB", color: "#A32D2D", dot: "#A32D2D" },
  unable_to_assess: { label: "unable to assess", bg: "#FAEEDA", color: "#854F0B", dot: "#854F0B" },
  not_applicable: { label: "not applicable", bg: "var(--color-background-secondary)", color: "var(--color-text-secondary)", dot: "var(--color-text-tertiary)" },
  unlikely_eligible: { label: "unlikely eligible", bg: "#FBEAF0", color: "#993556", dot: "#993556" },
};

function Badge({ cls, small }) {
  const conf = CLS_CONFIG[cls] || CLS_CONFIG["not_applicable"];
  return (
    <span style={{
      display: "inline-block",
      background: conf.bg,
      color: conf.color,
      fontSize: small ? 11 : 11,
      fontWeight: 500,
      padding: "3px 9px",
      borderRadius: 20,
      whiteSpace: "nowrap",
    }}>
      {conf.label}
    </span>
  );
}

function Spinner({ size = 16 }) {
  return (
    <span style={{
      display: "inline-block", width: size, height: size,
      border: `2px solid var(--color-border-tertiary)`,
      borderTopColor: "var(--color-text-secondary)",
      borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
    }} />
  );
}

function ProgressBar({ current, total }) {
  const pct = total === 0 ? 0 : Math.round((current / total) * 100);
  return (
    <div style={{ width: "100%", height: 3, background: "var(--color-border-tertiary)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{
        height: "100%", width: `${pct}%`,
        background: "var(--color-text-secondary)",
        borderRadius: 2,
        transition: "width 0.4s ease",
      }} />
    </div>
  );
}

function StepCard({ stepKey, result, isRunning, isPending }) {
  const [open, setOpen] = useState(false);
  const [dataOpen, setDataOpen] = useState(false);

  const statusDot = isRunning
    ? <Spinner size={13} />
    : isPending
      ? <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-border-secondary)", display: "inline-block" }} />
      : result
        ? <span style={{ width: 8, height: 8, borderRadius: "50%", background: CLS_CONFIG[result.classification]?.dot || "#888", display: "inline-block" }} />
        : null;

  return (
    <div style={{
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: "var(--border-radius-md)",
      overflow: "hidden",
      marginBottom: 8,
      background: "var(--color-background-primary)",
      opacity: isPending ? 0.5 : 1,
      transition: "opacity 0.3s",
    }}>
      <div
        onClick={() => result && setOpen(o => !o)}
        style={{
          padding: "12px 14px",
          display: "flex",
          alignItems: "center",
          gap: 10,
          cursor: result ? "pointer" : "default",
          userSelect: "none",
          background: "var(--color-background-primary)",
        }}
        onMouseEnter={e => { if (result) e.currentTarget.style.background = "var(--color-background-secondary)"; }}
        onMouseLeave={e => { if (result) e.currentTarget.style.background = "var(--color-background-primary)"; }}
      >
        <span style={{ flexShrink: 0, display: "flex", alignItems: "center" }}>{statusDot}</span>
        <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>
          {STEP_LABELS[stepKey]}
        </span>
        {result && <Badge cls={result.classification} small />}
        {result && (
          <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s", display: "inline-block" }}>▾</span>
        )}
      </div>

      {open && result && (
        <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: 14, background: "var(--color-background-primary)" }}>
          {result.summary && (
            <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6, margin: "0 0 12px" }}>
              {result.summary}
            </p>
          )}
          {result.reasoning && (
            <>
              <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-tertiary)", margin: "0 0 6px" }}>
                Reasoning
              </p>
              <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6, margin: "0 0 12px", whiteSpace: "pre-wrap" }}>
                {typeof result.reasoning === "string"
                  ? (() => {
                    try {
                      const parsed = JSON.parse(result.reasoning);
                      return parsed.reason || result.reasoning;
                    } catch { return result.reasoning; }
                  })()
                  : JSON.stringify(result.reasoning, null, 2)
                }
              </p>
            </>
          )}
          {result.metadata?.warnings?.length > 0 && (
            <div style={{ marginBottom: 10 }}>
              {result.metadata.warnings.map((w, i) => (
                <div key={i} style={{
                  fontSize: 12, color: "#854F0B", background: "#FAEEDA",
                  borderRadius: "var(--border-radius-md)", padding: "6px 10px", marginBottom: 4, lineHeight: 1.4,
                }}>
                  {w}
                </div>
              ))}
            </div>
          )}
          {result.data_used && Object.keys(result.data_used).length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>Data used</span>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setDataOpen(o => !o); }}
                  style={{
                    fontSize: 11,
                    color: "var(--color-text-tertiary)",
                    background: "var(--color-background-secondary)",
                    border: "0.5px solid var(--color-border-tertiary)",
                    borderRadius: "var(--border-radius-md)",
                    padding: "3px 8px",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  {dataOpen ? "Hide" : "Show"} sources
                </button>
              </div>
              {dataOpen && (
                <pre style={{
                  background: "var(--color-background-secondary)",
                  borderRadius: "var(--border-radius-md)",
                  padding: "10px 12px",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--color-text-secondary)",
                  overflowX: "auto",
                  maxHeight: 280,
                  lineHeight: 1.5,
                  margin: 0,
                }}>
                  {JSON.stringify(result.data_used, null, 2)}
                </pre>
              )}
            </div>
          )}
          {result.token_usage && Object.keys(result.token_usage).length > 0 && (
            <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 6 }}>
              {Object.entries(result.token_usage).map(([model, usage]) => (
                <span key={model} style={{
                  fontSize: 11,
                  color: "var(--color-text-tertiary)",
                  background: "var(--color-background-secondary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: "var(--border-radius-md)",
                  padding: "3px 8px",
                  fontFamily: "var(--font-mono)",
                }}>
                  {model}: {(usage.total_tokens || 0).toLocaleString()} tokens
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StrategyCard({ stratKey, strat }) {
  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: "var(--border-radius-md)",
      padding: 14,
    }}>
      <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 6 }}>
        {STRATEGY_LABELS[stratKey]}
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge
          cls={String(strat?.classification ?? "not_applicable")
            .toLowerCase()
            .replace(/\s+/g, "_")}
        />
      </div>
      <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.5, margin: "8px 0" }}>
        {strat?.key_evidence}
      </p>
      {strat?.caveats && strat.caveats !== "None" && (
        <div style={{
          borderTop: "0.5px solid var(--color-border-tertiary)",
          paddingTop: 8,
          marginTop: 4,
          fontSize: 12,
          color: "var(--color-text-tertiary)",
          lineHeight: 1.5,
        }}>
          {strat.caveats}
        </div>
      )}
    </div>
  );
}

function FinalReport({ report, onDownload }) {
  const summaryRaw = report.summary;
  const s = summaryRaw !== null && typeof summaryRaw === "object" && !Array.isArray(summaryRaw)
    ? summaryRaw
    : null;
  const summaryText = typeof summaryRaw === "string" ? summaryRaw : null;

  const steps = report.step_results ?? report.steps ?? {};
  const strategies =
    s?.strategy_assessments !== null && typeof s.strategy_assessments === "object"
      ? s.strategy_assessments
      : {};

  const meta = [
    {
      label: "Gene",
      val: report.gene_id || "—",
      mono: true,
    },
    {
      label: "Inheritance",
      val: steps.inheritance_pattern?.metadata?.inheritance_pattern?.replace(/_/g, " ") || "—",
    },
    {
      label: "Pathomechanism",
      val: steps.pathomechanism?.metadata?.pathomechanism?.replace(/_/g, " ") || "—",
    },
    {
      label: "Haploinsufficient",
      val: steps.pathomechanism?.metadata?.is_haploinsufficient ? "Yes" : "No",
    },
  ];

  const sectionMb = "1.25rem";
  const cardHeaderPad = "14px 16px 12px";
  const cardBodyPad = "14px 16px";

  return (
    <div>
      <div style={{ marginBottom: "1.5rem" }}>
        <span style={{
          display: "inline-block",
          background: "var(--color-background-secondary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderRadius: "var(--border-radius-md)",
          padding: "3px 10px",
          fontSize: 12,
          color: "var(--color-text-secondary)",
          marginBottom: 8,
          fontFamily: "var(--font-mono)",
        }}>
          {report.gene_id || "—"}
        </span>
        <h2 style={{ fontSize: 22, fontWeight: 500, color: "var(--color-text-primary)", margin: "0 0 4px" }}>
          {report.hgvs ?? "—"}
        </h2>
        <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0 }}>
          {s?.variant_description ?? "—"}
        </p>
        <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", margin: "8px 0 0" }}>
          Assessment date: {report.date || "—"} · Model: {report.model_name || "—"}
        </p>
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
        gap: 10,
        marginBottom: sectionMb,
      }}>
        {meta.map(m => (
          <div key={m.label} style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-md)",
            padding: "12px 14px",
          }}>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 4 }}>
              {m.label}
            </div>
            <div style={{
              fontSize: 13,
              fontWeight: 500,
              color: "var(--color-text-primary)",
              fontFamily: m.mono ? "var(--font-mono)" : "inherit",
            }}>
              {m.val}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginBottom: sectionMb }}>
        <div style={{
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
          overflow: "hidden",
        }}>
          <div style={{
            padding: cardHeaderPad,
            borderBottom: "0.5px solid var(--color-border-tertiary)",
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}>
            <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>Clinical summary</span>
          </div>
          <div style={{ padding: cardBodyPad }}>
            <p style={{ fontSize: 14, color: "var(--color-text-secondary)", lineHeight: 1.7, margin: "0 0 12px" }}>
              {summaryText ?? s?.overall_summary ?? "—"}
            </p>
            {s?.splicing_summary && (
              <>
                <div style={{ height: "0.5px", background: "var(--color-border-tertiary)", margin: "12px 0" }} />
                <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-tertiary)", margin: "0 0 6px" }}>
                  Splicing note
                </p>
                <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.65, margin: 0 }}>
                  {s.splicing_summary}
                </p>
              </>
            )}
          </div>
        </div>
      </div>

      {Object.keys(strategies).length > 0 && (
        <div style={{ marginBottom: sectionMb }}>
          <div style={{ fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 10 }}>
            Therapy strategy assessment
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10 }}>
            {Object.entries(strategies).map(([key, strat]) => (
              <StrategyCard key={key} stratKey={key} strat={strat} />
            ))}
          </div>
        </div>
      )}

      <div style={{ marginBottom: sectionMb }}>
        <div style={{ fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 10 }}>
          Pipeline steps
        </div>
        {STEP_ORDER.map(k => steps[k] && (
          <StepCard key={k} stepKey={k} result={steps[k]} />
        ))}
      </div>

      {s?.recommended_next_steps?.length > 0 && (
        <div style={{ marginBottom: sectionMb }}>
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            overflow: "hidden",
          }}>
            <div style={{ padding: cardHeaderPad, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
              <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>Recommended next steps</span>
            </div>
            <div style={{ padding: cardBodyPad }}>
              {s.recommended_next_steps.map((step, i) => (
                <div key={i} style={{
                  fontSize: 13,
                  color: "var(--color-text-secondary)",
                  padding: "6px 0",
                  borderBottom: i < s.recommended_next_steps.length - 1 ? "0.5px solid var(--color-border-tertiary)" : "none",
                  lineHeight: 1.5,
                  display: "flex",
                  gap: 8,
                }}>
                  <span style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary)", minWidth: 18, paddingTop: 2 }}>
                    {i + 1}.
                  </span>
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {s?.important_caveats?.length > 0 && (
        <div style={{ marginBottom: sectionMb }}>
          <div style={{ fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 10 }}>
            Important caveats
          </div>
          {s.important_caveats.map((c, i) => (
            <div key={i} style={{
              fontSize: 12,
              color: "#854F0B",
              background: "#FAEEDA",
              borderRadius: "var(--border-radius-md)",
              padding: "6px 10px",
              marginBottom: 4,
              lineHeight: 1.4,
            }}>
              {c}
            </div>
          ))}
        </div>
      )}

      <button
        type="button"
        onClick={onDownload}
        style={{
          marginTop: 8,
          display: "inline-flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 16px",
          borderRadius: "var(--border-radius-md)",
          background: "var(--color-background-primary)",
          color: "var(--color-text-primary)",
          border: "0.5px solid var(--color-border-secondary)",
          fontSize: 13,
          cursor: "pointer",
        }}
        onMouseOver={e => { e.currentTarget.style.background = "var(--color-background-secondary)"; }}
        onMouseOut={e => { e.currentTarget.style.background = "var(--color-background-primary)"; }}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
          <path d="M7 1v8M4 6l3 3 3-3M2 11h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Download report JSON
      </button>
    </div>
  );
}

export default function App() {
  const [refSeq, setRefSeq] = useState("NM_000329.3");
  const [codingChange, setCodingChange] = useState("c.1430A>G");
  const [useWebSearch, setUseWebSearch] = useState(false);
  const [verbose, setVerbose] = useState(false);

  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [stepStatuses, setStepStatuses] = useState({});
  const [stepResults, setStepResults] = useState({});
  const [currentStep, setCurrentStep] = useState(null);
  const [completedCount, setCompletedCount] = useState(0);
  const [finalReport, setFinalReport] = useState(null);
  const [error, setError] = useState(null);
  const [log, setLog] = useState([]);

  const abortRef = useRef(null);
  const reportEndRef = useRef(null);

  const addLog = (msg) => setLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const hgvs = `${refSeq}:${codingChange}`;

  const runPipeline = useCallback(async () => {
    setPhase("running");
    setStepStatuses({});
    setStepResults({});
    setCurrentStep(null);
    setCompletedCount(0);
    setFinalReport(null);
    setError(null);
    setLog([]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      addLog(`Starting pipeline for ${hgvs}`);

      let ctx = null;
      const allStepResults = {};

      for (let i = 0; i < STEP_ORDER.length; i++) {
        const step = STEP_ORDER[i];

        if (controller.signal.aborted) break;

        setCurrentStep(step);
        setStepStatuses(prev => ({ ...prev, [step]: "running" }));
        addLog(`Running step: ${STEP_LABELS[step]}`);

        const body = {
          hgvs,
          context: ctx,
          options: {
            use_web_search: useWebSearch,
            verbose,
          },
        };

        const res = await fetch(`${API_BASE}/assessment/steps/${step}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`Step ${step} failed (${res.status}): ${errText}`);
        }

        const data = await res.json();
        ctx = data.context;
        const result = data.step_result;
        allStepResults[step] = result;

        setStepResults(prev => ({ ...prev, [step]: result }));
        setStepStatuses(prev => ({ ...prev, [step]: "done" }));
        setCompletedCount(i + 1);
        addLog(`Completed: ${STEP_LABELS[step]} → ${result.classification}`);
      }

      if (!controller.signal.aborted) {
        setCurrentStep("final_report");
        addLog("Generating final report...");

        const finalBody = {
          hgvs,
          context: ctx,
          step_results: allStepResults,
          options: { use_web_search: useWebSearch, verbose },
        };

        const finalRes = await fetch(`${API_BASE}/assessment/steps/final_report`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(finalBody),
          signal: controller.signal,
        });

        if (!finalRes.ok) {
          const errText = await finalRes.text();
          throw new Error(`Final report failed (${finalRes.status}): ${errText}`);
        }

        const reportData = await finalRes.json();
        setFinalReport(reportData);
        setPhase("done");
        setCurrentStep(null);
        addLog("Assessment complete.");
        setTimeout(() => reportEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        addLog("Pipeline cancelled.");
        setPhase("idle");
      } else {
        setError(err.message);
        setPhase("error");
        addLog(`Error: ${err.message}`);
      }
    }
  }, [hgvs, useWebSearch, verbose]);

  const cancel = () => {
    abortRef.current?.abort();
  };

  const reset = () => {
    setPhase("idle");
    setStepStatuses({});
    setStepResults({});
    setCurrentStep(null);
    setCompletedCount(0);
    setFinalReport(null);
    setError(null);
    setLog([]);
  };

  const downloadReport = () => {
    const blob = new Blob([JSON.stringify(finalReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aso-report-${hgvs.replace(/[^a-zA-Z0-9]/g, "_")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isRunning = phase === "running";
  const isDone = phase === "done";

  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>

      <div>

        <div style={{ marginBottom: "1.5rem" }}>
          <h1 style={{ fontSize: 22, fontWeight: 500, margin: "0 0 4px", color: "var(--color-text-primary)" }}>
            N1C Variant ASO Assessor
          </h1>
          <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0, lineHeight: 1.5 }}>
            LLM-assisted tool for assessing variants for ASO eligibility according to the <a href="https://doi.org/10.1016/j.ajhg.2025.02.017" target="_blank" rel="noopener noreferrer" style={{ color: "blue" }}>N1C VARIANT</a> guidelines.
          </p>
        </div>

        {(phase === "idle" || phase === "error") && (
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "14px 16px",
            marginBottom: "1.25rem",
            animation: "fadeSlideIn 0.3s ease",
          }}>
            <div style={{ fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 10 }}>
              Variant input
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10, marginBottom: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 5 }}>
                  Reference sequence
                </label>
                <input
                  value={refSeq}
                  onChange={e => setRefSeq(e.target.value)}
                  placeholder="e.g. NM_000329.3"
                  style={{
                    width: "100%",
                    boxSizing: "border-box",
                    padding: "8px 12px",
                    fontSize: 13,
                    fontFamily: "var(--font-mono)",
                    border: "0.5px solid var(--color-border-secondary)",
                    borderRadius: "var(--border-radius-md)",
                    background: "var(--color-background-primary)",
                    color: "var(--color-text-primary)",
                    outline: "none",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 5 }}>
                  Coding change (HGVS)
                </label>
                <input
                  value={codingChange}
                  onChange={e => setCodingChange(e.target.value)}
                  placeholder="e.g. c.1430A>G"
                  style={{
                    width: "100%",
                    boxSizing: "border-box",
                    padding: "8px 12px",
                    fontSize: 13,
                    fontFamily: "var(--font-mono)",
                    border: "0.5px solid var(--color-border-secondary)",
                    borderRadius: "var(--border-radius-md)",
                    background: "var(--color-background-primary)",
                    color: "var(--color-text-primary)",
                    outline: "none",
                  }}
                />
              </div>
            </div>

            <div style={{
              display: "inline-block",
              background: "var(--color-background-secondary)",
              border: "0.5px solid var(--color-border-secondary)",
              borderRadius: "var(--border-radius-md)",
              padding: "3px 10px",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              color: "var(--color-text-secondary)",
              marginBottom: "1rem",
            }}>
              {hgvs}
            </div>

            {/* <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
              {[
                { key: "useWebSearch", val: useWebSearch, set: setUseWebSearch, label: "Web search", desc: "Augment with live literature" },
                { key: "verbose", val: verbose, set: setVerbose, label: "Verbose mode", desc: "Extended LLM reasoning" },
              ].map(opt => (
                <label key={opt.key} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                  <div
                    onClick={() => opt.set(v => !v)}
                    style={{
                      width: 32, height: 18, borderRadius: 9,
                      background: opt.val ? "#0F6E56" : "var(--color-border-secondary)",
                      position: "relative", cursor: "pointer", flexShrink: 0,
                      transition: "background 0.2s",
                    }}
                  >
                    <div style={{
                      position: "absolute", top: 2, left: opt.val ? 16 : 2,
                      width: 14, height: 14, borderRadius: "50%", background: "white",
                      transition: "left 0.2s",
                    }} />
                  </div>
                  <div>
                    <div style={{ fontSize: 12.5, fontWeight: 500, color: "var(--color-text-primary)" }}>{opt.label}</div>
                    <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{opt.desc}</div>
                  </div>
                </label>
              ))}
            </div> */}

            {error && (
              <div style={{
                background: "#FCEBEB",
                color: "#A32D2D",
                border: "0.5px solid rgba(163, 45, 45, 0.25)",
                borderRadius: "var(--border-radius-md)",
                padding: "8px 12px",
                fontSize: 12,
                marginBottom: "1rem",
                lineHeight: 1.5,
              }}>
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={runPipeline}
              disabled={!refSeq.trim() || !codingChange.trim()}
              style={{
                padding: "8px 16px",
                borderRadius: "var(--border-radius-md)",
                background: "var(--color-background-primary)",
                color: "var(--color-text-primary)",
                border: "0.5px solid var(--color-border-secondary)",
                fontSize: 13,
                cursor: (!refSeq.trim() || !codingChange.trim()) ? "not-allowed" : "pointer",
                opacity: (!refSeq.trim() || !codingChange.trim()) ? 0.5 : 1,
              }}
              onMouseOver={e => {
                if (!refSeq.trim() || !codingChange.trim()) return;
                e.currentTarget.style.background = "var(--color-background-secondary)";
              }}
              onMouseOut={e => {
                e.currentTarget.style.background = "var(--color-background-primary)";
              }}
            >
              Run assessment
            </button>
          </div>
        )}

        {/* Running state */}
        {isRunning && (
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: "var(--border-radius-lg)",
            padding: "14px 16px",
            marginBottom: "1.25rem",
            animation: "fadeSlideIn 0.3s ease",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Spinner size={15} />
                <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>
                  Running pipeline
                </span>
              </div>
              <button
                type="button"
                onClick={cancel}
                style={{
                  fontSize: 11,
                  color: "var(--color-text-tertiary)",
                  background: "var(--color-background-secondary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: "var(--border-radius-md)",
                  padding: "3px 8px",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                Cancel
              </button>
            </div>

            <div style={{ marginBottom: 10 }}>
              <ProgressBar current={completedCount} total={STEP_ORDER.length} />
            </div>

            <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 14, lineHeight: 1.5 }}>
              {completedCount}/{STEP_ORDER.length} steps complete
              {currentStep && currentStep !== "final_report" && (
                <span style={{ color: "var(--color-text-tertiary)" }}>
                  {" "}· {STEP_LABELS[currentStep]}
                  <span style={{ animation: "pulse 1.2s ease-in-out infinite", display: "inline-block", marginLeft: 4 }}>…</span>
                </span>
              )}
              {currentStep === "final_report" && (
                <span style={{ color: "var(--color-text-tertiary)" }}>
                  {" "}· Synthesizing final report
                  <span style={{ animation: "pulse 1.2s ease-in-out infinite", display: "inline-block", marginLeft: 4 }}>…</span>
                </span>
              )}
            </div>

            <div>
              {STEP_ORDER.map((step) => {
                const result = stepResults[step];
                const isActive = currentStep === step;
                const isPendingStep = !stepStatuses[step];

                return (
                  <StepCard
                    key={step}
                    stepKey={step}
                    result={result}
                    isRunning={isActive}
                    isPending={isPendingStep && !isActive}
                  />
                );
              })}
            </div>

            {/* Log */}
            <div style={{
              marginTop: 12,
              background: "var(--color-background-secondary)",
              borderRadius: "var(--border-radius-md)",
              padding: "10px 12px",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--color-text-secondary)",
              maxHeight: 100,
              overflowY: "auto",
              lineHeight: 1.5,
            }}>
              {log.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          </div>
        )}

        {/* Done state - show final report */}
        {isDone && finalReport && (
          <div style={{ animation: "fadeSlideIn 0.4s ease" }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "1rem",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--color-text-secondary)", display: "inline-block" }} />
                <span style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>
                  Assessment complete
                </span>
              </div>
              <button
                type="button"
                onClick={reset}
                style={{
                  fontSize: 13,
                  color: "var(--color-text-primary)",
                  background: "var(--color-background-primary)",
                  border: "0.5px solid var(--color-border-secondary)",
                  borderRadius: "var(--border-radius-md)",
                  padding: "8px 16px",
                  cursor: "pointer",
                }}
                onMouseOver={e => { e.currentTarget.style.background = "var(--color-background-secondary)"; }}
                onMouseOut={e => { e.currentTarget.style.background = "var(--color-background-primary)"; }}
              >
                New assessment
              </button>
            </div>

            <div style={{
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-lg)",
              padding: "14px 16px",
            }}>
              <FinalReport report={finalReport} onDownload={downloadReport} />
            </div>
          </div>
        )}

        <div ref={reportEndRef} />
      </div>
    </>
  );
}