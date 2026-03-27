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
  eligible: { label: "Eligible", bg: "#E1F5EE", color: "#0F6E56", dot: "#1D9E75" },
  likely_eligible: { label: "Likely eligible", bg: "#EAF3DE", color: "#3B6D11", dot: "#639922" },
  not_eligible: { label: "Not eligible", bg: "#FCEBEB", color: "#A32D2D", dot: "#E24B4A" },
  unable_to_assess: { label: "Unable to assess", bg: "#FAEEDA", color: "#854F0B", dot: "#BA7517" },
  not_applicable: { label: "Not applicable", bg: "#F1EFE8", color: "#5F5E5A", dot: "#888780" },
  unlikely_eligible: { label: "Unlikely eligible", bg: "#FBEAF0", color: "#993556", dot: "#D4537E" },
};

function Badge({ cls, small }) {
  const conf = CLS_CONFIG[cls] || CLS_CONFIG["not_applicable"];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      background: conf.bg, color: conf.color,
      fontSize: small ? 11 : 12, fontWeight: 500,
      padding: small ? "2px 7px" : "3px 9px",
      borderRadius: 20, whiteSpace: "nowrap",
      fontFamily: "'DM Sans', sans-serif",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: conf.dot, flexShrink: 0 }} />
      {conf.label}
    </span>
  );
}

function Spinner({ size = 16 }) {
  return (
    <span style={{
      display: "inline-block", width: size, height: size,
      border: `2px solid var(--color-border-tertiary)`,
      borderTopColor: "#178A6B",
      borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
    }} />
  );
}

function ProgressBar({ current, total }) {
  const pct = total === 0 ? 0 : Math.round((current / total) * 100);
  return (
    <div style={{ width: "100%", height: 3, background: "var(--color-background-tertiary)", borderRadius: 4, overflow: "hidden" }}>
      <div style={{
        height: "100%", width: `${pct}%`,
        background: "linear-gradient(90deg, #0F6E56, #1D9E75)",
        borderRadius: 4,
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
      borderRadius: 10, overflow: "hidden", marginBottom: 6,
      background: "var(--color-background-primary)",
      opacity: isPending ? 0.5 : 1,
      transition: "opacity 0.3s",
    }}>
      <div
        onClick={() => result && setOpen(o => !o)}
        style={{
          padding: "11px 14px", display: "flex", alignItems: "center",
          gap: 10, cursor: result ? "pointer" : "default",
          userSelect: "none",
        }}
      >
        <span style={{ flexShrink: 0, display: "flex", alignItems: "center" }}>{statusDot}</span>
        <span style={{ flex: 1, fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)", fontFamily: "'DM Sans', sans-serif" }}>
          {STEP_LABELS[stepKey]}
        </span>
        {result && <Badge cls={result.classification} small />}
        {result && (
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s", display: "inline-block" }}>▾</span>
        )}
      </div>

      {open && result && (
        <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", padding: "12px 14px" }}>
          {result.summary && (
            <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6, marginBottom: 10 }}>
              {result.summary}
            </p>
          )}
          {result.reasoning && (
            <>
              <p style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary)", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Reasoning
              </p>
              <p style={{ fontSize: 12.5, color: "var(--color-text-secondary)", lineHeight: 1.65, marginBottom: 10, whiteSpace: "pre-wrap" }}>
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
                  borderRadius: 6, padding: "5px 9px", marginBottom: 4, lineHeight: 1.45,
                }}>
                  {w}
                </div>
              ))}
            </div>
          )}
          {result.data_used && Object.keys(result.data_used).length > 0 && (
            <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 10, marginTop: 6 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>Source data</span>
                <button
                  onClick={() => setDataOpen(o => !o)}
                  style={{
                    fontSize: 11, color: "var(--color-text-secondary)",
                    background: "var(--color-background-secondary)",
                    border: "0.5px solid var(--color-border-tertiary)",
                    borderRadius: 6, padding: "2px 7px", cursor: "pointer",
                  }}
                >
                  {dataOpen ? "Hide" : "Show"} raw data
                </button>
              </div>
              {dataOpen && (
                <pre style={{
                  background: "var(--color-background-secondary)",
                  borderRadius: 8, padding: "8px 10px",
                  fontFamily: "'DM Mono', monospace", fontSize: 10.5,
                  color: "var(--color-text-secondary)",
                  overflowX: "auto", maxHeight: 260, lineHeight: 1.5,
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
                  fontSize: 10.5, color: "var(--color-text-tertiary)",
                  background: "var(--color-background-secondary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: 5, padding: "2px 7px",
                  fontFamily: "'DM Mono', monospace",
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
      borderRadius: 10, padding: 14,
    }}>
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 7, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {STRATEGY_LABELS[stratKey]}
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge
          cls={String(strat?.classification ?? "not_applicable")
            .toLowerCase()
            .replace(/\s+/g, "_")}
        />
      </div>
      <p style={{ fontSize: 12.5, color: "var(--color-text-secondary)", lineHeight: 1.55, marginBottom: 8 }}>
        {strat?.key_evidence}
      </p>
      {strat?.caveats && strat.caveats !== "None" && (
        <div style={{
          borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 8, marginTop: 4,
          fontSize: 12, color: "var(--color-text-tertiary)", lineHeight: 1.5,
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

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <div style={{
          display: "inline-block",
          background: "var(--color-background-secondary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderRadius: 7, padding: "2px 10px",
          fontSize: 11.5, color: "var(--color-text-secondary)",
          fontFamily: "'DM Mono', monospace", marginBottom: 8,
        }}>
          {report.hgvs}
        </div>
        <h2 style={{ fontSize: 19, fontWeight: 600, color: "var(--color-text-primary)", margin: "0 0 4px", fontFamily: "'DM Serif Display', serif" }}>
          {s?.variant_description ?? report.hgvs ?? "—"}
        </h2>
        <p style={{ fontSize: 13, color: "var(--color-text-tertiary)", margin: 0 }}>
          Assessment date: {report.date || "—"} · Model: {report.model_name || "—"}
        </p>
      </div>

      {/* Meta grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 16 }}>
        {meta.map(m => (
          <div key={m.label} style={{
            background: "var(--color-background-secondary)",
            borderRadius: 9, padding: "10px 12px",
            border: "0.5px solid var(--color-border-tertiary)",
          }}>
            <div style={{ fontSize: 10.5, color: "var(--color-text-tertiary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 500 }}>
              {m.label}
            </div>
            <div style={{
              fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)",
              fontFamily: m.mono ? "'DM Mono', monospace" : "'DM Sans', sans-serif",
            }}>
              {m.val}
            </div>
          </div>
        ))}
      </div>

      {/* Overall summary */}
      <div style={{
        background: "var(--color-background-primary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: 11, marginBottom: 16, overflow: "hidden",
      }}>
        <div style={{ padding: "11px 15px 10px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)" }}>Clinical summary</span>
        </div>
        <div style={{ padding: "12px 15px" }}>
          <p style={{ fontSize: 13.5, color: "var(--color-text-secondary)", lineHeight: 1.7, marginBottom: 12 }}>
            {summaryText ?? s?.overall_summary ?? "—"}
          </p>
          {s?.splicing_summary && (
            <>
              <div style={{ height: 1, background: "var(--color-border-tertiary)", marginBottom: 12 }} />
              <p style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary)", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Splicing note
              </p>
              <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.65, margin: 0 }}>
                {s.splicing_summary}
              </p>
            </>
          )}
        </div>
      </div>

      {/* Strategy grid */}
      {Object.keys(strategies).length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 10 }}>
            Therapy strategy assessment
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {Object.entries(strategies).map(([key, strat]) => (
              <StrategyCard key={key} stratKey={key} strat={strat} />
            ))}
          </div>
        </div>
      )}

      {/* Pipeline steps */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 10 }}>
          Pipeline step details
        </div>
        {STEP_ORDER.map(k => steps[k] && (
          <StepCard key={k} stepKey={k} result={steps[k]} />
        ))}
      </div>

      {/* Next steps */}
      {s?.recommended_next_steps?.length > 0 && (
        <div style={{
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: 11, marginBottom: 16, overflow: "hidden",
        }}>
          <div style={{ padding: "11px 15px 10px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)" }}>Recommended next steps</span>
          </div>
          <div style={{ padding: "10px 15px" }}>
            {s.recommended_next_steps.map((step, i) => (
              <div key={i} style={{
                display: "flex", gap: 10, padding: "7px 0",
                borderBottom: i < (s.recommended_next_steps?.length ?? 0) - 1 ? "0.5px solid var(--color-border-tertiary)" : "none",
              }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--color-text-tertiary)", minWidth: 18, paddingTop: 2, fontFamily: "'DM Mono', monospace" }}>
                  {i + 1}.
                </span>
                <span style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.55 }}>
                  {step}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Caveats */}
      {s?.important_caveats?.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 8 }}>
            Important caveats
          </div>
          {s.important_caveats.map((c, i) => (
            <div key={i} style={{
              fontSize: 12.5, color: "#854F0B",
              background: "#FAEEDA", borderRadius: 7,
              padding: "6px 10px", marginBottom: 5, lineHeight: 1.45,
            }}>
              {c}
            </div>
          ))}
        </div>
      )}

      <button
        onClick={onDownload}
        style={{
          marginTop: 18, display: "flex", alignItems: "center", gap: 8,
          padding: "9px 18px", borderRadius: 9,
          background: "#0F6E56", color: "#E1F5EE",
          border: "none", fontSize: 13, fontWeight: 500,
          cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
          transition: "background 0.15s",
        }}
        onMouseOver={e => e.currentTarget.style.background = "#085041"}
        onMouseOut={e => e.currentTarget.style.background = "#0F6E56"}
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
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
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        body { font-family: 'DM Sans', sans-serif; }
      `}</style>

      <div style={{ padding: "1.5rem 0", fontFamily: "'DM Sans', sans-serif" }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{
              width: 30, height: 30, borderRadius: 8,
              background: "#0F6E56", display: "flex", alignItems: "center", justifyContent: "center",
              flexShrink: 0,
            }}>
              <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                <path d="M7.5 1.5C4.186 1.5 1.5 4.186 1.5 7.5s2.686 6 6 6 6-2.686 6-6-2.686-6-6-6zm0 2a1.5 1.5 0 110 3 1.5 1.5 0 010-3zm0 9a4.5 4.5 0 01-3.897-6.75A3 3 0 007.5 9a3 3 0 003.897-3.25A4.5 4.5 0 017.5 12.5z" fill="#9FE1CB" />
              </svg>
            </div>
            <h1 style={{ fontSize: 18, fontWeight: 600, margin: 0, color: "var(--color-text-primary)", fontFamily: "'DM Sans', sans-serif" }}>
              N1C Variant ASO Assessor
            </h1>
          </div>
          <p style={{ fontSize: 13, color: "var(--color-text-tertiary)", margin: 0, paddingLeft: 40 }}>
            Multi-step pipeline assessing antisense oligonucleotide therapy eligibility for a given genetic variant
          </p>
        </div>

        {/* Input form */}
        {(phase === "idle" || phase === "error") && (
          <div style={{
            background: "var(--color-background-primary)",
            border: "0.5px solid var(--color-border-tertiary)",
            borderRadius: 12, padding: "18px 18px", marginBottom: 16,
            animation: "fadeSlideIn 0.3s ease",
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 14 }}>
              Variant input
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
              <div>
                <label style={{ display: "block", fontSize: 11.5, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Reference sequence
                </label>
                <input
                  value={refSeq}
                  onChange={e => setRefSeq(e.target.value)}
                  placeholder="e.g. NM_000329.3"
                  style={{
                    width: "100%", padding: "8px 11px", fontSize: 13,
                    fontFamily: "'DM Mono', monospace",
                    border: "0.5px solid var(--color-border-secondary)",
                    borderRadius: 8, background: "var(--color-background-secondary)",
                    color: "var(--color-text-primary)", outline: "none",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: 11.5, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Coding change (HGVS)
                </label>
                <input
                  value={codingChange}
                  onChange={e => setCodingChange(e.target.value)}
                  placeholder="e.g. c.1430A>G"
                  style={{
                    width: "100%", padding: "8px 11px", fontSize: 13,
                    fontFamily: "'DM Mono', monospace",
                    border: "0.5px solid var(--color-border-secondary)",
                    borderRadius: 8, background: "var(--color-background-secondary)",
                    color: "var(--color-text-primary)", outline: "none",
                  }}
                />
              </div>
            </div>

            <div style={{
              display: "inline-block",
              background: "var(--color-background-tertiary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: 7, padding: "4px 10px",
              fontSize: 11.5, fontFamily: "'DM Mono', monospace",
              color: "var(--color-text-secondary)", marginBottom: 14,
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
                background: "#FCEBEB", color: "#A32D2D",
                border: "0.5px solid #F7C1C1",
                borderRadius: 8, padding: "8px 12px",
                fontSize: 12.5, marginBottom: 12, lineHeight: 1.5,
              }}>
                {error}
              </div>
            )}

            <button
              onClick={runPipeline}
              disabled={!refSeq.trim() || !codingChange.trim()}
              style={{
                padding: "9px 20px", borderRadius: 9,
                background: "#0F6E56", color: "#E1F5EE",
                border: "none", fontSize: 13, fontWeight: 500,
                cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
                opacity: (!refSeq.trim() || !codingChange.trim()) ? 0.5 : 1,
                transition: "background 0.15s, opacity 0.15s",
              }}
              onMouseOver={e => e.currentTarget.style.background = "#085041"}
              onMouseOut={e => e.currentTarget.style.background = "#0F6E56"}
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
            borderRadius: 12, padding: "16px 18px", marginBottom: 16,
            animation: "fadeSlideIn 0.3s ease",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Spinner size={15} />
                <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)" }}>
                  Running pipeline
                </span>
              </div>
              <button
                onClick={cancel}
                style={{
                  fontSize: 12, color: "var(--color-text-tertiary)",
                  background: "none", border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: 6, padding: "3px 9px", cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>

            <div style={{ marginBottom: 10 }}>
              <ProgressBar current={completedCount} total={STEP_ORDER.length} />
            </div>

            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 14 }}>
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

            <div style={{ fontFamily: "'DM Mono', monospace" }}>
              {STEP_ORDER.map((step, i) => {
                const status = stepStatuses[step];
                const result = stepResults[step];
                const isActive = currentStep === step;
                const isDoneStep = status === "done";
                const isPendingStep = !status;

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
              background: "var(--color-background-tertiary)",
              borderRadius: 8, padding: "8px 10px",
              fontFamily: "'DM Mono', monospace", fontSize: 10.5,
              color: "var(--color-text-tertiary)", maxHeight: 100, overflowY: "auto",
              lineHeight: 1.6,
            }}>
              {log.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          </div>
        )}

        {/* Done state - show final report */}
        {isDone && finalReport && (
          <div style={{ animation: "fadeSlideIn 0.4s ease" }}>
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              marginBottom: 14,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#1D9E75", display: "inline-block" }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-text-primary)" }}>
                  Assessment complete
                </span>
              </div>
              <button
                onClick={reset}
                style={{
                  fontSize: 12, color: "var(--color-text-secondary)",
                  background: "none", border: "0.5px solid var(--color-border-secondary)",
                  borderRadius: 7, padding: "4px 11px", cursor: "pointer",
                  fontFamily: "'DM Sans', sans-serif",
                }}
              >
                New assessment
              </button>
            </div>

            <div style={{
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: 12, padding: "18px 18px",
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