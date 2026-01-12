"use client";

import { useEffect, useMemo, useState } from "react";

type PlaybookItem = {
  id: string;
  title?: string | null;
  version?: string | number | null;
  status?: string | null;
  error_message?: string | null;
};

type ReviewResult = {
  exec_summary?: string;
  risk_table?: Array<{
    check_id?: string;
    title?: string;
    severity?: string;
    recommendation?: string;
    what_good_looks_like?: string;
  }>;
};

type ExplainFinding = {
  check_id: string;
  title?: string;
  severity?: string;
  recommendation?: string;
  what_good_looks_like?: string;
  evidence_quotes?: Array<{
    quote: string;
    start_char: number;
    end_char: number;
    chunk_index: number;
  }>;
  playbook_guidance?: { heading?: string; content?: string };
  negotiation?: { ask?: string; fallback?: string; rationale?: string };
};

type ExplainResponse = {
  review_id: string;
  status: string;
  playbook_id: string | null;
  findings: ExplainFinding[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export default function NewReviewPage() {
  const [playbooks, setPlaybooks] = useState<PlaybookItem[]>([]);
  const [playbooksError, setPlaybooksError] = useState<string | null>(null);
  const [playbookId, setPlaybookId] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [reviewId, setReviewId] = useState<string | null>(null);
  const [results, setResults] = useState<ReviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingExplain, setLoadingExplain] = useState<string | null>(null);
  const [explainCache, setExplainCache] = useState<
    Record<string, ExplainFinding>
  >({});
  const [exporting, setExporting] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [exportKey, setExportKey] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  const readyPlaybooks = useMemo(
    () => playbooks.filter((item) => item.status === "ready" || !item.status),
    [playbooks]
  );

  const getPlaybookLabel = (item: PlaybookItem) => {
    const shortId = item.id ? item.id.slice(0, 8) : "unknown";
    const title = item.title && item.title.trim() ? item.title : `Playbook ${shortId}`;
    const version = item.version ? ` v${item.version}` : "";
    const statusLabel = item.status ? ` (${item.status})` : "";
    return `${title}${version}${statusLabel}`;
  };

  useEffect(() => {
    let cancelled = false;
    async function loadPlaybooks() {
      try {
        setPlaybooksError(null);
        const res = await fetch(`${API_BASE}/playbook/versions`);
        if (!res.ok) {
          throw new Error(`Failed to load playbooks (${res.status})`);
        }
        const data = await res.json();
        if (!cancelled) {
          setPlaybooks(data.playbooks || []);
        }
      } catch (err) {
        if (!cancelled) {
          setPlaybooksError(
            err instanceof Error ? err.message : "Failed to load playbooks"
          );
          setManualPlaybook(true);
        }
      }
    }
    loadPlaybooks();
    return () => {
      cancelled = true;
    };
  }, []);

  async function createReview(): Promise<string> {
    const payload: Record<string, unknown> = { context: {} };
    if (playbookId.trim()) {
      payload.playbook_id = playbookId.trim();
    }
    const res = await fetch(`${API_BASE}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`Create review failed (${res.status})`);
    }
    const data = await res.json();
    return data.id as string;
  }

  async function uploadFile(id: string, uploadFile: File) {
    const form = new FormData();
    form.append("file", uploadFile);
    const res = await fetch(`${API_BASE}/reviews/${id}/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      throw new Error(`Upload failed (${res.status})`);
    }
  }

  async function startReview(id: string) {
    const res = await fetch(`${API_BASE}/reviews/${id}/start`, {
      method: "POST",
    });
    if (!res.ok) {
      throw new Error(`Start failed (${res.status})`);
    }
  }

  async function pollUntilDone(id: string) {
    setStatus("processing");
    setResults(null);
    setExplainCache({});
    setExportUrl(null);
    setExportKey(null);
    setExportError(null);
    for (;;) {
      const res = await fetch(`${API_BASE}/reviews/${id}`);
      if (!res.ok) {
        throw new Error(`Status check failed (${res.status})`);
      }
      const data = await res.json();
      if (data.status === "completed") {
        setStatus("completed");
        const resultsRes = await fetch(`${API_BASE}/reviews/${id}/results`);
        if (!resultsRes.ok) {
          throw new Error(`Results fetch failed (${resultsRes.status})`);
        }
        setResults(await resultsRes.json());
        break;
      }
      if (data.status === "failed") {
        setStatus("failed");
        throw new Error(data.error_message || "Review failed");
      }
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  }

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Please select a PDF file.");
      return;
    }
    setError(null);
    setStatus("creating");
    try {
      const id = await createReview();
      setReviewId(id);
      setStatus("uploading");
      await uploadFile(id, file);
      setStatus("starting");
      await startReview(id);
      await pollUntilDone(id);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Something went wrong");
    }
  }

  async function explain(checkId: string) {
    if (!reviewId) return;
    if (explainCache[checkId]) return;
    setLoadingExplain(checkId);
    try {
      const res = await fetch(`${API_BASE}/reviews/${reviewId}/explain`);
      if (!res.ok) {
        throw new Error(`Explain failed (${res.status})`);
      }
      const data: ExplainResponse = await res.json();
      const match = data.findings.find((item) => item.check_id === checkId);
      if (match) {
        setExplainCache((prev) => ({ ...prev, [checkId]: match }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Explain failed");
    } finally {
      setLoadingExplain(null);
    }
  }

  async function generatePdf() {
    if (!reviewId) return;
    setExporting(true);
    setExportError(null);
    try {
      const res = await fetch(`${API_BASE}/reviews/${reviewId}/export/pdf`);
      if (!res.ok) {
        throw new Error(`Export failed (${res.status})`);
      }
      const data = await res.json();
      setExportUrl(data.url || null);
      setExportKey(data.object_key || null);
    } catch (err) {
      setExportError(
        err instanceof Error ? err.message : "Failed to generate PDF"
      );
    } finally {
      setExporting(false);
    }
  }

  async function copyExportUrl() {
    if (!exportUrl) return;
    try {
      await navigator.clipboard.writeText(exportUrl);
    } catch {
      setExportError("Could not copy link.");
    }
  }

  return (
    <main className="container">
      <h1>New Review</h1>
      <p>
        Upload a DPA, select a playbook, and generate a risk summary with
        evidence.
      </p>

      <form onSubmit={onSubmit} style={{ marginTop: 24 }}>
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: "block", fontWeight: 600 }}>
            Playbook
          </label>
          {playbooksError && (
            <div style={{ color: "#b00020", marginTop: 4 }}>
              {playbooksError}
            </div>
          )}
          {readyPlaybooks.length > 0 ? (
            <select
              value={playbookId}
              onChange={(event) => setPlaybookId(event.target.value)}
              style={{ padding: 8, width: "100%", marginTop: 6 }}
            >
              <option value="">(Optional) Select a playbook</option>
              {readyPlaybooks.map((item) => (
                <option key={item.id} value={item.id}>
                  {getPlaybookLabel(item)}
                </option>
              ))}
            </select>
          ) : null}
        </div>

        <div style={{ marginBottom: 16 }}>
          <label style={{ display: "block", fontWeight: 600 }}>
            DPA PDF
          </label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
            style={{ marginTop: 6 }}
          />
        </div>

        <button
          type="submit"
          disabled={status === "creating" || status === "uploading"}
          style={{ padding: "10px 16px", fontWeight: 600 }}
        >
          {status === "idle" && "Create Review"}
          {status === "creating" && "Creating..."}
          {status === "uploading" && "Uploading..."}
          {status === "starting" && "Starting..."}
          {status === "processing" && "Processing..."}
          {status === "completed" && "Completed"}
          {status === "error" && "Retry"}
        </button>
      </form>

      {reviewId && (
        <div style={{ marginTop: 16 }}>
          <strong>Review ID:</strong> {reviewId}
        </div>
      )}

      {error && (
        <div style={{ marginTop: 16, color: "#b00020" }}>{error}</div>
      )}

      {results && (
        <section style={{ marginTop: 32 }}>
          <h2>Executive Summary</h2>
          <p>{results.exec_summary || "No summary available."}</p>

          <div style={{ marginTop: 16 }}>
            <button
              type="button"
              onClick={generatePdf}
              disabled={exporting}
              style={{ padding: "10px 16px", fontWeight: 600 }}
            >
              {exporting ? "Generating PDF..." : "Generate PDF Report"}
            </button>
            {exportUrl && (
              <div style={{ marginTop: 12 }}>
                <a
                  href={exportUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{ marginRight: 12 }}
                >
                  Open PDF
                </a>
                <button type="button" onClick={copyExportUrl}>
                  Copy link
                </button>
              </div>
            )}
            {exportKey && (
              <div style={{ marginTop: 8, fontSize: 12, color: "#555" }}>
                {exportKey}
              </div>
            )}
            {exportError && (
              <div style={{ marginTop: 8, color: "#b00020" }}>
                {exportError} Link expired, generate again.
              </div>
            )}
          </div>

          <h2 style={{ marginTop: 24 }}>Risk Table</h2>
          <div style={{ display: "grid", gap: 12 }}>
            {results.risk_table?.map((item) => {
              const checkId = item.check_id || "";
              const explainData = explainCache[checkId];
              return (
                <div
                  key={checkId}
                  style={{
                    border: "1px solid #ddd",
                    borderRadius: 8,
                    padding: 16,
                    background: "#fff",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <div>
                      <div style={{ fontWeight: 600 }}>
                        {item.title || checkId}
                      </div>
                      <div style={{ fontSize: 14, color: "#444" }}>
                        Severity: {item.severity || "n/a"}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => explain(checkId)}
                      disabled={!checkId || loadingExplain === checkId}
                    >
                      {loadingExplain === checkId ? "Loading..." : "Explain"}
                    </button>
                  </div>

                  {explainData && (
                    <div style={{ marginTop: 12 }}>
                      <h3>Evidence</h3>
                      <ul>
                        {(explainData.evidence_quotes || []).map((quote, idx) => (
                          <li key={idx} style={{ marginBottom: 8 }}>
                            <blockquote style={{ margin: 0 }}>
                              {quote.quote}
                            </blockquote>
                          </li>
                        ))}
                      </ul>

                      {explainData.playbook_guidance && (
                        <>
                          <h3>Playbook Guidance</h3>
                          <div style={{ fontWeight: 600 }}>
                            {explainData.playbook_guidance.heading}
                          </div>
                          <p>{explainData.playbook_guidance.content}</p>
                        </>
                      )}

                      {explainData.negotiation && (
                        <>
                          <h3>Negotiation</h3>
                          <div>
                            <strong>Ask:</strong> {explainData.negotiation.ask}
                          </div>
                          <div>
                            <strong>Fallback:</strong>{" "}
                            {explainData.negotiation.fallback}
                          </div>
                          <div>
                            <strong>Rationale:</strong>{" "}
                            {explainData.negotiation.rationale}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}
    </main>
  );
}
