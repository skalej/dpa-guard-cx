"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

import {
  ExplainFinding,
  ReviewItem,
  ReviewResults,
  exportPdf,
  getReview,
  getReviewExplain,
  getReviewResults,
  rememberReviewId,
} from "../../../lib/api";

export default function ReviewDetailPage() {
  const params = useParams<{ id: string }>();
  const reviewId = params?.id;
  const [review, setReview] = useState<ReviewItem | null>(null);
  const [results, setResults] = useState<ReviewResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [explainCache, setExplainCache] = useState<
    Record<string, ExplainFinding>
  >({});
  const [loadingExplain, setLoadingExplain] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [exportKey, setExportKey] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  const statusLabel = useMemo(() => review?.status || "loading", [review]);

  useEffect(() => {
    if (!reviewId) return;
    rememberReviewId(reviewId);
  }, [reviewId]);

  useEffect(() => {
    if (!reviewId) return;
    let cancelled = false;

    async function loadReview() {
      try {
        setError(null);
        setLoading(true);
        for (;;) {
          const data = await getReview(reviewId);
          if (cancelled) return;
          setReview(data);
          if (data.status === "completed") {
            const resultsData = await getReviewResults(reviewId);
            if (!cancelled) {
              setResults(resultsData);
            }
            break;
          }
          if (data.status === "failed") {
            throw new Error(data.error_message || "Review failed");
          }
          await new Promise((resolve) => setTimeout(resolve, 1500));
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load review");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadReview();
    return () => {
      cancelled = true;
    };
  }, [reviewId]);

  async function explain(checkId: string) {
    if (!reviewId) return;
    if (explainCache[checkId]) return;
    setLoadingExplain(checkId);
    try {
      const data = await getReviewExplain(reviewId);
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
      const data = await exportPdf(reviewId);
      setExportUrl(data.url || null);
      setExportKey(data.object_key || null);
      if (data.url) {
        window.open(data.url, "_blank", "noopener,noreferrer");
      }
    } catch (err) {
      setExportError(
        err instanceof Error ? err.message : "Failed to generate PDF"
      );
    } finally {
      setExporting(false);
    }
  }

  return (
    <main className="container">
      <Link href="/reviews">← Back to reviews</Link>
      <h1 style={{ marginTop: 12 }}>Review {reviewId}</h1>

      {loading && <p>Loading review status…</p>}
      {error && <p style={{ color: "#b00020" }}>{error}</p>}

      {review && (
        <div style={{ marginTop: 12 }}>
          <div>Status: {statusLabel}</div>
          <div style={{ fontSize: 14, color: "#444" }}>
            File: {review.source_filename || "Pending upload"}
          </div>
          <div style={{ fontSize: 12, color: "#666" }}>
            Created: {review.created_at || "unknown"}
          </div>
        </div>
      )}

      {review?.status === "completed" && results && (
        <section style={{ marginTop: 28 }}>
          <h2>Executive Summary</h2>
          <p>{results.exec_summary || "No summary available."}</p>

          <div style={{ marginTop: 16 }}>
            <button
              type="button"
              onClick={generatePdf}
              disabled={exporting}
              style={{ padding: "10px 16px", fontWeight: 600 }}
            >
              {exporting ? "Generating PDF..." : "Download PDF"}
            </button>
            {exportUrl && (
              <div style={{ marginTop: 8, fontSize: 14 }}>
                <a href={exportUrl} target="_blank" rel="noreferrer">
                  Open PDF
                </a>
              </div>
            )}
            {exportKey && (
              <div style={{ marginTop: 6, fontSize: 12, color: "#555" }}>
                {exportKey}
              </div>
            )}
            {exportError && (
              <div style={{ marginTop: 8, color: "#b00020" }}>
                {exportError}. Link expired, generate again.
              </div>
            )}
          </div>

          <h2 style={{ marginTop: 24 }}>Risk Table</h2>
          <div style={{ display: "grid", gap: 12 }}>
            {results.risk_table?.map((item) => {
              const checkId = item.check_id || "";
              const explainData = checkId ? explainCache[checkId] : undefined;
              return (
                <div
                  key={checkId || item.title}
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
