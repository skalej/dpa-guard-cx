"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import {
  PlaybookItem,
  createReview,
  listPlaybooks,
  rememberReviewId,
  startReview,
  uploadReviewFile,
} from "../../../lib/api";

export default function NewReviewPage() {
  const router = useRouter();
  const [playbooks, setPlaybooks] = useState<PlaybookItem[]>([]);
  const [playbooksError, setPlaybooksError] = useState<string | null>(null);
  const [playbookId, setPlaybookId] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [error, setError] = useState<string | null>(null);

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
        const data = await listPlaybooks();
        if (!cancelled) {
          setPlaybooks(data || []);
        }
      } catch (err) {
        if (!cancelled) {
          setPlaybooksError(
            err instanceof Error ? err.message : "Failed to load playbooks"
          );
        }
      }
    }
    loadPlaybooks();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Please select a PDF file.");
      return;
    }
    setError(null);
    setStatus("creating");
    try {
      const data = await createReview(playbookId.trim() || undefined);
      const id = data.id;
      rememberReviewId(id);
      setStatus("uploading");
      await uploadReviewFile(id, file);
      setStatus("starting");
      await startReview(id);
      router.push(`/reviews/${id}`);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Something went wrong");
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
          ) : (
            <input
              type="text"
              placeholder="Optional playbook ID"
              value={playbookId}
              onChange={(event) => setPlaybookId(event.target.value)}
              style={{ padding: 8, width: "100%", marginTop: 6 }}
            />
          )}
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
          {status === "error" && "Retry"}
        </button>
        {status !== "idle" && status !== "error" && (
          <div style={{ marginTop: 10, color: "#444" }}>
            {status === "creating" && "Creating review..."}
            {status === "uploading" && "Uploading document..."}
            {status === "starting" && "Starting analysis..."}
          </div>
        )}
      </form>

      {error && (
        <div style={{ marginTop: 16, color: "#b00020" }}>{error}</div>
      )}
    </main>
  );
}
