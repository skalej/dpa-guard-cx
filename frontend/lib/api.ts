export type PlaybookItem = {
  id: string;
  title?: string | null;
  version?: string | number | null;
  status?: string | null;
  error_message?: string | null;
};

export type ReviewItem = {
  id: string;
  status: string;
  playbook_id?: string | null;
  source_filename?: string | null;
  source_object_key?: string | null;
  source_mime?: string | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type ReviewResults = {
  exec_summary?: string;
  risk_table?: Array<{
    check_id?: string;
    title?: string;
    severity?: string;
    recommendation?: string;
    what_good_looks_like?: string;
  }>;
  negotiation_pack?: Array<{
    check_id?: string;
    title?: string;
    severity?: string;
    ask?: string;
    fallback?: string;
    rationale?: string;
  }>;
};

export type ExplainFinding = {
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

export type ExplainResponse = {
  review_id: string;
  status: string;
  playbook_id: string | null;
  findings: ExplainFinding[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

const RECENT_REVIEWS_KEY = "dpa_guard_recent_reviews";

export function rememberReviewId(id: string) {
  if (typeof window === "undefined") return;
  try {
    const existing = JSON.parse(
      window.localStorage.getItem(RECENT_REVIEWS_KEY) || "[]"
    ) as string[];
    const next = [id, ...existing.filter((item) => item !== id)].slice(0, 20);
    window.localStorage.setItem(RECENT_REVIEWS_KEY, JSON.stringify(next));
  } catch {
    window.localStorage.removeItem(RECENT_REVIEWS_KEY);
  }
}

export function loadRecentReviewIds(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const existing = JSON.parse(
      window.localStorage.getItem(RECENT_REVIEWS_KEY) || "[]"
    ) as string[];
    return Array.isArray(existing) ? existing : [];
  } catch {
    return [];
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export async function listPlaybooks(): Promise<PlaybookItem[]> {
  const data = await request<{ playbooks: PlaybookItem[] }>("/playbook/versions");
  return data.playbooks || [];
}

export async function createReview(playbookId?: string): Promise<{ id: string }> {
  const payload: Record<string, unknown> = { context: {} };
  if (playbookId) {
    payload.playbook_id = playbookId;
  }
  return request<{ id: string }>("/reviews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function uploadReviewFile(reviewId: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  await request(`/reviews/${reviewId}/upload`, { method: "POST", body: form });
}

export async function startReview(reviewId: string): Promise<void> {
  await request(`/reviews/${reviewId}/start`, { method: "POST" });
}

export async function getReview(reviewId: string): Promise<ReviewItem> {
  return request<ReviewItem>(`/reviews/${reviewId}`);
}

export async function getReviewResults(reviewId: string): Promise<ReviewResults> {
  return request<ReviewResults>(`/reviews/${reviewId}/results`);
}

export async function getReviewExplain(reviewId: string): Promise<ExplainResponse> {
  return request<ExplainResponse>(`/reviews/${reviewId}/explain`);
}

export async function exportPdf(reviewId: string): Promise<{
  url?: string;
  object_key?: string;
}> {
  return request(`/reviews/${reviewId}/export/pdf`);
}
