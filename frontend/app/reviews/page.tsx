"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { ReviewItem, getReview, loadRecentReviewIds } from "../../lib/api";

export default function ReviewsPage() {
  const [reviews, setReviews] = useState<ReviewItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function loadReviews() {
      try {
        const ids = loadRecentReviewIds();
        if (ids.length === 0) {
          setReviews([]);
          setLoading(false);
          return;
        }
        const results = await Promise.all(
          ids.map(async (id) => {
            try {
              return await getReview(id);
            } catch {
              return { id, status: "unknown" } as ReviewItem;
            }
          })
        );
        if (!cancelled) {
          setReviews(results);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load reviews");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    loadReviews();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="container">
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h1>Reviews</h1>
        <Link href="/reviews/new" style={{ alignSelf: "center" }}>
          New Review
        </Link>
      </div>

      {loading && <p>Loading reviews...</p>}
      {error && <p style={{ color: "#b00020" }}>{error}</p>}
      {!loading && reviews.length === 0 && (
        <p>No local reviews yet. Create a new review to get started.</p>
      )}

      <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
        {reviews.map((review) => (
          <Link
            key={review.id}
            href={`/reviews/${review.id}`}
            style={{
              border: "1px solid #ddd",
              borderRadius: 10,
              padding: 16,
              background: "#fff",
              textDecoration: "none",
              color: "inherit",
            }}
          >
            <div style={{ fontWeight: 600 }}>{review.source_filename || review.id}</div>
            <div style={{ fontSize: 14, color: "#444", marginTop: 4 }}>
              Status: {review.status}
            </div>
            <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
              Created: {review.created_at || "unknown"}
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}
