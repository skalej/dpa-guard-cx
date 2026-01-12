export default function Home() {
  return (
    <main className="container">
      <h1>DPA Guard</h1>
      <p>Start a new review or browse recent reviews.</p>
      <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
        <a href="/reviews/new">New Review</a>
        <a href="/reviews">Reviews</a>
      </div>
    </main>
  );
}
