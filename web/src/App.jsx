import { useState } from "react";

// One search box over /api/search: shows the Cohere embedding time, the retrieval
// time and the top 10 passages.
export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");

  async function onSubmit(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setStatus("Searching...");
    setResult(null);
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    if (!res.ok) {
      setStatus(`Error: ${(await res.json()).detail}`);
      return;
    }
    setResult(await res.json());
    setStatus("");
  }

  return (
    <main>
      <h1>Wikipedia search</h1>
      <p className="sub">
        41.5M English Wikipedia passages. Binary HNSW + IVF-PQ union, int8 rescore from SSD.
      </p>
      <form onSubmit={onSubmit}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question" autoFocus />
        <button>Search</button>
      </form>
      <p className="stats">
        {result ? `Embedding (Cohere API) ${result.embed_ms} ms · retrieval ${result.search_ms} ms` : status}
      </p>
      {result && (
        <ol>
          {result.passages.map((p) => (
            <li key={p.id}>
              <small>passage {p.id}</small>
              {p.text}
            </li>
          ))}
        </ol>
      )}
    </main>
  );
}
