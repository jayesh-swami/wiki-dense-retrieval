import { useEffect, useState } from "react";

// Retro arcade terminal over /api/search: type at the prompt, the top 10 passages print
// out one by one, and the HUD shows the Cohere embedding time and the retrieval time.
// ?q=... in the URL runs that search on load, so a search can be shared as a link.
export default function App() {
  const [query, setQuery] = useState(() => new URLSearchParams(location.search).get("q") ?? "");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | searching | done | error
  const [error, setError] = useState("");
  const [asked, setAsked] = useState("");

  async function run(q) {
    q = q.trim();
    if (!q || status === "searching") return;
    history.replaceState(null, "", `?q=${encodeURIComponent(q)}`);
    setStatus("searching");
    setAsked(q);
    setResult(null);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail);
      setResult(body);
      setStatus("done");
    } catch (err) {
      setError(String(err.message || err));
      setStatus("error");
    }
  }

  useEffect(() => {
    if (query) run(query);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- only the ?q= on first load

  const hud = (label, value) => (
    <span className="hud-item">
      {label} <b>{value}</b>
    </span>
  );

  return (
    <div className="crt">
      <main>
        <header>
          <h1>WIKI SEARCH</h1>
          <p className="sub">41,500,000 PASSAGES · BINARY HNSW + IVF-PQ · INT8 RESCORE FROM SSD</p>
        </header>

        <div className="hud" aria-live="polite">
          {hud("EMBED", result ? `${result.embed_ms} MS` : "---")}
          {hud("SEARCH", result ? `${result.search_ms} MS` : "---")}
          {hud("HITS", result ? String(result.passages.length).padStart(2, "0") : "00")}
        </div>

        <form onSubmit={(e) => { e.preventDefault(); run(query); }} className="prompt">
          <label htmlFor="q">wiki&gt;</label>
          <input
            id="q"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="type a question and press enter"
            autoComplete="off"
            spellCheck="false"
            autoFocus
          />
        </form>

        <section className="output">
          {status === "idle" && <p className="blink">PRESS ENTER TO START</p>}
          {status === "searching" && <p className="blink">SEARCHING 41.5M PASSAGES...</p>}
          {status === "error" && <p className="err">ERROR: {error}</p>}
          {status === "done" && (
            <>
              <p className="echo">&gt; {asked}</p>
              <ol>
                {result.passages.map((p, i) => (
                  <li key={p.id} style={{ animationDelay: `${i * 60}ms` }}>
                    <span className="rank">[{String(i + 1).padStart(2, "0")}]</span>
                    <span className="pid">passage {p.id}</span>
                    <p>{p.text}</p>
                  </li>
                ))}
              </ol>
              <p className="dim">-- END OF RESULTS --</p>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
