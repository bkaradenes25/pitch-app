import { useEffect, useState } from "react";
import { API, api } from "../api";
import { Results } from "../components/Results";
import { Scoreboard } from "../components/Scoreboard";
import { errMsg } from "../format";
import type { LiveGame, LiveSnapshot } from "../types";

export function Live() {
  const [games, setGames] = useState<LiveGame[]>([]);
  const [pk, setPk] = useState("");
  const [snap, setSnap] = useState<LiveSnapshot | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    api<LiveGame[]>("/live/games")
      .then(setGames)
      .catch((e: unknown) => setErr("Could not load games: " + errMsg(e)));
  }, []);

  useEffect(() => {
    if (!pk) return;
    setSnap(null);
    setErr("");
    const es = new EventSource(`${API}/live/${pk}/stream`);
    es.onmessage = (e) => setSnap(JSON.parse(e.data) as LiveSnapshot);
    es.onerror = () => setErr("Stream interrupted (game may not be live).");
    return () => es.close();
  }, [pk]);

  return (
    <>
      <div className="controls">
        <label>
          game
          <select value={pk} onChange={(e) => setPk(e.target.value)}>
            <option value="">Select a game…</option>
            {games.map((g) => (
              <option key={g.game_pk} value={g.game_pk}>
                {g.away} @ {g.home} ({g.status})
              </option>
            ))}
          </select>
        </label>
        {games.length === 0 && !err && <span className="muted">No games today. Use Sandbox for the demo.</span>}
      </div>
      {err && <p className="err">{err}</p>}
      {snap && (
        <>
          <div className="matchup">
            <b>{snap.pitcher_name}</b> vs <b>{snap.batter_name}</b>
            <span className="badge">
              {snap.state.balls}-{snap.state.strikes}
            </span>
            <span className="badge">{snap.state.outs_when_up} out</span>
            <span className="badge">Inn {snap.state.inning}</span>
            <span className="badge">{snap.state.on_base ? "Runners on" : "Bases empty"}</span>
          </div>
          <Results data={snap.recommendation} pitches={snap.pitches_this_at_bat} />
          <Scoreboard sb={snap.scoreboard} outcomes={false} title="This live game: model call vs pitch thrown" />
        </>
      )}
    </>
  );
}
