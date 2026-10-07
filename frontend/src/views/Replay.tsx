import { useEffect, useState } from "react";
import { api, post } from "../api";
import { Results } from "../components/Results";
import { Scoreboard } from "../components/Scoreboard";
import { errMsg } from "../format";
import type { ReplayGame, ReplayStep, RunCreated, ScoreboardStats } from "../types";

export function Replay() {
  const [games, setGames] = useState<ReplayGame[]>([]);
  const [gk, setGk] = useState("");
  const [runId, setRunId] = useState<number | null>(null);
  const [idx, setIdx] = useState(0);
  const [step, setStep] = useState<ReplayStep | null>(null);
  const [sb, setSb] = useState<ScoreboardStats | null>(null);
  const [backtest, setBacktest] = useState<ScoreboardStats | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1200);
  const [err, setErr] = useState("");
  const total = games.find((g) => String(g.game_pk) === gk)?.n_pitches ?? 0;

  useEffect(() => {
    api<ReplayGame[]>("/replay/games")
      .then(setGames)
      .catch((e: unknown) => setErr("Could not load games: " + errMsg(e)));
    api<ScoreboardStats>("/scoreboard?source=backtest")
      .then(setBacktest)
      .catch(() => {});
  }, []);

  const start = async () => {
    try {
      const r = await post<RunCreated>("/runs", { source: "replay", label: `game ${gk}` });
      setRunId(r.run_id);
      setIdx(0);
      setStep(null);
      setSb(null);
      setErr("");
      setPlaying(true);
    } catch (e) {
      setErr(errMsg(e));
    }
  };

  useEffect(() => {
    if (!playing || runId == null) return;
    if (idx >= total) {
      setPlaying(false);
      return;
    }
    let alive = true;
    const t = setTimeout(async () => {
      try {
        const s = await post<ReplayStep>(`/replay/${gk}/step`, { index: idx, run_id: runId });
        if (!alive) return;
        setStep(s);
        setSb(s.scoreboard);
        setIdx(idx + 1);
      } catch (e) {
        setErr(errMsg(e));
        setPlaying(false);
      }
    }, speed);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [playing, idx, runId, total, speed, gk]);

  return (
    <>
      <div className="controls">
        <label>
          held-out game
          <select
            value={gk}
            onChange={(e) => {
              setGk(e.target.value);
              setPlaying(false);
              setRunId(null);
              setStep(null);
              setSb(null);
            }}
          >
            <option value="">Select a game…</option>
            {games.map((g) => (
              <option key={g.game_pk} value={g.game_pk}>
                {g.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={2500}>Slow</option>
            <option value={1200}>Normal</option>
            <option value={300}>Fast</option>
          </select>
        </label>
        <button onClick={start} disabled={!gk}>
          ▶ Start replay
        </button>
        {playing && <button onClick={() => setPlaying(false)}>Pause</button>}
        {!playing && runId != null && idx > 0 && idx < total && (
          <button onClick={() => setPlaying(true)}>Resume</button>
        )}
        {total > 0 && (
          <span className="muted">
            pitch {idx} / {total}
          </span>
        )}
      </div>
      {err && <p className="err">{err}</p>}
      {step && (
        <>
          <div className="matchup">
            <b>{step.pitcher_name}</b> vs {step.state.stand === "L" ? "LHB" : "RHB"}
            <span className="badge">
              {step.state.balls}-{step.state.strikes}
            </span>
            <span className="badge">{step.state.outs_when_up} out</span>
            <span className="badge">Inn {step.state.inning}</span>
            <span className={"badge " + (step.matched ? "ok" : "")}>
              Model: {step.rec_pitch} · Thrown: {step.actual_pitch} {step.matched ? "✓" : ""}
            </span>
            <span className="badge">{step.actual_success ? "Good outcome" : "Not a good outcome"}</span>
          </div>
          <Results data={step} pitches={[{ plate_x: step.actual_x, plate_z: step.actual_z }]} />
        </>
      )}
      <Scoreboard sb={sb} title="This replay" />
      <Scoreboard sb={backtest} title="Full held-out backtest (python backtest.py)" />
      <p className="muted">
        The model was trained only on earlier regular-season games, so these pitches are unseen. A mismatch is not
        an error (the pitcher's choice isn't ground truth); the outcome comparison is confounded by game situation.
      </p>
    </>
  );
}
