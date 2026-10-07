import { useEffect, useState } from "react";
import { post, api } from "../api";
import { Results } from "../components/Results";
import { errMsg } from "../format";
import type { GameState, Hand, Pitcher, Recommendation } from "../types";

const DEMO: Array<[number, number]> = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]]; // scripted at-bat (balls, strikes)
const HANDS: Hand[] = ["R", "L"];
const INITIAL: GameState = { stand: "R", p_throws: "R", on_base: 0, balls: 0, strikes: 0, outs_when_up: 0, inning: 1 };

interface SelProps<K extends keyof GameState> {
  name: K;
  opts: GameState[K][];
  state: GameState;
  onChange: (key: K, value: GameState[K]) => void;
}

function Sel<K extends keyof GameState>({ name, opts, state, onChange }: SelProps<K>) {
  return (
    <label>
      {name.replace("_", " ")}
      <select
        value={String(state[name])}
        onChange={(e) => {
          const picked = opts.find((o) => String(o) === e.target.value);
          if (picked !== undefined) onChange(name, picked);
        }}
      >
        {opts.map((o) => (
          <option key={String(o)}>{String(o)}</option>
        ))}
      </select>
    </label>
  );
}

export function Sandbox() {
  const [pitchers, setPitchers] = useState<Pitcher[]>([]);
  const [pid, setPid] = useState<number | null>(null);
  const [s, setS] = useState<GameState>(INITIAL);
  const [data, setData] = useState<Recommendation | null>(null);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState("");

  const set = <K extends keyof GameState>(key: K, value: GameState[K]) =>
    setS((x) => ({ ...x, [key]: value }));

  useEffect(() => {
    api<Pitcher[]>("/pitchers")
      .then((p) => {
        setPitchers(p);
        setPid(p[0]?.pitcher_id ?? null);
      })
      .catch((e: unknown) => setErr("Backend unreachable: " + errMsg(e)));
  }, []);

  useEffect(() => {
    if (pid == null) return;
    let alive = true;
    post<Recommendation>("/recommend", { pitcher_id: pid, state: s })
      .then((d) => {
        if (!alive) return;
        setData(d);
        setErr("");
      })
      .catch((e: unknown) => {
        if (alive) setErr(errMsg(e));
      });
    return () => {
      alive = false;
    };
  }, [pid, s]);

  // replay mode: step through a scripted at-bat without needing a live game
  useEffect(() => {
    if (!playing) return;
    let i = 0;
    const id = setInterval(() => {
      if (i >= DEMO.length) {
        setPlaying(false);
        return;
      }
      const [balls, strikes] = DEMO[i++];
      setS((x) => ({ ...x, balls, strikes }));
    }, 2500);
    return () => clearInterval(id);
  }, [playing]);

  return (
    <>
      <div className="controls">
        <label>
          pitcher
          <select value={pid ?? ""} onChange={(e) => setPid(Number(e.target.value))}>
            {pitchers.map((p) => (
              <option key={p.pitcher_id} value={p.pitcher_id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <Sel name="p_throws" opts={HANDS} state={s} onChange={set} />
        <Sel name="stand" opts={HANDS} state={s} onChange={set} />
        <Sel name="balls" opts={[0, 1, 2, 3]} state={s} onChange={set} />
        <Sel name="strikes" opts={[0, 1, 2]} state={s} onChange={set} />
        <Sel name="outs_when_up" opts={[0, 1, 2]} state={s} onChange={set} />
        <Sel name="inning" opts={[1, 2, 3, 4, 5, 6, 7, 8, 9]} state={s} onChange={set} />
        <Sel name="on_base" opts={[0, 1]} state={s} onChange={set} />
        <button onClick={() => setPlaying(!playing)}>{playing ? "Stop" : "▶ Replay demo at-bat"}</button>
      </div>
      {err && <p className="err">{err}</p>}
      <Results data={data} />
    </>
  );
}
