import { useEffect, useMemo, useState } from "react";
import { post, api } from "../api";
import { Results } from "../components/Results";
import { batterOptions, errMsg, findBatter, standFor } from "../format";
import type { BatterListItem, GameState, Hand, Pitcher, Recommendation } from "../types";

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
  const [batters, setBatters] = useState<BatterListItem[]>([]);
  const [batterText, setBatterText] = useState("");
  const [pid, setPid] = useState<number | null>(null);
  const [s, setS] = useState<GameState>(INITIAL);
  const [data, setData] = useState<Recommendation | null>(null);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState("");

  const options = useMemo(() => batterOptions(batters), [batters]);
  const batter = useMemo(() => findBatter(options, batterText), [options, batterText]);
  const batterId = batter?.batter_id ?? null;

  const set = <K extends keyof GameState>(key: K, value: GameState[K]) =>
    setS((x) => {
      const next: GameState = { ...x, [key]: value };
      // a switch hitter changes sides when the pitcher's hand changes
      return key === "p_throws" ? { ...next, stand: standFor(batter, next.p_throws, next.stand) } : next;
    });

  const pickPitcher = (p: Pitcher | undefined) => {
    if (!p) return;
    setPid(p.pitcher_id);
    const hand = p.throws;
    if (hand) setS((x) => ({ ...x, p_throws: hand, stand: standFor(batter, hand, x.stand) }));
  };

  const pickBatter = (text: string) => {
    setBatterText(text);
    const b = findBatter(options, text);
    if (b) setS((x) => ({ ...x, stand: standFor(b, x.p_throws, x.stand) }));
  };

  useEffect(() => {
    api<Pitcher[]>("/pitchers")
      .then((p) => {
        setPitchers(p);
        const first = p[0];
        setPid(first?.pitcher_id ?? null);
        const hand = first?.throws;
        if (hand) setS((x) => ({ ...x, p_throws: hand }));
      })
      .catch((e: unknown) => setErr("Backend unreachable: " + errMsg(e)));
    api<BatterListItem[]>("/batters")
      .then(setBatters)
      .catch(() => setBatters([])); // older backend without batter profiles: the picker just stays empty
  }, []);

  useEffect(() => {
    if (pid == null) return;
    let alive = true;
    post<Recommendation>("/recommend", {
      pitcher_id: pid,
      state: s,
      ...(batterId != null ? { batter_id: batterId } : {}),
    })
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
  }, [pid, s, batterId]);

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
          <select
            value={pid ?? ""}
            onChange={(e) => pickPitcher(pitchers.find((p) => p.pitcher_id === Number(e.target.value)))}
          >
            {pitchers.map((p) => (
              <option key={p.pitcher_id} value={p.pitcher_id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          batter
          <input
            list="batter-options"
            value={batterText}
            placeholder="league-average hitter"
            onChange={(e) => pickBatter(e.target.value)}
          />
          <datalist id="batter-options">
            {options.map((o) => (
              <option key={o.batter.batter_id} value={o.label} />
            ))}
          </datalist>
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
      {batterText.trim() && !batter && (
        <p className="muted">No batter called "{batterText.trim()}" in the list: using a league-average hitter.</p>
      )}
      <Results data={data} />
    </>
  );
}
