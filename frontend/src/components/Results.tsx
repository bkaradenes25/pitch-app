import { useState } from "react";
import { pct } from "../format";
import type { PitchRec, PlotPitch, Recommendation } from "../types";
import { StrikeZone } from "./StrikeZone";

export type ResultsData = Pick<Recommendation, "recommendations" | "used_default_arsenal">;

interface BarsProps {
  recs: PitchRec[];
  current: string;
  onSelect: (pitchType: string) => void;
}

function Bars({ recs, current, onSelect }: BarsProps) {
  const max = Math.max(...recs.map((r) => r.success_prob));
  return (
    <>
      {recs.map((r) => (
        <div
          key={r.pitch_type}
          onClick={() => onSelect(r.pitch_type)}
          className={"bar-row" + (r.pitch_type === current ? " sel" : "")}
        >
          <span className="pt">{r.pitch_type}</span>
          <div className="bar">
            <div style={{ width: `${(r.success_prob / max) * 100}%` }} />
          </div>
          <span className="val">{pct(r.success_prob)}</span>
        </div>
      ))}
    </>
  );
}

interface Props {
  data: ResultsData | null;
  pitches?: PlotPitch[];
}

export function Results({ data, pitches }: Props) {
  const [sel, setSel] = useState<string | null>(null);
  if (!data) return <p className="muted">Loading…</p>;
  const recs = data.recommendations;
  const cur = recs.find((r) => r.pitch_type === sel) ?? recs[0];
  return (
    <div className="results">
      <div>
        <h3>
          Recommended: <span className="hl">{recs[0].pitch_type}</span>
        </h3>
        <Bars recs={recs} current={cur.pitch_type} onSelect={setSel} />
        <p className="muted">
          Bar = model's average success if thrown where this pitcher usually locates it in this kind of count.
          Heatmap = success if located at that exact spot.
        </p>
        {data.used_default_arsenal && (
          <p className="muted">Pitcher not in training data: using a default arsenal.</p>
        )}
      </div>
      <div>
        <h3>
          {cur.pitch_type} location heatmap <small className="muted">(catcher's view)</small>
        </h3>
        <StrikeZone rec={cur} pitches={pitches} />
      </div>
    </div>
  );
}
