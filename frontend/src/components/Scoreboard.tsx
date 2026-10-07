import { pp } from "../format";
import type { ScoreboardStats } from "../types";

interface StatProps {
  label: string;
  value: string;
  sub?: string;
}

function Stat({ label, value, sub }: StatProps) {
  return (
    <div className="stat">
      <div className="stat-val">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="muted">{sub}</div>}
    </div>
  );
}

interface Props {
  sb: ScoreboardStats | null | undefined;
  title: string;
  /** Hide the real-outcome stats (live pitches have no outcome yet). */
  outcomes?: boolean;
}

export function Scoreboard({ sb, title, outcomes = true }: Props) {
  return (
    <div className="board">
      <h3>
        {title} {sb?.n ? <small className="muted">({sb.n.toLocaleString()} pitches)</small> : null}
      </h3>
      {!sb?.n ? (
        <p className="muted">Nothing scored yet.</p>
      ) : (
        <div className="stats">
          <Stat label="Model's top pick = actual pitch" value={pp(sb.top1_pct)} />
          <Stat label="Actual pitch in model's top 3" value={pp(sb.top3_pct)} />
          <Stat label="Model score: its top pick" value={pp(sb.avg_rec_prob)} />
          <Stat label="Model score: pitch thrown" value={pp(sb.avg_actual_prob)} />
          {outcomes && (
            <Stat
              label="Good outcome when pitcher matched"
              value={pp(sb.success_matched)}
              sub={`n=${sb.n_matched}`}
            />
          )}
          {outcomes && (
            <Stat
              label="Good outcome when not matched"
              value={pp(sb.success_unmatched)}
              sub={`n=${sb.n_unmatched}`}
            />
          )}
        </div>
      )}
    </div>
  );
}
