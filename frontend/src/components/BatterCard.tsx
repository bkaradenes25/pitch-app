import { pp, xwoba } from "../format";
import type { BatterInfo } from "../types";

const LABELS: Array<[string, string]> = [
  ["fastballs", "Fastballs"],
  ["breaking_balls", "Breaking balls"],
  ["offspeed", "Offspeed"],
];

/** What the model knows about the batter: his results against each kind of pitch. */
export function BatterCard({ batter }: { batter: BatterInfo }) {
  if (!batter.known) {
    return (
      <p className="muted">
        {batter.name}: no profile, so he is treated as a league-average hitter against every pitch type.
      </p>
    );
  }
  const rows = LABELS.flatMap(([key, label]) => {
    const v = batter.vs[key];
    return v ? [{ key, label, v }] : [];
  });
  return (
    <div className="batter-card">
      <h4>
        {batter.name} <small className="muted">against each pitch type</small>
      </h4>
      <table>
        <thead>
          <tr>
            <th></th>
            <th>Whiff rate</th>
            <th>xwOBA on contact</th>
            <th>Swings seen</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ key, label, v }) => (
            <tr key={key}>
              <td>{label}</td>
              <td>{pp(v.whiff)}</td>
              <td>{xwoba(v.xwoba_con)}</td>
              <td>{v.swings}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">
        Rates are pulled toward the league average when a hitter has seen few pitches of a type.
      </p>
    </div>
  );
}
