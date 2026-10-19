/** Shapes of the JSON returned by the FastAPI backend. */

export type Hand = "L" | "R";

export interface GameState {
  stand: Hand;
  p_throws: Hand;
  on_base: number;
  balls: number;
  strikes: number;
  outs_when_up: number;
  inning: number;
}

export interface GridCell {
  plate_x: number;
  plate_z: number;
  prob: number;
}

/** What this pitcher's version of the pitch looks like (averages), shown next to the heatmap. */
export interface Stuff {
  velo: number | null;
  spin: number | null;
  ext: number | null;
  ivb_in: number | null; // induced vertical break, inches
  hb_in: number | null; // horizontal break, inches
  source: "pitcher" | "league"; // league = he has too few of this pitch, so league average is used
}

export interface PitchRec {
  pitch_type: string;
  success_prob: number;
  n_samples: number;
  locations_source: string;
  stuff?: Stuff | null;
  best_location: { plate_x: number; plate_z: number };
  best_cell_prob: number;
  grid: GridCell[];
}

/** One batter's results against a pitch category (shrunk toward the league average). */
export interface BatterVs {
  whiff: number | null;
  xwoba_con: number | null;
  swings: number;
}

export interface BatterInfo {
  batter_id: number;
  name: string;
  known: boolean; // false -> no profile, the model treated him as a league-average hitter
  stand: Hand | null;
  switch: boolean;
  vs: Record<string, BatterVs>;
}

export interface BatterListItem {
  batter_id: number;
  name: string;
  stand: Hand;
  switch: boolean;
}

export interface Recommendation {
  batter?: BatterInfo | null;
  pitcher_id: number;
  pitcher_name: string;
  used_default_arsenal: boolean;
  pitcher_cluster: number;
  hitter_cluster: number;
  count_bucket: string;
  recommendations: PitchRec[];
}

export interface Pitcher {
  pitcher_id: number;
  name: string;
  pitches: string[];
  throws?: Hand | null;
}

export interface ScoreboardStats {
  n: number;
  run_id?: number;
  top1_pct?: number | null;
  top3_pct?: number | null;
  avg_rec_prob?: number | null;
  avg_actual_prob?: number | null;
  success_matched?: number | null;
  success_unmatched?: number | null;
  n_matched?: number;
  n_unmatched?: number;
}

/** A pitch to draw on the strike zone (coordinates can be missing in live data). */
export interface PlotPitch {
  plate_x: number | null;
  plate_z: number | null;
}

export interface LivePitch extends PlotPitch {
  type: string | null;
  call?: string | null;
  speed?: number | null;
  spin?: number | null;
}

export interface LiveGame {
  game_pk: number;
  status: string;
  away: string;
  home: string;
}

export interface LiveSnapshot {
  pitcher_id: number;
  pitcher_name: string;
  batter_id: number;
  batter_name: string;
  state: GameState;
  score: { away: number | null; home: number | null };
  pitches_this_at_bat: LivePitch[];
  recommendation: Recommendation;
  scoreboard?: ScoreboardStats;
}

export interface ReplayGame {
  game_pk: number;
  date: string;
  n_pitches: number;
  postseason: boolean;
  label: string;
}

export interface ReplayStep extends Recommendation {
  rec_pitch: string;
  actual_pitch: string;
  actual_success: number | null;
  matched: boolean;
  state: GameState;
  actual_x: number;
  actual_z: number;
  scoreboard: ScoreboardStats;
}

export interface RunCreated {
  run_id: number;
}
