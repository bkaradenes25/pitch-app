import { useState } from "react";
import { Live } from "./views/Live";
import { Replay } from "./views/Replay";
import { Sandbox } from "./views/Sandbox";

type Tab = "live" | "sandbox" | "replay";

export default function App() {
  const [tab, setTab] = useState<Tab>("sandbox");
  const tabClass = (t: Tab) => (tab === t ? "on" : "");
  return (
    <div className="app">
      <header>
        <h1>⚾ Pitch Recommender</h1>
        <nav>
          <button className={tabClass("live")} onClick={() => setTab("live")}>
            Live games
          </button>
          <button className={tabClass("sandbox")} onClick={() => setTab("sandbox")}>
            Sandbox
          </button>
          <button className={tabClass("replay")} onClick={() => setTab("replay")}>
            Replay &amp; Scoreboard
          </button>
        </nav>
      </header>
      {tab === "live" ? <Live /> : tab === "replay" ? <Replay /> : <Sandbox />}
    </div>
  );
}
