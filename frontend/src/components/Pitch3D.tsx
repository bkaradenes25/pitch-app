import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { pct } from "../format";
import {
  buildField, buildHeatmap, buildPitches, buildTarget, CAMERA_PRESETS, disposeTree, pointAlong,
  type CameraPreset, type CellInfo, type Flight,
} from "../scene3d";
import type { PitchRec, PlotPitch } from "../types";

const NO_PITCHES: PlotPitch[] = []; // a stable default, so the effect below does not re-run on every render
const SLOW_MOTION = 0.15; // real time -> shown time, so a 0.4 s pitch takes about 2.7 s

interface Props {
  rec: PitchRec;
  pitches?: PlotPitch[];
}

interface Rig {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  dynamic: THREE.Group;
  flights: Flight[];
  tween: { fromPos: THREE.Vector3; toPos: THREE.Vector3; fromTarget: THREE.Vector3; toTarget: THREE.Vector3; start: number } | null;
  play: { start: number } | null;
  heat: THREE.Object3D[];
  raycaster: THREE.Raycaster;
}

const reducedMotion = () =>
  typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export default function Pitch3D({ rec, pitches = NO_PITCHES }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const rig = useRef<Rig | null>(null);
  const [failed, setFailed] = useState(false);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [preset, setPreset] = useState<CameraPreset>("catcher");
  const [hasFlight, setHasFlight] = useState(false);

  // renderer, camera, controls, render loop: once
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailed(true); // no WebGL (old GPU, blocked, or a test environment)
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(6, 12, 10);
    scene.add(sun);
    scene.add(buildField());
    const dynamic = new THREE.Group();
    scene.add(dynamic);

    const p = CAMERA_PRESETS.catcher;
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 300);
    camera.position.set(...p.position);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...p.target);
    controls.enableDamping = !reducedMotion();
    controls.minDistance = 2;
    controls.maxDistance = 80;
    controls.update();

    const r: Rig = {
      renderer, scene, camera, controls, dynamic, flights: [], tween: null, play: null, heat: [],
      raycaster: new THREE.Raycaster(),
    };
    rig.current = r;

    const resize = () => {
      const w = el.clientWidth || 480;
      const h = Math.max(260, Math.round(w * 0.75));
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(resize);
      ro.observe(el);
    } else window.addEventListener("resize", resize);

    let raf = 0;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (r.tween) {
        const k = Math.min(1, (now - r.tween.start) / 600);
        const e = k * k * (3 - 2 * k);
        camera.position.lerpVectors(r.tween.fromPos, r.tween.toPos, e);
        controls.target.lerpVectors(r.tween.fromTarget, r.tween.toTarget, e);
        if (k >= 1) r.tween = null;
      }
      if (r.play && r.flights.length) {
        const last = r.flights[r.flights.length - 1];
        const progress = ((now - r.play.start) / 1000) * SLOW_MOTION / last.duration;
        last.ball.position.copy(pointAlong(last.path, progress));
        if (progress >= 1) r.play = null;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro?.disconnect();
      window.removeEventListener("resize", resize);
      controls.dispose();
      disposeTree(scene);
      renderer.dispose();
      renderer.domElement.remove();
      rig.current = null;
    };
  }, []);

  // heatmap, target and pitches: whenever the data changes
  useEffect(() => {
    const r = rig.current;
    if (!r) return;
    for (const child of [...r.dynamic.children]) {
      r.dynamic.remove(child);
      disposeTree(child);
    }
    const heat = buildHeatmap(rec.grid);
    r.dynamic.add(heat);
    r.heat = heat.children.filter((c: THREE.Object3D) => (c.userData as CellInfo).prob !== undefined);
    r.dynamic.add(buildTarget(rec.best_location.plate_x, rec.best_location.plate_z));
    const { group, flights } = buildPitches(pitches);
    r.dynamic.add(group);
    r.flights = flights;
    r.play = null;
    setHasFlight(flights.length > 0);
  }, [rec, pitches]);

  const choose = (name: CameraPreset) => {
    setPreset(name);
    const r = rig.current;
    if (!r) return;
    const p = CAMERA_PRESETS[name];
    const toPos = new THREE.Vector3(...p.position);
    const toTarget = new THREE.Vector3(...p.target);
    if (reducedMotion()) {
      r.camera.position.copy(toPos);
      r.controls.target.copy(toTarget);
    } else {
      r.tween = {
        fromPos: r.camera.position.clone(), toPos,
        fromTarget: r.controls.target.clone(), toTarget, start: performance.now(),
      };
    }
  };

  const playFlight = () => {
    const r = rig.current;
    if (!r || !r.flights.length) return;
    const last = r.flights[r.flights.length - 1];
    if (reducedMotion()) {
      last.ball.position.copy(last.path[last.path.length - 1]);
      return;
    }
    last.ball.position.copy(last.path[0]);
    r.play = { start: performance.now() };
  };

  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const r = rig.current;
    if (!r) return;
    const box = r.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - box.left) / box.width) * 2 - 1,
      -(((e.clientY - box.top) / box.height) * 2 - 1),
    );
    r.raycaster.setFromCamera(ndc, r.camera);
    const hit = r.raycaster.intersectObjects(r.heat, false)[0];
    if (!hit) return setTip(null);
    const c = hit.object.userData as CellInfo;
    setTip({
      x: e.clientX - box.left + 12,
      y: e.clientY - box.top + 12,
      text: `${pct(c.prob)}${c.best ? " (best spot)" : ""}`,
    });
  };

  if (failed) {
    return <p className="muted">3D view needs WebGL, which isn't available here. Use the 2D view instead.</p>;
  }
  return (
    <div className="pitch3d">
      <div className="pitch3d-bar">
        {(Object.keys(CAMERA_PRESETS) as CameraPreset[]).map((name) => (
          <button key={name} className={name === preset ? "on" : ""} onClick={() => choose(name)}>
            {name[0].toUpperCase() + name.slice(1)}
          </button>
        ))}
        <button onClick={playFlight} disabled={!hasFlight} title={hasFlight ? "" : "No flight data for this pitch"}>
          ▶ Play flight
        </button>
      </div>
      <div className="pitch3d-canvas" ref={host} onPointerMove={onMove} onPointerLeave={() => setTip(null)}>
        {tip && (
          <div className="pitch3d-tip" style={{ left: tip.x, top: tip.y }}>
            {tip.text}
          </div>
        )}
      </div>
      {!hasFlight && (
        <p className="muted">
          Play flight needs a real pitch with tracking data: pick a Replay step or a Live game. The Sandbox has no real pitch, and
          Replay needs the backend restarted after <code>python get_data.py --refresh</code>.
        </p>
      )}
      <p className="muted">Drag to orbit, scroll to zoom. Bar height and colour = success if located there.</p>
    </div>
  );
}
