# PhysLab: Video-Based Physics Experiment Analyser

Project plan and working instructions for Claude Code.
Built for the **Meta Global AI Developer Hackathon** (track: **Push the Frontier**).

---

## 1. What we're building

A tool that takes a phone video of a simple physics experiment, tracks the moving object, and extracts physical constants (e.g. g) **with uncertainty estimates**, then produces a short verified report.

Pipeline:

```
video upload
  → frame extraction (OpenCV)
  → object segmentation & tracking (SAM 3.1 via Meta Model API)
  → pixel→metre calibration (reference object of known size in frame)
  → trajectory cleaning (outlier rejection + Kalman filter)
  → physics model fitting (SciPy)
  → uncertainty estimation
  → plausibility checks
  → report (Muse Spark explains results; it never computes numbers)
```

### Supported experiments (in priority order)
1. **Free fall / dropped ball** → estimate g (fit y = y0 + v0·t + ½·g·t², optionally with linear/quadratic drag)
2. **Simple pendulum** → estimate g from period and length; also damped-oscillation fit
3. **Bouncing ball** → coefficient of restitution from successive peak heights
4. **Ramp / incline** (stretch goal) → acceleration and effective friction

Only build experiment 2+ once experiment 1 works end to end.

---

## 2. Core principles (follow these throughout)

1. **Every number is computed deterministically by our code.** The LLM (Muse Spark) may classify the experiment, explain results and write the report, but it must never produce or alter a numerical result. Report text must reference values passed in from the pipeline.
2. **Verifiable outputs.** Every result includes: value, uncertainty, fit quality (R², residuals), and the checks it passed or failed.
3. **Meta models must be used meaningfully** (hackathon requirement): SAM 3.1 for segmentation/tracking, Muse Spark for experiment classification + report generation.
4. **Do not invent API details.** For the Meta Model API, consult the official docs (https://dev.meta.ai/docs/overview, https://dev.meta.ai/docs/quickstart, https://dev.meta.ai/docs/models) and the cookbook (https://github.com/meta-models/meta-model-cookbook). If an endpoint or parameter is unclear, stop and ask rather than guessing.
5. **Control API cost.** Budget is ~$150 in credits. Prefer SAM 3.1's video tracking/propagation if supported; otherwise sample frames and interpolate. Cache all API responses to disk keyed by (video hash, frame index, params) so re-runs are free.
6. **Only use data we have rights to:** our own phone videos and synthetic videos we generate. No downloaded YouTube clips.
7. **Commit small and often** with clear messages. The commit history is evidence that work was done during the hackathon (building period: Oct 27 – Nov 4, 2026).
8. **Secrets** (API keys) live in `.env`, never committed. Provide `.env.example`.

---

## 3. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| Backend API | FastAPI |
| Video / CV | OpenCV, NumPy |
| Segmentation & tracking | SAM 3.1 (Meta Model API) |
| Baseline tracker (for comparison) | OpenCV colour/background-subtraction tracker |
| Filtering & fitting | filterpy (or hand-written Kalman), SciPy (`curve_fit`, `least_squares`) |
| Reasoning & reports | Muse Spark (Meta Model API) |
| Frontend | Streamlit (default). Switch to React only if time allows. |
| Charts | Plotly |
| Tests | pytest |
| Hosting (demo) | Modal |

---

## 4. Repository layout

```
physlab/
  CLAUDE.md
  README.md
  .env.example
  pyproject.toml
  physlab/
    __init__.py
    config.py            # settings, env loading
    video.py             # frame extraction, fps detection, metadata
    meta_client.py       # thin wrapper around Meta Model API + disk cache
    tracking/
      sam_tracker.py     # SAM 3.1 masks → centroids per frame
      baseline_tracker.py# plain OpenCV tracker for benchmarking
    calibration.py       # reference object detection, px→m scale
    trajectory.py        # outlier rejection, Kalman smoothing
    physics/
      freefall.py
      pendulum.py
      bounce.py
    uncertainty.py       # error propagation, bootstrap
    checks.py            # plausibility & sanity checks
    report.py            # builds structured result + calls Muse Spark for prose
    pipeline.py          # orchestrates everything
  api/
    main.py              # FastAPI endpoints
  app/
    streamlit_app.py     # UI
  synth/
    generate.py          # synthetic video generator with known ground truth
  data/
    real/                # our phone videos (gitignored if large) + metadata.json
    synthetic/           # generated videos + ground_truth.json
    cache/               # API response cache (gitignored)
  eval/
    run_eval.py          # runs pipeline on test sets, outputs accuracy table
  tests/
```

---

## 5. Build phases

Each phase has acceptance criteria. Don't move on until they pass.

### Phase 0: Setup (Day 1, morning)
- Init repo, `pyproject.toml`, `.env.example`, `.gitignore`, basic README.
- `meta_client.py`: authenticate, make one successful call each to SAM 3.1 and Muse Spark, with disk caching.
- **Done when:** a script segments one frame of a test image with SAM 3.1 and Muse Spark returns a text response; second run hits the cache.

### Phase 1: Synthetic data generator (Day 1)
- `synth/generate.py` renders simple videos with known physics: falling ball (configurable g, drop height, fps, optional drag), pendulum (length, amplitude, damping), bouncing ball (restitution).
- Include a reference object of known size in frame (e.g. a rectangle labelled as 0.297 m), plain background, optional noise/blur/camera shake.
- Write `ground_truth.json` alongside each video.
- **Done when:** 20+ synthetic videos generated across experiment types with ground truth.

### Phase 2: Tracking (Day 1–2)
- `video.py`: extract frames, read true fps from metadata (handle slow-motion files correctly).
- `sam_tracker.py`: get object mask per frame (use video propagation if API supports it, else prompt on first frame and sample), output centroid + mask area + confidence per frame.
- `baseline_tracker.py`: simple OpenCV tracker for comparison.
- **Done when:** trajectories from both trackers can be plotted over a synthetic video and SAM tracks the object through the full clip.

### Phase 3: Calibration & trajectory cleaning (Day 2)
- `calibration.py`: detect reference object (start with user clicking two points of known distance in the UI as fallback; automate detection if time allows). Output metres-per-pixel with its own uncertainty.
- `trajectory.py`: reject outlier frames (low confidence, mask area jumps), Kalman smooth, convert to metres and seconds.
- **Done when:** synthetic free-fall trajectory in metres matches ground truth within 2% at each frame.

### Phase 4: Physics fitting & uncertainty (Day 2–3)
- `physics/freefall.py`: fit quadratic (and drag model variant); pick model via residuals/AIC.
- `physics/pendulum.py`: fit damped sinusoid to angle vs time; derive period, g from L.
- `physics/bounce.py`: find successive peaks, compute restitution.
- `uncertainty.py`: combine fit covariance, calibration uncertainty and frame-timing uncertainty (1/fps); also offer bootstrap estimate.
- `checks.py`: e.g. g within plausible range, enough frames, residuals not systematic, object never lost for too long. Each check returns pass/warn/fail with a reason.
- **Done when:** on synthetic free-fall videos, estimated g is within 3% of truth and the true value lies inside the reported uncertainty in most cases.

### Phase 5: Evaluation (Day 3)
- Film 15–20 real clips (see section 7) and record known values (string length, drop height) in `data/real/metadata.json`.
- `eval/run_eval.py`: run pipeline on synthetic + real sets, output a table: experiment, true value, estimate, error %, uncertainty, whether truth is within error bars, SAM vs baseline tracker.
- **Done when:** eval table generated as CSV + markdown. These numbers go in the README, demo and CV.

### Phase 6: Agent & report layer (Day 4)
- Muse Spark classifies experiment type from a few sampled frames (user can override).
- `report.py` builds a structured JSON result first (all numbers, checks), then asks Muse Spark to write a short explanation: what was measured, how reliable it is, likely error sources, how to film a better clip. Prompt must instruct it to use only the provided values.
- Add a post-check: verify every number in the generated text appears in the structured result; if not, regenerate or fall back to a template.
- **Done when:** reports are generated for all eval videos with zero mismatched numbers.

### Phase 7: API & UI (Day 5)
- FastAPI: `POST /analyse` (video upload, optional experiment type and reference length) → job id; `GET /result/{id}`.
- Streamlit: upload video, click calibration points if needed, show tracked overlay video/frames, trajectory and fit plots (Plotly), results with error bars, check list, and the report.
- **Done when:** a new real video can be analysed end to end through the UI in under ~2 minutes.

### Phase 8: Polish & submission (Day 6, by Nov 4)
- README: problem, how it works (pipeline diagram), what was built during the hackathon, setup instructions, eval results table, limitations.
- Deploy demo on Modal if stable.
- Record demo video: one live real clip, the tracked overlay, the fit, the error bars, the checks, and the eval table.
- Confirm repo is **public**, video link works, all team members listed.
- **Deadline: Wed Nov 4, 9:00 PM ET (02:00 Thu Nov 5 UK time).** Aim to submit a day early.

---

## 6. Stretch goals (only if Phases 0–8 are done)
- Ramp/incline experiment.
- Automatic reference-object detection (no clicking).
- Multiple objects in one video.
- Run Muse Spark Open locally/on Modal as an alternative to the API.
- Lens distortion / perspective correction.

---

## 7. Filming guide for real test videos
- Phone on a stable surface, filming side-on to the motion.
- Plain background, good lighting.
- Ruler or A4 sheet (0.297 m long side) in the same plane as the motion.
- Use slow-motion (120/240 fps) for drops and bounces.
- Record the known values (drop height, string length) for each clip.

---

## 8. Definition of done
- Free fall and pendulum work end to end on real phone video.
- Eval table shows accuracy on synthetic and real sets, with SAM vs baseline comparison.
- Every result has uncertainty and pass/fail checks.
- Report text never contains numbers not produced by the pipeline.
- Public repo, README, demo video, submitted before the deadline.
