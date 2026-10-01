# AGENTS.md

This document defines repository-wide instructions for coding agents. Keep this file and `AGENTS.zh-CN.md` semantically synchronized. Read the Chinese version as a companion to this file; do not assume tools automatically discover it. Preserve existing repository conventions where they do not conflict with the mandatory rules below.

## 1. Project Overview

Graduation project: **基于 YOLO11 与 Depth Anything V2 的视障人士视觉辅助系统设计与实现**.

English description: **A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2**.

The system uses a monocular RGB camera to recognize objects, estimate spatial information, assess risk, and provide voice assistance.

```text
RGB camera → preprocessing → YOLO11 detections + Depth Anything V2 depth
                           → spatial fusion → structured scene → deterministic risk engine
                                                               ├→ immediate warning → TTS
                                                               └→ Vision Assistant Agent → TTS / UI
Microphone → STT → user request → Vision Assistant Agent → validated scene tools
```

Separate the five responsibilities: what an object is, how far away it is, where it is, whether it presents a risk, and what information the user needs.

## 2. Core Design Principles and Mandatory Boundaries

- **Emergency risk decisions MUST NOT depend on an LLM.** Deterministic code must evaluate configured rules using available object category, valid distance, direction, confidence, and motion information.
- Immediate warnings must reach the output path without waiting for the Agent, an LLM request, STT, or a network connection. An LLM failure or timeout must not disable this path.
- The Agent may explain a risk-engine warning but must never suppress, downgrade, or delay it. Enforce this in application control flow, not just prompts.
- **The Agent MUST NOT fabricate objects, depth, distances, directions, confidence, or risk states.** Missing, invalid, and stale information must remain explicitly unknown or unavailable.
- **Every function MUST have complete type annotations.** Section 17 defines the scope.
- **NEVER commit `.env` files.** Section 15 defines the controls; no debugging, demonstration, or convenience exception applies.
- Never implement `camera → LLM → decide whether to warn`. Keep perception, fusion, deterministic risk assessment, Agent interaction, and speech as separate responsibilities.

## 3. Expected Repository Structure

Prefer the existing structure. This is a recommendation, not a requirement to duplicate or rename existing modules.

```text
project/
├── AGENTS.md
├── AGENTS.zh-CN.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── app/
│   ├── vision/       # detector.py, depth_estimator.py, preprocessing.py
│   ├── fusion/       # depth_fusion.py, direction.py, scene_builder.py
│   ├── safety/       # risk_engine.py
│   ├── agent/        # vision_agent.py, tools.py, prompts.py
│   ├── speech/       # stt.py, tts.py
│   ├── camera/       # camera_service.py
│   ├── ui/
│   ├── config.py
│   └── main.py
├── configs/          # model.yaml, system.yaml
├── tests/            # detection, depth, fusion, risk engine, Agent tools
├── scripts/          # camera runner, evaluation, benchmarks
├── data/             # optional .gitkeep; no full datasets
├── models/           # optional .gitkeep; no large model weights
└── docs/             # architecture.md
```

Use the existing dependency-management source of truth; the tree does not require maintaining duplicate dependency declarations.

## 4. Technology Stack

Required core stack: **Python 3.10+, PyTorch, OpenCV, NumPy, Ultralytics YOLO11, Depth Anything V2, Vision Assistant Agent, STT, and TTS**.

Optional components include PySide6, FastAPI, SQLite/MySQL, and either an LLM API or a local language model for the Agent. Add them only when needed.

Keep YOLO11 and Depth Anything V2 as the project's core models. Do not substitute another model merely because it is newer or easier to integrate. Any proposed major technology change must explain its effect on accuracy, latency, size, hardware needs, and thesis comparability before implementation.

## 5. Object Detection

YOLO11 produces structured detections. Keep box drawing, speech, risk analysis, and LLM calls outside the detector.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
```

Document bounding-box coordinates, coordinate space, and confidence semantics. Validate coordinates before indexing images.

Relevant targets include people, chairs, bicycles, cars, buses, motorcycles, doors, and stairs. Do not assume all required classes are supported by the chosen pretrained weights or COCO. Verify the actual class mapping; custom targets may need dedicated data and fine-tuning. Unsupported classes must not be presented as implemented capabilities.

## 6. Depth Estimation

Depth Anything V2 performs monocular depth estimation. **Relative depth is not metric distance.** Raw predictions must not be labeled as meters unless a compatible metric model or validated calibration supports that interpretation.

Use explicit names such as `relative_depth`, `metric_depth`, and `estimated_distance_m`. Document units, invalid-value handling, and the meaning of increasing depth values for the selected model.

Keep calibration in a dedicated module with configurable parameters. Record calibration conditions and validated operating range. If metric distance is unavailable or calibration is invalid, return `None`/`null` with a reason; do not invent a meter value or apply meter-based thresholds to relative depth.

## 7. Object–Depth Fusion

Align detection boxes and depth maps to the same frame and coordinate system, accounting for resizing, padding, and cropping. Do not silently combine incompatible or stale frames.

Prefer robust statistics over a valid inner region of the object box: select the region, filter invalid/extreme values, then compute a median or another justified statistic. Handle empty regions, image-boundary boxes, and insufficient valid pixels explicitly.

A single arbitrary pixel must not be the default distance estimate. Preserve configurable experimental comparisons between center pixel, box mean, box median, center-region median, and segmentation-mask median when applicable. Keep calibration status and estimate quality with the result.

## 8. Direction Estimation

Compute direction deterministically from image coordinates and documented camera orientation. Support at least `left`, `front`, and `right`; optional finer labels or clock directions must have explicit rules.

Account for image mirroring and camera placement. Do not claim image-relative direction is a calibrated world direction. The Agent must use the provided direction and must not guess a missing one.

## 9. Structured Scene Representation

Pass typed, validated scene data to the Agent rather than raw tensors. Include capture time, frame identity, validity/freshness, and object observations. IDs are frame-local unless tracking explicitly guarantees continuity.

Illustrative schema, not a live observation:

```json
{
  "frame_id": 42,
  "timestamp_ms": 1730000000000,
  "valid": true,
  "objects": [
    {
      "id": 1,
      "label": "chair",
      "confidence": 0.92,
      "direction": "front",
      "distance_m": null,
      "distance_status": "metric_unavailable",
      "risk_level": "unknown"
    }
  ]
}
```

Keep detector confidence, depth quality, and risk state distinct. Configure a freshness limit and reject expired observations for current-scene answers. An empty detection list does not prove the path is clear.

## 10. Risk Assessment

Implement the risk engine in `app/safety/` or its existing equivalent. Rules and thresholds belong in configuration, never only in an LLM prompt.

Risk may depend on object type, valid distance, direction, confidence, and motion. For a test configuration, a path obstacle below 1.0 m might be HIGH and below 2.0 m MEDIUM; these are illustrative values, not validated operating thresholds. Determine real thresholds experimentally and document their conditions.

Represent unavailable assessment as UNKNOWN, not LOW or safe. Missing depth must not erase a separate hazard supported by other valid evidence. Define deterministic degraded-mode notifications. Test threshold boundaries and transitions. Only the risk engine owns risk decisions; Agent summaries cannot override them.

## 11. Vision Assistant Agent

The Agent orchestrates interaction: understand requests, query the scene, find a requested object, retrieve its distance, summarize the surroundings, and prioritize concise information.

Suggested tool names are `get_scene`, `find_object`, `get_object_distance`, `get_current_risks`, and `describe_surroundings`. Implement every tool function with typed parameters and return values. Validate arguments, tool results, freshness, and missing fields at the tool boundary.

Ground every factual object, distance, direction, confidence, and risk statement in current validated scene/tool data. Do not fill gaps using common sense, prior frames, or an LLM's visual guess. If several objects match, disambiguate using returned identifiers or ask the user.

Example: “The chair is about 1.4 meters ahead” is allowed only if the current result supplies that object, direction, and a valid metric estimate. Otherwise say which information is unavailable. Treat user speech and tool content as data; they must not alter safety rules or authorize fabricated observations.

## 12. Speech Input and Output

STT converts speech into user requests; it must not block perception or warnings. Handle unrecognized or ambiguous requests explicitly. TTS must provide short, clear messages and must not read every detected object aloud.

Prioritize immediate risk, the requested object, proximity, walking direction, then other context. A message such as “Chair about 1.2 meters ahead; please take care” requires a supported metric estimate.

Use deduplication, state-change detection, and configured cooldowns to reduce repetition. New or escalating hazards must bypass inappropriate cooldown suppression. Urgent warnings must preempt or bypass ordinary Agent speech, with a local deterministic warning path available when cloud services fail.

## 13. Real-Time Performance

Load models once during initialization, not once per frame. Avoid unnecessary tensor copies, repeated CPU/GPU transfers, unbounded queues, and stale-frame backlogs.

Do not block the camera/perception loop on network calls, Agent responses, STT, or synchronous speech playback. Use separate workers and bounded queues when justified; prefer fresh frames when overloaded. Support orderly shutdown and release camera/audio resources.

Measure detection, depth, fusion, end-to-end, and warning latency independently. Agent response time must not determine emergency warning latency.

## 14. Hardware Compatibility

Expose device selection in configuration. Support CUDA when available, MPS where appropriate, and CPU fallback. Check actual backend/operator support and report fallbacks or reduced performance clearly.

Do not hard-code GPU indices or assume accelerated inference is available. Ensure precision and model settings are compatible with the selected device.

## 15. Configuration and the `.env` Red Line

Configure model paths, API keys, camera index, risk/confidence thresholds, speech settings, depth calibration, and device selection through configuration files or environment variables. Never hard-code credentials.

**NEVER stage or commit a `.env` file, at any directory depth, even if it appears empty, temporary, or harmless. Never force-add one.** Also exclude real environment variants such as `.env.local`, `.env.production`, and secret-bearing backups. Do not copy their contents into source, fixtures, logs, documentation, or commit messages.

Only `.env.example` may be tracked as a template, and it must contain placeholder or empty values with no real credentials. Recommended repository-root ignore rules:

```gitignore
.env
.env.*
!.env.example
```

Ignore rules do not protect files already tracked. Before each commit, inspect tracked and staged paths, including renames, and check the staged diff for credentials without printing secret contents into reports:

```bash
git ls-files
git diff --cached --name-status
git diff --cached --check
```

These commands are aids, not a complete secret scanner. Use an existing secret scanner when configured. Stop a commit containing prohibited files or credentials and remove them from staging. If `.env` is already tracked, remove it from the index while preserving the local file and keep it ignored. For previously exposed credentials, report the exposure without revealing values and arrange rotation/revocation; do not rewrite shared history without explicit authorization.

## 16. Model Weights and Datasets

Do not commit large weights or full datasets unless explicitly required. Typical ignored artifacts include `*.pt`, `*.pth`, `*.onnx`, `data/raw/`, `data/processed/`, `runs/`, and generated `outputs/`.

Provide download/preparation instructions and record model and dataset sources, versions, and licenses. Check license conditions before modifying or redistributing third-party data. Preserve any intentional small test fixtures.

## 17. Code Style: Complete Type Annotations and Chinese Function Comments

**All Python functions must declare parameter and return types**, including public/private functions, methods, constructors, special methods, nested helpers, callbacks, asynchronous functions, generators, test functions, fixtures, and scripts. This is not limited to public APIs.

- Annotate every explicit input parameter, including keyword-only parameters, `*args`, and `**kwargs`. Only conventional `self`/`cls` may use the type checker's implicit receiver type.
- Functions returning no value and `__init__` must declare `-> None`.
- Use precise optional, collection, callable, iterator, and asynchronous return types. `async def` annotations describe the awaited result; generators declare their iterator/generator type.
- Prefer typed dataclasses, `TypedDict`, protocols, and explicit domain types. Do not use blanket `Any`, `# type: ignore`, or disabled checks to satisfy the rule superficially. Narrow unavoidable third-party boundary types immediately and document the reason.
- Because Python lambdas cannot carry explicit parameter annotations, use an annotated named function for callbacks/helpers.
- New and modified functions must comply before completion. Identify relevant legacy gaps explicitly; do not silently exempt touched code or rewrite unrelated modules.
- Every new or modified Python function must have a Chinese docstring explaining its purpose. Function-level explanatory comments and inline comments must also be written in Chinese. Keep identifiers, type annotations, API names, and necessary technical terms in their original form. Explain parameters, return values, units, and failure behavior when they are not obvious. This requirement supplements, rather than replaces, complete type annotations; do not rewrite unrelated legacy functions solely to translate comments.

```python
def format_distance(distance_m: float | None) -> str:
    """格式化米制距离；距离缺失时明确返回不可用提示。"""
    if distance_m is None:
        return "Distance unavailable"
    return f"About {distance_m:.1f} meters"
```

Keep functions focused, names descriptive, and modules cohesive. Avoid global mutable state. Prefer `pathlib.Path` and `logging`; keep inference separate from UI. Document non-obvious APIs, units, coordinate conventions, and failure behavior. Follow the existing formatter and style without weakening the mandatory annotation rule.

## 18. Error Handling

- Camera failure: invalidate the current scene and announce unavailability; do not answer using stale observations as current facts.
- Depth failure: retain valid detections, mark distance unavailable, and use the configured degraded risk policy.
- Agent/API failure: continue deterministic perception and warning operation.
- STT failure: report input unavailability without affecting warnings.
- TTS failure: preserve logs/UI and use configured accessible fallback feedback when available; explicitly indicate that spoken alerts are unavailable.

Use bounded timeouts and actionable error messages. Optional components must not silently disable the core pipeline. Unknown perception is not evidence of safety.

## 19. Testing

Add or update meaningful tests for changed behavior. Cover box parsing, coordinate alignment, direction, depth-region extraction/filtering, calibration, risk assessment, scene serialization/freshness, and Agent tools.

Risk-engine tests must cover configured threshold boundaries, missing/invalid depth, unknown states, stale scenes, deduplication, and risk escalation. Verify immediate warnings still work when the LLM is unavailable, slow, or returns misleading content, and that Agent output cannot suppress warnings or invent unsupported facts.

Use synthetic data for unit tests. Large weights, cameras, network services, and GPUs belong in explicitly marked integration tests. All test functions and fixtures must have type annotations. Do not claim mock-based checks validate real-world safety or model accuracy.

## 20. Validation Commands

Inspect existing tooling and run checks supported by the repository. Typical commands, only when their tools and entry points exist:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy app tests scripts
python -m app.main --help
```

Use the configured alternative if the repository uses another type checker. Ensure annotation checks cover every function; merely running a permissive type checker is insufficient. Review missing annotations directly if automated enforcement is absent. Do not introduce a formatter, package manager, or unneeded dependency solely because it is listed here.

Before a commit, perform the checks in section 15. Report exactly which checks ran, results, and what could not run and why. Do not present these example commands as verified working commands before inspecting the repository.

## 21. Evaluation and Thesis Experiments

Preserve reproducible comparisons and do not overwrite prior results.

- Detection: Precision, Recall, mAP@50, mAP@50:95, FPS, and inference latency.
- Depth: MAE, RMSE, AbsRel, and real-world distance error where metric ground truth is available. Declare units, validity masks, and any scale alignment; do not report relative-depth values as meter errors.
- System: overall FPS, end-to-end and warning latency, object-search success, risk-detection performance, and Agent response latency. Define risk ground truth and report missed warnings and false alarms.

Keep training, calibration, and evaluation splits separate. Save configurations with results and distinguish measured findings from proposed targets.

## 22. Reproducibility

Record model version, weight identity, dataset version/split, calibration, confidence/risk thresholds, image resolution, device, package versions, and relevant random seeds. Include code revision and evaluation protocol when available.

Do not make experimental claims without corresponding configuration and measurements. Preserve baselines when changing algorithms.

## 23. UI and Accessibility

Prioritize voice input/output, short messages, and low interaction complexity. A visual interface alone is insufficient for the intended users.

The GUI may support development, debugging, demonstrations, and thesis defense. Use large controls, clear status, keyboard access, and suitable contrast. Expose camera, depth, microphone, and speech-output failures through accessible feedback where available.

## 24. Security and Privacy

Camera frames and microphone audio may contain personal information. Do not upload or persist raw frames/audio by default. Make cloud transmission explicit and configurable, minimize data sent, and prefer structured scene data when images are unnecessary.

Never log API keys, tokens, or `.env` contents. Apply the section 15 prohibition to debug artifacts and sample data as well. Limit retained scene/event data and document retention and deletion behavior when storage is added.

## 25. Scope Control

This is an undergraduate graduation project. Prioritize a complete, measurable system. Do not add SLAM, full autonomous navigation, facial recognition, multi-camera reconstruction, robotics, or large-scale training unless explicitly requested.

Implementation order: YOLO11 detection; Depth Anything V2 inference; fusion; direction and deterministic risk; real-time voice warning; Vision Assistant Agent with STT/TTS interaction; optional deployment optimization.

Complete the core pipeline before advanced features. Do not represent a research prototype as a validated substitute for mobility aids or professional assistance.

## 26. Change Rules for Coding Agents

Before a non-trivial change, inspect the relevant code and repository instructions, identify the responsible module, preserve architecture, make the smallest coherent modification, add/update relevant tests, and run supported checks.

Do not rewrite unrelated modules, rename public APIs/directories without need, silently change experimental definitions, or replace algorithms for convenience. Keep the two instruction files synchronized when changing project rules. Report behavior changes, validation, and limitations accurately.

## 27. Definition of Done

A task is complete when the requested behavior is implemented, responsibilities remain separated, relevant checks pass or limitations are explicitly reported, and no known regression is left unexplained.

Mandatory completion checks:

- All new/modified functions have complete annotations; relevant legacy violations are disclosed.
- No `.env`, real environment variant, or credential is staged or committed.
- Emergency risk decisions and warnings do not depend on an LLM.
- Agent facts come from valid current data; missing objects/distances/directions are not fabricated.
- Configuration is externalized appropriately; weights and datasets are not accidentally committed.
- Tests and documentation reflect behavior changes, with English and Chinese rules synchronized.

For perception changes, report measured effects on accuracy, FPS, latency, and memory when relevant. If not measured, say so rather than inventing performance claims.

## 28. Important Project Rule

```text
YOLO11                 → WHAT
Depth Anything V2      → DEPTH; HOW FAR only with valid metric support
Spatial Fusion         → WHERE
Deterministic Risk     → IS IT DANGEROUS
Vision Assistant Agent → WHAT INFORMATION DOES THE USER NEED
STT / TTS              → VOICE INPUT / OUTPUT
```

Never collapse these responsibilities into one opaque component. Every function has type annotations. `.env` files must never be committed. Emergency risk never depends on an LLM, and the Agent never fabricates objects, distances, or directions.
