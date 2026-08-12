# CardScope v0.7 Platform Consolidation Design

## Goal

Make CardScope v0.7 the only deployed service. Testers sign in through the
web application to run either traditional frame detection or reference-image
registration, correct geometry manually, and submit reviewed samples for model
training. Each completed inspection has an access-controlled JSON result file.

## Scope

- Remove the legacy FastAPI application entry point and its deployment files.
- Promote the v0.7 application, its web UI, deployment scripts, and
  `ml_backend` to the canonical contents of `算法`.
- Preserve the existing reference-registration algorithm by adapting it to the
  v0.7 service and web application; do not run it as a second HTTP service.
- Keep all mutable state outside release code in the v0.7 workspace.

## Application Layout

`算法/platform_server.py` is the only application entry point. It starts the
v0.7 HTTP server, website, batch worker, model pre-label engine, feedback
review flow, and auto-training worker. `算法/ml_backend` is the only shipped
model and inference package. `算法/platform_app`, `算法/studio`, `算法/web`, and
`算法/deployment/server` remain release code.

Legacy FastAPI files (`app.py`, `api_pipeline.py`, its Docker image, and its
runtime-specific deployment configuration) are removed only after the v0.7
integration tests pass. Existing user worktree changes are not reverted.

## Detection Modes

### Traditional Frame Detection

The existing v0.7 upload flow remains available. The tester uploads one card
photo; the service detects outer corners, rectifies the card, detects inner
lines, calculates centering, saves the inspection, and displays the result.

### Reference-Image Registration

The website presents this as a separate workflow, not as extra fields hidden
inside the traditional uploader. It requires two files from the tester:

1. A photographed card image.
2. The matching standard/reference card image.

The service normalizes both inputs, runs the shared outer-frame and
rectification inference once per image as needed, then applies the existing
feature/phase/ECC registration and fusion algorithm. The resulting inspection
stores the selected mode, the reference upload metadata, registration offset,
confidence, diagnostics, and a registration overlay. A failed registration is
recorded as a failed inspection with an explicit error code rather than a
fabricated measurement.

The two uploads are handled as one short-lived, authenticated registration
job. Separate binary upload endpoints prevent image bytes from being placed in
JSON and avoid a second public service. The job is processed only after both
files have passed size and image-type validation.

## Results And Access Control

On every terminal inspection state, the service atomically writes:

`<workspace>/exports/inspections/<inspection-id>/result.json`

The document contains the public inspection result, mode, model version,
geometry, centering or registration result, and relative API image links. It
contains no absolute server paths, session tokens, passwords, or private model
state. The inspection response includes a protected result-download URL.

The download handler authenticates the session and reuses inspection ownership
checks: an enterprise can access only its own result, while authorized
administrators can access all records. A missing result file returns a normal
not-found response and never exposes a filesystem path.

## Manual Annotation And Training

Both result views provide editable outer corners and inner guide lines. The
reference-registration view also shows its reference image and registration
overlay separately from the traditional outer/inner-frame display. A tester
can submit corrected geometry and notes, but cannot train or deploy a model.

Corrections enter the existing feedback review queue. An administrator either
rejects, requests annotation, discards, or approves the feedback. Only
approved geometry is exported into the training pool and can be used by the
existing auto-training workflow. Promotion of a trained model continues to be
an administrator operation.

## Deployment

The supported remote target is Ubuntu 22.04 or 24.04. `systemd` runs
`platform_server.py` on localhost and Caddy terminates HTTPS. Persistent state
remains at `/var/lib/cardscope/platform_workspace`; release code remains under
`/opt/cardscope/releases`. Release updates preserve images, SQLite data,
feedback, result JSON files, and training state.

## Verification

- Unit tests verify both detection modes, invalid/missing reference upload
  handling, and no inference before both reference files exist.
- Tests verify atomic JSON output and that another enterprise receives no
  result-file access.
- Tests verify reference-registration geometry is stored in the inspection and
  that approved manual corrections reach the training-export path.
- A deployment smoke check starts the v0.7 service, checks health, logs in,
  uploads a traditional image and a two-file registration job, and retrieves
  each JSON result through the authenticated endpoint.
