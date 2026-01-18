# Implementation Plan

## Phase 1: Data model + migrations
- Define schema for breakpoints, feedback, events, and queue jobs.
- Provide initial SQL migration.

## Phase 2: API service
- Implement REST endpoints for agents and humans.
- Add auth middleware (simple token-based for now).
- Persist events and state changes.

## Phase 3: Queue worker
- Implement job polling and execution.
- Handle TTL expiration and notifications (stubbed).
- Ensure idempotent updates.

## Phase 4: Web UI
- Provide list, detail, feedback, and release actions.
- Poll API for updates.

## Phase 5: Docs + scripts
- README with setup/run instructions.
- Dev scripts for Windows and Unix.
