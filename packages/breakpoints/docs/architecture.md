# Architecture

## Overview
The breakpoint manager is composed of four parts:
- API service: handles agent and human requests.
- Web UI: lists and resolves breakpoints.
- Queue worker: processes async work (TTL expiration, notifications).
- Database: stores breakpoints, events, and feedback.

## Components
### API Service
- Node.js service exposing REST endpoints.
- Validates requests and persists state transitions.
- Emits queue jobs for expiration and notifications.

### Web UI
- Static HTML/JS app that calls the API.
- Supports listing, filtering, viewing details, and releasing breakpoints.

### Queue Worker
- Polls a jobs table in the database for pending work.
- Executes TTL expiration and notification jobs.
- Idempotent job execution; job status recorded in DB.

### Database
- SQLite for local development (single file DB).
- Tables for breakpoints, feedback, events, and jobs.

## Data Flow
1. Agent POSTs a breakpoint to the API.
2. API writes breakpoint + event rows.
3. API enqueues TTL job (if configured).
4. Human reviews in the Web UI and releases breakpoint.
5. API writes feedback and release events.
6. Agent polls for status until released.

## Security
- API token for agent requests.
- Basic auth or token for human UI in production (placeholder for now).
- All transitions are recorded with actor identifiers.

## Deployment
- API and worker run as separate processes.
- Web UI served from the API (static) or a separate host.
- SQLite used for dev; can be swapped for Postgres by replacing db adapter.
