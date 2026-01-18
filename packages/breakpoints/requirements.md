# Breakpoint Manager Requirements

## Overview
Provide a web app, API, queue, and database to manage "breakpoints" posted by agents. A breakpoint pauses agent execution until a human reviews, provides feedback, and releases it. Agents poll the API to detect releases.

## Goals
- Allow agents to create breakpoints with context and optionally request specific feedback.
- Provide a human-facing UI to view, filter, and resolve waiting breakpoints.
- Offer a stable API for agents to poll and for UIs/tools to manage breakpoints.
- Ensure reliable ordering and processing via a queue-backed workflow.

## Non-goals
- Building a full agent runtime or task runner.
- Replacing existing ticketing systems.

## Actors
- Agent: creates and polls breakpoints.
- Human reviewer: views, comments, and releases breakpoints.
- System: manages state transitions, notifications, and auditing.

## User Stories
- As an agent, I can POST a breakpoint with payload and metadata.
- As an agent, I can poll for status and retrieve feedback when released.
- As a human, I can see all waiting breakpoints and open one to review.
- As a human, I can add feedback and release a breakpoint.
- As an admin, I can search and filter breakpoints by status, tags, or time.

## Functional Requirements
### Breakpoint Lifecycle
- States: `waiting`, `released`, `expired`, `cancelled`.
- Transition rules:
  - `waiting` -> `released` via human action.
  - `waiting` -> `expired` via TTL policy.
  - `waiting` -> `cancelled` via agent/system action.
- Each transition is recorded with timestamp and actor.

### Web Application
- List view of breakpoints with filters (status, tags, agentId, createdAt).
- Detail view showing:
  - Payload/context from the agent.
  - History of events.
  - Feedback/comments.
  - Release action.
- Release flow requires optional comment and confirmation.
- Real-time updates are optional; polling or refresh is acceptable.

### API
- Agents poll the API until the breakpoint is released.
- API returns payload, status, and any human feedback.
- Errors should be explicit (not found, unauthorized, invalid state).

### Queue
- A queue component handles side effects:
  - Notification or webhook on new breakpoint.
  - TTL/expiration processing.
  - Audit/event persistence if needed.
- Queue processing must be idempotent.

### Database
- Store breakpoints, events, and feedback.
- Store agent metadata (agentId, runId, tags).

## API Endpoints (Initial)
### Agent-facing
- `POST /api/breakpoints`
  - Body: `{ agentId, runId?, title?, payload, tags?, ttlSeconds? }`
  - Response: `{ breakpointId, status, createdAt }`
- `GET /api/breakpoints/{id}`
  - Response: `{ breakpointId, status, payload, feedback?, releasedAt? }`
- `GET /api/breakpoints/{id}/status`
  - Response: `{ status, updatedAt }`
- `POST /api/breakpoints/{id}/cancel`
  - Response: `{ status }`

### Human-facing
- `GET /api/breakpoints?status=waiting&tag=...`
- `POST /api/breakpoints/{id}/feedback`
  - Body: `{ comment, release?: boolean }`
  - If `release=true`, status transitions to `released`.

### Admin/System
- `POST /api/breakpoints/{id}/expire`

## Data Model
- Breakpoint:
  - `id` (string, unique)
  - `status` (enum)
  - `agentId` (string)
  - `runId` (string, optional)
  - `title` (string, optional)
  - `payload` (json)
  - `tags` (array of string)
  - `createdAt`, `updatedAt`
  - `releasedAt`, `expiredAt`, `cancelledAt`
  - `ttlSeconds` (optional)
- Feedback:
  - `id`, `breakpointId`, `author`, `comment`, `createdAt`
- Event:
  - `id`, `breakpointId`, `type`, `actor`, `timestamp`, `metadata`

## Polling Behavior
- Agents poll `GET /api/breakpoints/{id}/status` on a fixed interval.
- Recommended interval: 2-10 seconds (configurable).
- API should return `released` with feedback if available.

## Nonfunctional Requirements
- Availability: 99.5%+ for API.
- Performance: p95 < 200ms for read endpoints under normal load.
- Scalability: support 10k+ breakpoints/day; horizontal scaling for API and queue.
- Security: auth required for human endpoints; agent auth via token.
- Auditability: all state transitions recorded.
- Data retention: configurable TTL for old breakpoints and events.

## Observability
- Structured logs for API and queue.
- Metrics: request rate, error rate, queue lag, pending breakpoints.
- Tracing optional; include correlation ids in responses.

## Open Questions
- Authentication model for agents (API key vs. signed tokens).
- UI access control and role model.
- Required payload schema (free-form vs. typed).
- Notification channels (email, Slack, webhook).
- Should agents support long-polling or SSE?
