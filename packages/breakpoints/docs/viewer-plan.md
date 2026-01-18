# Viewer Enhancement Plan

## Goals
- Render breakpoint context files in the UI.
- Support markdown rendering and code syntax highlighting.
- Add a safe API endpoint for context file retrieval.
- Define a stable breakpoint payload structure for context.

## API Changes
- Add `GET /api/breakpoints/:id/context?path=...` for file content.
- Validate that requested files are listed in the breakpoint payload.
- Enforce repo-root path restrictions and allowlist extensions.
- Return metadata: `path`, `format`, `language`, `content`.

## UI Changes
- Show a "Context" panel with file list from payload.
- Render markdown using a client renderer.
- Render code with syntax highlighting.
- Disable feedback actions when breakpoint is not waiting.

## Payload Structure
Prefer:
```json
{
  "context": {
    "runId": "run-...",
    "files": [
      { "path": "path/to/file.md", "format": "markdown" },
      { "path": "path/to/file.js", "format": "code", "language": "javascript" }
    ]
  }
}
```

Fallback support for legacy `context.paths` (array of strings).
