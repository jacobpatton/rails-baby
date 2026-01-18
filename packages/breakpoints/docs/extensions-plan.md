# Extensions Framework Plan

## Goals
- Provide a registry of extensions with hooks.
- Store extension configs in the database.
- Support extension-specific polling (e.g., Telegram).
- Provide UI + API for enabling/configuring extensions.

## Hook points
- `onBreakpointCreated(breakpoint)` for dispatch.
- `onBreakpointReleased(breakpoint)` for notifications.
- `poll(db, config)` for periodic feedback checks.

## Configuration
- Table: `extensions_config` with `name`, `enabled`, `config_json`, timestamps.
- API: `GET /api/extensions` and `POST /api/extensions/:name`.
- UI: settings panel to view and update configs.

## Telegram Extension
- Config: `token`, `chatId`, `allowedUserId`.
- Dispatch: send message with breakpoint id + summary.
- Poll: check updates; if "release <id>", mark released + add feedback.
