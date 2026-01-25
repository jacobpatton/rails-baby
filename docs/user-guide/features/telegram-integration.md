# Telegram Integration for Breakpoint Notifications

**Version:** 1.0
**Last Updated:** 2026-01-25
**Category:** Feature Guide

---

## Overview

The Telegram integration for Babysitter breakpoints allows you to receive notifications and interact with breakpoints directly from your mobile device or desktop Telegram app. Instead of keeping a browser tab open, you can review breakpoints, download context files, and release approvals through Telegram messages.

### Why Use Telegram Integration

- **Mobile Accessibility**: Approve breakpoints from anywhere using your phone
- **Instant Notifications**: Receive push notifications when breakpoints are created or released
- **Quick Responses**: Release breakpoints by simply replying to a message
- **File Access**: Download and preview context files directly in Telegram
- **Asynchronous Workflows**: Enable team members to review and approve without being at their desk

---

## Setup Instructions

### Step 1: Create a Telegram Bot via @BotFather

1. Open Telegram and search for `@BotFather` (or click [this link](https://t.me/botfather))
2. Start a conversation and send the command:
   ```
   /newbot
   ```
3. BotFather will ask you to choose a name for your bot. Enter a descriptive name:
   ```
   My Breakpoint Bot
   ```
4. BotFather will ask for a username. This must end with `bot`:
   ```
   my_breakpoint_bot
   ```
5. BotFather will provide your **bot token**. It looks like this:
   ```
   1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ123456789
   ```

**Important:** Keep your bot token private. Anyone with access to this token can control your bot.

### Step 2: Get Your Telegram Username

Your Telegram username is used to authorize which users can interact with the bot.

1. Open Telegram and go to **Settings**
2. Your username is displayed near the top (e.g., `@your_username`)
3. If you do not have a username set, tap to create one

### Step 3: Enable the Telegram Extension

Run the following command to configure and enable the Telegram extension:

```bash
breakpoints extension enable telegram --token <your-bot-token> --username <your-telegram-username>
```

**Example:**

```bash
breakpoints extension enable telegram \
  --token "1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ123456789" \
  --username "johndoe"
```

### Step 4: Connect to Your Bot

1. Search for your bot in Telegram using the username you created (e.g., `@my_breakpoint_bot`)
2. Start a conversation with your bot by clicking **Start** or sending:
   ```
   /start
   ```
3. The bot will confirm the connection and display any waiting breakpoints

**Expected response:**

```
Telegram connected. I will notify you about breakpoints here.

You have 0 waiting breakpoints.

Commands:
- list: Show all waiting breakpoints
- preview <number>: View breakpoint details
- file <number>: Download a context file
- Reply to any breakpoint message to release it
```

---

## Features

### Automatic Notifications

Once connected, you will automatically receive notifications for:

- **New breakpoints**: When a workflow creates a breakpoint requiring your approval
- **Released breakpoints**: Confirmation when a breakpoint is released (either by you or another reviewer)

**New breakpoint notification example:**

```
New Breakpoint

Title: Plan Approval
ID: abc-123-def
Run: run-20260125-refactor
Created: just now

Question: Review the implementation plan. Should I proceed with the refactoring?

Context files:
1. artifacts/plan.md (markdown)
2. code/main.js (javascript)

Reply to this message to release the breakpoint.
```

### Viewing Breakpoints

**List all waiting breakpoints:**

Send `list`, `ls`, or `waiting` to see all breakpoints awaiting approval:

```
list
```

**Response:**

```
You have 2 waiting breakpoints:

1. Approve refactoring plan
   ID: abc-123
   Run: run-20260125-refactor
   Created: 5 mins ago
   Question: Should I proceed with this refactoring?

2. Review API changes
   ID: def-456
   Run: run-20260125-api
   Created: 2 mins ago
   Question: Are these API changes acceptable?
```

**Preview a specific breakpoint:**

Send `preview <number>` or `show <number>` to view full details:

```
preview 1
```

**Response:**

```
Breakpoint Details

Title: Approve refactoring plan
ID: abc-123
Run: run-20260125-refactor
Created: 5 mins ago
Status: waiting

Question: Should I proceed with this refactoring?

Context files:
1. artifacts/plan.md (markdown)
2. artifacts/risk-analysis.md (markdown)
3. code/main.js (javascript)

Reply to release this breakpoint.
```

### Releasing Breakpoints

There are multiple ways to release a breakpoint:

**Method 1: Reply to a breakpoint message**

Swipe to reply on any breakpoint notification and type your feedback:

```
Looks good, proceed with the refactoring.
```

**Method 2: Send the breakpoint ID**

Simply send the breakpoint ID:

```
abc-123
```

**Method 3: Release the most recent breakpoint**

Send any text message (not a command) to release the most recently created breakpoint:

```
Approved
```

**Release confirmation:**

```
Breakpoint released

ID: abc-123
Feedback: Looks good, proceed with the refactoring.
```

### File Preview and Download

**Download a context file by number:**

```
file 1
```

The bot sends the file as a document attachment that you can open or save.

**Download a context file by path:**

```
file artifacts/plan.md
```

**View file content inline:**

For shorter files, use `raw` to display content with syntax highlighting:

```
raw artifacts/plan.md
```

**Response:**

```
artifacts/plan.md

# Implementation Plan

## Phase 1: Database Migration
- Create new schema
- Migrate existing data
...
```

Note: If the file is too large for inline display, use `file` to download it instead.

---

## Commands Reference

| Command | Aliases | Description |
|---------|---------|-------------|
| `list` | `ls`, `waiting` | Show all waiting breakpoints |
| `preview <number>` | `show <number>` | View full details of a breakpoint |
| `file <number>` | - | Download context file by its index number |
| `file <path>` | - | Download context file by its path |
| `raw <path>` | - | View file content inline with syntax highlighting |
| `/start` | - | Connect to the bot and show waiting breakpoints |
| `/help` | - | Display available commands |

### Releasing Breakpoints

| Action | Effect |
|--------|--------|
| Reply to a breakpoint message | Release that specific breakpoint with your reply as feedback |
| Send a breakpoint ID | Release the breakpoint with that ID |
| Send any text message | Release the most recent waiting breakpoint |

---

## Example Workflows

### Workflow 1: Mobile Plan Approval

A developer starts a Babysitter workflow that generates an implementation plan and requires approval.

```
[Telegram notification arrives]

Bot: New Breakpoint

     Title: Plan Approval
     ID: abc-123
     Run: run-20260125-feature
     Created: just now

     Question: Review the implementation plan for the user
     authentication feature. Approve to proceed?

     Context files:
     1. artifacts/plan.md (markdown)
     2. artifacts/risk-analysis.md (markdown)

You: preview 1

Bot: [Shows full breakpoint details]

You: file 1

Bot: [Sends plan.md as document]

[You review the plan on your phone]

You: [Reply to the breakpoint message]
     Plan looks good. Proceed with implementation.

Bot: Breakpoint released
     ID: abc-123
     Feedback: Plan looks good. Proceed with implementation.
```

### Workflow 2: Quick Approval from Notification

For simple approvals when you trust the context:

```
[Telegram notification arrives]

Bot: New Breakpoint

     Title: Deploy to staging
     ID: xyz-789
     Question: Tests passed. Deploy to staging environment?

You: Yes, deploy to staging

Bot: Breakpoint released
     ID: xyz-789
     Feedback: Yes, deploy to staging
```

### Workflow 3: Reviewing Multiple Breakpoints

When multiple workflows are waiting for approval:

```
You: /start

Bot: Telegram connected.

     You have 3 waiting breakpoints:

     1. Code Review - API Refactor
        ID: bp-001
        Run: run-api-refactor
        Question: Approve the API changes?

     2. Plan Approval - Database Migration
        ID: bp-002
        Run: run-db-migration
        Question: Proceed with schema migration?

     3. Deploy Approval
        ID: bp-003
        Run: run-deploy
        Question: Deploy v2.1.0 to production?

You: preview 2

Bot: [Shows details for Database Migration breakpoint]

You: file 1

Bot: [Sends first context file]

You: [Review file, then reply to preview message]
     Migration plan approved. Ensure backups are complete first.

Bot: Breakpoint released
     ID: bp-002
     Feedback: Migration plan approved. Ensure backups are complete first.

You: list

Bot: You have 2 waiting breakpoints:
     [Shows remaining breakpoints]
```

### Workflow 4: Downloading and Reviewing Code

When you need to examine code files before approval:

```
Bot: New Breakpoint

     Title: Code Implementation Review
     ID: impl-456
     Question: Review the authentication implementation

     Context files:
     1. src/auth/login.ts (typescript)
     2. src/auth/middleware.ts (typescript)
     3. tests/auth.test.ts (typescript)

You: raw src/auth/login.ts

Bot: src/auth/login.ts

     import { hash, compare } from 'bcrypt';
     import { createToken } from './jwt';

     export async function login(email: string, password: string) {
       const user = await findUserByEmail(email);
       ...

You: file 3

Bot: [Sends tests/auth.test.ts as document]

You: Implementation looks secure. Approved.

Bot: Breakpoint released
     ID: impl-456
```

---

## Troubleshooting

### Bot Not Responding to /start

**Symptom:** You send `/start` but the bot does not respond.

**Possible causes:**
1. The breakpoints service is not running
2. The Telegram extension is not enabled
3. Your username does not match the configured username

**Solutions:**

1. Verify the breakpoints service is running:
   ```bash
   curl http://localhost:3184/health
   ```

2. Check extension status:
   ```bash
   breakpoints extension list
   ```

3. Re-enable the extension with the correct username:
   ```bash
   breakpoints extension enable telegram --token <token> --username <username>
   ```

### Not Receiving Notifications

**Symptom:** Breakpoints are created but no Telegram notifications arrive.

**Possible causes:**
1. You have not connected to the bot with `/start`
2. The bot token is incorrect
3. Network connectivity issues

**Solutions:**

1. Send `/start` to your bot to establish the connection
2. Verify your bot token is correct with BotFather (`/mybots` -> select your bot)
3. Check that Telegram notifications are enabled in your device settings

### "Unauthorized" Error When Releasing Breakpoint

**Symptom:** Bot responds with "Unauthorized" when you try to release a breakpoint.

**Possible cause:** Your Telegram username does not match the configured username.

**Solution:**

Re-configure the extension with your exact Telegram username (without the @ symbol):

```bash
breakpoints extension enable telegram --token <token> --username your_actual_username
```

### Cannot Download Context Files

**Symptom:** The `file` command responds with "File not found" or no response.

**Possible causes:**
1. The file path is incorrect
2. The file does not exist in the run directory
3. The file type is not in the allowed list

**Solutions:**

1. Use `preview <number>` to see the correct file paths
2. Use the file number instead of the path: `file 1`
3. Verify the breakpoint includes context files in its payload

### Breakpoint Already Released

**Symptom:** You try to release a breakpoint but it has already been released.

**Explanation:** Another team member (or you via the Web UI) may have already released the breakpoint.

**Solution:** Use `list` to see current waiting breakpoints and verify which ones still need approval.

### Multiple Team Members Using the Same Bot

**Current limitation:** The Telegram extension is designed for single-user operation. Each configured username receives notifications for all breakpoints.

**For team usage:**

1. Each team member creates their own bot via @BotFather
2. Enable the extension with each team member's bot and username
3. Alternatively, use the Web UI at http://localhost:3184 for multi-user access

---

## Related Documentation

- [Breakpoints: Human-in-the-Loop Approval](./breakpoints.md) - Core breakpoints documentation
- [Run Resumption](./run-resumption.md) - Resume workflows after breakpoint approval
- [Process Definitions](./process-definitions.md) - Learn how to create workflows with breakpoints

---

## Summary

The Telegram integration extends Babysitter breakpoints to your mobile device, enabling you to receive notifications and approve workflows from anywhere. Set up is straightforward: create a bot via @BotFather, configure the extension, and connect by sending `/start`. Use commands like `list`, `preview`, and `file` to review breakpoints, and release them by replying with your feedback.
