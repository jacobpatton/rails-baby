- [x] add hooks (using shell scripts) on-breakpoint in the babysitter breakpoint skill (so that the skill is instructed to call the main hook script that dispatches the per repo or per user hooks for the breakpoint), then have the breakpoint cli just one potential hook implementation. also write the main hook scripts ✅ Completed 2026-01-19 - See HOOK_IMPLEMENTATION_2026-01-19.md and HOOKS.md
- [x] find potential gaps, bugs, inconsistencies, missing functionalty, copy-paste leftovers, inline todos in the plugin and fix them. ✅ Completed 2026-01-19 - See PLUGIN_CONSISTENCY_REPORT_2026-01-19.md for full results
- [x] generalize hook system so that processes can call hooks directly and not just through the skill instructions. (for example, pre-commit hook, pre-branch hook, post-planning-hook, score hook, etc.), make sure to adapt the babysitter skill to use the new hook system and add some examples in the instructions of using it in various use cases.  add native hooks (shell scripts) for start-run, resume-run, step-dispatch, orchestration-iteration-start, orchestration-iteration-end (on-breakpoint ✅ done), on-run-start, on-run-complete, on-run-fail, on-task-start, on-task-complete, etc. ✅ Completed 2026-01-19 - See HOOK_SYSTEM_V2_SUMMARY.md and updated HOOKS.md
- [x] refactor and implement all the existing AND missing functionality in(breakpoints, orchestration through sdk, etc.) through native hooks and not just skill instructions and embedded shell logic. ✅ Completed 2026-01-19 - See NATIVE_HOOKS_INTEGRATION_2026-01-19.md for full implementation details
- [x] in the breakpoint worker, when using the telegram extension. when it says: "Telegram connected. I will notify you about breakpoints here.", it should also return the existing waiting breakpoints and allow the user to select one to preview and approve, etc. ✅ Completed 2026-01-20 - Enhanced connection message to show all waiting breakpoints with list, preview, and file commands. See packages/breakpoints/TELEGRAM_ENHANCEMENT_2026-01-20.md
- [ ] create the standard library of processes. core, per roles, per methodology, etc
- [x] allow packaging processes with skills (/skill/some-skill/process/some-process.js) , meaning documenting it and adding it to the babysitter skill instructions. ✅ Completed 2026-01-20 - Created comprehensive PACKAGING_PROCESSES_WITH_SKILLS.md documentation, comprehensive advanced example process (tdd-quality-convergence.js) demonstrating agent planning, TDD workflow, quality convergence, parallel execution, and breakpoints. Fixed skill invocation pattern (removed incorrect `args` field, added `instructions` to context). Updated SKILL.md and ADVANCED_PATTERNS.md with correct patterns. See PROCESS_EXAMPLES_WITH_AGENT_SKILL_INVOCATION.md
- [x] populate the plugins/babysitter/commands/ and files with the adapted instructions and verify setup-babysitter-run and the on-stop hooks are implemented correctly. ✅ Completed 2026-01-20 - Created setup-babysitter-run-resume.sh, updated command docs with architecture sections, verified all hooks. See COMMANDS_AND_HOOKS_VERIFICATION_2026-01-20.md
- [x] write a spec for babysitter plugin ✅ Completed 2026-01-20 - Created comprehensive BABYSITTER_PLUGIN_SPECIFICATION.md covering architecture, components, hooks, CLI, workflows, API reference, and best practices for Version 4.0
- [x] README.md for the babysitter plugin ✅ Completed 2026-01-20 - Created comprehensive README.md with quick start, core concepts, usage examples, CLI reference, advanced features, documentation links, troubleshooting, and contribution guidelines
- [ ] research each of these methodologies and techniques and create an example process for each of them:
ralph.js (simple ralph wiggum loops that iterates until DONE is emmited)
plan-and-execute.js
self-assessment.js
state-machine-orchestration.js
consensus-and-voting-mechanisms.js
devin.js ( plan, code, debug, and deploy )
base44.js
adversarial-spec-debates.js ( adversarial spec debates between two or more LLMs or even other coding agents )
tdd.js ✓ DONE - see tdd-quality-convergence.js
graph-of-thoughts.js
evolutionary.js
build-realtime-remediation.js ( pulls ci state and remediates it )
agile.js ( agile development loop with springs, release cycles and product->develop->qa->release iterations )
top-down.js
bottom-up.js
score-gated-iterative-convergence.js ✓ DONE - see tdd-quality-convergence.js (combines TDD + quality convergence)
