/**
 * @process babysitter/tdd-quality-convergence
 * @description Advanced TDD workflow with quality convergence, planning, parallelization, and agent scoring
 * @inputs { feature: string, targetQuality: number, maxIterations: number }
 * @outputs { success: boolean, iterations: number, finalQuality: number, artifacts: object }
 */

import { defineTask } from '@a5c-ai/babysitter-sdk';

/**
 * TDD Quality Convergence Process
 *
 * Demonstrates:
 * - Agent-based planning
 * - TDD workflow (test → implement → refine)
 * - Quality convergence with agent scoring
 * - Parallel test execution
 * - Breakpoints for approval
 * - Iterative feedback loops
 *
 * @param {Object} inputs - Process inputs
 * @param {string} inputs.feature - Feature to implement
 * @param {number} inputs.targetQuality - Target quality score (0-100)
 * @param {number} inputs.maxIterations - Maximum convergence iterations
 * @param {Object} ctx - Process context
 * @returns {Promise<Object>} Process result
 */
export async function process(inputs, ctx) {
  const {
    feature = 'User authentication',
    targetQuality = 85,
    maxIterations = 5
  } = inputs;

  ctx.log(`Starting TDD quality convergence for: ${feature}`);
  ctx.log(`Target quality: ${targetQuality}, Max iterations: ${maxIterations}`);

  // ============================================================================
  // PHASE 1: PLANNING WITH AGENT
  // ============================================================================

  ctx.log('Phase 1: Generating implementation plan with agent');

  const planningResult = await ctx.task(agentPlanningTask, {
    feature,
    requirements: inputs.requirements || [],
    constraints: inputs.constraints || []
  });

  ctx.log(`Plan generated: ${planningResult.approach}`);

  // Breakpoint: Review plan before implementation
  await ctx.breakpoint({
    question: `Review the implementation plan for "${feature}" and approve to proceed?`,
    title: 'Implementation Plan Review',
    context: {
      runId: ctx.runId,
      files: [
        { path: `artifacts/plan.md`, format: 'markdown' }
      ]
    }
  });

  // ============================================================================
  // PHASE 2: TDD CONVERGENCE LOOP
  // ============================================================================

  let iteration = 0;
  let currentQuality = 0;
  let converged = false;
  const iterationResults = [];

  while (iteration < maxIterations && !converged) {
    iteration++;
    ctx.log(`\n=== Iteration ${iteration}/${maxIterations} ===`);
    ctx.log(`Current quality: ${currentQuality}/${targetQuality}`);

    // Step 1: Write/update tests based on plan and previous feedback
    const testsResult = await ctx.task(writeTestsTask, {
      feature,
      plan: planningResult,
      iteration,
      previousFeedback: iteration > 1 ? iterationResults[iteration - 2].feedback : null
    });

    ctx.log(`Tests written: ${testsResult.testCount} tests`);

    // Step 2: Run tests (expect failures on first iteration)
    const testRunResult = await ctx.task(runTestsTask, {
      testFiles: testsResult.testFiles,
      expectFailures: iteration === 1
    });

    ctx.log(`Test run: ${testRunResult.passed}/${testRunResult.total} passed`);

    // Step 3: Implement/refine code to pass tests
    const implementationResult = await ctx.task(implementCodeTask, {
      feature,
      plan: planningResult,
      tests: testsResult,
      testResults: testRunResult,
      iteration,
      previousFeedback: iteration > 1 ? iterationResults[iteration - 2].feedback : null
    });

    ctx.log(`Implementation: ${implementationResult.filesModified.length} files modified`);

    // Step 4: Run tests again
    const finalTestResult = await ctx.task(runTestsTask, {
      testFiles: testsResult.testFiles,
      expectFailures: false
    });

    ctx.log(`Final test run: ${finalTestResult.passed}/${finalTestResult.total} passed`);

    // Step 5: Parallel quality checks
    ctx.log('Running parallel quality checks');

    const [
      coverageResult,
      lintResult,
      typeCheckResult,
      securityResult
    ] = await ctx.parallel.all([
      () => ctx.task(coverageCheckTask, { testFiles: testsResult.testFiles }),
      () => ctx.task(lintCheckTask, { files: implementationResult.filesModified }),
      () => ctx.task(typeCheckTask, { files: implementationResult.filesModified }),
      () => ctx.task(securityCheckTask, { files: implementationResult.filesModified })
    ]);

    ctx.log(`Quality checks complete: coverage=${coverageResult.percentage}%, lint=${lintResult.issueCount} issues, types=${typeCheckResult.errorCount} errors, security=${securityResult.vulnerabilityCount} vulnerabilities`);

    // Step 6: Agent-based quality scoring
    ctx.log('Agent scoring implementation quality');

    const qualityScore = await ctx.task(agentQualityScoringTask, {
      feature,
      plan: planningResult,
      tests: testsResult,
      testResults: finalTestResult,
      implementation: implementationResult,
      qualityChecks: {
        coverage: coverageResult,
        lint: lintResult,
        typeCheck: typeCheckResult,
        security: securityResult
      },
      iteration,
      targetQuality
    });

    currentQuality = qualityScore.overallScore;

    ctx.log(`Quality score: ${currentQuality}/100`);
    ctx.log(`Agent analysis: ${qualityScore.summary}`);

    // Store iteration results
    iterationResults.push({
      iteration,
      quality: currentQuality,
      tests: testsResult,
      testResults: finalTestResult,
      implementation: implementationResult,
      qualityChecks: {
        coverage: coverageResult,
        lint: lintResult,
        typeCheck: typeCheckResult,
        security: securityResult
      },
      agentScore: qualityScore,
      feedback: qualityScore.recommendations
    });

    // Check convergence
    if (currentQuality >= targetQuality) {
      converged = true;
      ctx.log(`✓ Quality target reached! (${currentQuality} >= ${targetQuality})`);
    } else {
      ctx.log(`✗ Quality below target (${currentQuality} < ${targetQuality})`);
      ctx.log(`Agent recommendations: ${qualityScore.recommendations.slice(0, 3).join(', ')}`);

      // Breakpoint: Review iteration results before continuing
      if (iteration < maxIterations) {
        await ctx.breakpoint({
          question: `Iteration ${iteration} complete. Quality: ${currentQuality}/${targetQuality}. Continue iteration ${iteration + 1}?`,
          title: `Iteration ${iteration} Review`,
          context: {
            runId: ctx.runId,
            files: [
              { path: `artifacts/iteration-${iteration}-report.md`, format: 'markdown' },
              { path: `artifacts/quality-score.json`, format: 'code', language: 'json' }
            ]
          }
        });
      }
    }
  }

  // ============================================================================
  // PHASE 3: FINAL VERIFICATION
  // ============================================================================

  ctx.log('\nPhase 3: Final verification');

  // Parallel final checks
  const [
    finalTestResult,
    finalCoverageResult,
    integrationTestResult
  ] = await ctx.parallel.all([
    () => ctx.task(runTestsTask, { testFiles: iterationResults[iteration - 1].tests.testFiles }),
    () => ctx.task(coverageCheckTask, { testFiles: iterationResults[iteration - 1].tests.testFiles }),
    () => ctx.task(integrationTestTask, { feature })
  ]);

  // Agent-based final review
  const finalReview = await ctx.task(agentFinalReviewTask, {
    feature,
    iterations: iterationResults,
    finalQuality: currentQuality,
    targetQuality,
    converged,
    finalTests: finalTestResult,
    finalCoverage: finalCoverageResult,
    integrationTests: integrationTestResult
  });

  ctx.log(`Final review: ${finalReview.verdict}`);

  // Final breakpoint for approval
  await ctx.breakpoint({
    question: `Implementation complete. Quality: ${currentQuality}/${targetQuality}. ${finalReview.verdict}. Approve for merge?`,
    title: 'Final Implementation Review',
    context: {
      runId: ctx.runId,
      files: [
        { path: `artifacts/final-report.md`, format: 'markdown' },
        { path: `artifacts/coverage-report.html`, format: 'html' },
        { path: `artifacts/quality-history.json`, format: 'code', language: 'json' }
      ]
    }
  });

  // Return results
  return {
    success: converged,
    feature,
    iterations: iteration,
    finalQuality: currentQuality,
    targetQuality,
    converged,
    iterationResults,
    finalReview,
    artifacts: {
      plan: `artifacts/plan.md`,
      finalReport: `artifacts/final-report.md`,
      coverageReport: `artifacts/coverage-report.html`,
      qualityHistory: `artifacts/quality-history.json`
    },
    metadata: {
      processId: 'babysitter/tdd-quality-convergence',
      timestamp: ctx.now()
    }
  };
}

// ============================================================================
// TASK DEFINITIONS
// ============================================================================

/**
 * Agent-based planning task
 * Generates implementation plan using LLM reasoning
 */
export const agentPlanningTask = defineTask('agent-planner', (args, taskCtx) => ({
  kind: 'agent',
  title: `Plan implementation: ${args.feature}`,
  description: 'Generate detailed implementation plan with agent',

  agent: {
    name: 'feature-planner',
    prompt: {
      role: 'senior software architect and technical lead',
      task: 'Generate a detailed, actionable implementation plan for the requested feature using TDD principles',
      context: {
        feature: args.feature,
        requirements: args.requirements,
        constraints: args.constraints,
        methodology: 'Test-Driven Development (TDD)'
      },
      instructions: [
        'Analyze the feature requirements and constraints',
        'Break down the feature into testable units',
        'Define test cases that cover happy path, edge cases, and error conditions',
        'Outline the implementation approach following TDD red-green-refactor cycle',
        'Identify potential quality concerns (security, performance, maintainability)',
        'Generate acceptance criteria',
        'Provide estimated complexity and risk assessment'
      ],
      outputFormat: 'JSON with approach (string), testCases (array), implementationSteps (array), qualityConcerns (array), acceptanceCriteria (array), complexity (low|medium|high), risks (array)'
    },
    outputSchema: {
      type: 'object',
      required: ['approach', 'testCases', 'implementationSteps', 'acceptanceCriteria'],
      properties: {
        approach: { type: 'string' },
        testCases: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              scenario: { type: 'string' },
              expectedBehavior: { type: 'string' }
            }
          }
        },
        implementationSteps: { type: 'array', items: { type: 'string' } },
        qualityConcerns: { type: 'array', items: { type: 'string' } },
        acceptanceCriteria: { type: 'array', items: { type: 'string' } },
        complexity: { type: 'string', enum: ['low', 'medium', 'high'] },
        risks: { type: 'array', items: { type: 'string' } }
      }
    }
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['agent', 'planning']
}));

/**
 * Write tests task
 */
export const writeTestsTask = defineTask('write-tests', (args, taskCtx) => ({
  kind: 'node',
  title: `Write tests for ${args.feature}`,
  description: 'Generate test files based on plan',

  node: {
    entry: '.a5c/orchestrator_scripts/tdd/write-tests.js',
    args: [
      '--feature', args.feature,
      '--iteration', String(args.iteration),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['tests', 'tdd', `iteration-${args.iteration}`]
}));

/**
 * Run tests task
 */
export const runTestsTask = defineTask('run-tests', (args, taskCtx) => ({
  kind: 'node',
  title: 'Run test suite',
  description: `Run tests (expect failures: ${args.expectFailures})`,

  node: {
    entry: '.a5c/orchestrator_scripts/tdd/run-tests.js',
    args: [
      '--test-files', JSON.stringify(args.testFiles),
      '--expect-failures', String(args.expectFailures),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['tests', 'execution']
}));

/**
 * Implement code task
 */
export const implementCodeTask = defineTask('implement-code', (args, taskCtx) => ({
  kind: 'node',
  title: `Implement ${args.feature}`,
  description: `Write/refine implementation (iteration ${args.iteration})`,

  node: {
    entry: '.a5c/orchestrator_scripts/tdd/implement-code.js',
    args: [
      '--feature', args.feature,
      '--iteration', String(args.iteration),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['implementation', 'tdd', `iteration-${args.iteration}`]
}));

/**
 * Coverage check task
 */
export const coverageCheckTask = defineTask('coverage-check', (args, taskCtx) => ({
  kind: 'node',
  title: 'Check test coverage',
  description: 'Run coverage analysis',

  node: {
    entry: '.a5c/orchestrator_scripts/quality/coverage-check.js',
    args: [
      '--test-files', JSON.stringify(args.testFiles),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['quality', 'coverage']
}));

/**
 * Lint check task
 */
export const lintCheckTask = defineTask('lint-check', (args, taskCtx) => ({
  kind: 'node',
  title: 'Run linter',
  description: 'Check code style and common issues',

  node: {
    entry: '.a5c/orchestrator_scripts/quality/lint-check.js',
    args: [
      '--files', JSON.stringify(args.files),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['quality', 'lint']
}));

/**
 * Type check task
 */
export const typeCheckTask = defineTask('type-check', (args, taskCtx) => ({
  kind: 'node',
  title: 'Run type checker',
  description: 'Verify TypeScript types',

  node: {
    entry: '.a5c/orchestrator_scripts/quality/type-check.js',
    args: [
      '--files', JSON.stringify(args.files),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['quality', 'types']
}));

/**
 * Security check task
 */
export const securityCheckTask = defineTask('security-check', (args, taskCtx) => ({
  kind: 'node',
  title: 'Run security scan',
  description: 'Check for security vulnerabilities',

  node: {
    entry: '.a5c/orchestrator_scripts/quality/security-check.js',
    args: [
      '--files', JSON.stringify(args.files),
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['quality', 'security']
}));

/**
 * Agent quality scoring task
 * Comprehensive quality assessment using LLM
 */
export const agentQualityScoringTask = defineTask('agent-quality-scorer', (args, taskCtx) => ({
  kind: 'agent',
  title: `Score quality (iteration ${args.iteration})`,
  description: 'Comprehensive quality assessment with agent',

  agent: {
    name: 'quality-assessor',
    prompt: {
      role: 'senior quality assurance engineer and code reviewer',
      task: 'Analyze implementation quality across multiple dimensions and provide quantitative score with actionable feedback',
      context: {
        feature: args.feature,
        plan: args.plan,
        tests: args.tests,
        testResults: args.testResults,
        implementation: args.implementation,
        qualityChecks: args.qualityChecks,
        iteration: args.iteration,
        targetQuality: args.targetQuality
      },
      instructions: [
        'Review test quality: coverage, edge cases, assertions, test structure (weight: 25%)',
        'Review implementation quality: correctness, readability, maintainability, performance (weight: 30%)',
        'Review code quality metrics: lint issues, type safety, complexity (weight: 20%)',
        'Review security: vulnerabilities, input validation, error handling (weight: 15%)',
        'Review alignment with plan and TDD principles (weight: 10%)',
        'Calculate weighted overall score (0-100)',
        'Provide specific, prioritized recommendations for improvement',
        'Identify critical issues that must be addressed',
        'Assess progress towards target quality'
      ],
      outputFormat: 'JSON with overallScore (number 0-100), scores (object with dimensions), summary (string), recommendations (array of strings), criticalIssues (array), progress (string)'
    },
    outputSchema: {
      type: 'object',
      required: ['overallScore', 'scores', 'summary', 'recommendations'],
      properties: {
        overallScore: { type: 'number', minimum: 0, maximum: 100 },
        scores: {
          type: 'object',
          properties: {
            tests: { type: 'number' },
            implementation: { type: 'number' },
            codeQuality: { type: 'number' },
            security: { type: 'number' },
            alignment: { type: 'number' }
          }
        },
        summary: { type: 'string' },
        recommendations: { type: 'array', items: { type: 'string' } },
        criticalIssues: { type: 'array', items: { type: 'string' } },
        progress: { type: 'string' }
      }
    }
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['agent', 'quality-scoring', `iteration-${args.iteration}`]
}));

/**
 * Integration test task
 */
export const integrationTestTask = defineTask('integration-test', (args, taskCtx) => ({
  kind: 'node',
  title: 'Run integration tests',
  description: 'Test feature in integrated environment',

  node: {
    entry: '.a5c/orchestrator_scripts/tdd/integration-test.js',
    args: [
      '--feature', args.feature,
      '--output', `tasks/${taskCtx.effectId}/result.json`
    ]
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['tests', 'integration']
}));

/**
 * Agent final review task
 * Comprehensive final assessment
 */
export const agentFinalReviewTask = defineTask('agent-final-reviewer', (args, taskCtx) => ({
  kind: 'agent',
  title: 'Final implementation review',
  description: 'Comprehensive final review with agent',

  agent: {
    name: 'implementation-reviewer',
    prompt: {
      role: 'principal engineer and technical reviewer',
      task: 'Conduct final comprehensive review of the implementation and provide verdict on readiness for production',
      context: {
        feature: args.feature,
        iterations: args.iterations,
        finalQuality: args.finalQuality,
        targetQuality: args.targetQuality,
        converged: args.converged,
        finalTests: args.finalTests,
        finalCoverage: args.finalCoverage,
        integrationTests: args.integrationTests
      },
      instructions: [
        'Review convergence history and quality trajectory',
        'Assess final quality against target',
        'Review test suite completeness and quality',
        'Review integration test results',
        'Assess production readiness',
        'Identify any blocking issues',
        'Provide merge recommendation',
        'Suggest follow-up tasks if any'
      ],
      outputFormat: 'JSON with verdict (string), approved (boolean), confidence (number 0-100), strengths (array), concerns (array), blockingIssues (array), followUpTasks (array)'
    },
    outputSchema: {
      type: 'object',
      required: ['verdict', 'approved', 'confidence'],
      properties: {
        verdict: { type: 'string' },
        approved: { type: 'boolean' },
        confidence: { type: 'number', minimum: 0, maximum: 100 },
        strengths: { type: 'array', items: { type: 'string' } },
        concerns: { type: 'array', items: { type: 'string' } },
        blockingIssues: { type: 'array', items: { type: 'string' } },
        followUpTasks: { type: 'array', items: { type: 'string' } }
      }
    }
  },

  io: {
    inputJsonPath: `tasks/${taskCtx.effectId}/input.json`,
    outputJsonPath: `tasks/${taskCtx.effectId}/result.json`
  },

  labels: ['agent', 'final-review']
}));
