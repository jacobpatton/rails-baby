# Process Library

The Babysitter Process Library is a comprehensive collection of **2,000+ pre-built process definitions** (and growing) that you can use immediately or customize for your specific needs. This extensive library covers software development, business operations, scientific research, and dozens of specialized domains.

## What is the Process Library?

The Process Library provides ready-to-use orchestration workflows for virtually any task you might encounter. Each process is:

- **Battle-tested**: Designed with best practices and quality gates built-in
- **Composable**: Can be combined with other processes to create complex workflows
- **Customizable**: Extend or modify any process to match your requirements
- **Self-documenting**: Includes clear inputs, outputs, and execution flow

Instead of writing orchestration logic from scratch, you can leverage these pre-built processes and focus on what matters: your actual work.

## How Babysitter Uses the Process Library

When you describe a task to Babysitter, the agent **automatically selects the most relevant processes** from the library and adapts them to your specific needs. This happens in several ways:

### Automatic Process Selection

Simply describe what you want to accomplish in natural language:

```
Use babysitter to build a Next.js app with authentication and PostgreSQL
```

Babysitter will:
1. Identify relevant processes (e.g., `nextjs-fullstack-app`, `jwt-authentication`, `database-setup`)
2. Combine and adapt them to your requirements
3. Execute the orchestrated workflow with quality gates

### Explicit Process Selection

You can also request a specific process by name:

```
Use babysitter with the tdd-quality-convergence process to implement user authentication
```

Or:

```
Use babysitter with the devin methodology to build a REST API
```

### Mix and Match

For complex tasks, Babysitter intelligently combines multiple processes:

```
Use babysitter to create a microservices architecture with:
- GraphQL gateway
- Kubernetes deployment
- Monitoring and alerting
```

This might pull from `graphql-api-apollo`, `kubernetes-setup`, `monitoring-setup`, and `cicd-pipeline-setup` processes, adapting each to work together coherently.

### Customizing for Your Use Case

You can also modify existing processes or provide additional context:

```
Use babysitter with the security-audit process but focus specifically on
OWASP Top 10 vulnerabilities and include PCI-DSS compliance checks
```

The agent will take the base process and adapt it to your specific requirements.

## Library Structure

The Process Library is organized into four main areas:

```
process/
├── methodologies/          # Development methodologies (TDD, Agile, Devin, etc.)
├── gsd/                    # Get Shit Done workflows
├── specializations/        # Domain-specific processes (30+ categories)
│   ├── [Development Processes]
│   │   ├── web-development/
│   │   ├── mobile-development/
│   │   ├── devops-sre-platform/
│   │   ├── ai-agents-conversational/
│   │   ├── security-research/
│   │   └── ... (20+ more dev categories)
│   ├── domains/
│   │   ├── business/       # Finance, HR, Marketing, Sales, etc.
│   │   ├── science/        # Physics, Chemistry, Engineering, etc.
│   │   └── social-sciences-humanities/
│   └── meta/               # Process creation and validation
└── tdd-quality-convergence # Featured TDD workflow
```

## Browsing and Discovering Processes

### Using the CLI

List available processes by category:

```bash
# List all specializations
ls plugins/babysitter/skills/babysit/process/specializations/

# Browse a specific category
ls plugins/babysitter/skills/babysit/process/specializations/web-development/

# View process documentation
cat plugins/babysitter/skills/babysit/process/specializations/web-development/README.md
```

### Process Naming Convention

Process files follow a consistent naming pattern:

- `feature-name.js` - The process definition
- `README.md` - Category documentation
- `examples/` - Example inputs and usage
- `agents/` - Specialized agents for the category

### Finding the Right Process

1. **Start with the category** that matches your domain
2. **Review the README.md** in that category for an overview
3. **Check the JSDoc** at the top of each `.js` file for inputs/outputs
4. **Look at examples/** for sample usage patterns

## Categories Overview

### Development Processes (680+ processes)

| Category | Processes | Description |
|----------|-----------|-------------|
| **web-development** | 61 | Full-stack web development, frameworks, deployment |
| **algorithms-optimization** | 45 | Algorithm implementation, performance tuning |
| **ai-agents-conversational** | 39 | LLM applications, RAG, multi-agent systems |
| **cryptography-blockchain** | 33 | Smart contracts, DeFi, cryptographic protocols |
| **security-research** | 32 | Penetration testing, vulnerability research |
| **robotics-simulation** | 30 | ROS2, simulation, autonomous systems |
| **performance-optimization** | 30 | Profiling, optimization, benchmarking |
| **network-programming** | 30 | Protocols, distributed systems, networking |
| **game-development** | 30 | Game engines, mechanics, production |
| **cli-mcp-development** | 30 | CLI tools, MCP servers, developer tooling |
| **mobile-development** | 26 | iOS, Android, React Native, Flutter |
| **embedded-systems** | 26 | Firmware, drivers, real-time systems |
| **sdk-platform-development** | 25 | SDKs, APIs, platform engineering |
| **programming-languages** | 25 | Compilers, interpreters, language design |
| **gpu-programming** | 25 | CUDA, compute shaders, parallel processing |
| **fpga-programming** | 25 | HDL, synthesis, hardware design |
| **code-migration-modernization** | 25 | Legacy modernization, framework upgrades |
| **security-compliance** | 24 | Security standards, compliance automation |
| **desktop-development** | 23 | Electron, native apps, cross-platform |
| **ux-ui-design** | 21 | Design systems, accessibility, prototyping |
| **technical-documentation** | 21 | API docs, guides, documentation systems |
| **software-architecture** | 20 | System design, patterns, architecture reviews |
| **qa-testing-automation** | 20 | Test automation, quality assurance |
| **devops-sre-platform** | 20 | CI/CD, infrastructure, observability |
| **data-science-ml** | 18 | ML pipelines, model training, MLOps |
| **data-engineering-analytics** | 18 | Data pipelines, analytics, ETL |
| **product-management** | 17 | Roadmaps, specifications, product strategy |
| **meta** | 5 | Process creation, validation, tooling |

### Business Domains (430+ processes)

| Category | Processes | Description |
|----------|-----------|-------------|
| **decision-intelligence** | 33 | Decision frameworks, analysis models |
| **legal** | 28 | Contract analysis, compliance, legal ops |
| **operations** | 26 | Business process optimization |
| **business-strategy** | 26 | Strategic planning, competitive analysis |
| **venture-capital** | 25 | Due diligence, portfolio management |
| **supply-chain** | 25 | Logistics, inventory, procurement |
| **sales** | 25 | Sales processes, CRM workflows |
| **public-relations** | 25 | Communications, media relations |
| **marketing** | 25 | Campaigns, analytics, content strategy |
| **logistics** | 25 | Distribution, routing, fulfillment |
| **knowledge-management** | 25 | Documentation, wikis, knowledge bases |
| **human-resources** | 25 | Recruiting, onboarding, HR processes |
| **finance-accounting** | 25 | Financial analysis, reporting, auditing |
| **entrepreneurship** | 25 | Startup workflows, business planning |
| **digital-marketing** | 25 | SEO, PPC, social media, analytics |
| **customer-experience** | 25 | CX design, feedback loops, journey mapping |
| **project-management** | 5 | Project planning, tracking, delivery |

### Science and Engineering (550+ processes)

| Category | Processes | Description |
|----------|-----------|-------------|
| **scientific-discovery** | 168 | Research methodologies, reasoning patterns |
| **quantum-computing** | 27 | Quantum algorithms, circuit design |
| **mechanical-engineering** | 26 | CAD, simulation, manufacturing |
| **computer-science** | 25 | Theory, algorithms, formal methods |
| **civil-engineering** | 25 | Structural analysis, infrastructure |
| **chemical-engineering** | 25 | Process design, reaction engineering |
| **biomedical-engineering** | 25 | Medical devices, biomechanics |
| **automotive-engineering** | 25 | Vehicle systems, ADAS, EV |
| **aerospace-engineering** | 25 | Flight systems, propulsion, avionics |
| **physics** | 24 | Simulation, modeling, analysis |
| **nanotechnology** | 24 | Nanofabrication, characterization |
| **mathematics** | 24 | Proofs, modeling, computation |
| **materials-science** | 24 | Material characterization, discovery |
| **industrial-engineering** | 24 | Process optimization, operations research |
| **environmental-engineering** | 24 | Environmental modeling, sustainability |
| **electrical-engineering** | 24 | Circuit design, signal processing |
| **bioinformatics** | 12 | Genomics, proteomics, computational biology |

### Social Sciences and Humanities (150+ processes)

Processes for research methodologies, analysis frameworks, and academic workflows in social sciences, humanities, and interdisciplinary fields.

## Example Processes by Category

### Web Development

```
web-development/
├── nextjs-fullstack-app.js          # Complete Next.js application
├── graphql-api-apollo.js            # GraphQL API with Apollo
├── jwt-authentication.js            # JWT auth implementation
├── e2e-testing-playwright.js        # Playwright E2E testing
├── micro-frontend-module-federation.js  # Micro-frontend architecture
├── accessibility-audit-remediation.js   # WCAG compliance
└── docker-containerization.js       # Docker deployment
```

### AI Agents and Conversational

```
ai-agents-conversational/
├── multi-agent-system.js            # Multi-agent orchestration
├── advanced-rag-patterns.js         # Advanced RAG implementation
├── langgraph-workflow-design.js     # LangGraph workflows
├── conversational-memory-system.js  # Long-term memory for agents
├── function-calling-agent.js        # Tool-using agents
├── agent-evaluation-framework.js    # Agent testing and eval
└── llm-observability-monitoring.js  # LLM monitoring setup
```

### Security Research

```
security-research/
├── binary-reverse-engineering.js    # Binary analysis
├── exploit-development.js           # Exploit writing workflow
├── fuzzing-campaign.js              # Fuzzing setup and execution
├── malware-analysis.js              # Malware analysis workflow
├── network-penetration-testing.js   # Network pentesting
├── capture-the-flag-challenges.js   # CTF solving workflow
└── bug-bounty-workflow.js           # Bug bounty methodology
```

### DevOps and SRE

```
devops-sre-platform/
├── kubernetes-setup.js              # Kubernetes cluster setup
├── cicd-pipeline-setup.js           # CI/CD pipeline creation
├── monitoring-setup.js              # Observability stack
├── incident-response.js             # Incident management
├── disaster-recovery-plan.js        # DR planning and testing
├── slo-sli-tracking.js              # SLO/SLI implementation
└── secrets-management.js            # Secrets management setup
```

### Scientific Discovery

```
scientific-discovery/
├── hypothesis-formulation-testing.js    # Scientific method
├── causal-inference.js                  # Causal analysis
├── bayesian-probabilistic-reasoning.js  # Bayesian reasoning
├── experiment-design-reasoning.js       # Experiment planning
├── literature-review-synthesis.js       # Literature review
├── reproducible-research-pipeline.js    # Reproducibility
└── systems-thinking.js                  # Systems analysis
```

## Using a Pre-Built Process

### Step 1: Create a Run

```bash
babysitter run:create \
  --process-id specializations/web-development/nextjs-fullstack-app \
  --entry plugins/babysitter/skills/babysit/process/specializations/web-development/nextjs-fullstack-app.js#process \
  --inputs inputs.json
```

### Step 2: Prepare Your Inputs

Create an `inputs.json` file with the required parameters:

```json
{
  "projectName": "my-nextjs-app",
  "features": ["authentication", "database", "api-routes"],
  "database": "postgresql",
  "deployment": "vercel"
}
```

### Step 3: Run the Orchestration

```bash
babysitter run:continue .a5c/runs/<runId> --auto-node-tasks --auto-node-max 10
```

### Step 4: Handle Breakpoints

The process will pause at breakpoints for human review:

```
[BREAKPOINT] Review the implementation plan for "my-nextjs-app"
Approve to proceed with setup? (yes/no)
```

## Methodologies Reference

The library includes 38+ development methodologies that can be applied to any project:

### Core Methodologies

| Methodology | Description | Best For |
|-------------|-------------|----------|
| **devin** | Plan -> Code -> Debug -> Deploy with autonomous iteration | Full feature implementation |
| **ralph** | Simple iterative loop until task completion | Persistent tasks with unclear scope |
| **plan-and-execute** | Detailed planning phase followed by execution | Complex, well-defined features |
| **tdd-quality-convergence** | TDD with iterative quality scoring | High-quality, tested code |
| **spec-driven-development** | Executable specifications drive implementation | Enterprise, governance-heavy projects |

### Agile and Iterative

| Methodology | Description | Best For |
|-------------|-------------|----------|
| **agile** | Sprint-based iterative development | Team-based projects |
| **scrum** | Full Scrum implementation with ceremonies | Scrum teams |
| **kanban** | Continuous flow with WIP limits | Continuous delivery |
| **extreme-programming** | XP practices (pair programming, TDD) | High-quality code |
| **feature-driven-development** | Feature-centric development | Large codebases |

### Architecture and Design

| Methodology | Description | Best For |
|-------------|-------------|----------|
| **top-down** | Architecture-first development | New systems, clear requirements |
| **bottom-up** | Component-first development | Exploratory, uncertain requirements |
| **domain-driven-design** | DDD strategic and tactical patterns | Complex business domains |
| **event-storming** | Collaborative domain discovery | Domain modeling |
| **evolutionary** | Incremental architecture evolution | Legacy modernization |

### Specialized Approaches

| Methodology | Description | Best For |
|-------------|-------------|----------|
| **graph-of-thoughts** | Multi-path reasoning exploration | Complex problem solving |
| **adversarial-spec-debates** | Red team/blue team specification | Critical systems |
| **consensus-and-voting-mechanisms** | Multi-agent consensus building | Distributed decisions |
| **state-machine-orchestration** | State-based workflow management | Complex state transitions |
| **build-realtime-remediation** | Real-time error detection and fixing | CI/CD pipelines |

### GSD (Get Shit Done) Workflows

The GSD methodology provides systematic project development:

| Workflow | Purpose |
|----------|---------|
| **new-project** | Project initialization with vision capture |
| **discuss-phase** | Capture implementation preferences |
| **plan-phase** | Generate verified task plans |
| **execute-phase** | Parallel task execution with commits |
| **verify-work** | User acceptance testing |
| **audit-milestone** | Milestone completion verification |
| **map-codebase** | Brownfield project analysis |
| **iterative-convergence** | Quality-gated development loop |

## Customizing Processes

### Extending an Existing Process

```javascript
import { process as baseProcess } from './nextjs-fullstack-app.js';

export async function process(inputs, ctx) {
  // Add pre-processing
  const enhancedInputs = {
    ...inputs,
    additionalChecks: true,
    customConfig: myConfig
  };

  // Run base process
  const result = await baseProcess(enhancedInputs, ctx);

  // Add post-processing
  await ctx.task(myCustomValidation, result);

  return {
    ...result,
    customData: myCustomData
  };
}
```

### Composing Multiple Processes

```javascript
import { process as planPhase } from './gsd/plan-phase.js';
import { process as executePhase } from './gsd/execute-phase.js';
import { process as tddConvergence } from './tdd-quality-convergence.js';

export async function process(inputs, ctx) {
  // Planning with GSD
  const plan = await planPhase(inputs, ctx);

  // Execute with TDD quality gates
  const implementation = await tddConvergence({
    ...inputs,
    plan: plan.tasks,
    targetQuality: 90
  }, ctx);

  // Verify with GSD
  const verification = await executePhase({
    ...inputs,
    tasks: implementation.artifacts
  }, ctx);

  return { plan, implementation, verification };
}
```

### Modifying Process Parameters

Most processes accept configuration through inputs:

```json
{
  "feature": "User authentication",
  "targetQuality": 95,
  "maxIterations": 10,
  "requirements": [
    "Support OAuth2",
    "Include MFA"
  ],
  "constraints": [
    "Must use existing user table",
    "No breaking API changes"
  ]
}
```

## Quick Reference: All Categories

| Category | Count | Focus Area |
|----------|-------|------------|
| domains/science/scientific-discovery | 168 | Research reasoning patterns |
| web-development | 61 | Full-stack web development |
| algorithms-optimization | 45 | Algorithm implementation |
| ai-agents-conversational | 39 | LLM and agent development |
| domains/business/decision-intelligence | 33 | Decision frameworks |
| cryptography-blockchain | 33 | Blockchain and crypto |
| security-research | 32 | Security research and testing |
| robotics-simulation | 30 | Robotics and simulation |
| performance-optimization | 30 | Performance tuning |
| network-programming | 30 | Network and protocols |
| game-development | 30 | Game development |
| cli-mcp-development | 30 | CLI and MCP tools |
| domains/business/legal | 28 | Legal operations |
| domains/science/quantum-computing | 27 | Quantum computing |
| domains/science/mechanical-engineering | 26 | Mechanical engineering |
| domains/business/operations | 26 | Business operations |
| domains/business/business-strategy | 26 | Strategic planning |
| mobile-development | 26 | Mobile app development |
| embedded-systems | 26 | Embedded and firmware |
| sdk-platform-development | 25 | SDK development |
| programming-languages | 25 | Language implementation |
| gpu-programming | 25 | GPU and parallel computing |
| fpga-programming | 25 | FPGA design |
| code-migration-modernization | 25 | Code modernization |
| security-compliance | 24 | Security compliance |
| desktop-development | 23 | Desktop applications |
| ux-ui-design | 21 | UX/UI design |
| technical-documentation | 21 | Documentation |
| software-architecture | 20 | Architecture design |
| qa-testing-automation | 20 | Test automation |
| devops-sre-platform | 20 | DevOps and SRE |
| data-science-ml | 18 | Machine learning |
| data-engineering-analytics | 18 | Data engineering |
| product-management | 17 | Product management |
| meta | 5 | Process tooling |

## Best Practices

### Choosing the Right Process

1. **Match your domain**: Start with the specialization that matches your work
2. **Check the methodology**: Consider which methodology fits your project style
3. **Review inputs carefully**: Understand what configuration options are available
4. **Read the examples**: Look at example inputs in the `examples/` directories

### Customization Tips

1. **Start simple**: Use processes as-is before customizing
2. **Layer changes**: Extend rather than modify base processes
3. **Preserve breakpoints**: Keep human approval gates in critical paths
4. **Test incrementally**: Validate customizations with small inputs first

### Quality Considerations

1. **Use quality convergence**: Processes with quality scoring help ensure high standards
2. **Enable breakpoints**: Human review catches issues early
3. **Compose methodologies**: Combine TDD with your domain process for better results
4. **Track iterations**: Monitor how many iterations processes require

## See Also

- [Process Definitions](./process-definitions.md) - How to create your own processes
- [Quality Convergence](./quality-convergence.md) - Quality gates and scoring
- [Breakpoints](./breakpoints.md) - Human-in-the-loop approval
- [Parallel Execution](./parallel-execution.md) - Running tasks concurrently
