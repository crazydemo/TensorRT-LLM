# Redefining the QA Engineer's Value in the Age of AI

> Reflections from building a change-based test selection module with AI

---

## Background

This sharing stems from my experience building a change-based test selection module. The module itself is modest in scope, but the development process forced me to re-examine two questions:

1. How can we get more value out of AI?
2. As AI grows more capable, where does a QA engineer's value lie?

My current use of AI covers only a tiny fraction of its capability. If this module — validated at small scale — proves stable at a larger scale, it means AI's capability boundary far exceeds our current use cases. And AI is growing faster than we are. This forces us to think about how to use it, and what irreplaceable value we ourselves can provide.

---

## I. How to Get More Value from AI

### 1.1 The Human Needs a Clear Mental Model

There's a recent concept called **harness engineering** — it explores how to create a stable environment that enables AI to work effectively.

When I started, I simply told Claude: "I want to build a change-based testing module." It quickly produced a first version — it could read core code, read test definitions, build mappings, and even had a polished CLI. Impressive at first glance.

But it had no visibility into our QA workflow:
- How many GPU resources do I want to use?
- How long should it run?
- How should test cases be prioritized?
- How does the waive list work?

These had to be fed to it incrementally through conversation. And in that process, it wasn't just the AI organizing information — **I was also clarifying my own thinking, sorting out the tiers and priorities**. The first step is for the human to have a clear mental model, or at least the willingness to develop one through dialogue with AI.

### 1.2 Infrastructure Must Be Unified

A key discovery: **inconsistency in test infrastructure barely affects humans, but it's fatal noise for AI.**

Specifically:
- The accuracy test suite and integration tests follow different design patterns — structurally they are two separate systems, and a single parser cannot handle both
- Test case names have no consistent convention — some use key-value pairs, some require special parsing, some are parsed from parameter lists, some from function names, some even from comments

These inconsistencies caused a specific phenomenon when working with AI: **it would go in circles**. For example, after the parser and selector were written, when I found a case that wasn't being selected, the AI would say "the selector is too strict." But the real cause was that the case's feature/model was already parsed incorrectly in the parsing stage, and the existing code logic didn't account for that scenario.

The deeper issue: **AI tends to patch on top of existing abstractions rather than questioning the abstractions themselves.** It won't step back and say "the parser's design approach is fundamentally wrong." This means that if the underlying infrastructure isn't unified and clean, AI efficiency drops dramatically.

### 1.3 The Need for an Architect Role

Building on this observation, I believe standardization and refactoring at the test infrastructure layer is essential. An architect role is needed to:
- Unify the test framework abstractions
- Establish naming conventions for test cases
- Reduce the cognitive load on AI when comprehending the test system

This role should currently be filled by a human. AI can assist with analysis, but decisions involving standard-setting and cross-team constraints still require human judgment.

---

## II. Opening a Window in the Black Box — A Division of Labor Model

### 2.1 The Core Concept

If we treat the AI agent as a black box that handles all the routine work, what does our job become?

If we only use AI to speed through the daily pipeline, that's a waste — for us, for the project, and for the AI.

But what if we **open a window in this black box?**

### 2.2 Impact Rules as an Example

This module has a set of impact rules that require static maintenance. From the beginning, I felt this mapping was the module's core, but I wasn't sure whether AI should generate it dynamically or humans should confirm it.

My conclusion: **there still needs to be a static, human-confirmed piece.**

This "window" exposes the truly core thing — the impact rules. Through this window, our attention can shift from:
- ~~Is this case a regression? How do I reproduce it?~~ (routine)

To:
- **What are the project's current priority areas?**
- **Where is test coverage weakest?**
- **What's the future direction?**

We can do more, go deeper, and think more strategically.

### 2.3 Extending to the Entire QA Workflow

Following this logic, we can re-examine the entire QA workflow:
- **Which parts are mechanical, routine work?** → Hand them to the black box
- **Which parts deserve a window for careful thought?** → Keep them for humans

The decision of "where to open the window" is itself the most essential capability for QA engineers in the age of AI.

---

## III. The Boundary Between Agents and Skills

### 3.1 Agents Should Be Built on Workflows

For QA, we don't necessarily need many agents on the codebase. Our agents should be built on **workflows**:
- For Jenkins
- For NVBugs
- For Jira

Because QA's core value chain isn't in the code itself — it's in the processes surrounding the code.

### 3.2 The Limitations of Skills

Skills are an important technique that fill gaps where conventional code falls short. But skills have two inherent problems:

| Problem | Meaning | Mitigation |
|---------|---------|------------|
| **Drift** | The same skill gradually deviates from its original expected behavior as the underlying model is upgraded | Requires regression testing |
| **Volatility** | The same execution produces different results with different models/agents | Requires constraining output format or falling back to code |

We cannot require all users to use the same agent or model to execute a skill. Therefore:

> **Where determinism and collaboration are needed → use code**
> **Where flexibility and judgment are needed → use skills/agents**

For parts that require strong logic, or where human-to-human, human-to-AI, or AI-to-AI collaboration is involved, code is still needed to guarantee consistency.

---

## IV. Conclusion: Redefining the QA Engineer's Value

In the age of AI, the core value of a QA engineer is not executing tests — it is **designing the boundary between human and AI collaboration**:

1. **Deciding what goes into the black box** — identifying routine work and building automation
2. **Deciding what needs a window** — finding the critical decision points that require human judgment
3. **Deciding where the window opens** — designing mechanisms that expose AI's intermediate artifacts for review and intervention

This is itself a new kind of architectural capability. It requires understanding testing, the project, and AI's capability boundaries, as well as the strategic vision to judge what deserves human attention.

Routine work will increasingly be taken over by AI. But **defining what is routine and what is not** — that itself will not be automated.
