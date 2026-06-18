# CS 7641 — Project Website (Citation Impact)

This repository is the **project website** for our CS 7641 (Summer 2026) group project,
*Predicting Citation Impact of Scientific Papers*. It is a static site intended to be hosted on
**Georgia Tech GitHub Pages**, structured around **Progress Documents** (Proposal → Midterm → Final).

## Source of truth

- The written content originates from the Overleaf project at `~/Documents/MachineLearning/Project/Overleaf/main.tex`.
  When that LaTeX changes, the corresponding HTML page under `progress-documents/` must be updated to match.
- The proposal grading guideline lives here:
  <https://mahdi-roozbahani.github.io/CS46417641-summer2026/docs/grading/project-breakdown/proposal/>

## Layout

```
index.html                              # Home: overview, team, progress-document index, links
progress-documents/proposal/index.html  # Full proposal (mirrors main.tex)
progress-documents/midterm/index.html   # Placeholder
progress-documents/final/index.html     # Placeholder
assets/css/style.css                    # Shared styles
assets/img/citation_skew.png            # Figure 1 (rendered from Overleaf PDF)
```

---

## ⚙️ Required workflow: run a guideline-compliance check

**Whenever you (Claude) are asked to verify, review, or update this website — or after any edit to a
`progress-documents/*` page — you MUST run a guideline-compliance review.**

Do this by **spawning a sub-agent** (Agent tool, `subagent_type: general-purpose`), but first:

1. **Check whether one already exists.** Use `TaskList` (and/or look back through this
   conversation) for a running or recently-completed agent doing this same compliance check.
   - If a **running** agent exists → continue it via `SendMessage` instead of spawning a new one.
   - If a **recent completed** agent's findings are still valid (no page edits since) → reuse its result.
   - Only **spawn a new agent if none exists** or the pages changed since the last check.

2. Spawn the agent with a prompt equivalent to:

   > Fetch the CS 7641 Summer 2026 proposal grading guideline at
   > `https://mahdi-roozbahani.github.io/CS46417641-summer2026/docs/grading/project-breakdown/proposal/`.
   > Then read the rendered website pages under
   > `~/Documents/MachineLearning/Project/CS7641-Web/` (start at `index.html` and
   > `progress-documents/proposal/index.html`). Produce a checklist verdict (✅ / ⚠️ / ❌ with a
   > one-line reason) on whether the site meets every guideline requirement listed below, and a
   > prioritized list of what is missing or needs fixing. Do not edit files — report only.

3. Relay the agent's checklist verdict back to me, then fix any ❌/⚠️ items.

### Guideline checklist the agent must verify

**Five graded sections (these count toward the word limit):**
- [ ] **1. Introduction / Background** — literature review + dataset description + dataset link
- [ ] **2. Problem Definition** — the problem and why a solution is needed
- [ ] **3. Methods** — **3+ data preprocessing methods** and **3+ ML algorithms/models**, with supervised *and* unsupervised methods
- [ ] **4. (Potential) Results & Discussion** — **3+ quantitative metrics**, project goals, expected results, plus **sustainability & ethics**
- [ ] **5. References** — **IEEE format**, **3+ references** (preferably peer-reviewed), **≥1 in-text citation per reference**

**Format & word limit:**
- [ ] Proposal body (sections 1–5) is **< 800 words** (Gantt chart, contribution table, and references do **not** count)
- [ ] Hosted as a **website on GT GitHub Pages**

**Required additional components:**
- [ ] **Gantt chart** showing each member's planned responsibilities
- [ ] **Contribution table** — every member's name + explicit proposal contributions
- [ ] **3-minute YouTube *Unlisted* video** link present
- [ ] **Private GT GitHub Enterprise repo** with teammates + mentor as collaborators (link present)
- [ ] **"Outstanding Project" award opt-in** status stated

### Known TODOs to flag until resolved
- Add the real **GT GitHub Enterprise repository** link (placeholder `#` in `index.html`).
- Add the real **YouTube Unlisted video** link (placeholder `#` in `index.html`).
- Re-confirm the proposal stays **under 800 words** after any content edit.

## Local preview

```bash
cd ~/Documents/MachineLearning/Project/CS7641-Web
python3 -m http.server 8000   # then open http://localhost:8000
```
