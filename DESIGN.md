# Design

## Visual Theme

Precise, lab-notebook rigor rather than SaaS-dashboard polish. Both light and dark are first-class (reviewers open this in normal office light most often; dark is a genuine second mode, not a hacker-aesthetic default). The purge/embargo gap (a literal separation between train and test data in time) is the recurring structural motif: gapped rules, a visible "gap" divider between paired honest/leaky elements, used sparingly and deliberately, not as wallpaper.

## Color Strategy

Full palette (product, data-heavy). Named roles, each used consistently and only for its role, never decoratively:

- **Ink / surface**: neutrals tinted toward a very low-chroma indigo (chroma 0.006-0.01), not pure black/white. Light surface near L=98, dark surface near L=14.
- **Honest** (the real, leakage-proof result): a desaturated deep teal/cyan. This color always means "the number computed the correct, leakage-proof way."
- **Leaky** (the naive-CV control): a warm coral/amber, deliberately positioned as teal's opposite on the wheel, so the honest-vs-leaky pairing reads as a contrast without either color meaning "good."
- **Pass**: a green distinct from honest-teal (avoid overloading teal with "good"), used only for audit badges and the PASS verdict state.
- **Fail / null-caution**: a warm red-orange, used only for failed audits and the NULL verdict pill (NULL is not styled as an alarming failure red; use a quieter amber-slate so a null result doesn't visually read as "broken").
- Accent budget: honest-teal and leaky-coral together may carry real surface weight in the hero comparison section (Committed-level, 30-50% of that section); everywhere else stays Restrained (tinted neutrals, color only on data/state).

## Typography

- Prose/UI: system sans stack (`-apple-system, "Segoe UI", Inter, sans-serif`), no external font dependency (keeps the file:// / offline degradation story honest, matching the anti-references principle against silent failures).
- Numerals/stats/code: system mono stack (`"Cascadia Code", "SF Mono", Consolas, monospace`), used for every statistic, threshold, hash, and the verdict pill, not just inside code blocks. This is the primary typographic signature of the page.
- Scale ratio >= 1.25 between steps, at least 4 distinct steps (display verdict, section h2, body, stat/mono).

## Layout

- No nested cards. Sections separated by rhythm (varied vertical spacing) and the gap motif, not by boxing everything.
- The honest-vs-leaky comparison is the hero, placed directly under the verdict banner, above the equity curve.
- Stat values render in mono with generous tabular alignment where multiple numbers are compared (e.g. IC across regimes, CPCV CI bounds).

## Motion

Ease-out-quart/quint only, no bounce. Respect prefers-reduced-motion (disable non-essential transitions). Motion budget is small and purposeful: tooltip reveal, a count-up on the verdict pill, hover states, nothing decorative or looping.

## Components

- Audit badges: pass/fail conveyed by icon + text token + color (three redundant cues), not color alone.
- Verdict banner: mono statistic block (IC, t-stat, threshold) with the criterion written as a literal inequality, PASS/NULL state visually distinct but both rendered with equal typographic confidence.
- Charts: Plotly, restyled off the CSS custom properties (no default Plotly palette), with a visible loading state and an explicit typeof-guard fallback if the CDN script fails.
