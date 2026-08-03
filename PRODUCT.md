# Product

## Register

product

## Users

Recruiters, hiring managers, and quant/ML engineers reviewing Alex's portfolio, typically on a laptop, comparing this against other candidates' GitHub Pages projects for signal of genuine methodological rigor. They skim first (30-60 seconds) then dig into the one or two artifacts that prove the claim. A minority are quant practitioners who will scrutinize the actual statistics (IC, Newey-West t-stats, CPCV confidence intervals) and will notice if the visualization misrepresents them.

## Product Purpose

Present the honest result of a pre-registered test (does gradient boosting beat plain momentum on 5-day cross-sectional equity returns, under leakage-proof evaluation) without spin. The product IS the receipt: a viewer should be able to verify, in under two minutes, that the methodology was genuinely pre-registered, that the 8 leakage-audit tests are real and passing, and that the null result was reported as-is rather than reframed as a near-win. Success looks like a reviewer thinking "this person understands why most public ML-for-trading projects are untrustworthy, and built the harness that would catch the mistake."

## Brand Personality

Rigorous, unhurried, quietly confident. Not hype, not apologetic about the null result, not trying to look like a funded startup's SaaS dashboard. Voice is precise and slightly dry, the way a good lab notebook or a de Prado paper reads, not marketing copy. Three words: exacting, transparent, unshowy.

## Anti-references

Not a generic SaaS analytics dashboard (hero metric + gradient text + icon-in-a-circle card grid). Not a "finance app" navy-and-gold or crypto neon-on-black cliche, chosen by category reflex rather than intent. Not a dark-mode-by-default hacker terminal aesthetic either, this is meant to be read in an office during a five-minute review, not glanced at in a dim server room. Avoid anything that reads as templated Bootstrap/Tailwind-starter-kit output.

## Design Principles

- Show the receipt before the claim: the 8-badge audit strip and the honest-vs-leaky comparison earn trust before any equity curve is shown.
- The null result is not a bug to hide, the page's typographic and color hierarchy should treat "NULL, criterion not met" with the same visual confidence as a PASS would get.
- One motif, used throughout: the purge/embargo gap (a literal separation between train and test data) is the project's real technical idea, reflect it structurally somewhere in the layout, not just in a chart legend.
- Numbers are the content. Typographic hierarchy and a tabular/mono treatment for statistics matter more than decorative color.
- Every interactive element must degrade honestly: a failed fetch, a missing chart library, or a file:// viewing context should say exactly what's wrong and how to fix it, never a silent blank page.

## Accessibility & Inclusion

WCAG AA minimum (4.5:1 body text, 3:1 large text/UI components). Pass/fail state must never rely on color alone (colorblind users need a second cue: icon or text token). Keyboard-navigable tooltips that also work on touch. Respect prefers-reduced-motion. Charts need a text-accessible summary (aria-label or adjacent caption), not just a bare canvas/SVG.
