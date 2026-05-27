# Submission LaTeX

`paper.tex` + `references.bib` — a venue-agnostic (`article` class) LaTeX version of the
canonical content in `../results/PAPER.md`. Figures are pulled from `../results/` via
`\graphicspath`.

## Build
No LaTeX toolchain is installed on this machine. Build with either:
- **Overleaf**: upload `paper.tex`, `references.bib`, and the referenced PNGs from
  `../results/` (`fig_money_grn_effect.png`, `fig_comparison_tahoe.png`).
- **Local** (after installing TeX Live / MiKTeX):
  ```
  pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
  ```

## Before submitting — required edits
1. **Authors/affiliations** — the `\author{...}` field is a `\todo` placeholder.
2. **References** — every entry in `references.bib` is a **stub** (`PLACEHOLDER`/`TODO`).
   Replace all 12 with real citations; keys are already wired into `paper.tex`.
3. **Venue style** — swap `\documentclass{article}` for the venue template (NeurIPS/ICML/
   journal) and adjust the bibliography style.
4. Pick the target venue and trim to its page limit (current draft is full-length).

## Consistency
Numbers/claims mirror `../results/PAPER.md` (the source of truth) as of the latest commit,
including the leakage-free OOD audit (`../results/REAL_TAHOE_OOD_AUDIT.md`): no GRN prior
helps, and the deep model does not beat linear baselines on real OOD.
