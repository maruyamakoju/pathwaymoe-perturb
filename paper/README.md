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
1. **Authors/affiliations** — the `\author{...}` field is still a `\todo` placeholder.
   Fill in real name + affiliation + contact before upload.
2. **Venue style** — swap `\documentclass{article}` for the venue template (NeurIPS/ICML/
   journal) and adjust the bibliography style.
3. Pick the target venue and trim to its page limit (current draft is full-length).

## Already done
- References are complete and verified (`references.bib`, 12 entries with DOIs).
- Numerical tables reflect the corrected statistics pipeline (bootstrap p-value
  continuity correction + leakage-free protocol).

## Consistency
Numbers/claims mirror `../results/PAPER.md` (the source of truth) as of the latest commit,
including the leakage-free OOD audit (`../results/REAL_TAHOE_OOD_AUDIT.md`): no GRN prior
helps, and the deep model does not beat linear baselines on real OOD.
