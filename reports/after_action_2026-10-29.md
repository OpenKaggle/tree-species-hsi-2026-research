# Tree Species HSI 2026 after-action review — 2026-10-29

> After-action review prepared 2026-10-29 from public post-competition sources;
> internal evidence grades mapped to Verified/Unverified.

Data sources: the Kaggle competition API (participant view), a 55-row final
leaderboard CSV (2026-10-28 snapshot) plus paginated leaderboard JSON, and the
competition discussion pages (JavaScript-rendered; not machine-readable, and
web search returned no indexed writeups for this 55-team community
competition).

## A. Gap quantification

- Final board [Verified]: 55 teams. #1 "tiger" 0.50275 (59 submissions), #2
  "谱树队" 0.38658 (78), #3 "OUC AI LAB" 0.37683 (30). Median 0.15007.
- Our entry [Verified]: rank 52/55 at 0.06450 (4 submissions; the final board
  auto-selected the 09-20 submission, not the 09-23 Phase 2 submission). The
  gap to #1 is 7.8x; to #3, 5.9x.
- Public/private shake [Unverified]: comparing the paginated JSON (a 7% public
  pagination snapshot) against the full 55-team CSV shows same-team score
  differences ≤0.0004 and zero Top-10 order change, so the public/private shake
  is negligible. The CSV is filename-tagged `publicleaderboard`, but its score
  set differs from the public snapshot and contains all 55 teams; attributing
  it to the private final board is an Unverified inference.
- Same-code, two-slice evidence [Verified]: during Phase 1 the v3 model scored
  0.10387 (rank 31) on the public slice; after the Phase 2 slice change the
  same pipeline rescored 0.06450 — the fragility of the ~7% public slice is
  confirmed by our own data.

## B. What the leaders did vs us (their mechanism overall Unverified)

- Submission density [Verified]: Top-10 submission counts were
  59/78/30/28/59/82/5/69/35/76 (mean ≈ 52), while the bottom of the board was
  mostly 1–9. The head teams ran high-frequency targeted search against the 7%
  public slice; we used 4 submissions.
- Our method family [Verified]: LightGBM on spectral statistics plus radius-1
  spatial features; connected-region OOF 0.688 (AA 0.626) → LB 0.064, a
  validation-proxy collapse of 10.7x.
- Competitor method families [Unverified]: the competition's academic
  background is 3D-CNN/attention/Mamba two-branch spatial-spectral networks
  (matched 2026 literature), and several head team names include domain
  institutions. Inferred (not confirmed): the winner used a deep
  spatial-spectral model with cross-scene augmentation.

## C. Floor-gate retrospective (0.15)

- Position [Verified]: 28/55 teams (51%) finished at ≥0.15, so 0.15 sits almost
  exactly at the median; top-3 required 0.377+, top-10% required 0.30+.
- Verdict [Verified/Unverified mix]: the stop-loss "saved the right amount."
  Even passing the gate would only have meant median placement on a kudos-only
  line with no prize (a zero-prize line should not have consumed a submission
  slot at all); and the cause of death was proxy collapse (0.688→0.064), not
  miscalibration of the floor — more submissions cannot rescue a broken proxy.
  At 0.064 there were already 52 teams ahead; the ceiling was structural, not a
  calibration issue.
- Honest caveats [Unverified]: the derivation of the 0.15 value itself is not
  on record (why not 0.25 or 0.10), so the post-hoc "≈ median" match looks more
  like a lucky interval than a derived one; and the Phase 1 anchor of 0.104 was
  already below 0.15, meaning the proxy was already dead before the gate was
  ever evaluated — the gate was not front-loaded into Phase 1. The floor-gate
  mechanism was right; its value and timing were not rigorous.

## D. Lessons to institutionalize (hyperspectral / remote-sensing detection class)

1. [Verified] **The validation proxy comes before the ceiling.** If
   connected-region OOF and the cross-scene LB differ by more than 3x, the
   proxy is broken and the ceiling is "unmeasurable." The single biggest
   leverage if replayed: spend one sanity submission on day one to calibrate
   the OOF↔LB correlation (0.688 vs 0.104 was already visible in Phase 1);
   every model improvement is worth zero until the proxy is calibrated.
2. [Verified] **Zero-prize lines: read-only/delivery first.** On a kudos-only
   line with an uncalibrated proxy, submission slots are a negative asset. The
   only positive action available on this line was a workshop presentation /
   code package — **0 submissions would have beaten 1.**
3. [Unverified] Under a ~7%-of-test public-slice format, head teams squeeze the
   slice with 50–80 submissions. If a future remote-sensing line carries a real
   prize, use the floor gate on day one only, then iterate at full daily
   budget — "one-submission stop-loss" fits the kudos-only context, not a
   prize-bearing one.

## E. Open questions

- Winner / top-3 mechanisms: Unverified (requires manually opening the Kaggle
  pages; JS rendering is not machine-readable).
- Whether the 55-team CSV is the public or private board; the record of which
  of our 4 submissions was auto-selected as the final entry.
- A retroactive write-up of the 0.15 floor's entry-time derivation (guards
  against post-hoc rationalization recurring).
