# Public discussion audit

Captured from the official Kaggle competition forum on 2026-09-09.

Only one public topic was returned by the official API:

- Topic 735532, “Are the images corrupt?”, posted 2026-08-16, 3 votes,
  0 replies.
- The author displayed band index 80 over a crop of `data_hsi.mat` and reported
  unexpected horizontal stripes.
- There was no organizer response at capture time.

Operational consequence: include per-band stripe/outlier diagnostics and a
band-ablation experiment. Do not assume corruption or drop the band before an
offline validation result supports it.

