# Attribution — SQuAD 2.0 (basis for `data/fact800.jsonl`)

`data/fact800.jsonl` is derived from the **SQuAD 2.0** dataset by Rajpurkar
et al. (2018). The shipped file contains a sampled subset (seed=42) of the
SQuAD 2.0 train/validation split, restructured into matched A/U pairs using
the official `is_impossible` labels. The original passages, questions, and
answer spans are reproduced verbatim where used.

- **Upstream**: <https://rajpurkar.github.io/SQuAD-explorer/>
- **License**: Creative Commons Attribution-ShareAlike 4.0 International
  (CC-BY-SA-4.0).

Per CC-BY-SA-4.0:

- Attribution: Rajpurkar et al., "Know What You Don't Know: Unanswerable
  Questions for SQuAD" (ACL 2018).
- ShareAlike: any redistribution of `data/fact800.jsonl` or derivatives must
  carry the same CC-BY-SA-4.0 terms. This file (the .jsonl) is therefore
  released under CC-BY-SA-4.0 even though the repo's overall LICENSE is MIT.

Reconstruction script: `src/data/prepare_squad2.py` rebuilds `fact800.jsonl`
directly from the upstream SQuAD 2.0 download, so the in-repo .jsonl is a
convenience copy.
