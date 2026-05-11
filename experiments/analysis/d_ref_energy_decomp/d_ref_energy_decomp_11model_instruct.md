# d_ref / d_imp Energy Decomposition + cos_full_full Attribution (11-model main grid)

Re-run of the v1 closed-form / saved-reps decomposition (see `analysis/d_ref_energy_decomp.py`) on the 22 (model, dataset) cells of the 11-model main grid defined in `experiments/main_grid_facts_v2.json` (`orthogonality.cells`). Algorithm is unchanged; only the cell list is swapped from the legacy 8-model grid to the 11-model main grid. No model forward.

## Per-cell decomposition (22 instruct cells)

| model | dataset | L | D | verified | $\cos_\text{matched,full}$ | $\cos_\text{full,full}$ | $E_{\mathrm{PC}}(d_\text{ref})$ | $E_\text{null}(d_\text{ref})$ | $E_{\mathrm{PC}}(d_\text{imp,full})$ | $E_\text{null}(d_\text{imp,full})$ | PC-PC | null-null | PC-PC share | null-null share |
|---|---|---:|---:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smollm2 | math800 | 11 | 2048 | ✓ | +0.0521 | +0.1008 | 0.1140 | 0.8860 | 0.6595 | 0.3405 | +0.0704 | +0.0304 | 0.698 | 0.302 |
| smollm2 | code800 | 11 | 2048 | ✓ | +0.0475 | +0.1536 | 0.1774 | 0.8226 | 0.8430 | 0.1570 | +0.1348 | +0.0188 | 0.877 | 0.123 |
| phi4mini | math800 | 14 | 3072 | ✓ | +0.1237 | +0.2524 | 0.1488 | 0.8512 | 0.8352 | 0.1648 | +0.2022 | +0.0502 | 0.801 | 0.199 |
| phi4mini | code800 | 29 | 3072 | ✓ | +0.0752 | +0.1185 | 0.1089 | 0.8911 | 0.8376 | 0.1624 | +0.0882 | +0.0303 | 0.744 | 0.256 |
| gemma3_4b | math800 | 16 | 2560 | ✓ | +0.1216 | +0.5312 | 0.3773 | 0.6227 | 0.9206 | 0.0794 | +0.4969 | +0.0343 | 0.935 | 0.065 |
| gemma3_4b | code800 | 15 | 2560 | ✓ | +0.0859 | +0.7811 | 0.6839 | 0.3161 | 0.9761 | 0.0239 | +0.7678 | +0.0133 | 0.983 | 0.017 |
| mistral | math800 | 15 | 4096 | ✓ | +0.1204 | +0.2720 | 0.1591 | 0.8409 | 0.8138 | 0.1862 | +0.2201 | +0.0519 | 0.809 | 0.191 |
| mistral | code800 | 15 | 4096 | ✓ | +0.0897 | +0.2446 | 0.1838 | 0.8162 | 0.9211 | 0.0789 | +0.2194 | +0.0252 | 0.897 | 0.103 |
| qwen3_8b | math800 | 21 | 4096 | ✓ | +0.1088 | +0.2379 | 0.1204 | 0.8796 | 0.8057 | 0.1943 | +0.1900 | +0.0480 | 0.798 | 0.202 |
| qwen3_8b | code800 | 19 | 4096 | ✓ | +0.1156 | +0.2779 | 0.1293 | 0.8707 | 0.8253 | 0.1747 | +0.2296 | +0.0483 | 0.826 | 0.174 |
| llama | math800 | 15 | 4096 | — | +0.0682 | +0.2568 | 0.2180 | 0.7820 | 0.8233 | 0.1767 | +0.2281 | +0.0287 | 0.888 | 0.112 |
| llama | code800 | 15 | 4096 | — | +0.0720 | +0.2439 | 0.1832 | 0.8168 | 0.8361 | 0.1639 | +0.2148 | +0.0292 | 0.880 | 0.120 |
| qwen3_14b | math800 | 25 | 5120 | ✓ | +0.0803 | +0.2511 | 0.1273 | 0.8727 | 0.8504 | 0.1496 | +0.2200 | +0.0311 | 0.876 | 0.124 |
| qwen3_14b | code800 | 24 | 5120 | ✓ | +0.1296 | +0.2342 | 0.1196 | 0.8804 | 0.7994 | 0.2006 | +0.1762 | +0.0581 | 0.752 | 0.248 |
| olmo13b | math800 | 23 | 5120 | ✓ | +0.1010 | +0.1974 | 0.0869 | 0.9131 | 0.7825 | 0.2175 | +0.1503 | +0.0471 | 0.761 | 0.239 |
| olmo13b | code800 | 15 | 5120 | ✓ | +0.0972 | +0.1043 | 0.0779 | 0.9221 | 0.8221 | 0.1779 | +0.0633 | +0.0410 | 0.607 | 0.393 |
| mistral_small | math800 | 28 | 5120 | ✓ | +0.0649 | +0.1232 | 0.1669 | 0.8331 | 0.8667 | 0.1333 | +0.0995 | +0.0237 | 0.808 | 0.192 |
| mistral_small | code800 | 28 | 5120 | ✓ | +0.0195 | +0.1796 | 0.1412 | 0.8588 | 0.8822 | 0.1178 | +0.1729 | +0.0067 | 0.963 | 0.037 |
| qwen3_32b | math800 | 48 | 5120 | ✓ | +0.0977 | +0.2404 | 0.1376 | 0.8624 | 0.8024 | 0.1976 | +0.1970 | +0.0434 | 0.819 | 0.181 |
| qwen3_32b | code800 | 47 | 5120 | ✓ | +0.1096 | +0.1919 | 0.1343 | 0.8657 | 0.7578 | 0.2422 | +0.1379 | +0.0540 | 0.719 | 0.281 |
| llama70b | math800 | 31 | 8192 | ✓ | +0.0976 | +0.2391 | 0.1192 | 0.8808 | 0.7989 | 0.2011 | +0.1953 | +0.0438 | 0.817 | 0.183 |
| llama70b | code800 | 72 | 8192 | ✓ | +0.0371 | +0.0565 | 0.0287 | 0.9713 | 0.6585 | 0.3415 | +0.0348 | +0.0217 | 0.617 | 0.383 |

## Summary stats (22 instruct cells)

- $E_{\mathrm{PC}}(d_\text{ref})$: mean **0.170**, range $[0.029, 0.684]$, n=22
- $E_{\mathrm{PC}}(d_\text{imp,full})$: mean **0.824**, range $[0.659, 0.976]$, n=22
- $|\cos_\text{full,full}|$: mean **0.240**, range $[0.056, 0.781]$, n=22
- PC-PC share: mean **0.813**, range $[0.607, 0.983]$, n=22
- null-null share: mean **0.187**, range $[0.017, 0.393]$, n=22
- Cells with sign-disagreement among (PC-PC, null-null, $\cos_\text{full,full}$): **0/22**

## Notes

PC-PC share := $|\text{PC-PC contrib}| / (|\text{PC-PC}| + |\text{null-null}|)$ — the magnitude share of $\cos_\text{full,full}$ attributable to PC-PC overlap. The two shares sum to 1 by construction. When PC-PC and null-null have the same sign as $\cos_\text{full,full}$, magnitude share equals signed share. We track sign agreement explicitly to flag any cell where 'share' wording would be ambiguous.
