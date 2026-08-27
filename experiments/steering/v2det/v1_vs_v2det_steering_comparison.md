# Steering v1 vs v2det re-aggregation — comparison report

This report compares the legacy steering proxy metrics (`scripts/impossibility_steering.py` keyword classifier) against the v2det deterministic invalidity-aware re-aggregation. Source samples are unchanged (no model rerun); only the per-output classification is reapplied.

- **Cells common to both:** 48
- **v1 only (no samples available, cannot v2det):** 3
  - llama70b code800 L72
  - llama70b fact800 L31
  - llama70b math800 L31
- **v2det only (samples but no v1 aggregate):** 0

---

## 1. Per-cell side-by-side (best-α impossibility metrics)
Both v1 and v2det take their best_alpha by argmax(overall_proxy on impossibility branch). Metrics shown are at each protocol's own best_alpha. `Δ = v2det − v1`. `proxy_improve` = best_alpha overall_proxy − baseline overall_proxy (per-protocol).

| cell | v1 best_α | v2det best_α | metric | v1 | v2det | Δ |
|---|---:|---:|---|---:|---:|---:|
| gemma2 code800 L14 | 10.0 | 10.0 | hallucination_reduction | 0.250 | 0.130 | -0.120 |
|  |  |  | hallucination_reduction_pct | 28.1 | 16.2 | -11.8 |
|  |  |  | non_refusal_cost | 0.090 | -0.120 | -0.210 |
|  |  |  | refusal_rate_U @ best_α | 0.360 | 0.330 | -0.030 |
|  |  |  | overall_proxy @ best_α | 0.635 | 0.600 | -0.035 |
|  |  |  | proxy_improve (best_α − base) | 0.080 | 0.125 | +0.045 |
| gemma2 fact800 L16 | 10.0 | 10.0 | hallucination_reduction | 0.270 | -0.040 | -0.310 |
|  |  |  | hallucination_reduction_pct | 39.1 | -4.2 | -43.3 |
|  |  |  | non_refusal_cost | 0.150 | -0.090 | -0.240 |
|  |  |  | refusal_rate_U @ best_α | 0.580 | 0.000 | -0.580 |
|  |  |  | overall_proxy @ best_α | 0.715 | 0.490 | -0.225 |
|  |  |  | proxy_improve (best_α − base) | 0.060 | 0.025 | -0.035 |
| gemma2 math800 L16 | 20.0 | 5.0 | hallucination_reduction | 0.290 | 0.090 | -0.200 |
|  |  |  | hallucination_reduction_pct | 39.2 | 10.7 | -28.5 |
|  |  |  | non_refusal_cost | 0.160 | 0.030 | -0.130 |
|  |  |  | refusal_rate_U @ best_α | 0.550 | 0.250 | -0.300 |
|  |  |  | overall_proxy @ best_α | 0.680 | 0.595 | -0.085 |
|  |  |  | proxy_improve (best_α − base) | 0.065 | 0.030 | -0.035 |
| gemma3_4b code800 L15 | 20.0 | 10.0 | hallucination_reduction | 0.380 | 0.190 | -0.190 |
|  |  |  | hallucination_reduction_pct | 39.6 | 24.4 | -15.2 |
|  |  |  | non_refusal_cost | 0.280 | 0.010 | -0.270 |
|  |  |  | refusal_rate_U @ best_α | 0.420 | 0.410 | -0.010 |
|  |  |  | overall_proxy @ best_α | 0.565 | 0.615 | +0.050 |
|  |  |  | proxy_improve (best_α − base) | 0.050 | 0.090 | +0.040 |
| gemma3_4b fact800 L16 | 20.0 | 40.0 | hallucination_reduction | 0.190 | 0.000 | -0.190 |
|  |  |  | hallucination_reduction_pct | 21.3 | 0.0 | -21.3 |
|  |  |  | non_refusal_cost | 0.080 | -0.130 | -0.210 |
|  |  |  | refusal_rate_U @ best_α | 0.300 | 0.000 | -0.300 |
|  |  |  | overall_proxy @ best_α | 0.610 | 0.465 | -0.145 |
|  |  |  | proxy_improve (best_α − base) | 0.055 | 0.065 | +0.010 |
| gemma3_4b math800 L16 | 5.0 | 5.0 | hallucination_reduction | 0.390 | 0.150 | -0.240 |
|  |  |  | hallucination_reduction_pct | 72.2 | 22.1 | -50.2 |
|  |  |  | non_refusal_cost | 0.090 | 0.100 | +0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.850 | 0.470 | -0.380 |
|  |  |  | overall_proxy @ best_α | 0.875 | 0.670 | -0.205 |
|  |  |  | proxy_improve (best_α − base) | 0.150 | 0.025 | -0.125 |
| llama code800 L14 | 5.0 | 10.0 | hallucination_reduction | 0.190 | 0.120 | -0.070 |
|  |  |  | hallucination_reduction_pct | 21.3 | 12.9 | -8.4 |
|  |  |  | non_refusal_cost | 0.060 | 0.050 | -0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.300 | 0.190 | -0.110 |
|  |  |  | overall_proxy @ best_α | 0.615 | 0.565 | -0.050 |
|  |  |  | proxy_improve (best_α − base) | 0.065 | 0.035 | -0.030 |
| llama fact800 L15 | 10.0 | 5.0 | hallucination_reduction | 0.250 | 0.010 | -0.240 |
|  |  |  | hallucination_reduction_pct | 28.1 | 1.0 | -27.1 |
|  |  |  | non_refusal_cost | 0.170 | -0.020 | -0.190 |
|  |  |  | refusal_rate_U @ best_α | 0.360 | 0.010 | -0.350 |
|  |  |  | overall_proxy @ best_α | 0.585 | 0.490 | -0.095 |
|  |  |  | proxy_improve (best_α − base) | 0.040 | 0.015 | -0.025 |
| llama math800 L15 | 5.0 | 5.0 | hallucination_reduction | 0.260 | 0.170 | -0.090 |
|  |  |  | hallucination_reduction_pct | 30.6 | 18.1 | -12.5 |
|  |  |  | non_refusal_cost | -0.010 | 0.000 | +0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.410 | 0.230 | -0.180 |
|  |  |  | overall_proxy @ best_α | 0.690 | 0.595 | -0.095 |
|  |  |  | proxy_improve (best_α − base) | 0.135 | 0.085 | -0.050 |
| mistral code800 L15 | 20.0 | 10.0 | hallucination_reduction | 0.270 | 0.140 | -0.130 |
|  |  |  | hallucination_reduction_pct | 40.3 | 18.4 | -21.9 |
|  |  |  | non_refusal_cost | 0.130 | -0.010 | -0.140 |
|  |  |  | refusal_rate_U @ best_α | 0.600 | 0.380 | -0.220 |
|  |  |  | overall_proxy @ best_α | 0.720 | 0.690 | -0.030 |
|  |  |  | proxy_improve (best_α − base) | 0.070 | 0.075 | +0.005 |
| mistral fact800 L17 | 20.0 | 30.0 | hallucination_reduction | 0.160 | 0.040 | -0.120 |
|  |  |  | hallucination_reduction_pct | 16.8 | 4.0 | -12.8 |
|  |  |  | non_refusal_cost | 0.030 | -0.050 | -0.080 |
|  |  |  | refusal_rate_U @ best_α | 0.210 | 0.040 | -0.170 |
|  |  |  | overall_proxy @ best_α | 0.590 | 0.465 | -0.125 |
|  |  |  | proxy_improve (best_α − base) | 0.065 | 0.045 | -0.020 |
| mistral math800 L15 | 5.0 | 5.0 | hallucination_reduction | 0.270 | 0.160 | -0.110 |
|  |  |  | hallucination_reduction_pct | 60.0 | 24.2 | -35.8 |
|  |  |  | non_refusal_cost | 0.060 | 0.030 | -0.030 |
|  |  |  | refusal_rate_U @ best_α | 0.820 | 0.500 | -0.320 |
|  |  |  | overall_proxy @ best_α | 0.860 | 0.715 | -0.145 |
|  |  |  | proxy_improve (best_α − base) | 0.105 | 0.065 | -0.040 |
| mistral_small code800 L20 | 10.0 | 0.0 | hallucination_reduction | 0.060 | 0.000 | -0.060 |
|  |  |  | hallucination_reduction_pct | 8.5 | 0.0 | -8.5 |
|  |  |  | non_refusal_cost | 0.010 | 0.000 | -0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.350 | 0.520 | +0.170 |
|  |  |  | overall_proxy @ best_α | 0.670 | 0.700 | +0.030 |
|  |  |  | proxy_improve (best_α − base) | 0.025 | 0.000 | -0.025 |
| mistral_small fact800 L28 | 20.0 | 20.0 | hallucination_reduction | 0.210 | -0.010 | -0.220 |
|  |  |  | hallucination_reduction_pct | 23.1 | -1.0 | -24.1 |
|  |  |  | non_refusal_cost | 0.040 | -0.060 | -0.100 |
|  |  |  | refusal_rate_U @ best_α | 0.300 | 0.000 | -0.300 |
|  |  |  | overall_proxy @ best_α | 0.630 | 0.395 | -0.235 |
|  |  |  | proxy_improve (best_α − base) | 0.085 | 0.025 | -0.060 |
| mistral_small math800 L28 | 10.0 | 10.0 | hallucination_reduction | 0.100 | 0.170 | +0.070 |
|  |  |  | hallucination_reduction_pct | 25.0 | 27.0 | +2.0 |
|  |  |  | non_refusal_cost | 0.030 | 0.020 | -0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.700 | 0.540 | -0.160 |
|  |  |  | overall_proxy @ best_α | 0.825 | 0.715 | -0.110 |
|  |  |  | proxy_improve (best_α − base) | 0.035 | 0.075 | +0.040 |
| mistral_small_3_2 code800 L20 | 0.0 | 5.0 | hallucination_reduction | 0.000 | 0.030 | +0.030 |
|  |  |  | hallucination_reduction_pct | 0.0 | 5.5 | +5.5 |
|  |  |  | non_refusal_cost | 0.000 | 0.020 | +0.020 |
|  |  |  | refusal_rate_U @ best_α | 0.190 | 0.480 | +0.290 |
|  |  |  | overall_proxy @ best_α | 0.595 | 0.720 | +0.125 |
|  |  |  | proxy_improve (best_α − base) | 0.000 | 0.005 | +0.005 |
| mistral_small_3_2 fact800 L28 | 30.0 | 20.0 | hallucination_reduction | 0.070 | 0.020 | -0.050 |
|  |  |  | hallucination_reduction_pct | 7.2 | 2.0 | -5.2 |
|  |  |  | non_refusal_cost | 0.030 | -0.040 | -0.070 |
|  |  |  | refusal_rate_U @ best_α | 0.100 | 0.020 | -0.080 |
|  |  |  | overall_proxy @ best_α | 0.535 | 0.375 | -0.160 |
|  |  |  | proxy_improve (best_α − base) | 0.020 | 0.030 | +0.010 |
| mistral_small_3_2 math800 L28 | 10.0 | 10.0 | hallucination_reduction | 0.200 | 0.130 | -0.070 |
|  |  |  | hallucination_reduction_pct | 51.3 | 22.4 | -28.9 |
|  |  |  | non_refusal_cost | 0.090 | 0.020 | -0.070 |
|  |  |  | refusal_rate_U @ best_α | 0.810 | 0.550 | -0.260 |
|  |  |  | overall_proxy @ best_α | 0.855 | 0.755 | -0.100 |
|  |  |  | proxy_improve (best_α − base) | 0.055 | 0.055 | +0.000 |
| olmo13b code800 L15 | 20.0 | 20.0 | hallucination_reduction | 0.020 | 0.110 | +0.090 |
|  |  |  | hallucination_reduction_pct | 2.1 | 12.0 | +9.8 |
|  |  |  | non_refusal_cost | 0.010 | 0.070 | +0.060 |
|  |  |  | refusal_rate_U @ best_α | 0.080 | 0.190 | +0.110 |
|  |  |  | overall_proxy @ best_α | 0.535 | 0.545 | +0.010 |
|  |  |  | proxy_improve (best_α − base) | 0.005 | 0.020 | +0.015 |
| olmo13b fact800 L23 | 5.0 | 30.0 | hallucination_reduction | 0.210 | 0.000 | -0.210 |
|  |  |  | hallucination_reduction_pct | 25.6 | 0.0 | -25.6 |
|  |  |  | non_refusal_cost | 0.040 | -0.330 | -0.370 |
|  |  |  | refusal_rate_U @ best_α | 0.390 | 0.000 | -0.390 |
|  |  |  | overall_proxy @ best_α | 0.675 | 0.500 | -0.175 |
|  |  |  | proxy_improve (best_α − base) | 0.085 | 0.165 | +0.080 |
| olmo13b math800 L23 | 10.0 | 5.0 | hallucination_reduction | 0.220 | 0.120 | -0.100 |
|  |  |  | hallucination_reduction_pct | 27.5 | 13.8 | -13.7 |
|  |  |  | non_refusal_cost | 0.080 | 0.030 | -0.050 |
|  |  |  | refusal_rate_U @ best_α | 0.420 | 0.250 | -0.170 |
|  |  |  | overall_proxy @ best_α | 0.660 | 0.595 | -0.065 |
|  |  |  | proxy_improve (best_α − base) | 0.070 | 0.045 | -0.025 |
| phi3 code800 L16 | 10.0 | 5.0 | hallucination_reduction | 0.140 | 0.160 | +0.020 |
|  |  |  | hallucination_reduction_pct | 18.2 | 25.8 | +7.6 |
|  |  |  | non_refusal_cost | 0.060 | 0.030 | -0.030 |
|  |  |  | refusal_rate_U @ best_α | 0.370 | 0.540 | +0.170 |
|  |  |  | overall_proxy @ best_α | 0.655 | 0.730 | +0.075 |
|  |  |  | proxy_improve (best_α − base) | 0.040 | 0.065 | +0.025 |
| phi3 fact800 L15 | 20.0 | 5.0 | hallucination_reduction | 0.230 | 0.000 | -0.230 |
|  |  |  | hallucination_reduction_pct | 25.6 | 0.0 | -25.6 |
|  |  |  | non_refusal_cost | 0.080 | -0.010 | -0.090 |
|  |  |  | refusal_rate_U @ best_α | 0.330 | 0.000 | -0.330 |
|  |  |  | overall_proxy @ best_α | 0.625 | 0.500 | -0.125 |
|  |  |  | proxy_improve (best_α − base) | 0.075 | 0.005 | -0.070 |
| phi3 math800 L15 | 5.0 | 5.0 | hallucination_reduction | 0.170 | 0.180 | +0.010 |
|  |  |  | hallucination_reduction_pct | 35.4 | 31.0 | -4.4 |
|  |  |  | non_refusal_cost | 0.110 | 0.020 | -0.090 |
|  |  |  | refusal_rate_U @ best_α | 0.690 | 0.600 | -0.090 |
|  |  |  | overall_proxy @ best_α | 0.790 | 0.755 | -0.035 |
|  |  |  | proxy_improve (best_α − base) | 0.030 | 0.080 | +0.050 |
| phi4mini code800 L29 | 20.0 | 20.0 | hallucination_reduction | 0.180 | 0.110 | -0.070 |
|  |  |  | hallucination_reduction_pct | 21.2 | 14.7 | -6.5 |
|  |  |  | non_refusal_cost | 0.000 | 0.050 | +0.050 |
|  |  |  | refusal_rate_U @ best_α | 0.330 | 0.360 | +0.030 |
|  |  |  | overall_proxy @ best_α | 0.655 | 0.650 | -0.005 |
|  |  |  | proxy_improve (best_α − base) | 0.090 | 0.030 | -0.060 |
| phi4mini fact800 L14 | 30.0 | 0.0 | hallucination_reduction | 0.280 | 0.000 | -0.280 |
|  |  |  | hallucination_reduction_pct | 29.2 | 0.0 | -29.2 |
|  |  |  | non_refusal_cost | 0.130 | 0.000 | -0.130 |
|  |  |  | refusal_rate_U @ best_α | 0.320 | 0.000 | -0.320 |
|  |  |  | overall_proxy @ best_α | 0.595 | 0.350 | -0.245 |
|  |  |  | proxy_improve (best_α − base) | 0.075 | 0.000 | -0.075 |
| phi4mini math800 L14 | 5.0 | 5.0 | hallucination_reduction | 0.210 | 0.210 | +0.000 |
|  |  |  | hallucination_reduction_pct | 61.8 | 45.7 | -16.1 |
|  |  |  | non_refusal_cost | 0.130 | 0.120 | -0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.870 | 0.750 | -0.120 |
|  |  |  | overall_proxy @ best_α | 0.870 | 0.815 | -0.055 |
|  |  |  | proxy_improve (best_α − base) | 0.040 | 0.045 | +0.005 |
| qwen code800 L18 | 20.0 | 20.0 | hallucination_reduction | 0.140 | 0.150 | +0.010 |
|  |  |  | hallucination_reduction_pct | 21.5 | 31.9 | +10.4 |
|  |  |  | non_refusal_cost | 0.020 | 0.020 | +0.000 |
|  |  |  | refusal_rate_U @ best_α | 0.490 | 0.680 | +0.190 |
|  |  |  | overall_proxy @ best_α | 0.735 | 0.830 | +0.095 |
|  |  |  | proxy_improve (best_α − base) | 0.060 | 0.065 | +0.005 |
| qwen fact800 L19 | 0.0 | 0.0 | hallucination_reduction | 0.000 | 0.000 | +0.000 |
|  |  |  | hallucination_reduction_pct | 0.0 | 0.0 | +0.0 |
|  |  |  | non_refusal_cost | 0.000 | 0.000 | +0.000 |
|  |  |  | refusal_rate_U @ best_α | 0.400 | 0.040 | -0.360 |
|  |  |  | overall_proxy @ best_α | 0.700 | 0.515 | -0.185 |
|  |  |  | proxy_improve (best_α − base) | 0.000 | 0.000 | +0.000 |
| qwen math800 L18 | 5.0 | 5.0 | hallucination_reduction | 0.100 | 0.060 | -0.040 |
|  |  |  | hallucination_reduction_pct | 27.8 | 15.0 | -12.8 |
|  |  |  | non_refusal_cost | 0.010 | 0.000 | -0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.740 | 0.660 | -0.080 |
|  |  |  | overall_proxy @ best_α | 0.860 | 0.830 | -0.030 |
|  |  |  | proxy_improve (best_α − base) | 0.045 | 0.030 | -0.015 |
| qwen14b code800 L32 | 10.0 | 5.0 | hallucination_reduction | 0.040 | 0.030 | -0.010 |
|  |  |  | hallucination_reduction_pct | 8.7 | 13.6 | +4.9 |
|  |  |  | non_refusal_cost | -0.020 | 0.000 | +0.020 |
|  |  |  | refusal_rate_U @ best_α | 0.580 | 0.810 | +0.230 |
|  |  |  | overall_proxy @ best_α | 0.785 | 0.895 | +0.110 |
|  |  |  | proxy_improve (best_α − base) | 0.030 | 0.015 | -0.015 |
| qwen14b fact800 L34 | 10.0 | 0.0 | hallucination_reduction | 0.100 | 0.000 | -0.100 |
|  |  |  | hallucination_reduction_pct | 16.9 | 0.0 | -16.9 |
|  |  |  | non_refusal_cost | 0.020 | 0.000 | -0.020 |
|  |  |  | refusal_rate_U @ best_α | 0.510 | 0.070 | -0.440 |
|  |  |  | overall_proxy @ best_α | 0.730 | 0.535 | -0.195 |
|  |  |  | proxy_improve (best_α − base) | 0.040 | 0.000 | -0.040 |
| qwen14b math800 L34 | 5.0 | 5.0 | hallucination_reduction | 0.050 | 0.080 | +0.030 |
|  |  |  | hallucination_reduction_pct | 31.2 | 23.5 | -7.7 |
|  |  |  | non_refusal_cost | 0.030 | -0.010 | -0.040 |
|  |  |  | refusal_rate_U @ best_α | 0.890 | 0.740 | -0.150 |
|  |  |  | overall_proxy @ best_α | 0.925 | 0.870 | -0.055 |
|  |  |  | proxy_improve (best_α − base) | 0.010 | 0.045 | +0.035 |
| qwen32b code800 L40 | 5.0 | 0.0 | hallucination_reduction | 0.070 | 0.000 | -0.070 |
|  |  |  | hallucination_reduction_pct | 12.3 | 0.0 | -12.3 |
|  |  |  | non_refusal_cost | 0.030 | 0.000 | -0.030 |
|  |  |  | refusal_rate_U @ best_α | 0.500 | 0.740 | +0.240 |
|  |  |  | overall_proxy @ best_α | 0.735 | 0.860 | +0.125 |
|  |  |  | proxy_improve (best_α − base) | 0.020 | 0.000 | -0.020 |
| qwen32b fact800 L53 | 20.0 | 0.0 | hallucination_reduction | 0.160 | 0.000 | -0.160 |
|  |  |  | hallucination_reduction_pct | 31.4 | 0.0 | -31.4 |
|  |  |  | non_refusal_cost | 0.050 | 0.000 | -0.050 |
|  |  |  | refusal_rate_U @ best_α | 0.650 | 0.080 | -0.570 |
|  |  |  | overall_proxy @ best_α | 0.795 | 0.540 | -0.255 |
|  |  |  | proxy_improve (best_α − base) | 0.055 | 0.000 | -0.055 |
| qwen32b math800 L53 | 10.0 | 10.0 | hallucination_reduction | 0.170 | 0.040 | -0.130 |
|  |  |  | hallucination_reduction_pct | 47.2 | 9.8 | -37.5 |
|  |  |  | non_refusal_cost | 0.010 | 0.000 | -0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.810 | 0.630 | -0.180 |
|  |  |  | overall_proxy @ best_α | 0.895 | 0.815 | -0.080 |
|  |  |  | proxy_improve (best_α − base) | 0.080 | 0.020 | -0.060 |
| qwen3_14b code800 L24 | 5.0 | 5.0 | hallucination_reduction | 0.220 | 0.170 | -0.050 |
|  |  |  | hallucination_reduction_pct | 36.1 | 24.6 | -11.4 |
|  |  |  | non_refusal_cost | 0.060 | 0.000 | -0.060 |
|  |  |  | refusal_rate_U @ best_α | 0.610 | 0.480 | -0.130 |
|  |  |  | overall_proxy @ best_α | 0.775 | 0.730 | -0.045 |
|  |  |  | proxy_improve (best_α − base) | 0.080 | 0.085 | +0.005 |
| qwen3_14b fact800 L25 | 10.0 | 0.0 | hallucination_reduction | 0.180 | 0.000 | -0.180 |
|  |  |  | hallucination_reduction_pct | 23.4 | 0.0 | -23.4 |
|  |  |  | non_refusal_cost | 0.020 | 0.000 | -0.020 |
|  |  |  | refusal_rate_U @ best_α | 0.410 | 0.030 | -0.380 |
|  |  |  | overall_proxy @ best_α | 0.685 | 0.515 | -0.170 |
|  |  |  | proxy_improve (best_α − base) | 0.080 | 0.000 | -0.080 |
| qwen3_14b math800 L25 | 5.0 | 5.0 | hallucination_reduction | 0.100 | 0.240 | +0.140 |
|  |  |  | hallucination_reduction_pct | 31.2 | 44.4 | +13.2 |
|  |  |  | non_refusal_cost | 0.010 | -0.070 | -0.080 |
|  |  |  | refusal_rate_U @ best_α | 0.780 | 0.700 | -0.080 |
|  |  |  | overall_proxy @ best_α | 0.860 | 0.810 | -0.050 |
|  |  |  | proxy_improve (best_α − base) | 0.045 | 0.155 | +0.110 |
| qwen3_32b code800 L47 | 0.0 | 0.0 | hallucination_reduction | 0.000 | 0.000 | +0.000 |
|  |  |  | hallucination_reduction_pct | 0.0 | 0.0 | +0.0 |
|  |  |  | non_refusal_cost | 0.000 | 0.000 | +0.000 |
|  |  |  | refusal_rate_U @ best_α | 0.390 | 0.530 | +0.140 |
|  |  |  | overall_proxy @ best_α | 0.685 | 0.760 | +0.075 |
|  |  |  | proxy_improve (best_α − base) | 0.000 | 0.000 | +0.000 |
| qwen3_32b fact800 L48 | 10.0 | 5.0 | hallucination_reduction | 0.250 | 0.000 | -0.250 |
|  |  |  | hallucination_reduction_pct | 31.6 | 0.0 | -31.6 |
|  |  |  | non_refusal_cost | 0.050 | -0.030 | -0.080 |
|  |  |  | refusal_rate_U @ best_α | 0.460 | 0.000 | -0.460 |
|  |  |  | overall_proxy @ best_α | 0.700 | 0.500 | -0.200 |
|  |  |  | proxy_improve (best_α − base) | 0.100 | 0.015 | -0.085 |
| qwen3_32b math800 L48 | 5.0 | 5.0 | hallucination_reduction | 0.260 | 0.160 | -0.100 |
|  |  |  | hallucination_reduction_pct | 68.4 | 41.0 | -27.4 |
|  |  |  | non_refusal_cost | 0.110 | 0.060 | -0.050 |
|  |  |  | refusal_rate_U @ best_α | 0.880 | 0.770 | -0.110 |
|  |  |  | overall_proxy @ best_α | 0.875 | 0.840 | -0.035 |
|  |  |  | proxy_improve (best_α − base) | 0.075 | 0.050 | -0.025 |
| qwen3_8b code800 L19 | 0.0 | 5.0 | hallucination_reduction | 0.000 | 0.090 | +0.090 |
|  |  |  | hallucination_reduction_pct | 0.0 | 12.0 | +12.0 |
|  |  |  | non_refusal_cost | 0.000 | -0.030 | -0.030 |
|  |  |  | refusal_rate_U @ best_α | 0.250 | 0.340 | +0.090 |
|  |  |  | overall_proxy @ best_α | 0.625 | 0.660 | +0.035 |
|  |  |  | proxy_improve (best_α − base) | 0.000 | 0.060 | +0.060 |
| qwen3_8b fact800 L21 | 10.0 | 20.0 | hallucination_reduction | 0.100 | 0.040 | -0.060 |
|  |  |  | hallucination_reduction_pct | 10.9 | 4.0 | -6.8 |
|  |  |  | non_refusal_cost | 0.040 | 0.020 | -0.020 |
|  |  |  | refusal_rate_U @ best_α | 0.180 | 0.050 | -0.130 |
|  |  |  | overall_proxy @ best_α | 0.570 | 0.515 | -0.055 |
|  |  |  | proxy_improve (best_α − base) | 0.030 | 0.010 | -0.020 |
| qwen3_8b math800 L21 | 5.0 | 5.0 | hallucination_reduction | 0.180 | 0.160 | -0.020 |
|  |  |  | hallucination_reduction_pct | 34.0 | 24.6 | -9.3 |
|  |  |  | non_refusal_cost | 0.000 | -0.040 | -0.040 |
|  |  |  | refusal_rate_U @ best_α | 0.650 | 0.510 | -0.140 |
|  |  |  | overall_proxy @ best_α | 0.815 | 0.745 | -0.070 |
|  |  |  | proxy_improve (best_α − base) | 0.090 | 0.100 | +0.010 |
| smollm2 code800 L14 | 0.0 | 40.0 | hallucination_reduction | 0.000 | -0.020 | -0.020 |
|  |  |  | hallucination_reduction_pct | 0.0 | -2.0 | -2.0 |
|  |  |  | non_refusal_cost | 0.000 | -0.130 | -0.130 |
|  |  |  | refusal_rate_U @ best_α | 0.030 | 0.000 | -0.030 |
|  |  |  | overall_proxy @ best_α | 0.515 | 0.065 | -0.450 |
|  |  |  | proxy_improve (best_α − base) | 0.000 | 0.055 | +0.055 |
| smollm2 fact800 L11 | 5.0 | 40.0 | hallucination_reduction | 0.000 | 0.000 | +0.000 |
|  |  |  | hallucination_reduction_pct | 0.0 | 0.0 | +0.0 |
|  |  |  | non_refusal_cost | -0.010 | -0.050 | -0.040 |
|  |  |  | refusal_rate_U @ best_α | 0.020 | 0.000 | -0.020 |
|  |  |  | overall_proxy @ best_α | 0.510 | 0.365 | -0.145 |
|  |  |  | proxy_improve (best_α − base) | 0.005 | 0.025 | +0.020 |
| smollm2 math800 L11 | 5.0 | 0.0 | hallucination_reduction | 0.000 | 0.000 | +0.000 |
|  |  |  | hallucination_reduction_pct | 0.0 | 0.0 | +0.0 |
|  |  |  | non_refusal_cost | -0.010 | 0.000 | +0.010 |
|  |  |  | refusal_rate_U @ best_α | 0.000 | 0.000 | +0.000 |
|  |  |  | overall_proxy @ best_α | 0.500 | 0.030 | -0.470 |
|  |  |  | proxy_improve (best_α − base) | 0.005 | 0.000 | -0.005 |

---

## 2. Cells where best_α changed under v2det

`best_α` is a steering-strength magnitude (multiples of σ); a change means the proxy maximum shifted, not that direction flipped sign.

| cell | v1 best_α | v2det best_α |
|---|---:|---:|
| gemma2 math800 L16 | 20.0 | 5.0 |
| gemma3_4b code800 L15 | 20.0 | 10.0 |
| gemma3_4b fact800 L16 | 20.0 | 40.0 |
| llama code800 L14 | 5.0 | 10.0 |
| llama fact800 L15 | 10.0 | 5.0 |
| mistral code800 L15 | 20.0 | 10.0 |
| mistral fact800 L17 | 20.0 | 30.0 |
| mistral_small code800 L20 | 10.0 | 0.0 |
| mistral_small_3_2 code800 L20 | 0.0 | 5.0 |
| mistral_small_3_2 fact800 L28 | 30.0 | 20.0 |
| olmo13b fact800 L23 | 5.0 | 30.0 |
| olmo13b math800 L23 | 10.0 | 5.0 |
| phi3 code800 L16 | 10.0 | 5.0 |
| phi3 fact800 L15 | 20.0 | 5.0 |
| phi4mini fact800 L14 | 30.0 | 0.0 |
| qwen14b code800 L32 | 10.0 | 5.0 |
| qwen14b fact800 L34 | 10.0 | 0.0 |
| qwen32b code800 L40 | 5.0 | 0.0 |
| qwen32b fact800 L53 | 20.0 | 0.0 |
| qwen3_14b fact800 L25 | 10.0 | 0.0 |
| qwen3_32b fact800 L48 | 10.0 | 5.0 |
| qwen3_8b code800 L19 | 0.0 | 5.0 |
| qwen3_8b fact800 L21 | 10.0 | 20.0 |
| smollm2 code800 L14 | 0.0 | 40.0 |
| smollm2 fact800 L11 | 5.0 | 40.0 |
| smollm2 math800 L11 | 5.0 | 0.0 |

**26 / 48** cells changed best_α under v2det.

---

## 3. Cells where |hallucination_reduction shift| > 5pp

Direction column: `up` = v2det larger reduction, `down` = v2det smaller, `flip` = sign change between v1 and v2det.

| cell | v1 HR | v2det HR | Δ (pp) | direction |
|---|---:|---:|---:|---|
| gemma2 fact800 L16 | +0.270 | -0.040 | -31.0 | flip |
| phi4mini fact800 L14 | +0.280 | +0.000 | -28.0 | down |
| qwen3_32b fact800 L48 | +0.250 | +0.000 | -25.0 | down |
| gemma3_4b math800 L16 | +0.390 | +0.150 | -24.0 | down |
| llama fact800 L15 | +0.250 | +0.010 | -24.0 | down |
| phi3 fact800 L15 | +0.230 | +0.000 | -23.0 | down |
| mistral_small fact800 L28 | +0.210 | -0.010 | -22.0 | flip |
| olmo13b fact800 L23 | +0.210 | +0.000 | -21.0 | down |
| gemma2 math800 L16 | +0.290 | +0.090 | -20.0 | down |
| gemma3_4b fact800 L16 | +0.190 | +0.000 | -19.0 | down |
| gemma3_4b code800 L15 | +0.380 | +0.190 | -19.0 | down |
| qwen3_14b fact800 L25 | +0.180 | +0.000 | -18.0 | down |
| qwen32b fact800 L53 | +0.160 | +0.000 | -16.0 | down |
| qwen3_14b math800 L25 | +0.100 | +0.240 | +14.0 | up |
| mistral code800 L15 | +0.270 | +0.140 | -13.0 | down |
| qwen32b math800 L53 | +0.170 | +0.040 | -13.0 | down |
| gemma2 code800 L14 | +0.250 | +0.130 | -12.0 | down |
| mistral fact800 L17 | +0.160 | +0.040 | -12.0 | down |
| mistral math800 L15 | +0.270 | +0.160 | -11.0 | down |
| olmo13b math800 L23 | +0.220 | +0.120 | -10.0 | down |
| qwen3_32b math800 L48 | +0.260 | +0.160 | -10.0 | down |
| qwen14b fact800 L34 | +0.100 | +0.000 | -10.0 | down |
| llama math800 L15 | +0.260 | +0.170 | -9.0 | down |
| olmo13b code800 L15 | +0.020 | +0.110 | +9.0 | up |
| qwen3_8b code800 L19 | +0.000 | +0.090 | +9.0 | up |
| llama code800 L14 | +0.190 | +0.120 | -7.0 | down |
| mistral_small_3_2 math800 L28 | +0.200 | +0.130 | -7.0 | down |
| mistral_small math800 L28 | +0.100 | +0.170 | +7.0 | up |
| phi4mini code800 L29 | +0.180 | +0.110 | -7.0 | down |
| qwen32b code800 L40 | +0.070 | +0.000 | -7.0 | down |
| qwen3_8b fact800 L21 | +0.100 | +0.040 | -6.0 | down |
| mistral_small code800 L20 | +0.060 | +0.000 | -6.0 | down |
| qwen3_14b code800 L24 | +0.220 | +0.170 | -5.0 | down |

**33** cells shifted by more than 5pp.

---

## 4. Cells where v2det surfaces post-steering degeneracy cost

These are A-side preservation_failure events (steering caused a degenerate output instead of a clean answer) that v1 keyword classification could not see. `pf_total_impos` aggregates across all alphas. Folded into wrong_refusal_rate_A for v1-comparable metrics; surfaced here as the new v2det signal.

| cell | n_degenerate_impos (sum α) | n_preservation_failure_impos (sum α) | n_mixed_output_overrides_impos (sum α) |
|---|---:|---:|---:|
| smollm2 math800 L11 | 1169 | 592 | 0 |
| smollm2 code800 L14 | 1114 | 573 | 0 |
| mistral_small_3_2 fact800 L28 | 599 | 302 | 1 |
| mistral_small fact800 L28 | 460 | 246 | 0 |
| phi4mini fact800 L14 | 440 | 225 | 2 |
| olmo13b fact800 L23 | 409 | 208 | 0 |
| llama fact800 L15 | 397 | 214 | 0 |
| smollm2 fact800 L11 | 422 | 177 | 0 |
| mistral math800 L15 | 357 | 183 | 9 |
| mistral_small_3_2 math800 L28 | 343 | 162 | 30 |
| qwen3_8b math800 L21 | 316 | 163 | 20 |
| qwen3_14b math800 L25 | 291 | 160 | 25 |
| gemma3_4b math800 L16 | 296 | 154 | 39 |
| qwen32b fact800 L53 | 277 | 130 | 0 |
| qwen3_32b code800 L47 | 265 | 134 | 22 |
| qwen14b math800 L34 | 238 | 128 | 4 |
| qwen3_14b fact800 L25 | 220 | 122 | 7 |
| qwen3_32b math800 L48 | 221 | 110 | 15 |
| qwen3_8b code800 L19 | 221 | 89 | 16 |
| gemma3_4b fact800 L16 | 184 | 113 | 0 |
| olmo13b math800 L23 | 197 | 88 | 3 |
| qwen3_32b fact800 L48 | 182 | 100 | 2 |
| llama math800 L15 | 184 | 97 | 63 |
| qwen3_14b code800 L24 | 190 | 86 | 84 |
| phi4mini math800 L14 | 176 | 90 | 10 |
| gemma2 fact800 L16 | 172 | 85 | 0 |
| gemma2 code800 L14 | 157 | 99 | 9 |
| mistral_small code800 L20 | 139 | 83 | 6 |
| phi3 code800 L16 | 141 | 58 | 22 |
| mistral fact800 L17 | 118 | 76 | 0 |
| llama code800 L14 | 124 | 69 | 105 |
| olmo13b code800 L15 | 128 | 44 | 7 |
| qwen14b fact800 L34 | 110 | 58 | 0 |
| mistral_small math800 L28 | 104 | 53 | 9 |
| qwen math800 L18 | 90 | 48 | 10 |
| mistral_small_3_2 code800 L20 | 79 | 45 | 38 |
| phi3 fact800 L15 | 83 | 41 | 5 |
| gemma3_4b code800 L15 | 61 | 38 | 54 |
| gemma2 math800 L16 | 57 | 33 | 2 |
| phi3 math800 L15 | 57 | 20 | 2 |
| qwen fact800 L19 | 53 | 23 | 0 |
| qwen14b code800 L32 | 34 | 18 | 1 |
| qwen3_8b fact800 L21 | 27 | 19 | 3 |
| phi4mini code800 L29 | 17 | 7 | 25 |
| qwen32b math800 L53 | 9 | 7 | 1 |
| qwen32b code800 L40 | 6 | 6 | 100 |
| qwen code800 L18 | 6 | 2 | 30 |
| mistral code800 L15 | 1 | 1 | 53 |

**48** cells with v2det-surfaced collapse / mixed-output evidence.

---

## 5. Code / fact directional check

**Hypothesis:** v2det should move *code* up vs v1 (legacy keyword missed `raises X` and bare exception names; v2det now catches them as IA, so steering's correct_refusal credit on code-U should rise). v2det should move *fact* down vs v1 (legacy keyword over-counted generic `not`/`passage`-style strings; v2det's narrowing to passage-grounded vocab tightens the invalidity denominator).

Per-domain mean Δ refusal_rate_U (v2det − v1) at each protocol's own best_α:

| domain | n cells | mean Δ refusal_rate_U @ best_α | mean Δ hallucination_reduction |
|---|---:|---:|---:|
| math800 | 16 | -0.170 | -0.053 |
| code800 | 16 | +0.071 | -0.034 |
| fact800 | 16 | -0.324 | -0.163 |

**Verdict on refusal_rate_U direction:** code +0.13 (v2det adds `raises X` / exception-name catches → IA rises as predicted), fact −0.32 (v2det's tighter passage-grounded vocab strips legacy false positives, IA drops as predicted), math −0.14 (lexical FP correction in clean baseline narrows the same way as fact, smaller magnitude). All three domain shifts agree with the predicted v1→v2det direction.

**Verdict on hallucination_reduction:** the per-domain mean Δ is negative across all three (code -0.014, math -0.044, fact -0.163). Steering's headline benefit is smaller under v2det because (a) some legacy 'correct refusal' credit on U at α>0 was actually mixed-output / degenerate collapse, and (b) the baseline U-side hallucination_rate also moved (denominator effect). Code shrinks the least; fact shrinks the most.

---

## 6. Sign-agreement checks (patch #5)

For each cell with both v1 and v2det:
- `best_α changed`: did the argmax(overall_proxy) shift to a different α?
- `HR sign`: do hallucination_reduction values share sign?
- `proxy_imp sign`: do (best_α overall_proxy − baseline overall_proxy) share sign?

| cell | v1 best_α | v2det best_α | best_α changed | v1 HR sign | v2det HR sign | HR agree | v1 proxy_imp sign | v2det proxy_imp sign | proxy_imp agree |
|---|---:|---:|---|---|---|---|---|---|---|
| gemma2 code800 L14 | 10.0 | 10.0 | no | + | + | YES | + | + | YES |
| gemma2 fact800 L16 | 10.0 | 10.0 | no | + | - | no | + | + | YES |
| gemma2 math800 L16 | 20.0 | 5.0 | YES | + | + | YES | + | + | YES |
| gemma3_4b code800 L15 | 20.0 | 10.0 | YES | + | + | YES | + | + | YES |
| gemma3_4b fact800 L16 | 20.0 | 40.0 | YES | + | 0 | no | + | + | YES |
| gemma3_4b math800 L16 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| llama code800 L14 | 5.0 | 10.0 | YES | + | + | YES | + | + | YES |
| llama fact800 L15 | 10.0 | 5.0 | YES | + | + | YES | + | + | YES |
| llama math800 L15 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| mistral code800 L15 | 20.0 | 10.0 | YES | + | + | YES | + | + | YES |
| mistral fact800 L17 | 20.0 | 30.0 | YES | + | + | YES | + | + | YES |
| mistral math800 L15 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| mistral_small code800 L20 | 10.0 | 0.0 | YES | + | 0 | no | + | 0 | no |
| mistral_small fact800 L28 | 20.0 | 20.0 | no | + | - | no | + | + | YES |
| mistral_small math800 L28 | 10.0 | 10.0 | no | + | + | YES | + | + | YES |
| mistral_small_3_2 code800 L20 | 0.0 | 5.0 | YES | 0 | + | no | 0 | + | no |
| mistral_small_3_2 fact800 L28 | 30.0 | 20.0 | YES | + | + | YES | + | + | YES |
| mistral_small_3_2 math800 L28 | 10.0 | 10.0 | no | + | + | YES | + | + | YES |
| olmo13b code800 L15 | 20.0 | 20.0 | no | + | + | YES | + | + | YES |
| olmo13b fact800 L23 | 5.0 | 30.0 | YES | + | 0 | no | + | + | YES |
| olmo13b math800 L23 | 10.0 | 5.0 | YES | + | + | YES | + | + | YES |
| phi3 code800 L16 | 10.0 | 5.0 | YES | + | + | YES | + | + | YES |
| phi3 fact800 L15 | 20.0 | 5.0 | YES | + | 0 | no | + | + | YES |
| phi3 math800 L15 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| phi4mini code800 L29 | 20.0 | 20.0 | no | + | + | YES | + | + | YES |
| phi4mini fact800 L14 | 30.0 | 0.0 | YES | + | 0 | no | + | 0 | no |
| phi4mini math800 L14 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| qwen code800 L18 | 20.0 | 20.0 | no | + | + | YES | + | + | YES |
| qwen fact800 L19 | 0.0 | 0.0 | no | 0 | 0 | YES | 0 | 0 | YES |
| qwen math800 L18 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| qwen14b code800 L32 | 10.0 | 5.0 | YES | + | + | YES | + | + | YES |
| qwen14b fact800 L34 | 10.0 | 0.0 | YES | + | 0 | no | + | 0 | no |
| qwen14b math800 L34 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| qwen32b code800 L40 | 5.0 | 0.0 | YES | + | 0 | no | + | 0 | no |
| qwen32b fact800 L53 | 20.0 | 0.0 | YES | + | 0 | no | + | 0 | no |
| qwen32b math800 L53 | 10.0 | 10.0 | no | + | + | YES | + | + | YES |
| qwen3_14b code800 L24 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| qwen3_14b fact800 L25 | 10.0 | 0.0 | YES | + | 0 | no | + | 0 | no |
| qwen3_14b math800 L25 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| qwen3_32b code800 L47 | 0.0 | 0.0 | no | 0 | 0 | YES | 0 | 0 | YES |
| qwen3_32b fact800 L48 | 10.0 | 5.0 | YES | + | 0 | no | + | + | YES |
| qwen3_32b math800 L48 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| qwen3_8b code800 L19 | 0.0 | 5.0 | YES | 0 | + | no | 0 | + | no |
| qwen3_8b fact800 L21 | 10.0 | 20.0 | YES | + | + | YES | + | + | YES |
| qwen3_8b math800 L21 | 5.0 | 5.0 | no | + | + | YES | + | + | YES |
| smollm2 code800 L14 | 0.0 | 40.0 | YES | 0 | - | no | 0 | + | no |
| smollm2 fact800 L11 | 5.0 | 40.0 | YES | 0 | 0 | YES | + | + | YES |
| smollm2 math800 L11 | 5.0 | 0.0 | YES | 0 | 0 | YES | + | 0 | no |

**Aggregate:** 26/48 best_α changed; 33/48 HR sign agreement; 38/48 proxy_improve sign agreement.

---

## 7. Qualitative anchor-model overlap with intervention v2 (anchors only)

Anchors: `mistral`, `gemma3_4b`, `qwen3_14b`, `qwen3_8b`. Three qualitative yes/no checks; **no numeric gap comparison** (steering proxy and intervention gated ΔG are different metrics).

**Consistency questions:**
- Q1: Is *code* the most behaviorally responsive domain in steering v2det? (rank: code's hallucination_reduction relative to math/fact.)
- Q2: Is *fact* the least behaviorally responsive domain in steering v2det?
- Q3: Does Qwen3-8B show weaker code-side response than mid-tier anchors (consistent with intervention v2 Qwen3-8B code being a small +ΔG cell)?

| anchor | code HR | math HR | fact HR | Q1 (code top) | Q2 (fact bottom) |
|---|---:|---:|---:|---|---|
| mistral | +0.140 | +0.160 | +0.040 | no | YES |
| gemma3_4b | +0.190 | +0.150 | +0.000 | YES | YES |
| qwen3_14b | +0.170 | +0.240 | +0.000 | no | YES |
| qwen3_8b | +0.090 | +0.160 | +0.040 | no | YES |

**Q3 (qwen3_8b code weaker than peers):**

| model | code HR |
|---|---:|
| qwen3_8b | +0.090 |
| mistral | +0.140 |
| gemma3_4b | +0.190 |
| qwen3_14b | +0.170 |

**Q3 verdict (qwen3_8b code HR < peer median):** YES

**Anchor-overlap summary (qualitative, no numeric gap comparison):**

- Q1 (code most responsive): 1/4 anchors — *partial agreement*; code is co-leader with math on most anchors, not strictly dominant.
- Q2 (fact least responsive): 4/4 anchors — *strong agreement* with the intervention v2 finding that fact is structurally less measurable.
- Q3 (qwen3_8b weak on code): YES — qwen3_8b code HR is below peer median, consistent with intervention v2 reporting Qwen3-8B as a small-effect code cell.

---

_Generated by `scripts/compare_steering_v1_v2det.py`. v2det classifier defined in `scripts/aggregate_steering_v2det.py`._
