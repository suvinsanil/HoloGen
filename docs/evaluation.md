# HoloGen Evaluation Metrics & Known Limitations

## 1. Depth Evaluation Metrics (Phase 2 & Phase 8)

Given predicted depth $Z_i$ (aligned via closed-form least-squares scale-shift for relative models, or raw meters for metric models) and ground-truth depth $Z_i^*$ over $N$ valid pixels:

- **Abs Rel:** $\frac{1}{N} \sum_{i=1}^N \frac{|Z_i - Z_i^*|}{Z_i^*}$
- **Sq Rel:** $\frac{1}{N} \sum_{i=1}^N \frac{(Z_i - Z_i^*)^2}{Z_i^*}$
- **RMSE:** $\sqrt{\frac{1}{N} \sum_{i=1}^N (Z_i - Z_i^*)^2}$
- **RMSE log:** $\sqrt{\frac{1}{N} \sum_{i=1}^N (\log Z_i - \log Z_i^*)^2}$
- **Threshold Accuracy ($\delta_k$):** Percentage of pixels satisfying $\max\left(\frac{Z_i}{Z_i^*}, \frac{Z_i^*}{Z_i}\right) < 1.25^k$ for $k \in \{1, 2, 3\}$.

## 2. Known Fundamental Limitations

1. **Single-View Scale Ambiguity:** A single 2D projection cannot uniquely determine absolute metric scale without strong scene priors. Relative models (`Depth Anything V2 Small`, `DPT-Hybrid`) recover ordinal/affine structure only.
2. **Occluded & Unseen Surfaces:** Single-view back-projection only observes the front visible surface (2.5D frustum shell). Regions behind foreground objects are empty in the point cloud; watertight meshers like Poisson reconstruction will hallucinate smooth bubble surfaces across occluded voids.
3. **Reflective, Transparent, and Thin Structures:** Mirrors, glass, and fine foliage violate Lambertian assumptions and produce depth bleeding or disconnected point clusters.
4. **Uncalibrated Uncertainty Proxy:** The monocular uncertainty map is a heuristic proxy (depth gradient + flip consistency), not a calibrated statistical confidence interval.
