# HoloGen Evaluation Metrics, Dataset Provenance & Fundamental Limitations

## 1. Depth Evaluation Metrics (Experiment A)

Given predicted depth $Z_i$ (aligned via closed-form least-squares scale-shift for relative models, or raw meters for metric models) and ground-truth depth $Z_i^*$ over $N$ valid pixels:

- **Abs Rel:** $\frac{1}{N} \sum_{i=1}^N \frac{|Z_i - Z_i^*|}{Z_i^*}$
- **Sq Rel:** $\frac{1}{N} \sum_{i=1}^N \frac{(Z_i - Z_i^*)^2}{Z_i^*}$
- **RMSE:** $\sqrt{\frac{1}{N} \sum_{i=1}^N (Z_i - Z_i^*)^2}$
- **RMSE log:** $\sqrt{\frac{1}{N} \sum_{i=1}^N (\log Z_i - \log Z_i^*)^2}$
- **Threshold Accuracy ($\delta_k$):** Percentage of pixels satisfying $\max\left(\frac{Z_i}{Z_i^*}, \frac{Z_i^*}{Z_i}\right) < 1.25^k$ for $k \in \{1, 2, 3\}$.
- **Edge F1 & Gradient Correlation:** 1-pixel dilated boundary overlap F1 score and Pearson correlation between spatial Sobel gradient magnitudes $\|\nabla \log Z\|_2$ and $\|\nabla \log Z^*\|_2$.

### 1.1 Experiment A Benchmark Subset ($n = 8$ Scenes) & Statistical Caveat

> [!IMPORTANT]
> **Sample Size Caveat ($n = 8$):** Experiment A evaluates models across **$n = 8$ calibrated indoor RGB-D frames** to fit local workstation execution constraints. While sufficient to verify metric vs. relative scale behavior and runtime/VRAM profiles, **$n = 8$ is a small diagnostic subset and is NOT statistically conclusive** for ranking general-domain foundation models across diverse indoor/outdoor distributions.

| # | Scene ID | Dataset Origin | Sensor / Depth Encoding | Source Distribution | License |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `nyu_depth_v2_office` | **NYU Depth V2** (Silberman et al., ECCV 2012) | Microsoft Kinect v1 structured light ($351.3 / (1092.5 - d)$ m) | `open3d.data.SampleNYURGBDImage` | MIT (Open3D data package) / NYU Academic Dataset |
| 2 | `sun_rgbd_bedroom` | **SUN RGB-D** (Song et al., CVPR 2015) | Kinect / Xtion (`((d >> 3) \| (d << 13)) / 1000` m) | `open3d.data.SampleSUNRGBDImage` | MIT (Open3D data package) / Princeton SUN RGB-D |
| 3 | `tum_rgbd_desk` | **TUM RGB-D Benchmark** (Sturm et al., IROS 2012) | Kinect v1 (`d / 5000.0` m) | `open3d.data.SampleTUMRGBDImage` | **CC-BY-4.0** (Technical University of Munich CVPR Group) |
| 4 | `redwood_livingroom_00` | **Redwood Indoor RGB-D** (Choi et al., CVPR 2015) | Asus Xtion Live (`d / 1000.0` m) | `open3d.data.SampleRedwoodRGBDImages` (`00000`) | **Public Domain / CC0** (Redwood 3D Scans) |
| 5 | `redwood_livingroom_01` | **Redwood Indoor RGB-D** (Choi et al., CVPR 2015) | Asus Xtion Live (`d / 1000.0` m) | `open3d.data.SampleRedwoodRGBDImages` (`00001`) | **Public Domain / CC0** (Redwood 3D Scans) |
| 6 | `redwood_livingroom_02` | **Redwood Indoor RGB-D** (Choi et al., CVPR 2015) | Asus Xtion Live (`d / 1000.0` m) | `open3d.data.SampleRedwoodRGBDImages` (`00002`) | **Public Domain / CC0** (Redwood 3D Scans) |
| 7 | `redwood_livingroom_03` | **Redwood Indoor RGB-D** (Choi et al., CVPR 2015) | Asus Xtion Live (`d / 1000.0` m) | `open3d.data.SampleRedwoodRGBDImages` (`00003`) | **Public Domain / CC0** (Redwood 3D Scans) |
| 8 | `redwood_livingroom_04` | **Redwood Indoor RGB-D** (Choi et al., CVPR 2015) | Asus Xtion Live (`d / 1000.0` m) | `open3d.data.SampleRedwoodRGBDImages` (`00004`) | **Public Domain / CC0** (Redwood 3D Scans) |

---

## 2. Surface Reconstruction Evaluation Caveat & Alpha Selection Criterion (Experiment B)

> [!IMPORTANT]
> **Input-Cloud Reference vs. Ground Truth:** In Experiment B (`scripts/run_experiment_b.py`), the reference point set $Q$ is the **cleaned input back-projected point cloud** (derived from monocular depth), **NOT an independent 3D laser-scanned ground-truth mesh**. Consequently, `Completeness (@tau=0.04)`, `Surface Noise RMS`, `Chamfer L2`, and `Hausdorff P95` in Experiment B measure **surface reconstruction fidelity to the input point cloud** (and detect algorithmic artifacts such as Poisson's hallucinated back-side bubble or Alpha Shape's over-carving), rather than true 3D metric accuracy against the physical scene.

- **Explicit $\alpha$ Selection Criterion for 3D Alpha Shapes:** Across the candidate parameter sweep $\alpha \in \{0.06, 0.09, 0.12, 0.18, 0.30\}$, `scripts/run_experiment_b.py` deterministically selects the optimal $\alpha^*$ by **minimizing bidirectional Chamfer L2 distance (`chamfer_l2`) to the reference input point cloud**:
  $$\alpha^* = \arg\min_{\alpha \in \{0.06, 0.09, 0.12, 0.18, 0.30\}} \text{Chamfer}_{L_2}(\mathcal{M}(\alpha), Q_{\text{ref}})$$
  Applied consistently in code (`if ev_a.chamfer_l2 < best_alpha_chamfer:`), this criterion uniquely selects **$\alpha^* = 0.12$** (`chamfer_l2 = 0.03461`, compared to `0.03670` at $\alpha = 0.18$ and `0.06120` at $\alpha = 0.09$) for inclusion as `6_alpha_shape_best_a0.12` in Table B.1.

---

## 3. View-Frustum Coverage vs. 3D Scene Completeness

HoloGen reports **View-Frustum Coverage (`view_frustum_coverage_pct`)**: the percentage of 2D input camera pixels $(W \times H)$ subtended by reconstructed mesh triangles when rasterized back onto the camera image plane.

- **What it measures:** How much of the visible 2D photograph is represented in the exported textured 3D mesh after discontinuity cutting, outlier removal, and hole repair.
- **What it does NOT measure:** **It does NOT measure 3D volumetric scene completeness.** From a single viewpoint, the back sides of objects and regions occluded behind foreground surfaces are completely unobserved ($0\%$ coverage of hidden geometry).

---

## 4. Investigation of Heavy-Tailed Uncertainty Proxy Statistics (`P90 < Mean` on `sample_02`)

On `sample_02_aloe_indoor.jpg` ($640 \times 554 = 354,560$ pixels), the monocular gradient uncertainty proxy under bilateral refinement reported `Mean = 0.0438` and `P90 = 0.0431` (`P90 < Mean`), whereas under `refinement = "none"` it reported `Mean = 0.0444` and `P90 = 0.0446`.

### Measured Pixel Histogram (`sample_02_aloe_indoor.jpg`, $N = 354,560$ pixels)

| Uncertainty Proxy Bin | Count (`bilateral`) | Share (`bilateral`) | Count (`none`) | Share (`none`) | Physical Region in Scene |
| :--- | :---: | :---: | :---: | :---: | :--- |
| $[0.00, 0.01)$ | 253,150 | **71.40%** | 248,984 | **70.22%** | Smooth leaf interiors & flat background wall |
| $[0.01, 0.02)$ | 38,496 | **10.86%** | 41,010 | **11.57%** | Gentle leaf curvature |
| $[0.02, 0.05)$ | 31,806 | **8.97%** | 32,415 | **9.14%** | Moderate perspective slopes (contains $P_{90}$) |
| $[0.05, 0.10)$ | 10,940 | 3.09% | 11,448 | 3.23% | Leaf tips & pot rim |
| $[0.10, 0.25)$ | 5,340 | 1.51% | 5,734 | 1.62% | Near-boundary transition zone (contains $P_{95} = 0.1528$) |
| $[0.25, 0.50)$ | 3,416 | 0.96% | 3,576 | 1.01% | Moderate depth jumps |
| $[0.50, 0.75)$ | 2,254 | 0.64% | 2,242 | 0.63% | Steep occlusion cliffs |
| $[0.75, 1.00]$ | 9,158 | **2.58%** | 9,151 | **2.58%** | Sharp 1D leaf occlusion silhouettes ($P_{98} = P_{99} = 1.0000$) |

**Mathematical Explanation:**
1. **1D Contours Embedded in a 2D Image:** Depth discontinuities occur along 1D object silhouettes (the perimeter of the aloe leaves), which occupy only $\sim 5.7\%$ of the 2D image area, while **91.23%** of pixels lie on smooth 2D surface interiors with values $< 0.05$.
2. **Why $P_{90} < \text{Mean}$ Occurs:** Because $>91\%$ of pixels lie below $0.05$, the 90th percentile ($P_{90} = 0.04314$) still sits inside the smooth interior mode. Meanwhile, the top **2.58%** of pixels on sharp occlusion cliffs are saturated at $1.0$ (and $P_{95} = 0.1528$), contributing $+0.0258$ to the first moment and pulling the arithmetic mean up to $0.04379 > P_{90}$ (with standard deviation $\sigma = 0.1645 \gg \text{Mean}$).
3. **Reporting Resolution:** All reports (`UncertaintyProxyResult.to_metadata_dict()`) now include `P50`, `P90`, `P95`, `P99`, and the 7-bin `histogram_pct` breakdown so the heavy-tailed boundary structure is never obscured by a single percentile.

### 4.1 Empirical Spearman Correlation Validation Against True Depth Error ($n = 8$ RGB-D Scenes)

To test how well the spatial gradient uncertainty proxy $U(u, v) = \text{norm}(\|\nabla \log Z(u, v)\|_2)$ correlates with actual monocular depth errors, `scripts/validate_uncertainty_and_glb_uvs.py` computes the pixelwise **Spearman rank correlation ($\rho$)** between $U(u, v)$ and:
1. **True Absolute Depth Error:** $E_{\text{abs}}(u, v) = |\hat{Z}(u, v) - Z_{\text{gt}}(u, v)|$ (in meters, after Experiment A's least-squares disparity scale-shift alignment).
2. **True Relative Depth Error:** $E_{\text{rel}}(u, v) = |\hat{Z}(u, v) - Z_{\text{gt}}(u, v)| / Z_{\text{gt}}(u, v)$.

| Scene ID | Dataset | Valid Pixels | Mean Abs Error (m) | Mean Proxy $U$ | **Spearman $\rho$ ($U$ vs Abs Error)** | $p$-value (Abs) | **Spearman $\rho$ ($U$ vs Rel Error)** | $p$-value (Rel) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nyu_depth_v2_office` | `NYU_Depth_V2` | 273,147 | 0.5248 | 0.0777 | **+0.1368** | $< 10^{-300}$ | **+0.1520** | $< 10^{-300}$ |
| `sun_rgbd_bedroom` | `SUN_RGBD` | 242,679 | 0.2220 | 0.2201 | **+0.2882** | $< 10^{-300}$ | **+0.1061** | $< 10^{-300}$ |
| `tum_rgbd_desk` | `TUM_RGBD` | 235,699 | 0.0852 | 0.0530 | **+0.1209** | $< 10^{-300}$ | **+0.1410** | $< 10^{-300}$ |
| `redwood_livingroom_00` | `Redwood_Indoor_RGBD` | 262,682 | 0.2081 | 0.1414 | **-0.0278** | $3.32 \times 10^{-46}$ | **+0.1092** | $< 10^{-300}$ |
| `redwood_livingroom_01` | `Redwood_Indoor_RGBD` | 263,291 | 0.1969 | 0.1282 | **-0.0124** | $2.10 \times 10^{-10}$ | **+0.1242** | $< 10^{-300}$ |
| `redwood_livingroom_02` | `Redwood_Indoor_RGBD` | 263,643 | 0.1867 | 0.1039 | **+0.0260** | $1.14 \times 10^{-40}$ | **+0.1382** | $< 10^{-300}$ |
| `redwood_livingroom_03` | `Redwood_Indoor_RGBD` | 264,106 | 0.1881 | 0.1200 | **+0.0700** | $6.38 \times 10^{-284}$ | **+0.1824** | $< 10^{-300}$ |
| `redwood_livingroom_04` | `Redwood_Indoor_RGBD` | 264,564 | 0.2074 | 0.1564 | **+0.2644** | $< 10^{-300}$ | **+0.3107** | $< 10^{-300}$ |
| **Mean Across 8 Scenes** | — | 258,726 | 0.2274 | 0.1251 | **+0.1083** | — | **+0.1580** | — |

**Honest Scientific Interpretation:**
- The gradient uncertainty proxy exhibits a **weak-to-moderate positive rank correlation** with relative depth error across all 8 scenes ($\bar{\rho}_{\text{rel}} = +0.1580$, range $[+0.1061, +0.3107]$) and a **weak mean rank correlation** with absolute depth error ($\bar{\rho}_{\text{abs}} = +0.1083$, with two Redwood living-room frames showing slightly negative $\rho_{\text{abs}} \in [-0.0278, -0.0124]$).
- **Why?** A spatial gradient proxy peaks exclusively on **local 1D occlusion boundaries** (where flying-pixel errors occur), whereas the largest global absolute errors in monocular depth estimation occur on **smooth, low-gradient background walls and ceilings** (where global perspective tilt or far-range scale compression produces $0.2\text{--}0.5\text{ m}$ error despite near-zero spatial gradient). This empirically validates why the map is strictly labeled an **uncalibrated discontinuity uncertainty proxy** rather than a calibrated global error predictor.

---

## 5. Known Fundamental Limitations

1. **Single-View Scale Ambiguity:** A single 2D projection cannot uniquely determine absolute metric scale without strong scene priors. Relative models (`Depth Anything V2 Small`, `DPT-Hybrid`) recover ordinal/affine structure only.
2. **Occluded & Unseen Surfaces:** Single-view back-projection only observes the front visible surface (2.5D frustum shell). Regions behind foreground objects are empty in the point cloud; watertight meshers like Poisson reconstruction will hallucinate smooth bubble surfaces across occluded voids.
3. **Reflective, Transparent, and Thin Structures:** Mirrors, glass, and fine foliage violate Lambertian assumptions and produce depth bleeding or disconnected point clusters.
4. **Uncalibrated Uncertainty Proxy:** The monocular uncertainty map is a heuristic proxy (depth gradient + flip consistency), not a calibrated statistical confidence interval.
