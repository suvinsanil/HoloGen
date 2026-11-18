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

- **Explicit $\alpha$ Selection Criterion & In-Sample Circularity Caveat for 3D Alpha Shapes:** Across the candidate parameter sweep $\alpha \in \{0.06, 0.09, 0.12, 0.18, 0.30\}$, `scripts/run_experiment_b.py` deterministically selects $\alpha^*$ by minimizing bidirectional Chamfer L2 distance (`chamfer_l2`) to the reference input point cloud:
  $$\alpha^* = \arg\min_{\alpha \in \{0.06, 0.09, 0.12, 0.18, 0.30\}} \text{Chamfer}_{L_2}(\mathcal{M}(\alpha), Q_{\text{ref}})$$
  Applied consistently in code (`if ev_a.chamfer_l2 < best_alpha_chamfer:`), this selects **$\alpha^* = 0.12$** (`Chamfer L2 = 0.03461`, **Input-Cloud Completeness = `76.11%`**, compared to $\alpha = 0.18$ with `Chamfer L2 = 0.03670`, **Input-Cloud Completeness = `79.42%`**).
  *Circularity Note:* Selecting $\alpha^*$ on `sample_02` by minimizing `Chamfer L2` and then reporting `Chamfer L2` on that same cloud is in-sample / circular; therefore, both `Chamfer L2` (`0.03461`) and `Input-Cloud Completeness` (`76.11%`)—along with the full 5-value parameter sweep—are reported together so the completeness trade-off against $\alpha = 0.18$ (`79.42%`) is transparent.

---

## 3. View-Frustum Coverage vs. 3D Scene Completeness

HoloGen reports **View-Frustum Coverage (`view_frustum_coverage_pct`)**: the percentage of 2D input camera pixels $(W \times H)$ subtended by reconstructed mesh triangles when rasterized back onto the camera image plane.

- **What it measures:** How much of the visible 2D photograph is represented in the exported textured 3D mesh after discontinuity cutting, outlier removal, and hole repair.
- **What it does NOT measure:** **It does NOT measure 3D volumetric scene completeness.** From a single viewpoint, the back sides of objects and regions occluded behind foreground surfaces are completely unobserved ($0\%$ coverage of hidden geometry).

---

## 4. Investigation of Heavy-Tailed Depth-Discontinuity Map Statistics (`P90 < Mean` on `sample_02`)

On `sample_02_aloe_indoor.jpg` ($640 \times 554 = 354,560$ pixels), the monocular **depth-discontinuity map** under bilateral refinement reported `Mean = 0.0438` and `P90 = 0.0431` (`P90 < Mean`), whereas under `refinement = "none"` it reported `Mean = 0.0444` and `P90 = 0.0446`.

### Measured Pixel Histogram (`sample_02_aloe_indoor.jpg`, $N = 354,560$ pixels)

| Depth-Discontinuity Map Bin | Count (`bilateral`) | Share (`bilateral`) | Count (`none`) | Share (`none`) | Physical Region in Scene |
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
3. **Reporting Resolution:** All reports include `P50`, `P90`, `P95`, `P99`, and the 7-bin `histogram_pct` breakdown so the heavy-tailed boundary structure is never obscured by a single percentile.

### 4.1 Empirical Spearman Correlation Validation Against True Depth Error ($n = 8$ RGB-D Scenes)

To test how well the spatial log-depth gradient **depth-discontinuity map** $D_{\text{disc}}(u, v) = \text{norm}(\|\nabla \log Z(u, v)\|_2)$ correlates with actual monocular depth errors, `scripts/validate_uncertainty_and_glb_uvs.py` computes the pixelwise **Spearman rank correlation effect size ($\rho$)** between $D_{\text{disc}}(u, v)$ and:
1. **True Absolute Depth Error:** $E_{\text{abs}}(u, v) = |\hat{Z}(u, v) - Z_{\text{gt}}(u, v)|$ (in meters, after Experiment A's least-squares disparity scale-shift alignment).
2. **True Relative Depth Error:** $E_{\text{rel}}(u, v) = |\hat{Z}(u, v) - Z_{\text{gt}}(u, v)| / Z_{\text{gt}}(u, v)$.

> [!NOTE]
> **Why $p$-values are omitted:** Nearby pixels in a dense $640 \times 480$ depth map ($N \approx 2.6 \times 10^5$) exhibit strong 2D spatial autocorrelation. Because standard Spearman asymptotic $p$-values assume independent and identically distributed (i.i.d.) observations, effective degrees of freedom are orders of magnitude smaller than $N$, rendering raw $p$-values statistically meaningless. Only effect sizes ($\rho$) are reported below.

| Scene ID | Dataset | Valid Pixels | Mean Abs Error (m) | Mean Discontinuity $D_{\text{disc}}$ | **Spearman $\rho$ ($D_{\text{disc}}$ vs Abs Error)** | **Spearman $\rho$ ($D_{\text{disc}}$ vs Rel Error)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `nyu_depth_v2_office` | `NYU_Depth_V2` | 273,147 | 0.5248 | 0.0777 | **+0.1368** | **+0.1520** |
| `sun_rgbd_bedroom` | `SUN_RGBD` | 242,679 | 0.2220 | 0.2201 | **+0.2882** | **+0.1061** |
| `tum_rgbd_desk` | `TUM_RGBD` | 235,699 | 0.0852 | 0.0530 | **+0.1209** | **+0.1410** |
| `redwood_livingroom_00` | `Redwood_Indoor_RGBD` | 262,682 | 0.2081 | 0.1414 | **-0.0278** | **+0.1092** |
| `redwood_livingroom_01` | `Redwood_Indoor_RGBD` | 263,291 | 0.1969 | 0.1282 | **-0.0124** | **+0.1242** |
| `redwood_livingroom_02` | `Redwood_Indoor_RGBD` | 263,643 | 0.1867 | 0.1039 | **+0.0260** | **+0.1382** |
| `redwood_livingroom_03` | `Redwood_Indoor_RGBD` | 264,106 | 0.1881 | 0.1200 | **+0.0700** | **+0.1824** |
| `redwood_livingroom_04` | `Redwood_Indoor_RGBD` | 264,564 | 0.2074 | 0.1564 | **+0.2644** | **+0.3107** |
| **Mean Across 8 Scenes** | — | 258,726 | 0.2274 | 0.1251 | **+0.1083** | **+0.1580** |

**Honest Scientific Interpretation:**
- The depth-discontinuity map exhibits a **weak-to-moderate positive rank correlation** with relative depth error across all 8 scenes ($\bar{\rho}_{\text{rel}} = +0.1580$, range $[+0.1061, +0.3107]$) and a **weak mean rank correlation** with absolute depth error ($\bar{\rho}_{\text{abs}} = +0.1083$, with two Redwood living-room frames showing slightly negative $\rho_{\text{abs}} \in [-0.0278, -0.0124]$).
- **Why?** A spatial gradient map peaks exclusively on **local 1D occlusion boundaries** (where flying-pixel errors occur), whereas the largest global absolute errors in monocular depth estimation occur on **smooth, low-gradient background walls and ceilings** (where global perspective tilt or far-range scale compression produces $0.2\text{--}0.5\text{ m}$ error despite near-zero spatial gradient). Consequently, HoloGen names and uses this layer strictly as a **depth-discontinuity map** (for flying-pixel masking and boundary inspection) rather than a global uncertainty or confidence measure.

---

## 5. Multi-View Datasets, Protocols & Consolidated Results

### 5.1 Multi-View Datasets & Licences
| Dataset | Scene / Views | Reference / Ground-Truth Type | License / Source |
| :--- | :--- | :--- | :--- |
| **SampleFountainRGBDImages** | 33 color frames ($640\times 480$) | `scene/key.log` trajectory & `scene/integrated.ply` (**RGB-D reference reconstruction**, *not* laser ground truth) | Open3D `SampleFountainRGBDImages` (MIT) |
| **ETH3D `pipes`** | 14 DSLR JPEGs ($6048\times 4032$, `THIN_PRISM_FISHEYE` GT) | `dslr_calibration_jpg/` poses & `dslr_scan_eval/scan1.ply` (**Faro laser scan ground truth**) | ETH3D Multi-View Benchmark (**CC-BY-SA**) |
| **`datasets/custom_scene_01/`** | Qualitative user scene | Not present in workspace (`not_present_in_workspace`) — **not run** | — |

### 5.2 Consolidated Benchmark Summary
1. **Experiment A — Monocular Depth ($n=8$ RGB-D scenes):** `Depth Anything V2 Small` (`none` refinement) achieves `Abs Rel = 0.0945`, `RMSE = 0.4139 m`, `delta < 1.25 = 0.8754`, `Edge F1 = 0.6727`, `mean_inference_ms = 359.3 ms`, `peak_vram_mb = 243.6 MB` (`bilateral` / `guided` reduced Edge F1 to `0.6544` / `0.5873` without improving Abs Rel). `dpt_hybrid_midas` achieves `Abs Rel = 0.1109`, `RMSE = 0.5181 m`, `delta < 1.25 = 0.9026`, `Edge F1 = 0.6162` at `511.8 ms` / `658.0 MB` VRAM.
2. **Phase 7b.5 — Two-View Epipolar Geometry (10 pairs × 3 seeds = 30 runs/method/scene):**
   - **ETH3D `pipes` (undistorted `PINHOLE`, GT poses, Med [IQR]):** `from_scratch_8pt_with_gn` achieves Rot `0.247° [0.655°]`, Trans `0.904° [1.645°]`, Reproj RMSE `0.321 [0.171] px` (vs. `no_gn` `2.314 px`, `pycolmap` `0.267° / 0.883° / 0.343 px`, `OpenCV 5-pt` `0.400° / 1.155° / 0.401 px`).
   - **Fountain (RGB-D ref., Med [IQR]):** All four methods plateau near `~1.0°` rotation error (`0.967°–1.022°`), reflecting the RGB-D trajectory's noise floor.
3. **Phase 7b / 7c.5 — N-View SfM & 80/20 Held-Out Depth Alignment:**
   - **Fountain (33 views):** `SIMPLE_RADIAL`, `33/33` registered, `ATE RMSE = 0.0157 m`, `80/20` held-out `Abs Rel = 0.0158` (`10` views).
   - **ETH3D `pipes` (14 views):** Auto-fallback `SIMPLE_RADIAL -> OPENCV`, `14/14` registered, `ATE RMSE = 0.0067–0.0074 m`, `80/20` held-out `Abs Rel = 0.0519`.
4. **Experiment D v2 (`phase-7e.3`) — Equalized 3D Geometry & Per-View Depth (NO ICP Headline, Equal-Density):**
   - **Fountain (`tau=0.05 m`, RGB-D ref., in-sample tuned):** `N=1` oracle scale (`0.223` F1) < `N=1` oracle scale+shift (`0.742`) < `N=5` (`0.824`) < `N=10` (`0.831` AABB / `0.853` Frustum).
   - **ETH3D `pipes` (`tau=0.10 m`, Faro GT, out-of-sample):** `N=1` oracle scale (`0.099` AABB F1 / `0.226` Frustum F1), `N=1` oracle scale+shift (`0.257` AABB F1 / `0.802` Frustum F1), Multi-view `N=14` (`0.276` AABB F1 / `0.657` Frustum F1; `0.699` Frustum F1 with `default 0.10/0.18`; `0.473` AABB F1 with rigid ICP).
   - **ETH3D Per-View GT Depth (`14/14` views, Med [IQR]):** Single-view median-scale `Abs Rel = 0.1688 [0.1450]` | **Multi-view SfM-aligned `Abs Rel = 0.0963 [0.0370]`** (0 GT depth params; not better than oracle) | Single-view oracle scale+shift `Abs Rel = 0.0736 [0.0366]`. *(Note: In `phase-7e.2`, these three medians were `0.1974 / 0.0989 / 0.0914`; the shift in `phase-7e.3` comes from re-running multi-threaded `pycolmap` SfM + undistortion, which produces slightly different undistorted `PINHOLE` focal lengths `fx, fy` across non-deterministic runs.)*

---

## 6. System Metrics (Real Measured Runs)

| Pipeline Mode & Scene | Hardware | Stage Timings | Output Counts | Peak RSS / VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **Single-View** (`sample_01_building`, $640\times 442$, resized from $868\times 600$) | i5-1135G7, 7.7 GB RAM, MX330 2 GB VRAM | Preprocess `55 ms`, Depth `741 ms`, Refinement `10 ms`, Uncertainty `12 ms`, Backproj `23 ms`, Clean `1,515 ms`, Mesh `15,673 ms`, Texture `329 ms`, Export `3,299 ms`, **Total `22.97 s`** | `256,952` pts; `545,572` full faces $\to$ `120,000` viewer faces | **Peak VRAM: `243.6 MB`** (`Depth Anything V2 Small` from Exp. A) |
| **Multi-View** (`Fountain`, 10 views, `1600 px` SfM / `640 px` depth) | i5-1135G7, 7.7 GB RAM, CPU SfM + CUDA depth | SfM + Undistort `12.28 s`, Depth+Align+TSDF+Mesh `13.47 s`, **Total `25.74 s`** | `10/10` registered; `179,703` pts, `330,242` faces | **Whole-pipeline Peak RSS: `876.2 MB`** (`+409.0 MB` over baseline) |
| **Multi-View** (`ETH3D pipes`, 14 views, `6048×4032` -> `1600 px`) | i5-1135G7, 7.7 GB RAM, CPU SfM + CUDA depth | SfM (`OPENCV` fallback) + Undistort `51.30 s`, Depth+Align+TSDF+Mesh `11.12 s`, **Total `62.42 s`** | `14/14` registered; `302,585` pts, `523,060` faces | **Whole-pipeline Peak RSS: `1,961.3 MB`** (`+1,372.6 MB` over baseline) |

---

## 7. Known Fundamental Limitations

1. **Single-View Scale Ambiguity:** A single 2D image cannot uniquely determine metric scale; relative models (`Depth Anything V2 Small`, `DPT-Hybrid`) recover affine/ordinal structure only.
2. **Hallucinated Occluded Surfaces:** Single-view back-projection observes only the front 2.5D visible shell; watertight meshers (Screened Poisson) hallucinate smooth bubbles over unobserved backsides.
3. **Thin, Reflective & Transparent Structures:** Mirrors, glass, specular highlights, and fine foliage/wires violate Lambertian assumptions and cause depth bleeding or flying pixels.
4. **Depth-Discontinuity Map Is Not Calibrated Uncertainty:** The spatial log-depth gradient map detects 1D occlusion jumps (`mean Spearman rho = +0.1083` vs. absolute error, `+0.1580` vs. relative error across `n=8` scenes) and does not predict planar bias on smooth walls.
5. **Small Depth Benchmark ($n=8$):** Experiment A uses 8 indoor RGB-D scenes (`NYU V2`, `SUN RGB-D`, `TUM RGB-D`, 5× `Redwood`) due to local hardware constraints and is diagnostic, not statistically conclusive.
6. **Fountain Reference Is Not Ground Truth:** `SampleFountainRGBDImages` (`key.log`, `integrated.ply`) is an RGB-D reconstruction; `~1°` two-view/SfM rotation error on Fountain reflects reference trajectory noise.
7. **Multi-View Overlap & Registration Dependency:** Sparse SfM requires sufficient visual overlap and texture; disconnected or low-overlap view subsets (e.g., wide-baseline jumps in `ETH3D`) fail to register without dense visual overlap.
8. **Monocular-Depth Fusion vs. True MVS:** HoloGen aligns pretrained monocular depth to sparse SfM points rather than solving dense PatchMatch stereo across epipolar lines; fine geometric details not captured by the monocular prior (or smoothed by TSDF voxelization) are coarser than full MVS.
9. **ETH3D Alignment Sensitivity:** Because `ETH3D pipes` cameras are clustered in one half of the room (`collinearity SV ratio = 0.54`), centre-only `Sim(3)` Umeyama leaves residual global alignment drift (rigid ICP raises `N=14` AABB F1 from `0.276` to `0.473`).
10. **Hardware Limits (7.7 GB RAM, 2 GB VRAM MX330):** Larger foundation models (`ZoeDepth`, `Metric3D`, `Depth Anything V2 Large/Giant`) and full-resolution PatchMatch MVS cannot run within 2 GB VRAM / 7.7 GB RAM, motivating the 640 px depth cap, sequential view-by-view TSDF streaming, and concurrency-1 job queue.
