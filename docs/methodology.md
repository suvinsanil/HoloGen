# Mathematical & Algorithmic Methodology

## 1. Camera Intrinsics & Pinhole Model

For an image of width $W$ and height $H$ (after EXIF orientation correction and aspect-preserving resize), HoloGen constructs the $3 \times 3$ pinhole intrinsic matrix:

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad
K^{-1} = \begin{bmatrix} 1/f_x & 0 & -c_x/f_x \\ 0 & 1/f_y & -c_y/f_y \\ 0 & 0 & 1 \end{bmatrix}
$$

- **Principal Point:** Defaults to the geometric center $c_x = W / 2$, $c_y = H / 2$ unless calibrated parameters are provided.
- **Focal Length Priority:**
  1. **Explicit Override:** User-supplied $f_x = f_y = f_{\text{px}}$.
  2. **EXIF 35mm Equivalent (`FocalLengthIn35mmFilm`):**
     $$f_x = f_y = \frac{f_{35\text{mm}}}{36.0} \times \max(W, H)$$
  3. **EXIF Physical Focal Length (`FocalLength`):**
     $$f_x = f_y = \frac{f_{\text{mm}}}{w_{\text{sensor\_mm}}} \times \max(W, H)$$
  4. **Horizontal Field-of-View Fallback (Default $\text{FOV}_x = 60^\circ$):**
     $$f_x = f_y = \frac{W}{2 \tan(\text{FOV}_x \cdot \pi / 360)}$$

---

## 2. Monocular Depth Scale & Affine-Invariant Disparity Conversion

Monocular depth networks such as **Depth Anything V2** and **MiDaS / DPT** are trained with scale- and shift-invariant losses in inverse-depth (disparity) space:

$$
d(u, v) \approx \alpha \cdot \frac{1}{Z(u, v)} + \beta
$$

Because absolute scale $\alpha$ and shift $\beta$ are mathematically unobservable from a single image without metric priors, HoloGen:
1. Explicitly labels single-view relative reconstructions with `scale_semantics = "relative_unknown_scale"`.
2. Maps raw disparity $d(u,v)$ into a positive relative depth range $[z_{\text{near}}, z_{\text{far}}]$ (default $[1.0, 10.0]$) linearly in inverse-depth space so that 3D planes remain planar under perspective back-projection:
   $$
   \hat{d}(u, v) = \text{clip}\left(\frac{d(u, v) - P_1(d)}{P_{99}(d) - P_1(d)}, 0, 1\right)
   $$
   $$
   Z(u, v) = \left(\frac{1}{z_{\text{far}}} + \hat{d}(u, v)\left(\frac{1}{z_{\text{near}}} - \frac{1}{z_{\text{far}}}\right)\right)^{-1}
   $$
3. When evaluating relative models against metric ground truth (e.g., NYU Depth V2), solves the closed-form least-squares scale-shift alignment over valid pixels $\Omega$:
   $$
   (\hat{s}, \hat{t}) = \arg\min_{s, t} \sum_{(u,v) \in \Omega} \left( s \cdot \hat{Z}(u, v) + t - Z_{\text{gt}}(u, v) \right)^2
   $$

---

## 3. 3D Back-Projection

Each valid pixel $(u, v)$ with depth $Z = D(u, v) > 0$ and RGB color $\mathbf{c}(u, v) \in [0, 1]^3$ is back-projected into the right-handed camera coordinate frame ($+X$ right, $+Y$ down, $+Z$ forward along the optical axis):

$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = D(u, v)
$$

---

## 4. Monocular Depth-Discontinuity Map

Monocular depth networks do not output calibrated epistemic or aleatoric probabilities, and empirical Spearman validation on our 8-scene RGB-D suite shows that spatial depth gradients correlate only weakly with global absolute error ($\bar{\rho}_{\text{abs}} = +0.1083$, $\bar{\rho}_{\text{rel}} = +0.1580$). Accordingly, HoloGen defines and reports this layer strictly as a **Depth-Discontinuity Map** $D_{\text{disc}}(u, v) \in [0, 1]$:
1. **Depth-Gradient Discontinuity Component:** Normalized Sobel gradient magnitude of log-depth $\|\nabla \log Z(u,v)\|_2$, which peaks along 1D occlusion boundaries where flying pixels occur.
2. **Flip-Consistency Component (optional):** Pixelwise discrepancy between original inference $Z(I)$ and horizontally flipped inference $\text{flip}(Z(\text{flip}(I)))$:
   $$D_{\text{flip}}(u, v) = \frac{|Z(u, v) - Z_{\text{flip}}(u, v)|}{\max(Z(u, v), \epsilon)}$$
3. **Inter-Model Disagreement Component (when multiple models are run):** Pixelwise deviation after least-squares scale-shift alignment between models.


---

## 5. Point Cloud Cleaning, Normal Orientation & Filtering Parameters (Phase 3)

| Operation | Parameter | Default | Mathematical Role |
| :--- | :--- | :---: | :--- |
| **Back-Projection** | `min_depth`, `max_depth`, `stride` | $10^{-4}, 10^4, 1$ | Bounds valid depth frustum $Z \in [z_{\min}, z_{\max}]$ and optional pixel decimation stride. |
| **Normal Estimation** | `max_nn`, `radius` | $30$, $4.5 \bar{d}_{\text{nn}}$ | Local covariance PCA over hybrid KD-tree sphere; normals oriented toward camera origin $(0,0,0)$ so $\mathbf{n}_i \cdot (-\mathbf{p}_i) > 0$. |
| **Statistical Outlier Removal (SOR)** | `nb_neighbors`, `std_ratio` | $20$, $2.0$ | Prunes points whose mean $k$-NN distance $\bar{d}_i > \mu_d + \text{std\_ratio} \cdot \sigma_d$. |
| **Radius Outlier Removal (ROR)** | `nb_points`, `radius` | $12$, $3.5 \bar{d}_{\text{nn}}$ | Prunes isolated points with fewer than `nb_points` inside sphere of radius $r$. |
| **Voxel Downsampling** | `voxel_size` | $0.035$ | Aggregates points and RGB colors within cubic 3D grid cells of edge length `voxel_size`. |
| **Normal Grazing-Angle Filter** | `max_grazing_angle_deg` | $80.0^\circ$ | Removes monocular depth-cliff veil points where $\arccos(\mathbf{n}_i \cdot \hat{\mathbf{v}}_i) > \theta_{\max}$. |
| **Tangent-Plane Smoothing** | `alpha`, `k_neighbors` | $0.35$, $9$ | Damps high-frequency depth jitter along surface normal direction $\mathbf{p}_i \leftarrow \mathbf{p}_i + \alpha ((\bar{\mathbf{p}}_i - \mathbf{p}_i)\cdot \mathbf{n}_i)\mathbf{n}_i$. |


---

## 6. Surface Mesh Reconstruction & Cleanup (Phase 4)

### 6.1 Depth-Grid Triangulation with Discontinuity Edge Cutting (Baseline)
Given the 2D depth grid $D(u, v)$ of shape $(H, W)$, each $2 \times 2$ pixel quad $(u, v), (u+1, v), (u, v+1), (u+1, v+1)$ is split into two candidate counter-clockwise triangles:
$$T_1 = (p_{00}, p_{01}, p_{10}), \quad T_2 = (p_{10}, p_{01}, p_{11})$$
To prevent stretched "curtain" triangles across occlusion cliffs, any triangle containing an edge $(a, b)$ whose relative depth jump exceeds $\tau_{\text{jump}}$ (default $8\%$) is cut:
$$\frac{|Z_a - Z_b|}{\min(Z_a, Z_b)} > \tau_{\text{jump}}$$

### 6.2 Open3D Comparison Surface Reconstruction Methods
1. **Screened Poisson Reconstruction (`octree_depth=8`):** Solves a global indicator function $\nabla \cdot \nabla \chi = \nabla \cdot \mathbf{V}$ over an adaptive octree. Because Poisson enforces a closed watertight surface, applying it to a single-view 2.5D frustum cloud invents a large hallucinated back-side bubble unless vertices with low sampling density $w_i < Q_q(w)$ are trimmed (`density_quantile_trim = 0.08`).
2. **Ball Pivoting Algorithm (BPA):** Rolls virtual balls of radii $r \in \{1.5, 2.5, 4.0\} \bar{d}_{\text{nn}}$ across the point cloud; preserves exact input points with zero hallucination (`surface_noise_rms = 0.0227`), at the cost of leaving small unbridged interior holes where sampling density fluctuates.
3. **3D Alpha Shapes:** Extracts the boundary of the $\alpha$-complex of the 3D Delaunay tetrahedrization (with $2 \times 10^{-3} \bar{d}_{\text{nn}}$ general-position perturbation to prevent Qhull coplanar lattice degeneracy).


---

## 7. Projective UV Texturing, Frustum Coverage & glTF Export (Phase 5)

### 7.1 Projective Camera UV Mapping
Every 3D vertex $\mathbf{v}_i = (X_i, Y_i, Z_i)$ of the reconstructed mesh (regardless of whether it was smoothed, hole-repaired, or generated via Poisson/BPA) is projected onto the camera image plane via intrinsics $K$:
$$
u_i = f_x \frac{X_i}{Z_i} + c_x, \quad v_i = f_y \frac{Y_i}{Z_i} + c_y
$$
and converted to normalized glTF/OpenGL texture coordinates $(s_i, t_i) \in [0, 1]^2$:
$$
s_i = \text{clip}\left(\frac{u_i}{W - 1}, 0, 1\right), \quad t_i = \text{clip}\left(1.0 - \frac{v_i}{H - 1}, 0, 1\right)
$$

### 7.2 View-Frustum Coverage Definition
HoloGen computes and reports two distinct, reproducible coverage metrics:
1. **View-Frustum Coverage (`view_frustum_coverage_pct`):** All reconstructed mesh triangles are rasterized in 2D image coordinates $(u, v)$ onto an $H \times W$ binary occupancy canvas $\mathcal{M}$. The View-Frustum Coverage is defined as:
   $$\text{Coverage}_{\text{view-frustum}} = 100 \times \frac{1}{W \cdot H} \sum_{v=0}^{H-1} \sum_{u=0}^{W-1} \mathbb{I}[\mathcal{M}(v, u) > 0]$$
   *Important Note:* View-Frustum Coverage measures 2D camera-plane projection occupancy only; it **does not** measure 3D volumetric scene completeness (occluded and back-facing surfaces are unobserved from a single viewpoint).
2. **Cleaned Point Retention (`cleaned_point_retention_pct`):** The percentage of raw back-projected points surviving Statistical Outlier Removal and Normal Grazing-Angle Filtering.

### 7.3 Coordinate Frame Convention for glTF / GLB Export
OpenCV pinhole back-projection uses $(+X \text{ right}, +Y \text{ down}, +Z \text{ forward})$, whereas glTF 2.0 and Three.js use a right-handed Y-up frame $(+X \text{ right}, +Y \text{ up}, -Z \text{ forward})$. On export to `GLB`, `OBJ`, and `STL`, HoloGen applies the proper rotation $R_X(180^\circ) = \text{diag}(1, -1, -1)$:
$$
(X_{\text{gltf}}, Y_{\text{gltf}}, Z_{\text{gltf}}) = (X_{\text{cam}}, -Y_{\text{cam}}, -Z_{\text{cam}})
$$
Since $\det(R_X(180^\circ)) = +1$, triangle winding order and outward normal orientation are preserved without inside-out inversion.

---

## 9. Multi-View Epipolar Geometry & Triangulation (Phase 7a)

### 9.1 Honest Scope: From-Scratch vs. Library Primitives
To demonstrate first-principles mastery of two-view epipolar geometry while keeping feature detection practical, Phase 7a separates library-based 2D feature detection from pure-NumPy geometric solvers:
- **Library-backed (`backend/app/vision/features.py`):** 2D keypoint detection and descriptor computation (`cv2.SIFT_create`, `cv2.ORB_create`) and k-NN descriptor distance lookup (`cv2.BFMatcher`), followed by our explicit Lowe ratio test ($\|d_1\| < 0.75 \|d_2\|$) and mutual cross-check filter.
- **Implemented from scratch in pure NumPy (`backend/app/geometry/two_view.py`, zero OpenCV geometry calls):**
  1. **Hartley Isotropic 2D Normalization (`normalize_points_2d`):** Translates 2D points to zero centroid $\bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i$ and scales isotropically by $s = \sqrt{2} / \left(\frac{1}{N}\sum_i \|\mathbf{p}_i - \bar{\mathbf{p}}\|_2\right)$ via similarity matrix $T \in \mathbb{R}^{3\times 3}$.
  2. **Normalized 8-Point Fundamental Matrix (`eight_point_fundamental_matrix`):** Forms the $N \times 9$ Kronecker design matrix $A$ for $\tilde{\mathbf{x}}_2^\top \tilde{F} \tilde{\mathbf{x}}_1 = 0$, extracts the right singular vector $\mathbf{v}_9$ via SVD, enforces the internal rank-2 constraint $\det(\tilde{F}) = 0$ by setting $\sigma_3 = 0$ in the SVD of $\tilde{F}$, and denormalizes $F = T_2^\top \tilde{F}_{\text{rank2}} T_1$.
  3. **First-Order Sampson Epipolar Distance (`compute_sampson_distance`):**
     $$d_{\text{Sampson}}(\mathbf{x}_1, \mathbf{x}_2; F) = \frac{|\mathbf{x}_2^\top F \mathbf{x}_1|}{\sqrt{(F\mathbf{x}_1)_x^2 + (F\mathbf{x}_1)_y^2 + (F^\top\mathbf{x}_2)_x^2 + (F^\top\mathbf{x}_2)_y^2}}$$
  4. **Adaptive MSAC/RANSAC Loop (`ransac_fundamental_matrix`):** Samples minimal 8-point subsets, evaluates Sampson inliers ($d_{\text{Sampson}} \le \tau$), minimizes the M-estimator Sample Consensus (MSAC, Torr & Zisserman 2000) truncated quadratic cost $\sum_i \min(d_i^2, \tau^2)$ so noisy hypotheses that accidentally graze an outlier near $\tau$ do not beat clean hypotheses, updates the stopping iteration count adaptively for $99.9\%$ confidence, and performs a two-stage trimmed least-squares 8-point refit on the consensus inlier set.
  5. **Essential Matrix Projection (`essential_from_fundamental`):** Computes $E_{\text{raw}} = K_2^\top F K_1$ and projects onto the normalized Essential manifold via SVD with $\det(U)=\det(V^\top)=+1$:
     $$E = U \operatorname{diag}(1, 1, 0) V^\top$$
  6. **Exact 4-Hypothesis Essential Decomposition (`decompose_essential_matrix`):** Decomposes $E$ into the 4 candidate relative poses $(R, \mathbf{t}) \in \text{SO}(3) \times \mathbb{S}^2$ using $R \in \{U W V^\top, U W^\top V^\top\}$ and $\mathbf{t} = \pm \mathbf{u}_3$.
  7. **Linear Homogeneous DLT Triangulation (`triangulate_points_dlt`):** For each correspondence and camera matrices $P_1 = K_1 [I \mid \mathbf{0}]$, $P_2 = K_2 [R \mid \mathbf{t}]$, constructs the $4 \times 4$ linear system $A \mathbf{X} = \mathbf{0}$ from $\mathbf{x} \times (P \mathbf{X}) = \mathbf{0}$, preconditions rows to unit $\ell_2$ norm, and solves via SVD.
  8. **Cheirality Disambiguation (`recover_pose_cheirality`):** Triangulates inliers across all 4 $(R, \mathbf{t})$ candidates and selects the unique pose maximizing positive depths in both cameras ($Z_1 > 0 \land Z_2 > 0$).
- **Comparison Baselines Only:** `cv2.findEssentialMat` + `cv2.recoverPose` (`estimate_two_view_opencv_baseline`) and `pycolmap.estimate_calibrated_two_view_geometry` (`estimate_two_view_pycolmap_baseline`) are used strictly for side-by-side benchmarking against our from-scratch implementation.
- **Design note (`phase-7b.5`):** an earlier variant projected every 8-point RANSAC hypothesis onto the Essential manifold before scoring, which produced a degenerate consensus on noisy minimal samples. We now score plain $F$ hypotheses with MSAC, iteratively re-fit $F$ on consensus inliers (up to 5 rounds until convergence), project $K_2^\top F K_1$ onto $\operatorname{diag}(1,1,0)$ once, recover $(R, \mathbf{t})$ via cheirality, and optionally refine $(R, \mathbf{t})$ on $\text{SO}(3)\times\mathbb{S}^2$ via 5-DoF Levenberg–Marquardt (`refine_essential_sampson_gn`) minimizing signed pixel Sampson error while remaining strictly on the Essential manifold.

## 10. N-View SfM (Phase 7b / 7b.5) — `pycolmap`, not from scratch
- `backend/app/vision/sfm.py` wraps `pycolmap.extract_features` (`CameraMode.SINGLE`), `match_exhaustive`, `incremental_mapping`, and `undistort_images` on CPU. Feature extraction, matching, two-view verification, registration, triangulation, bundle adjustment, and undistortion are all COLMAP's; HoloGen contributes image staging at `sfm_max_image_size`, automatic camera-model fallback (`SIMPLE_RADIAL` default -> `OPENCV` only if `<100%` of views register), `PINHOLE` undistortion for downstream dense geometry, parsing into HoloGen data structures, telemetry (registered/total, per-stage runtime, peak RSS via a 20 ms background sampler), and evaluation.
- **Implemented by us (NumPy):**
  - `align_umeyama_sim3`: Umeyama (1991) closed-form similarity. With $\Sigma_{xy} = \frac{1}{N}\sum (y_i - \mu_y)(x_i - \mu_x)^\top = UDV^\top$ and $S = \operatorname{diag}(1, 1, \operatorname{sign}\det(UV^\top))$: $R = USV^\top$, $s = \operatorname{tr}(DS)/\sigma_x^2$, $t = \mu_y - sR\mu_x$. Unit-tested on a hand-computed case ($s = 2.5$, 90° about Z, $t = (10, -5, 3)$) and a reflection case ($\det R = +1$ enforced).
  - `best_global_rotation`: closed-form chordal-L2 frame rotation $R_g$ minimising $\sum_i \|R^{\text{ref}}_i - R^{\text{sfm}}_i R_g^\top\|_F^2$, used to separate per-camera rotation error from the conditioning of the centre-based alignment.
  - `trajectory_sv_ratio`: $\sigma_2/\sigma_1$ of centred camera positions; values near 0 flag nearly collinear trajectories, for which centre-only Sim(3) leaves rotation about the trajectory axis poorly constrained.
