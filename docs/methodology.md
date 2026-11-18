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

## 4. Monocular Uncertainty Proxy

Monocular depth networks do not output calibrated epistemic or aleatoric probabilities. HoloGen defines and reports an explicit **Uncertainty Proxy** $U(u, v) \in [0, 1]$ composed of:
1. **Depth-Gradient Discontinuity Proxy:** Normalized Sobel gradient magnitude $\|\nabla Z(u,v)\|_2$, which peaks along occlusion boundaries where flying pixels occur.
2. **Flip-Consistency Proxy:** Pixelwise discrepancy between original inference $Z(I)$ and horizontally flipped inference $\text{flip}(Z(\text{flip}(I)))$:
   $$U_{\text{flip}}(u, v) = \frac{|Z(u, v) - Z_{\text{flip}}(u, v)|}{\max(Z(u, v), \epsilon)}$$
3. **Inter-Model Disagreement Proxy (when multiple models are run):** Pixelwise variance after least-squares scale-shift alignment between models.


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
