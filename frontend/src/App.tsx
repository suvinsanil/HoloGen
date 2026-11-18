import React, { useEffect, useRef } from 'react'
import {
  Box,
  Camera,
  CheckCircle2,
  Cpu,
  Download,
  Eye,
  FileCode2,
  FileJson,
  Layers,
  Loader2,
  RotateCcw,
  Sliders,
  Sparkles,
  Sun,
  Upload,
  AlertTriangle,
} from 'lucide-react'
import {
  ImageOverlayMode,
  useHoloStore,
  ViewMode3D,
} from './store/useHoloStore'
import { SceneViewer3D } from './components/SceneViewer3D'

const VIEW_MODES: { id: ViewMode3D; label: string }[] = [
  { id: 'mesh', label: 'Textured Mesh' },
  { id: 'wireframe', label: 'Wireframe' },
  { id: 'rgb', label: 'RGB Points' },
  { id: 'depth', label: 'Depth 3D' },
  { id: 'point_cloud', label: 'Point Cloud' },
  { id: 'normals', label: 'Normals' },
  { id: 'uncertainty_proxy', label: 'Uncertainty Proxy' },
]

const IMAGE_TABS: { id: ImageOverlayMode; label: string }[] = [
  { id: 'input', label: 'Input RGB' },
  { id: 'heatmap', label: 'Depth Heatmap' },
  { id: 'grayscale', label: 'Depth Gray' },
  { id: 'inverted', label: 'Depth Inverted' },
  { id: 'uncertainty_proxy', label: 'Uncertainty Proxy' },
]

const BG_SWATCHES = [
  { color: '#0b0f19', label: 'Obsidian' },
  { color: '#1e293b', label: 'Slate' },
  { color: '#334155', label: 'Charcoal' },
  { color: '#e2e8f0', label: 'Studio Light' },
]

export function App() {
  const {
    projects,
    activeProject,
    pointCloudData,
    params,
    viewMode3D,
    imageTab,
    lightIntensity,
    bgColor,
    pointSize,
    showGrid,
    cameraResetCounter,
    isUploading,
    errorBanner,
    setParams,
    setViewMode3D,
    setImageTab,
    setLightIntensity,
    setBgColor,
    setPointSize,
    setShowGrid,
    triggerCameraReset,
    setErrorBanner,
    fetchProjects,
    selectProject,
    uploadFile,
    runReconstruction,
    pollActiveProject,
  } = useHoloStore()

  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  useEffect(() => {
    const timer = setInterval(() => {
      pollActiveProject()
    }, 1500)
    return () => clearInterval(timer)
  }, [pollActiveProject])

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await uploadFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await uploadFile(e.target.files[0])
    }
  }

  const isRunning =
    activeProject &&
    [
      'queued',
      'preprocessing',
      'estimating_depth',
      'building_point_cloud',
      'meshing_and_texturing',
      'exporting',
    ].includes(activeProject.status)

  const isCompleted = activeProject?.status === 'completed'
  const report = activeProject?.report || null

  const imageSrc = activeProject
    ? `/api/depth/${activeProject.project_id}?mode=${imageTab}&t=${activeProject.updated_at}`
    : null

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur px-6 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/15 border border-cyan-400/40 flex items-center justify-center text-cyan-400">
            <Box className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white">HoloGen</h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-medium">
                Phase 6 Web Studio
              </span>
            </div>
            <p className="text-xs text-slate-400">From pixels to spatial intelligence</p>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            type="button"
            onClick={() => setParams({ mode: 'single_view' })}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
              params.mode === 'single_view'
                ? 'bg-cyan-500 text-slate-950 font-semibold shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Single View
          </button>
          <button
            type="button"
            onClick={() => setParams({ mode: 'multi_view' })}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition flex items-center gap-1.5 ${
              params.mode === 'multi_view'
                ? 'bg-cyan-500 text-slate-950 font-semibold shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <span>Multi-View</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-cyan-300">
              Phase 7
            </span>
          </button>
          <button
            type="button"
            disabled
            title="Video reconstruction is out of current scope"
            className="px-3 py-1.5 rounded-md text-xs font-medium text-slate-500 cursor-not-allowed flex items-center gap-1.5"
          >
            <span>Video</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-900 text-slate-500 border border-slate-800">
              Coming Soon
            </span>
          </button>
        </div>

        {/* Project Selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="project-select" className="text-xs text-slate-400">
            Scene Project:
          </label>
          <select
            id="project-select"
            value={activeProject?.project_id || ''}
            onChange={(e) => selectProject(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-cyan-400"
          >
            {projects.map((p) => (
              <option key={p.project_id} value={p.project_id}>
                {p.name} ({p.status})
              </option>
            ))}
          </select>
        </div>
      </header>

      {/* Error Alert Banner */}
      {errorBanner && (
        <div
          role="alert"
          className="mx-6 mt-3 px-4 py-2.5 rounded-lg bg-rose-950/80 border border-rose-500/40 text-rose-200 text-xs flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorBanner}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorBanner(null)}
            className="text-rose-300 hover:text-white font-semibold"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Content Grid */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 max-w-[1920px] w-full mx-auto">
        {/* Left Control Column (Upload + Parameters) */}
        <aside className="lg:col-span-3 flex flex-col gap-4">
          {/* Drag-and-Drop Upload Card */}
          <section className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
              <Upload className="w-3.5 h-3.5 text-cyan-400" />
              <span>1. Image Input</span>
            </h2>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-700 hover:border-cyan-400/70 rounded-xl p-4 text-center cursor-pointer transition bg-slate-950/50 hover:bg-slate-900/60"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
                className="hidden"
              />
              <Camera className="w-6 h-6 text-cyan-400 mx-auto mb-1.5" />
              <p className="text-xs font-medium text-slate-200">
                {isUploading ? 'Uploading & validating...' : 'Drag & drop image or click to browse'}
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                JPG, PNG, WebP (Max 15 MB, up to 4096×4096 px)
              </p>
            </div>
          </section>

          {/* Reconstruction Parameters Panel */}
          <section className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              <span>2. Reconstruction Parameters</span>
            </h2>

            {/* Depth Model */}
            <div>
              <label htmlFor="param-model" className="block text-xs text-slate-300 mb-1">
                Pretrained Depth Model
              </label>
              <select
                id="param-model"
                value={params.model_name}
                onChange={(e) =>
                  setParams({ model_name: e.target.value as any })
                }
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100"
              >
                <option value="depth_anything_v2_small">
                  Depth Anything V2 Small (Relative, Apache-2.0)
                </option>
                <option value="dpt_hybrid_midas">
                  Intel DPT-Hybrid MiDaS (Relative, MIT)
                </option>
                <option value="depth_anything_v2_metric_indoor_small">
                  Depth Anything V2 Metric Indoor (Meters, Apache-2.0)
                </option>
              </select>
            </div>

            {/* Resolution + Quality Preset */}
            <div className="grid grid-cols-2 gap-2.5">
              <div>
                <label htmlFor="param-res" className="block text-xs text-slate-300 mb-1">
                  Max Resolution
                </label>
                <select
                  id="param-res"
                  value={params.max_dimension}
                  onChange={(e) =>
                    setParams({ max_dimension: Number(e.target.value) })
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100"
                >
                  <option value={384}>384 px (Fast)</option>
                  <option value={512}>512 px (Balanced)</option>
                  <option value={640}>640 px (Default)</option>
                  <option value={768}>768 px (Dense)</option>
                </select>
              </div>

              <div>
                <label htmlFor="param-preset" className="block text-xs text-slate-300 mb-1">
                  Mesh Density Preset
                </label>
                <select
                  id="param-preset"
                  value={params.quality_preset}
                  onChange={(e) =>
                    setParams({ quality_preset: e.target.value as any })
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100"
                >
                  <option value="viewer">Viewer (~120k faces, &lt;5MB)</option>
                  <option value="full">Full Resolution</option>
                </select>
              </div>
            </div>

            {/* Surface Mesher & Depth Refinement */}
            <div className="grid grid-cols-2 gap-2.5">
              <div>
                <label htmlFor="param-mesher" className="block text-xs text-slate-300 mb-1">
                  Surface Mesher
                </label>
                <select
                  id="param-mesher"
                  value={params.mesh_method}
                  onChange={(e) =>
                    setParams({ mesh_method: e.target.value as any })
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100"
                >
                  <option value="depth_grid">Depth-Grid (Cut Edges)</option>
                  <option value="poisson">Screened Poisson (Trimmed)</option>
                  <option value="ball_pivoting">Ball Pivoting (BPA)</option>
                  <option value="alpha_shape">3D Alpha Shapes</option>
                </select>
              </div>

              <div>
                <label htmlFor="param-refinement" className="block text-xs text-slate-300 mb-1">
                  Depth Filter
                </label>
                <select
                  id="param-refinement"
                  value={params.refinement_method}
                  onChange={(e) =>
                    setParams({ refinement_method: e.target.value as any })
                  }
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100"
                >
                  <option value="none">None (Default, Best Edge F1)</option>
                  <option value="bilateral">Bilateral Filter</option>
                  <option value="guided">Guided Filter</option>
                  <option value="guided_bilateral">Guided + Bilateral</option>
                </select>
              </div>
            </div>

            {/* Smoothing Iterations Slider */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="param-smooth" className="text-slate-300">
                  Taubin Mesh Smoothing
                </label>
                <span className="text-cyan-300 font-mono">
                  {params.smoothing_iterations} iter
                </span>
              </div>
              <input
                id="param-smooth"
                type="range"
                min={0}
                max={10}
                step={1}
                value={params.smoothing_iterations}
                onChange={(e) =>
                  setParams({ smoothing_iterations: Number(e.target.value) })
                }
                className="w-full"
              />
            </div>

            {/* Uncertainty Proxy Threshold Slider */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <label htmlFor="param-unc" className="text-slate-300">
                  Uncertainty Proxy Cutoff
                </label>
                <span className="text-cyan-300 font-mono">
                  {params.uncertainty_threshold.toFixed(2)}
                </span>
              </div>
              <input
                id="param-unc"
                type="range"
                min={0.2}
                max={1.0}
                step={0.05}
                value={params.uncertainty_threshold}
                onChange={(e) =>
                  setParams({ uncertainty_threshold: Number(e.target.value) })
                }
                className="w-full"
              />
            </div>

            {/* Action Button */}
            <button
              type="button"
              disabled={!activeProject || Boolean(isRunning)}
              onClick={runReconstruction}
              className="mt-1 w-full py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-semibold text-xs flex items-center justify-center gap-2 transition shadow-lg shadow-cyan-500/10"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Reconstructing Scene...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Reconstruct 3D Scene</span>
                </>
              )}
            </button>

            {/* Live Status & Native <progress> Bar */}
            {activeProject && (
              <div
                role="status"
                aria-live="polite"
                className="mt-1 bg-slate-950/90 border border-slate-800 rounded-lg p-3"
              >
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="font-medium text-slate-200 flex items-center gap-1.5">
                    {activeProject.status === 'completed' ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : isRunning ? (
                      <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
                    ) : null}
                    <span className="uppercase tracking-wider text-[11px]">
                      {activeProject.status}
                    </span>
                  </span>
                  <span className="font-mono text-cyan-300 text-xs">
                    {activeProject.progress_pct}%
                  </span>
                </div>
                <label htmlFor="recon-progress" className="sr-only">
                  Reconstruction progress
                </label>
                <progress
                  id="recon-progress"
                  className="hologen-progress w-full h-2"
                  max={100}
                  value={
                    activeProject.status === 'queued'
                      ? undefined
                      : activeProject.progress_pct
                  }
                />
                <p className="text-[11px] text-slate-400 mt-1.5">
                  {activeProject.stage_message}
                </p>
              </div>
            )}
          </section>
        </aside>

        {/* Center + Right Split Workspace (2D Input/Depth vs Interactive 3D Viewer) */}
        <section className="lg:col-span-9 flex flex-col gap-4">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 min-h-[500px]">
            {/* 2D Input / Depth / Uncertainty Proxy Inspector (5 cols) */}
            <div className="xl:col-span-5 bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-cyan-400" />
                  <span>2D Input &amp; Depth Inspector</span>
                </h2>
                <div className="flex flex-wrap gap-1">
                  {IMAGE_TABS.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setImageTab(t.id)}
                      className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                        imageTab === t.id
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                          : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex-1 bg-slate-950 rounded-lg border border-slate-800/80 flex items-center justify-center overflow-hidden p-2 min-h-[360px]">
                {imageSrc ? (
                  <img
                    src={imageSrc}
                    alt={`${activeProject?.name || 'Scene'} - ${imageTab}`}
                    className="max-h-[420px] w-auto object-contain rounded"
                  />
                ) : (
                  <p className="text-xs text-slate-500">
                    Upload or select a project to inspect 2D layers
                  </p>
                )}
              </div>

              {imageTab === 'uncertainty_proxy' && (
                <p className="text-[11px] text-amber-300/90 mt-2 bg-amber-950/40 border border-amber-500/30 rounded px-2.5 py-1.5">
                  <strong>Uncertainty Proxy Notice:</strong> Computed from spatial log-depth
                  gradients. Indicates occlusion boundary risk, not a calibrated statistical
                  probability.
                </p>
              )}
            </div>

            {/* Interactive 3D Viewport (7 cols) */}
            <div className="xl:col-span-7 bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col">
              {/* 3D View Mode Switcher */}
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                <div className="flex flex-wrap gap-1">
                  {VIEW_MODES.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => setViewMode3D(m.id)}
                      className={`px-2.5 py-1 rounded text-[11px] font-medium transition ${
                        viewMode3D === m.id
                          ? 'bg-cyan-500 text-slate-950 font-semibold'
                          : 'bg-slate-950 text-slate-300 hover:text-white border border-slate-800'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={triggerCameraReset}
                  className="px-2.5 py-1 rounded text-[11px] bg-slate-950 hover:bg-slate-800 text-slate-200 border border-slate-700 flex items-center gap-1"
                  title="Reset 3D camera orbit, pan, and zoom"
                >
                  <RotateCcw className="w-3 h-3 text-cyan-400" />
                  <span>Reset View</span>
                </button>
              </div>

              {/* Three.js / React Three Fiber Canvas */}
              <div className="flex-1 min-h-[380px]">
                <SceneViewer3D
                  projectId={activeProject?.project_id || null}
                  isCompleted={Boolean(isCompleted)}
                  viewMode={viewMode3D}
                  pointCloudData={pointCloudData}
                  lightIntensity={lightIntensity}
                  bgColor={bgColor}
                  pointSize={pointSize}
                  showGrid={showGrid}
                  cameraResetCounter={cameraResetCounter}
                />
              </div>

              {/* Viewport Lighting, Point Size & Background Controls */}
              <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4 text-xs">
                <div className="flex items-center gap-2">
                  <Sun className="w-3.5 h-3.5 text-amber-400" />
                  <label htmlFor="ctrl-light" className="text-slate-400">
                    Light:
                  </label>
                  <input
                    id="ctrl-light"
                    type="range"
                    min={0.2}
                    max={2.5}
                    step={0.1}
                    value={lightIntensity}
                    onChange={(e) => setLightIntensity(Number(e.target.value))}
                    className="w-24"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  <label htmlFor="ctrl-ptsize" className="text-slate-400">
                    Point Size:
                  </label>
                  <input
                    id="ctrl-ptsize"
                    type="range"
                    min={0.01}
                    max={0.12}
                    step={0.005}
                    value={pointSize}
                    onChange={(e) => setPointSize(Number(e.target.value))}
                    className="w-24"
                  />
                </div>

                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Bg:</span>
                  {BG_SWATCHES.map((s) => (
                    <button
                      key={s.color}
                      type="button"
                      onClick={() => setBgColor(s.color)}
                      title={s.label}
                      style={{ backgroundColor: s.color }}
                      className={`w-5 h-5 rounded-full border ${
                        bgColor === s.color
                          ? 'border-cyan-400 ring-2 ring-cyan-400/40'
                          : 'border-slate-600'
                      }`}
                    />
                  ))}
                  <label className="ml-2 inline-flex items-center gap-1 text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showGrid}
                      onChange={(e) => setShowGrid(e.target.checked)}
                    />
                    <span>Grid</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Export Assets & Measured Reconstruction Report Panel */}
          {report && (
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col gap-4">
              {/* Export Buttons Row */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Download className="w-3.5 h-3.5 text-cyan-400" />
                    <span>3. Export 3D Scene Assets &amp; Measured Reports</span>
                  </h2>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Scale semantics:{' '}
                    <span className="text-amber-300 font-mono">
                      {report.depth_estimation?.scale_semantics}
                    </span>{' '}
                    — {report.depth_estimation?.scale_disclaimer}
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <a
                    href={`/api/export/${activeProject?.project_id}/glb`}
                    download
                    className="px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>
                      GLB Viewer ({report.export_availability?.glb?.size_mb ?? '—'} MB)
                    </span>
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/glb_full`}
                    download
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700 text-xs flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5 text-cyan-400" />
                    <span>
                      GLB Full ({report.export_availability?.glb_full?.size_mb ?? '—'} MB)
                    </span>
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/obj`}
                    download
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs"
                  >
                    OBJ
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/ply`}
                    download
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs"
                  >
                    PLY (Cloud)
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/mesh_ply`}
                    download
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs"
                  >
                    PLY (Mesh)
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/stl`}
                    download
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs"
                  >
                    STL
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/report_json`}
                    download
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 text-xs flex items-center gap-1"
                  >
                    <FileJson className="w-3.5 h-3.5" />
                    <span>JSON</span>
                  </a>
                  <a
                    href={`/api/export/${activeProject?.project_id}/report_md`}
                    download
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 text-xs flex items-center gap-1"
                  >
                    <FileCode2 className="w-3.5 h-3.5" />
                    <span>Report MD</span>
                  </a>
                </div>
              </div>

              {/* Telemetry Metrics Cards */}
              <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5">
                  <span className="text-[11px] text-slate-400 block">
                    Processed Resolution
                  </span>
                  <span className="text-sm font-bold text-white font-mono">
                    {report.preprocessing?.processed_size?.width} ×{' '}
                    {report.preprocessing?.processed_size?.height}
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    Blur Var: {report.preprocessing?.quality?.blur_laplacian_variance} (
                    {report.preprocessing?.quality?.exposure?.status})
                  </span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5">
                  <span className="text-[11px] text-slate-400 block">
                    Camera Intrinsics ({report.camera_intrinsics?.source})
                  </span>
                  <span className="text-sm font-bold text-white font-mono">
                    fx = {report.camera_intrinsics?.fx} px
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    FOV: {report.camera_intrinsics?.fov_x_deg}° ×{' '}
                    {report.camera_intrinsics?.fov_y_deg}°
                  </span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5">
                  <span className="text-[11px] text-slate-400 block">
                    Depth Inference ({report.hardware?.compute_device})
                  </span>
                  <span className="text-sm font-bold text-cyan-300 font-mono">
                    {report.depth_estimation?.inference_time_ms} ms
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    VRAM: {report.depth_estimation?.vram_allocated_mb} MB
                  </span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5">
                  <span className="text-[11px] text-slate-400 block">
                    Cleaned 3D Point Cloud
                  </span>
                  <span className="text-sm font-bold text-white font-mono">
                    {report.point_cloud?.num_points?.toLocaleString()} pts
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    Retained:{' '}
                    {report.estimated_coverage?.cleaned_point_retention_pct}% of raw
                  </span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5">
                  <span className="text-[11px] text-slate-400 block">
                    Mesh Faces ({report.mesh?.quality_preset || 'viewer'})
                  </span>
                  <span className="text-sm font-bold text-white font-mono">
                    {report.mesh?.num_faces?.toLocaleString()} tris
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    Full-res: {report.mesh?.full_res_faces?.toLocaleString() || report.mesh?.num_faces?.toLocaleString()} tris
                  </span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5">
                  <span
                    className="text-[11px] text-slate-400 block"
                    title="2D camera-plane projection occupancy; does NOT measure 3D scene completeness"
                  >
                    View-Frustum Coverage*
                  </span>
                  <span className="text-sm font-bold text-emerald-400 font-mono">
                    {report.estimated_coverage?.view_frustum_coverage_pct ??
                      report.estimated_coverage?.frustum_mesh_coverage_pct}
                    %
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    2D frustum occupancy (not 3D completeness)
                  </span>
                </div>
              </div>

              {/* Stage Timings + Uncertainty Proxy Histogram */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
                  <h3 className="text-xs font-semibold text-slate-300 mb-2 flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Measured Stage Timings (Total: {report.timings_ms?.total_pipeline_ms} ms)</span>
                  </h3>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    {report.timings_ms &&
                      Object.entries(report.timings_ms).map(([k, v]) => (
                        <div
                          key={k}
                          className="flex items-center justify-between bg-slate-900/80 px-2 py-1 rounded border border-slate-800"
                        >
                          <span className="text-slate-400 truncate">
                            {k.replace('_ms', '')}:
                          </span>
                          <span className="font-mono text-slate-200">{String(v)}ms</span>
                        </div>
                      ))}
                  </div>
                </div>

                <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
                  <h3 className="text-xs font-semibold text-slate-300 mb-2">
                    Monocular Uncertainty Proxy Distribution (Mean:{' '}
                    {report.uncertainty_proxy?.mean_uncertainty_proxy}, P90:{' '}
                    {report.uncertainty_proxy?.p90_uncertainty_proxy}, P95:{' '}
                    {report.uncertainty_proxy?.p95_uncertainty_proxy})
                  </h3>
                  {report.uncertainty_proxy?.histogram_pct ? (
                    <div className="grid grid-cols-7 gap-1.5 text-[10px]">
                      {Object.entries(report.uncertainty_proxy.histogram_pct).map(
                        ([bin, pct]) => (
                          <div
                            key={bin}
                            className="bg-slate-900 p-1.5 rounded border border-slate-800 text-center"
                          >
                            <div className="font-mono text-cyan-300 font-semibold">
                              {String(pct)}%
                            </div>
                            <div className="text-slate-400 truncate mt-0.5">{bin}</div>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-400">
                      Reliable pixel fraction:{' '}
                      {Math.round(
                        (report.uncertainty_proxy?.reliable_pixel_fraction || 1) * 100
                      )}
                      %
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
