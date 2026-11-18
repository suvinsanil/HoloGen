import { create } from 'zustand'

export type ViewMode3D =
  | 'mesh'
  | 'wireframe'
  | 'rgb'
  | 'depth'
  | 'point_cloud'
  | 'normals'
  | 'uncertainty_proxy'

export type ImageOverlayMode =
  | 'input'
  | 'heatmap'
  | 'grayscale'
  | 'inverted'
  | 'uncertainty_proxy'

export interface ReconstructionParams {
  mode: 'single_view' | 'multi_view'
  model_name:
    | 'depth_anything_v2_small'
    | 'dpt_hybrid_midas'
    | 'depth_anything_v2_metric_indoor_small'
  max_dimension: number
  fov_deg: number
  refinement_method: 'none' | 'bilateral' | 'guided' | 'guided_bilateral'
  mesh_method: 'depth_grid' | 'poisson' | 'ball_pivoting' | 'alpha_shape'
  quality_preset: 'viewer' | 'full'
  viewer_target_faces: number
  smoothing_iterations: number
  uncertainty_threshold: number
  discontinuity_threshold: number
}

export interface ProjectStatus {
  project_id: string
  name: string
  mode: string
  status: string
  progress_pct: number
  stage_message: string
  queue_position: number
  created_at: string
  updated_at: string
  error_message?: string | null
  parameters: Record<string, any>
  report?: Record<string, any> | null
  available_exports: string[]
}

export interface PointCloudViewerPayload {
  count: number
  positions: number[]
  colors: number[]
  normals: number[]
  depths: number[]
  uncertainty: number[]
}

interface HoloState {
  projects: ProjectStatus[]
  activeProject: ProjectStatus | null
  pointCloudData: PointCloudViewerPayload | null
  params: ReconstructionParams
  viewMode3D: ViewMode3D
  imageTab: ImageOverlayMode
  lightIntensity: number
  bgColor: string
  pointSize: number
  showGrid: boolean
  cameraResetCounter: number
  isUploading: boolean
  errorBanner: string | null

  setParams: (partial: Partial<ReconstructionParams>) => void
  setViewMode3D: (mode: ViewMode3D) => void
  setImageTab: (tab: ImageOverlayMode) => void
  setLightIntensity: (val: number) => void
  setBgColor: (color: string) => void
  setPointSize: (size: number) => void
  setShowGrid: (show: boolean) => void
  triggerCameraReset: () => void
  setErrorBanner: (msg: string | null) => void

  fetchProjects: () => Promise<void>
  selectProject: (projectId: string) => Promise<void>
  uploadFile: (file: File) => Promise<void>
  runReconstruction: () => Promise<void>
  pollActiveProject: () => Promise<void>
}

export const useHoloStore = create<HoloState>((set, get) => ({
  projects: [],
  activeProject: null,
  pointCloudData: null,
  params: {
    mode: 'single_view',
    model_name: 'depth_anything_v2_small',
    max_dimension: 640,
    fov_deg: 60.0,
    refinement_method: 'none',
    mesh_method: 'depth_grid',
    quality_preset: 'viewer',
    viewer_target_faces: 120000,
    smoothing_iterations: 4,
    uncertainty_threshold: 0.75,
    discontinuity_threshold: 0.08,
  },
  viewMode3D: 'mesh',
  imageTab: 'input',
  lightIntensity: 1.3,
  bgColor: '#0b0f19',
  pointSize: 0.035,
  showGrid: true,
  cameraResetCounter: 0,
  isUploading: false,
  errorBanner: null,

  setParams: (partial) =>
    set((state) => ({ params: { ...state.params, ...partial } })),
  setViewMode3D: (viewMode3D) => set({ viewMode3D }),
  setImageTab: (imageTab) => set({ imageTab }),
  setLightIntensity: (lightIntensity) => set({ lightIntensity }),
  setBgColor: (bgColor) => set({ bgColor }),
  setPointSize: (pointSize) => set({ pointSize }),
  setShowGrid: (showGrid) => set({ showGrid }),
  triggerCameraReset: () =>
    set((state) => ({ cameraResetCounter: state.cameraResetCounter + 1 })),
  setErrorBanner: (errorBanner) => set({ errorBanner }),

  fetchProjects: async () => {
    try {
      const resp = await fetch('/api/projects')
      if (!resp.ok) return
      const data = await resp.json()
      const list: ProjectStatus[] = data.projects || []
      set({ projects: list })
      if (!get().activeProject && list.length > 0) {
        await get().selectProject(list[0].project_id)
      }
    } catch (err: any) {
      set({ errorBanner: `Failed to fetch projects: ${err.message}` })
    }
  },

  selectProject: async (projectId: string) => {
    try {
      set({ errorBanner: null })
      const resp = await fetch(`/api/status/${projectId}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const proj: ProjectStatus = await resp.json()
      set({ activeProject: proj })

      if (proj.status === 'completed') {
        const pcResp = await fetch(`/api/projects/${projectId}/point-cloud-json`)
        if (pcResp.ok) {
          const pcJson: PointCloudViewerPayload = await pcResp.json()
          set({ pointCloudData: pcJson })
        }
      } else {
        set({ pointCloudData: null })
      }
    } catch (err: any) {
      set({ errorBanner: `Could not load project ${projectId}: ${err.message}` })
    }
  },

  uploadFile: async (file: File) => {
    set({ isUploading: true, errorBanner: null })
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('name', file.name.replace(/\.[^.]+$/, ''))
      formData.append('mode', get().params.mode)

      const resp = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.detail || `Upload failed (HTTP ${resp.status})`)
      }
      const proj: ProjectStatus = await resp.json()
      set((state) => ({
        activeProject: proj,
        projects: [proj, ...state.projects.filter((p) => p.project_id !== proj.project_id)],
        pointCloudData: null,
        imageTab: 'input',
      }))
    } catch (err: any) {
      set({ errorBanner: err.message })
    } finally {
      set({ isUploading: false })
    }
  },

  runReconstruction: async () => {
    const active = get().activeProject
    if (!active) return
    set({ errorBanner: null })
    try {
      const resp = await fetch(`/api/reconstruct/${active.project_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...get().params,
          run_synchronously: false,
        }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.detail || `Reconstruct request failed (${resp.status})`)
      }
      const updated: ProjectStatus = await resp.json()
      set((state) => ({
        activeProject: updated,
        projects: state.projects.map((p) =>
          p.project_id === updated.project_id ? updated : p
        ),
      }))
    } catch (err: any) {
      set({ errorBanner: err.message })
    }
  },

  pollActiveProject: async () => {
    const active = get().activeProject
    if (!active) return
    if (!['queued', 'preprocessing', 'estimating_depth', 'building_point_cloud', 'meshing_and_texturing', 'exporting'].includes(active.status)) {
      return
    }
    try {
      const resp = await fetch(`/api/status/${active.project_id}`)
      if (!resp.ok) return
      const updated: ProjectStatus = await resp.json()
      set((state) => ({
        activeProject: updated,
        projects: state.projects.map((p) =>
          p.project_id === updated.project_id ? updated : p
        ),
      }))
      if (updated.status === 'completed') {
        const pcResp = await fetch(`/api/projects/${updated.project_id}/point-cloud-json`)
        if (pcResp.ok) {
          const pcJson: PointCloudViewerPayload = await pcResp.json()
          set({ pointCloudData: pcJson })
        }
      }
    } catch {
      // Ignore transient network poll errors
    }
  },
}))
