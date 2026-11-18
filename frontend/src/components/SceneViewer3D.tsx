import React, { Suspense, useEffect, useMemo, useRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Center, useGLTF, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { CameraFrustum, PointCloudViewerPayload, ViewMode3D } from '../store/useHoloStore'

interface GLBMeshSurfaceProps {
  glbUrl: string
  viewMode: ViewMode3D
}

function GLBMeshSurface({ glbUrl, viewMode }: GLBMeshSurfaceProps) {
  const { scene } = useGLTF(glbUrl)

  const clonedScene = useMemo(() => {
    const clone = scene.clone(true)
    clone.traverse((node) => {
      if ((node as THREE.Mesh).isMesh) {
        const mesh = node as THREE.Mesh
        if (viewMode === 'wireframe') {
          mesh.material = new THREE.MeshBasicMaterial({
            color: '#38bdf8',
            wireframe: true,
          })
        } else if (viewMode === 'normals') {
          mesh.material = new THREE.MeshNormalMaterial({
            flatShading: false,
            side: THREE.DoubleSide,
          })
        }
      }
    })
    return clone
  }, [scene, viewMode])

  return (
    <Center>
      <primitive object={clonedScene} />
    </Center>
  )
}

function infernoColor(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  // Smooth Inferno-like perceptual colormap in [0, 1]
  const r = Math.min(1, Math.max(0, 1.5 * c))
  const g = Math.min(1, Math.max(0, 1.6 * (c - 0.25)))
  const b = Math.min(1, Math.max(0, 1.0 - 1.8 * Math.abs(c - 0.35)))
  return [r, g, b]
}

interface PointCloudObjectProps {
  data: PointCloudViewerPayload
  viewMode: ViewMode3D
  pointSize: number
}

function PointCloudObject({ data, viewMode, pointSize }: PointCloudObjectProps) {
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry()
    const positions = new Float32Array(data.positions)
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))

    const n = data.count
    const colorArr = new Float32Array(n * 3)

    if (viewMode === 'normals' && data.normals.length === n * 3) {
      for (let i = 0; i < n * 3; i++) {
        colorArr[i] = Math.max(0, Math.min(1, data.normals[i] * 0.5 + 0.5))
      }
    } else if (viewMode === 'depth' && data.depths.length === n) {
      let dMin = Infinity
      let dMax = -Infinity
      for (let i = 0; i < n; i++) {
        if (data.depths[i] < dMin) dMin = data.depths[i]
        if (data.depths[i] > dMax) dMax = data.depths[i]
      }
      const span = Math.max(dMax - dMin, 1e-5)
      for (let i = 0; i < n; i++) {
        const norm = 1.0 - (data.depths[i] - dMin) / span
        const [r, g, b] = infernoColor(norm)
        colorArr[i * 3] = r
        colorArr[i * 3 + 1] = g
        colorArr[i * 3 + 2] = b
      }
    } else if (viewMode === 'uncertainty_proxy' && data.uncertainty.length === n) {
      for (let i = 0; i < n; i++) {
        const u = Math.max(0, Math.min(1, data.uncertainty[i] * 3.0))
        // Cyan (low uncertainty) -> Yellow -> Crimson Red (high uncertainty proxy)
        colorArr[i * 3] = Math.min(1, u * 1.6)
        colorArr[i * 3 + 1] = Math.max(0.1, 1.0 - Math.abs(u - 0.35) * 1.4)
        colorArr[i * 3 + 2] = Math.max(0, 1.0 - u * 1.5)
      }
    } else {
      for (let i = 0; i < n * 3; i++) {
        colorArr[i] = data.colors[i] ?? 0.8
      }
    }

    geom.setAttribute('color', new THREE.BufferAttribute(colorArr, 3))
    geom.computeBoundingSphere()
    return geom
  }, [data, viewMode])

  return (
    <Center>
      <points geometry={geometry}>
        <pointsMaterial
          size={pointSize}
          vertexColors
          sizeAttenuation
        />
      </points>
    </Center>
  )
}

function CameraResetController({ resetCounter }: { resetCounter: number }) {
  const { camera } = useThree()
  const controlsRef = useRef<any>(null)

  useEffect(() => {
    camera.position.set(0, 0.5, 6.5)
    camera.lookAt(0, 0, 0)
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0)
      controlsRef.current.update()
    }
  }, [resetCounter, camera])

  return <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08} />
}

function CameraFrustumsOverlay({ frustums }: { frustums: CameraFrustum[] }) {
  const geometry = useMemo(() => {
    const verts: number[] = []
    const scale = 0.25
    for (const f of frustums) {
      const c = new THREE.Vector3(...f.center)
      const r = new THREE.Vector3(...f.right)
      const u = new THREE.Vector3(...f.up)
      const fwd = new THREE.Vector3(...f.forward)
      const halfW = scale * Math.tan(((f.fov_x_deg || 60) * Math.PI) / 360)
      const halfH = scale * Math.tan(((f.fov_y_deg || 45) * Math.PI) / 360)
      const base = c.clone().addScaledVector(fwd, scale)
      const p0 = base.clone().addScaledVector(r, -halfW).addScaledVector(u, halfH)
      const p1 = base.clone().addScaledVector(r, halfW).addScaledVector(u, halfH)
      const p2 = base.clone().addScaledVector(r, halfW).addScaledVector(u, -halfH)
      const p3 = base.clone().addScaledVector(r, -halfW).addScaledVector(u, -halfH)
      const segs = [
        [c, p0], [c, p1], [c, p2], [c, p3],
        [p0, p1], [p1, p2], [p2, p3], [p3, p0],
      ]
      for (const [a, b] of segs) {
        verts.push(a.x, a.y, a.z, b.x, b.y, b.z)
      }
    }
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
    return geom
  }, [frustums])

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#38bdf8" opacity={0.85} transparent />
    </lineSegments>
  )
}

interface SceneViewer3DProps {
  projectId: string | null
  isCompleted: boolean
  viewMode: ViewMode3D
  pointCloudData: PointCloudViewerPayload | null
  cameraFrustums?: CameraFrustum[]
  showFrustums?: boolean
  lightIntensity: number
  bgColor: string
  pointSize: number
  showGrid: boolean
  cameraResetCounter: number
}

export const SceneViewer3D: React.FC<SceneViewer3DProps> = ({
  projectId,
  isCompleted,
  viewMode,
  pointCloudData,
  cameraFrustums = [],
  showFrustums = true,
  lightIntensity,
  bgColor,
  pointSize,
  showGrid,
  cameraResetCounter,
}) => {
  const useMeshRenderer =
    viewMode === 'mesh' || viewMode === 'wireframe' || (viewMode === 'normals' && !pointCloudData)

  const glbUrl = projectId ? `/api/export/${projectId}/glb?v=${projectId}` : null

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-slate-800">
      <Canvas
        camera={{ position: [0, 0.5, 6.5], fov: 50, near: 0.05, far: 200 }}
        gl={{ preserveDrawingBuffer: true, antialias: true }}
      >
        <color attach="background" args={[bgColor]} />
        <ambientLight intensity={lightIntensity * 0.75} />
        <directionalLight position={[4, 6, 5]} intensity={lightIntensity} />
        <directionalLight position={[-4, -2, 3]} intensity={lightIntensity * 0.4} />

        <CameraResetController resetCounter={cameraResetCounter} />

        {showGrid && (
          <Grid
            position={[0, -2.6, 0]}
            args={[16, 16]}
            cellSize={0.5}
            cellThickness={0.5}
            cellColor="#334155"
            sectionSize={2.0}
            sectionThickness={1.0}
            sectionColor="#475569"
            fadeDistance={24}
          />
        )}

        {isCompleted && projectId && (
          <Suspense fallback={null}>
            {useMeshRenderer && glbUrl ? (
              <GLBMeshSurface glbUrl={glbUrl} viewMode={viewMode} />
            ) : pointCloudData && pointCloudData.count > 0 ? (
              <PointCloudObject
                data={pointCloudData}
                viewMode={viewMode}
                pointSize={pointSize}
              />
            ) : null}
            {showFrustums && cameraFrustums.length > 0 && (
              <CameraFrustumsOverlay frustums={cameraFrustums} />
            )}
          </Suspense>
        )}
      </Canvas>
    </div>
  )
}
