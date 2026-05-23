import { useState, useEffect, useRef, useCallback } from "react";
import { CameraView } from "./components/CameraView";
import { Orientation } from "./components/SensorsShim";
import { motion, AnimatePresence } from "motion/react";
import {
  Info,
  RefreshCw,
  Zap,
  ShieldCheck,
  Play,
  Utensils,
  RotateCcw,
} from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { YoloDetector, type Detection } from "./services/yoloService";

// Helper for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

declare global {
  interface Window {
    cv: any;
  }
}

interface VisionState {
  luma: number;
  focusMetric: number;
  isCapturing: boolean;
  cvLoaded: boolean;
  yoloLoaded: boolean;
  detections: Detection[];
}

interface SensorData {
  alpha: number;
  beta: number;
  pitch: number;
  roll: number;
}

export default function App() {
  // --- Refs ---
  const cameraRef = useRef<any>(null);
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const requestRef = useRef<number>(0);

  const yoloDetector = useRef(new YoloDetector());
  const lastYoloRun = useRef(0);
  const photoDimsRef = useRef({ width: 0, height: 0 });
  const viewportRef = useRef<HTMLDivElement>(null);

  // --- State ---
  const [cameraActive, setCameraActive] = useState(false);
  const [hasCameraPermission, setHasCameraPermission] = useState<
    boolean | null
  >(null);
  const [sensorsActive, setSensorsActive] = useState(false);
  const [visionData, setVisionData] = useState<VisionState>({
    luma: 0,
    focusMetric: 0,
    isCapturing: false,
    cvLoaded: false,
    yoloLoaded: false,
    detections: [],
  });
  const [sensorData, setSensorData] = useState<SensorData>({
    alpha: 0,
    beta: 0,
    pitch: 0,
    roll: 0,
  });

  // --- Initialize CV & YOLO ---
  useEffect(() => {
    const checkCV = setInterval(() => {
      if (window.cv && window.cv.Mat) {
        setVisionData((prev) => ({ ...prev, cvLoaded: true }));
        clearInterval(checkCV);
      }
    }, 500);

    // Load YOLO
    yoloDetector.current.loadModel().then(() => {
      setVisionData((prev) => ({ ...prev, yoloLoaded: true }));
    });

    return () => clearInterval(checkCV);
  }, []);

  // --- Start everything on button press ---
  const startSystem = async () => {
    // 1. Start the camera (triggers permission prompt)
    setCameraActive(true);

    // 2. Request sensor permissions
    try {
      // iOS 13+ requires explicit permission for DeviceOrientation
      if (
        typeof (DeviceOrientationEvent as any).requestPermission === "function"
      ) {
        const response = await (
          DeviceOrientationEvent as any
        ).requestPermission();
        if (response !== "granted") {
          alert("Orientation permission denied");
          return;
        }
      }

      setSensorsActive(true);

      const subscription = Orientation.addListener((data) => {
        setSensorData((prev) => ({
          ...prev,
          alpha: data.alpha,
          beta: data.beta,
          pitch: data.beta, // beta = front-to-back tilt (-180 to 180)
          roll: data.gamma, // gamma = left-to-right tilt (-90 to 90)
        }));
      });

      return () => subscription.remove();
    } catch (err) {
      console.error("Error starting sensors:", err);
    }
  };

  // Retry camera permission
  const retryCamera = () => {
    setHasCameraPermission(null);
    setCameraActive(true);
  };

  // --- Processing Loop ---
  const processFrame = useCallback(async () => {
    if (!visionData.cvLoaded || !cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5,
        base64: false,
        skipProcessing: true,
      });

      if (!photo.uri) return;

      // Store photo dimensions for coordinate scaling
      photoDimsRef.current = { width: photo.width, height: photo.height };

      const img = new Image();
      img.src = photo.uri;

      img.onload = () => {
        const cv = window.cv;
        const canvas = document.getElementById(
          "processing-canvas",
        ) as HTMLCanvasElement;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Set canvas dims to image dims
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Map data to OpenCV Mat
        let src = cv.imread(canvas);
        let gray = new cv.Mat();
        let laplacian = new cv.Mat();

        // 1. Convert to Grayscale
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

        // 2. Calculate Luma (Average brightness)
        let mean = cv.mean(gray);
        const luma = mean[0];

        // 3. Laplacian Variance (Blur detection)
        cv.Laplacian(gray, laplacian, cv.CV_64F);
        let lapMean = new cv.Mat();
        let lapStdDev = new cv.Mat();
        cv.meanStdDev(laplacian, lapMean, lapStdDev);
        const variance = Math.pow(lapStdDev.data64F[0], 2);

        // 4. Staggered YOLO Detection (Every ~3s to save CPU)
        const now = Date.now();
        if (visionData.yoloLoaded && now - lastYoloRun.current > 3000) {
          lastYoloRun.current = now;
          yoloDetector.current.detect(canvas).then((detections) => {
            // Scale detection coordinates from photo size to viewport size
            const pw = photoDimsRef.current.width || 1;
            const ph = photoDimsRef.current.height || 1;
            const vw = viewportRef.current?.clientWidth || pw;
            const vh = viewportRef.current?.clientHeight || ph;
            const scaleX = vw / pw;
            const scaleY = vh / ph;
            const scaled = detections.map((d) => ({
              ...d,
              x: d.x * scaleX,
              y: d.y * scaleY,
              width: d.width * scaleX,
              height: d.height * scaleY,
            }));
            console.log(
              `Detected ${scaled.length} objects:`,
              scaled.map(
                (d) => `${d.class} ${(d.confidence * 100).toFixed(0)}%`,
              ),
            );
            setVisionData((prev) => ({ ...prev, detections: scaled }));
          });
        }

        setVisionData((prev) => ({
          ...prev,
          luma,
          focusMetric: variance,
        }));

        // CRITICAL: Memory Management
        src.delete();
        gray.delete();
        laplacian.delete();
        lapMean.delete();
        lapStdDev.delete();
      };
    } catch (err) {
      console.error("Frame processing error:", err);
    }
  }, [visionData.cvLoaded]);

  useEffect(() => {
    let interval: any;
    if (visionData.cvLoaded && sensorsActive) {
      interval = setInterval(processFrame, 1500); // Process every 1.5s for stability
    }
    return () => clearInterval(interval);
  }, [visionData.cvLoaded, sensorsActive, processFrame]);

  // --- UI Components ---
  const showCameraDenied = hasCameraPermission === false;
  const showCameraLoading = cameraActive && hasCameraPermission === null;
  const showCameraFeed = cameraActive && hasCameraPermission === true;

  return (
    <main className="relative flex flex-col h-full bg-black text-white font-sans overflow-hidden">
      {/* Camera Denied overlay */}
      {showCameraDenied && (
        <div className="absolute inset-0 z-[100] flex flex-col items-center justify-center p-8 bg-neutral-950">
          <Info className="w-16 h-16 text-red-500 mb-4" />
          <h2 className="text-2xl font-bold tracking-tight mb-2">
            Camera Denied
          </h2>
          <p className="text-neutral-400 text-center max-w-xs mb-6">
            Camera access was denied. Please allow camera access in your browser
            settings, then tap Retry below.
          </p>
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={retryCamera}
            className="bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-6 rounded-xl flex items-center gap-2 transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            RETRY
          </motion.button>
        </div>
      )}

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 p-6 flex justify-between items-center bg-gradient-to-b from-black/80 to-transparent pointer-events-none">
        <div>
          <h1 className="text-xl font-bold tracking-tighter uppercase flex items-center gap-2">
            <Zap className="fill-blue-500 text-blue-500 w-5 h-5" />
            <p className="text-[10px] text-neutral-500 uppercase tracking-widest font-mono">
              Experimental CV Runtime
            </p>
          </h1>
        </div>
        <div className="text-right flex flex-col items-end gap-1">
          <div
            className={cn(
              "text-[10px] font-mono uppercase px-2 py-0.5 rounded-full border border-current transition-colors",
              visionData.cvLoaded
                ? "text-emerald-500 bg-emerald-500/10"
                : "text-amber-500 bg-amber-500/10",
            )}
          >
            CV {visionData.cvLoaded ? "Initialized" : "Loading..."}
          </div>
          <div
            className={cn(
              "text-[10px] font-mono uppercase px-2 py-0.5 rounded-full border border-current transition-colors",
              visionData.yoloLoaded
                ? "text-blue-500 bg-blue-500/10"
                : "text-neutral-500 bg-neutral-500/10",
            )}
          >
            YOLO {visionData.yoloLoaded ? "Active" : "Standby"}
          </div>
        </div>
      </header>

      {/* Camera Viewport — starts only when `cameraActive` is true */}
      <section ref={viewportRef} className="flex-1 relative overflow-hidden">
        {/* Camera loading overlay */}
        {showCameraLoading && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black">
            <RefreshCw className="w-12 h-12 animate-spin text-blue-500 mb-4" />
            <h2 className="text-xl font-medium tracking-tight">
              Requesting Camera Access
            </h2>
          </div>
        )}

        <CameraView
          ref={cameraRef}
          style={{ width: "100%", height: "100%" }}
          facing="back"
          active={cameraActive}
          onPermissionResult={(granted) => setHasCameraPermission(granted)}
        />

        {/* Detection Overlays — only show when camera is running */}
        {showCameraFeed &&
          visionData.detections.map((det, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute border-2 border-emerald-500 bg-emerald-500/10 rounded-sm pointer-events-none"
              style={{
                left: `${det.x}px`,
                top: `${det.y}px`,
                width: `${det.width}px`,
                height: `${det.height}px`,
                zIndex: 40,
              }}
            >
              <div className="absolute top-0 left-0 -translate-y-full bg-emerald-500 text-black text-[10px] font-bold px-1 py-0.5 rounded-t-sm whitespace-nowrap">
                {det.class.toUpperCase()} {(det.confidence * 100).toFixed(0)}%
              </div>
            </motion.div>
          ))}

        {/* Reticle Overlay — show when camera is active */}
        {showCameraFeed && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-64 h-64 border-[0.5px] border-white/20 rounded-3xl relative">
              <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-blue-500 rounded-tl-lg" />
              <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-right-2 border-blue-500 rounded-tr-lg border-r-2" />
              <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-blue-500 rounded-bl-lg" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-right-2 border-blue-500 rounded-br-lg border-r-2" />
            </div>
          </div>
        )}

        {/* Real-time Telemetry Bubbles */}
        <AnimatePresence>
          {sensorsActive && (
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="absolute right-6 top-1/2 -translate-y-1/2 flex flex-col gap-3 pointer-events-none"
            >
              <TelemetryValue
                label="LUMA"
                value={visionData.luma.toFixed(1)}
                postfix="lx"
                color="blue"
              />
              <TelemetryValue
                label="FOCUS"
                value={visionData.focusMetric.toFixed(0)}
                postfix="sh"
                color="amber"
              />
              <TelemetryValue
                label="TILT"
                value={`${sensorData.pitch.toFixed(1)}°`}
                postfix=""
                color="neutral"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* Control Panel */}
      <footer className="p-8 bg-neutral-950 border-t border-white/5 space-y-6">
        {!sensorsActive ? (
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={startSystem}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-3 shadow-xl shadow-blue-900/20 transition-all"
          >
            <Play className="w-5 h-5 fill-current" />
            INITIATE SYSTEM
          </motion.button>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 rounded-2xl p-4 flex flex-col justify-between h-20">
              <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-widest">
                Pitch
              </span>
              <span className="text-xl font-mono">
                {sensorData.pitch.toFixed(1)}°
              </span>
            </div>
            <div className="bg-white/5 rounded-2xl p-4 flex flex-col justify-between h-20">
              <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-widest">
                Roll
              </span>
              <span className="text-xl font-mono">
                {sensorData.roll.toFixed(1)}°
              </span>
            </div>
          </div>
        )}

        {/* Detected Objects Bar */}
        <AnimatePresence>
          {visionData.detections.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="flex gap-2 overflow-x-auto pb-2 no-scrollbar"
            >
              {visionData.detections.map((det, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 bg-emerald-500/20 border border-emerald-500/30 px-3 py-1.5 rounded-full whitespace-nowrap"
                >
                  <Utensils className="w-3 h-3 text-emerald-500" />
                  <span className="text-xs font-bold text-emerald-100 uppercase tracking-tighter">
                    {det.class}
                  </span>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </footer>
    </main>
  );
}

function TelemetryValue({
  label,
  value,
  postfix,
  color,
}: {
  label: string;
  value: string;
  postfix: string;
  color: "blue" | "amber" | "neutral";
}) {
  const colors = {
    blue: "text-blue-400 border-blue-500/20 bg-blue-500/5",
    amber: "text-amber-400 border-amber-500/20 bg-amber-500/5",
    neutral: "text-neutral-400 border-neutral-500/20 bg-neutral-500/5",
  };

  return (
    <div
      className={cn(
        "px-3 py-2 rounded-xl border flex flex-col items-end backdrop-blur-md",
        colors[color],
      )}
    >
      <span className="text-[8px] font-bold tracking-widest uppercase opacity-70 leading-none mb-1">
        {label}
      </span>
      <div className="flex items-baseline gap-0.5">
        <span className="text-lg font-mono font-bold leading-none">
          {value}
        </span>
        <span className="text-[8px] font-mono opacity-50 uppercase">
          {postfix}
        </span>
      </div>
    </div>
  );
}
