import * as ort from "onnxruntime-web";

// Point ONNX Runtime to CDN-hosted WASM files
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/";
ort.env.wasm.numThreads = 1;

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
}

// COCO dataset 80 classes
const COCO_LABELS = ["food"];

const INPUT_SIZE = 320; // YOLO26n 640x640 input
const CONFIDENCE_THRESHOLD = 0.5;
const NMS_THRESHOLD = 0.4;

export class YoloDetector {
  private session: ort.InferenceSession | null = null;
  private isLoaded = false;
  private preprocessCanvas: HTMLCanvasElement | null = null;
  private preprocessCtx: CanvasRenderingContext2D | null = null;
  private lastInferenceTime = 0;
  private inferenceTimes: number[] = [];
  private fps = 0;

  async loadModel(modelPath: string = "/best.onnx") {
    try {
      console.log("Loading ONNX model from:", modelPath);
      this.session = await ort.InferenceSession.create(modelPath as any, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
        enableMemPattern: false,
      });
      this.isLoaded = true;
      console.log("ONNX model loaded successfully");
    } catch (error) {
      console.error("Failed to load ONNX model:", error);
    }
  }

  async detect(
    imageSource: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement,
  ): Promise<Detection[]> {
    if (!this.session || !this.isLoaded) return [];

    try {
      const startTime = performance.now();

      // Preprocess with letterbox padding (maintains aspect ratio)
      const input = this.preprocess(imageSource);

      // Run inference
      const results = await this.session.run({ images: input });

      // Get output tensor — log shape to debug
      const output = results["output0"] || Object.values(results)[0];
      const data = output.data as Float32Array;
      const dims = output.dims;
      console.log("Model output shape:", dims, "data length:", data.length);
      console.log("First 12 values:", Array.from(data.slice(0, 12)));

      // YOLO26n outputs [1, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
      // OR [1, 84, 8400] where 84 = [cx, cy, w, h, class_scores...]
      const numDetections = dims[1];
      const numFeatures = dims[2];
      console.log(
        `Output has ${numDetections} detections, ${numFeatures} features each`,
      );

      const detections: Detection[] = [];

      // Calculate padding offsets for coordinate transformation
      const srcAspect = imageSource.width / imageSource.height;
      const dstAspect = INPUT_SIZE / INPUT_SIZE; // 1 (square)

      let padX = 0,
        padY = 0,
        scaleX = 1,
        scaleY = 1;

      if (srcAspect > dstAspect) {
        // Image is wider — pad top/bottom
        scaleX = imageSource.width / INPUT_SIZE;
        scaleY = imageSource.width / INPUT_SIZE;
        padY = (INPUT_SIZE - INPUT_SIZE / srcAspect) / 2;
      } else {
        // Image is taller — pad left/right
        scaleX = imageSource.height / INPUT_SIZE;
        scaleY = imageSource.height / INPUT_SIZE;
        padX = (INPUT_SIZE - INPUT_SIZE * srcAspect) / 2;
      }

      for (let i = 0; i < numDetections; i++) {
        const idx = i * 6;
        const x1 = data[idx];
        const y1 = data[idx + 1];
        const x2 = data[idx + 2];
        const y2 = data[idx + 3];
        const confidence = data[idx + 4];
        const classId = Math.round(data[idx + 5]);

        if (confidence < CONFIDENCE_THRESHOLD) continue;

        // Transform coordinates back to original image space
        const tx1 = (x1 - padX) * scaleX;
        const ty1 = (y1 - padY) * scaleY;
        const tx2 = (x2 - padX) * scaleX;
        const ty2 = (y2 - padY) * scaleY;

        detections.push({
          x: Math.max(0, tx1),
          y: Math.max(0, ty1),
          width: Math.max(0, tx2 - tx1),
          height: Math.max(0, ty2 - ty1),
          confidence,
          class: COCO_LABELS[classId] || `class_${classId}`,
        });
      }

      // Apply NMS
      const finalDetections = this.applyNMS(detections);

      // Track FPS
      const elapsed = performance.now() - startTime;
      this.inferenceTimes.push(elapsed);
      if (this.inferenceTimes.length > 30) this.inferenceTimes.shift();
      const avg =
        this.inferenceTimes.reduce((a, b) => a + b, 0) /
        this.inferenceTimes.length;
      this.fps = Math.round(1000 / avg);
      this.lastInferenceTime = elapsed;

      return finalDetections;
    } catch (err) {
      console.error("Detection error:", err);
      return [];
    }
  }

  /**
   * Preprocess image with letterbox padding to maintain aspect ratio
   */
  private preprocess(
    source: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement,
  ): ort.Tensor {
    // Create/reuse preprocess canvas
    if (!this.preprocessCanvas) {
      this.preprocessCanvas = document.createElement("canvas");
      this.preprocessCanvas.width = INPUT_SIZE;
      this.preprocessCanvas.height = INPUT_SIZE;
      this.preprocessCtx = this.preprocessCanvas.getContext("2d", {
        willReadFrequently: true,
      })!;
    }

    const ctx = this.preprocessCtx!;

    // Fill with black (letterbox padding)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

    // Calculate aspect-ratio-preserved scaling
    const srcAspect = source.width / source.height;
    let drawWidth: number, drawHeight: number, offsetX: number, offsetY: number;

    if (srcAspect > 1) {
      drawWidth = INPUT_SIZE;
      drawHeight = INPUT_SIZE / srcAspect;
      offsetX = 0;
      offsetY = (INPUT_SIZE - drawHeight) / 2;
    } else {
      drawHeight = INPUT_SIZE;
      drawWidth = INPUT_SIZE * srcAspect;
      offsetX = (INPUT_SIZE - drawWidth) / 2;
      offsetY = 0;
    }

    ctx.drawImage(
      source,
      0,
      0,
      source.width,
      source.height,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight,
    );

    // Build NCHW tensor [1, 3, 320, 320]
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data;
    const input = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
      input[i] = pixels[i * 4] / 255.0; // R
      input[INPUT_SIZE * INPUT_SIZE + i] = pixels[i * 4 + 1] / 255.0; // G
      input[2 * INPUT_SIZE * INPUT_SIZE + i] = pixels[i * 4 + 2] / 255.0; // B
    }

    return new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  }

  /**
   * Apply Non-Maximum Suppression
   */
  private applyNMS(detections: Detection[]): Detection[] {
    detections.sort((a, b) => b.confidence - a.confidence);

    const filtered: Detection[] = [];
    const used = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue;
      filtered.push(detections[i]);
      used.add(i);

      for (let j = i + 1; j < detections.length; j++) {
        if (used.has(j)) continue;
        if (this.calculateIoU(detections[i], detections[j]) > NMS_THRESHOLD) {
          used.add(j);
        }
      }
    }

    return filtered;
  }

  private calculateIoU(a: Detection, b: Detection): number {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);

    if (x2 <= x1 || y2 <= y1) return 0;

    const intersection = (x2 - x1) * (y2 - y1);
    const union = a.width * a.height + b.width * b.height - intersection;
    return intersection / union;
  }

  getIsLoaded() {
    return this.isLoaded;
  }

  getFps() {
    return this.fps;
  }

  getLastInferenceTime() {
    return this.lastInferenceTime;
  }
}
