import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
}

const COCO_LABELS = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

export class YoloDetector {
  private model: tf.GraphModel | null = null;
  private isLoaded = false;
  private preprocessCanvas: HTMLCanvasElement | null = null;
  private preprocessCtx: CanvasRenderingContext2D | null = null;
  private lastInferenceTime = 0;
  private inferenceTimes: number[] = [];
  private fps = 0;

  async loadModel(modelPath: string = "/models/yolo26n_web_model/model.json") {
    try {
      console.log("Setting TF.js backend to: webgl");
      await tf.setBackend("webgl");
      await tf.ready();

      console.log("Loading TFJS model from:", modelPath);
      this.model = await tf.loadGraphModel(modelPath);

      // Warm up
      const warmup = tf.zeros([1, 320, 320, 3]);
      this.model.predict(warmup);
      warmup.dispose();

      this.isLoaded = true;
      console.log("TFJS model loaded successfully");
    } catch (error) {
      console.error("Failed to load TFJS model:", error);
    }
  }

  async detect(
    imageSource: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement,
  ): Promise<Detection[]> {
    if (!this.model || !this.isLoaded) return [];

    try {
      const startTime = performance.now();

      // Preprocess to 320x320 with letterbox (matches TFJS model input)
      const inputTensor = this.preprocess(imageSource);

      // Run inference
      const rawOutput = await this.model.predict(inputTensor);

      // Model has 2 outputs: [Identity:0 = bbox_data, TopKV2:0 = class_indices]
      // Handle both array and single tensor return
      let outputTensor: tf.Tensor;
      if (Array.isArray(rawOutput)) {
        outputTensor = rawOutput[0]; // Identity:0 — the main detection output
        // Dispose unused tensors
        for (let i = 1; i < rawOutput.length; i++) {
          rawOutput[i].dispose();
        }
      } else {
        outputTensor = rawOutput as tf.Tensor;
      }

      const outputData = (await outputTensor.data()) as Float32Array;
      const outputShape = outputTensor.shape;

      console.log(
        `Model output shape: [${outputShape}], first 10 values:`,
        Array.from(outputData.slice(0, 10)),
      );

      // Determine output format — likely [1, num_detections, 6] = [x1,y1,x2,y2,conf,class_id]
      // OR [1, 84, 8400] for standard YOLO
      const numDetections = outputShape[1];
      const numFeatures = outputShape[2];

      const detections: Detection[] = [];
      const srcAspect = imageSource.width / imageSource.height;

      // Calculate letterbox padding offsets
      let padX = 0,
        padY = 0,
        scaleX: number,
        scaleY: number;

      if (srcAspect > 1) {
        scaleX = imageSource.width / 320;
        scaleY = imageSource.width / 320;
        padY = (320 - 320 / srcAspect) / 2;
      } else {
        scaleX = imageSource.height / 320;
        scaleY = imageSource.height / 320;
        padX = (320 - 320 * srcAspect) / 2;
      }

      for (let i = 0; i < numDetections; i++) {
        const idx = i * 6;
        const x1 = outputData[idx];
        const y1 = outputData[idx + 1];
        const x2 = outputData[idx + 2];
        const y2 = outputData[idx + 3];
        const confidence = outputData[idx + 4];

        if (confidence < 0.5) continue;

        const tx1 = (x1 - padX) * scaleX;
        const ty1 = (y1 - padY) * scaleY;
        const tx2 = (x2 - padX) * scaleX;
        const ty2 = (y2 - padY) * scaleY;

        const classId = Math.round(outputData[idx + 5]);
        const label = COCO_LABELS[classId] || `class_${classId}`;

        detections.push({
          x: Math.max(0, tx1),
          y: Math.max(0, ty1),
          width: Math.max(1, tx2 - tx1),
          height: Math.max(1, ty2 - ty1),
          confidence,
          class: label,
        });
      }

      // Cleanup tensors
      inputTensor.dispose();
      outputTensor.dispose();

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

  private preprocess(
    source: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement,
  ): tf.Tensor4D {
    if (!this.preprocessCanvas) {
      this.preprocessCanvas = document.createElement("canvas");
      this.preprocessCanvas.width = 320;
      this.preprocessCanvas.height = 320;
      this.preprocessCtx = this.preprocessCanvas.getContext("2d", {
        willReadFrequently: true,
      })!;
    }

    const ctx = this.preprocessCtx!;

    // Fill black background
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, 320, 320);

    // Letterbox resize maintaining aspect ratio
    const srcAspect = source.width / source.height;
    let drawWidth: number, drawHeight: number, offsetX: number, offsetY: number;

    if (srcAspect > 1) {
      drawWidth = 320;
      drawHeight = 320 / srcAspect;
      offsetX = 0;
      offsetY = (320 - drawHeight) / 2;
    } else {
      drawHeight = 320;
      drawWidth = 320 * srcAspect;
      offsetX = (320 - drawWidth) / 2;
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

    // TF.js expects NHWC format [1, 320, 320, 3]
    return tf.tidy(() => {
      return tf.browser
        .fromPixels(this.preprocessCanvas!)
        .toFloat()
        .div(255.0)
        .expandDims(0) as tf.Tensor4D;
    });
  }

  private applyNMS(detections: Detection[]): Detection[] {
    if (detections.length <= 1) return detections;

    detections.sort((a, b) => b.confidence - a.confidence);
    const filtered: Detection[] = [];
    const used = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue;
      filtered.push(detections[i]);
      used.add(i);

      for (let j = i + 1; j < detections.length; j++) {
        if (used.has(j)) continue;
        if (this.calculateIoU(detections[i], detections[j]) > 0.4) {
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
