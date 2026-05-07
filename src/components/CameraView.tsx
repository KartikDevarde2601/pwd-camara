import { forwardRef, useImperativeHandle, useRef, useEffect } from "react";

interface CameraViewProps {
  style?: any;
  facing?: "front" | "back";
  active?: boolean;
  onPermissionResult?: (granted: boolean) => void;
}

export const CameraView = forwardRef((props: CameraViewProps, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useImperativeHandle(ref, () => ({
    takePictureAsync: async (options?: { quality?: number }) => {
      if (!videoRef.current || !videoRef.current.videoWidth) {
        throw new Error("Camera not ready");
      }

      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not create canvas context");

      ctx.drawImage(videoRef.current, 0, 0);
      const uri = canvas.toDataURL("image/jpeg", options?.quality || 0.8);

      return {
        uri,
        width: canvas.width,
        height: canvas.height,
      };
    },
  }));

  useEffect(() => {
    if (!props.active) return;

    async function startCamera() {
      try {
        const constraints = {
          video: {
            facingMode: props.facing === "back" ? "environment" : "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        props.onPermissionResult?.(true);
      } catch (err) {
        console.error("Error accessing camera:", err);
        props.onPermissionResult?.(false);
      }
    }

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [props.active, props.facing]);

  return (
    <div
      style={{
        ...props.style,
        overflow: "hidden",
        position: "relative",
        backgroundColor: "#000",
      }}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: props.facing === "front" ? "scaleX(-1)" : "none",
        }}
      />
    </div>
  );
});
