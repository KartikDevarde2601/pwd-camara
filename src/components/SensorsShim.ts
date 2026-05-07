interface OrientationListener {
  (data: { alpha: number; beta: number; gamma: number }): void;
}

/** Provides orientation angles (alpha, beta, gamma) from DeviceOrientationEvent */
class OrientationEmitter {
  private listeners: Set<OrientationListener> = new Set();
  private timer: any = null;

  constructor() {
    window.addEventListener("deviceorientation", this.handleOrientation);
  }

  private handleOrientation = (event: DeviceOrientationEvent) => {
    const data = {
      alpha: event.alpha || 0,
      beta: event.beta || 0,
      gamma: event.gamma || 0,
    };
    this.listeners.forEach((l) => l(data));
  };

  addListener(listener: OrientationListener) {
    this.listeners.add(listener);
    return {
      remove: () => this.listeners.delete(listener),
    };
  }
}

export const Orientation = new OrientationEmitter();
