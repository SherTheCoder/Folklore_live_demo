const WakeWord = (() => {
  const { Frontend, quantizeFeatures, STEP } = AudioFrontend;

  // ESPHome's rule: wake when the mean of the last 3 outputs passes the cutoff,
  // then ignore everything until it has stayed below the cutoff for a second.
  class Detector {
    constructor(cutoff, window = 3) {
      this.window = window;
      this.ring = new Array(window).fill(0);
      this.last = 0;
      this.ignore = -100;
      this.level = 0;
      this.threshold = Math.trunc(cutoff * 255);
    }

    slice(output) {
      if (output !== null) {
        this.last = (this.last + 1) % this.window;
        this.ring[this.last] = output;
        this.level = this.sum() / (this.window * 256);
      }
      if (this.ring[this.last] < this.threshold) this.ignore = Math.min(this.ignore + 1, 0);
      if (output === null || this.ignore < 0 || this.sum() <= this.threshold * this.window) return false;
      this.ring.fill(0);
      this.ignore = -100;
      return true;
    }

    sum() {
      return this.ring.reduce((a, b) => a + b, 0);
    }
  }

  class WakeWord {
    constructor(model, cutoff) {
      this.model = new Interpreter(model);
      this.frontend = new Frontend();
      this.detector = new Detector(cutoff);
      this.stride = this.model.input.shape[1];
      this.chunk = new Int16Array(STEP);
      this.buffered = 0;
      this.filled = 0;
    }

    feed(samples, onResult) {
      for (let i = 0; i < samples.length; ) {
        const take = Math.min(STEP - this.buffered, samples.length - i);
        this.chunk.set(samples.subarray(i, i + take), this.buffered);
        this.buffered += take;
        i += take;
        if (this.buffered < STEP) break;
        this.buffered = 0;
        this.step(onResult);
      }
    }

    step(onResult) {
      const features = this.frontend.push(this.chunk);
      if (!features) return;
      quantizeFeatures(features, this.model.input.data, this.filled * features.length);

      let output = null;
      if (++this.filled === this.stride) {
        this.filled = 0;
        output = this.model.invoke()[0];
      }
      const woke = this.detector.slice(output);
      if (output !== null) onResult({ level: this.detector.level, woke });
    }
  }

  return WakeWord;
})();
