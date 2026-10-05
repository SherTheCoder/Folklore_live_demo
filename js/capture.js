// Runs on the audio thread. app.js loads this function's source as the worklet module.
function captureWorklet() {
  const TARGET_RATE = 16000;
  const CHUNK = 480;

  // Only needed when the browser won't hand us 16 kHz (Firefox). Same filter as
  // the terminal tester: windowed sinc low-pass at 7.2 kHz, then linear interpolation.
  class Resampler {
    constructor(rate) {
      this.ratio = rate / TARGET_RATE;
      const n = 2 * Math.floor(8 * this.ratio) + 1;
      this.taps = new Float64Array(n);
      let total = 0;
      for (let i = 0; i < n; i++) {
        const x = ((2 * 7200) / rate) * (i - (n >> 1));
        const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
        this.taps[i] = sinc * (0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1)));
        total += this.taps[i];
      }
      this.taps = this.taps.map((t) => t / total);
      this.history = new Float64Array(n - 1);
      this.filtered = new Float64Array(0);
      this.previous = 0;
      this.position = 0;
    }

    process(input, emit) {
      const { taps, history } = this;
      const size = input.length;
      if (this.filtered.length !== size) this.filtered = new Float64Array(size);
      const filtered = this.filtered;

      for (let j = 0; j < size; j++) {
        let acc = 0;
        for (let k = 0; k < taps.length; k++) {
          const at = j + k;
          acc += (at < history.length ? history[at] : input[at - history.length]) * taps[k];
        }
        filtered[j] = acc;
      }
      const keep = history.length;
      if (size >= keep) {
        history.set(input.subarray(size - keep));
      } else {
        history.copyWithin(0, size);
        history.set(input, keep - size);
      }

      let p = this.position;
      for (; p < size - 1; p += this.ratio) {
        const i = Math.floor(p);
        const a = i < 0 ? this.previous : filtered[i];
        emit(a + (p - i) * (filtered[i + 1] - a));
      }
      this.position = p - size;
      this.previous = filtered[size - 1];
    }
  }

  class Capture extends AudioWorkletProcessor {
    constructor() {
      super();
      this.resampler = sampleRate === TARGET_RATE ? null : new Resampler(sampleRate);
      this.chunk = new Int16Array(CHUNK);
      this.length = 0;
      this.emit = this.emit.bind(this);
    }

    emit(value) {
      const v = Math.round(value * 32768);
      this.chunk[this.length++] = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
      if (this.length === CHUNK) {
        this.port.postMessage(this.chunk, [this.chunk.buffer]);
        this.chunk = new Int16Array(CHUNK);
        this.length = 0;
      }
    }

    process(inputs) {
      const channel = inputs[0][0];
      if (!channel) return true;
      if (this.resampler) this.resampler.process(channel, this.emit);
      else for (let i = 0; i < channel.length; i++) this.emit(channel[i]);
      return true;
    }
  }

  registerProcessor('capture', Capture);
}
