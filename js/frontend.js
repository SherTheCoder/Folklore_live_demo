const AudioFrontend = (() => {
  const SAMPLE_RATE = 16000;
  const STEP = 160;

  const WINDOW = 480;
  const FFT = 512;
  const CHANNELS = 40;
  const LOWER_HZ = 125;
  const UPPER_HZ = 7500;
  const SMOOTHING_BITS = 10;
  const CORRECTION_BITS = 3;

  const LOG_LUT = [
    0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163, 2329, 2490, 2646, 2797,
    2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848, 3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633,
    4714, 4791, 4864, 4934, 5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
    5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633, 5626, 5615, 5602, 5586,
    5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370, 5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000,
    4944, 4885, 4825, 4762, 4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
    3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659, 2549, 2437, 2323, 2207,
    2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224, 1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
  ];

  const f32 = Math.fround;
  const s16 = (x) => (x << 16) >> 16;
  const msb = (x) => (x === 0 ? 0 : 32 - Math.clz32(x));
  const mel = (hz) => f32(1127 * f32(Math.log1p(f32(hz / 700))));

  // Port of the TFLite Micro audio frontend that ESPHome runs. Every step rounds
  // like the C code, so the features match the device bit for bit.
  class Frontend {
    constructor() {
      this.window = hann(WINDOW);
      this.fft = new RealFFT(FFT);
      this.bank = filterbank(CHANNELS, LOWER_HZ, UPPER_HZ, FFT / 2 + 1);
      this.gainLut = pcanLut(SMOOTHING_BITS - CORRECTION_BITS);
      this.input = new Int16Array(WINDOW);
      this.frame = new Int16Array(FFT);
      this.estimate = new Float64Array(CHANNELS);
      this.signal = new Float64Array(CHANNELS);
      this.used = 0;
    }

    // 10 ms of audio in, 40 features out (or null until the first 30 ms window is full).
    push(chunk) {
      this.input.set(chunk, this.used);
      this.used += STEP;
      if (this.used < WINDOW) return null;

      let peak = 0;
      for (let i = 0; i < WINDOW; i++) {
        const v = s16((this.input[i] * this.window[i]) >> 12);
        this.frame[i] = v;
        const a = s16(v < 0 ? -v : v);
        if (a > peak) peak = a;
      }
      this.input.copyWithin(0, STEP);
      this.used -= STEP;

      const shift = 15 - msb(peak);
      for (let i = 0; i < WINDOW; i++) this.frame[i] = this.frame[i] << shift;
      this.frame.fill(0, WINDOW);
      const { re, im } = this.fft.run(this.frame);

      this.accumulate(re, im, shift);
      this.reduceNoise();
      this.applyPcan();
      return this.logScale();
    }

    accumulate(re, im, shift) {
      const { starts, weights, unweights } = this.bank;
      let carry = 0;
      for (let ch = 0; ch <= CHANNELS; ch++) {
        let sum = carry;
        carry = 0;
        for (let j = 0, bin = starts[ch]; j < weights[ch].length; j++, bin++) {
          const energy = re[bin] * re[bin] + im[bin] * im[bin];
          sum += weights[ch][j] * energy;
          carry += unweights[ch][j] * energy;
        }
        if (ch > 0) this.signal[ch - 1] = Math.floor(isqrt(sum) / 2 ** shift);
      }
    }

    reduceNoise() {
      for (let i = 0; i < CHANNELS; i++) {
        const smoothing = i & 1 ? 983 : 409;
        const x = this.signal[i];
        const scaled = (x * 2 ** SMOOTHING_BITS) % 4294967296;
        let estimate = Math.floor((scaled * smoothing + this.estimate[i] * (16384 - smoothing)) / 16384) % 4294967296;
        this.estimate[i] = estimate;
        if (estimate > scaled) estimate = scaled;
        const floor = Math.floor((x * 819) / 16384) % 4294967296;
        const subtracted = Math.floor((scaled - estimate) / 2 ** SMOOTHING_BITS);
        this.signal[i] = Math.max(subtracted, floor);
      }
    }

    applyPcan() {
      for (let i = 0; i < CHANNELS; i++) {
        let gain = wideDynamic(this.estimate[i], this.gainLut);
        if (gain < 0) gain += 4294967296;
        const snr = Math.floor((this.signal[i] * gain) / 64) % 4294967296;
        this.signal[i] = snr < 8192 ? Math.floor((snr * snr) / 1048576) : Math.floor(snr / 64) - 64;
      }
    }

    logScale() {
      const out = new Uint16Array(CHANNELS);
      for (let i = 0; i < CHANNELS; i++) {
        const value = (this.signal[i] * 2 ** CORRECTION_BITS) % 4294967296;
        out[i] = value > 1 ? Math.min(log(value), 65535) : 0;
      }
      return out;
    }
  }

  function quantizeFeatures(values, out, offset = 0) {
    for (let i = 0; i < values.length; i++) {
      const v = Math.floor((values[i] * 256 + 333) / 666) - 128;
      out[offset + i] = v < -128 ? -128 : v > 127 ? 127 : v;
    }
  }

  function hann(size) {
    const arg = f32(f32(f32(Math.PI) * 2) / size);
    const coeffs = new Int16Array(size);
    for (let i = 0; i < size; i++) {
      const v = f32(0.5 - f32(0.5 * f32(Math.cos(f32(arg * f32(i + 0.5))))));
      coeffs[i] = Math.floor(f32(f32(v * 4096) + 0.5));
    }
    return coeffs;
  }

  // Each FFT bin splits its energy between two neighbouring mel channels.
  function filterbank(channels, lower, upper, spectrum) {
    const melLow = mel(lower);
    const spacing = f32(f32(mel(upper) - melLow) / (channels + 1));
    const centers = [];
    for (let i = 0; i <= channels; i++) centers.push(f32(melLow + f32(spacing * (i + 1))));

    const hzPerBin = f32(f32(0.5 * SAMPLE_RATE) / f32(spectrum - 1));
    const starts = [];
    const weights = [];
    const unweights = [];
    let bin = Math.trunc(f32(1.5 + f32(lower / hzPerBin)));

    for (let ch = 0; ch <= channels; ch++) {
      const below = ch === 0 ? melLow : centers[ch - 1];
      const w = [];
      const u = [];
      starts.push(bin);
      while (mel(f32(bin * hzPerBin)) <= centers[ch]) {
        const weight = f32(f32(centers[ch] - mel(f32(bin * hzPerBin))) / f32(centers[ch] - below));
        w.push(Math.floor(f32(f32(weight * 4096) + 0.5)));
        u.push(Math.floor(f32(f32(f32(1 - weight) * 4096) + 0.5)));
        bin++;
      }
      weights.push(w);
      unweights.push(u);
    }
    return { starts, weights, unweights };
  }

  function pcanLut(inputBits) {
    const gain = (x) => {
      const v = f32(2097152 * f32(Math.pow(f32(f32(f32(x) / 2 ** inputBits) + 80), f32(-0.95))));
      return v > 32767 ? 32767 : Math.trunc(f32(v + 0.5));
    };
    const lut = new Int16Array(4 * 32 - 3);
    lut[0] = gain(0);
    lut[1] = gain(1);
    for (let interval = 2; interval <= 32; interval++) {
      const x0 = 2 ** (interval - 1);
      const x1 = x0 + x0 / 2;
      const x2 = interval === 32 ? x0 + (x0 - 1) : 2 * x0;
      const y0 = gain(x0);
      const a1 = 4 * (gain(x1) - y0) - (gain(x2) - y0);
      const a2 = gain(x2) - y0 - a1;
      const at = 4 * interval - 6;
      lut[at] = y0;
      lut[at + 1] = a1;
      lut[at + 2] = a2;
    }
    return lut;
  }

  function wideDynamic(x, lut) {
    if (x <= 2) return lut[x];
    const interval = msb(x >>> 0);
    const at = 4 * interval - 6;
    const frac = (interval < 11 ? x * 2 ** (11 - interval) : x >>> (interval - 11)) & 0x3ff;
    let result = (lut[at + 2] * frac) >> 5;
    result = (result + (lut[at + 1] << 5)) | 0;
    result = Math.imul(result, frac);
    result = (result + 16384) >> 15;
    return s16(result + lut[at]);
  }

  // FilterbankSqrt, done on doubles so 64-bit sums stay exact.
  function isqrt(num) {
    if (num === 0) return 0;
    const wide = num >= 4294967296;
    const bits = wide ? 32 + msb(Math.floor(num / 4294967296)) : msb(num);
    let maxBit = ((wide ? 64 : 32) - bits) | 1;
    let bit = 2 ** ((wide ? 63 : 31) - maxBit);
    let iterations = Math.floor(((wide ? 63 : 31) - maxBit) / 2) + 1;
    let res = 0;
    while (iterations--) {
      if (num >= res + bit) {
        num -= res + bit;
        res = Math.floor(res / 2) + bit;
      } else {
        res = Math.floor(res / 2);
      }
      bit /= 4;
    }
    if (num > res && res !== (wide ? 4294967295 : 65535)) res++;
    return res;
  }

  function log(x) {
    const integer = msb(x) - 1;
    let frac = x - 2 ** integer;
    frac = integer < 16 ? frac * 2 ** (16 - integer) : Math.floor(frac / 2 ** (integer - 16));
    const seg = frac >> 9;
    const c0 = LOG_LUT[seg];
    const c1 = LOG_LUT[seg + 1];
    const fraction = frac + c0 + (((c1 - c0) * (frac - seg * 512)) >> 16);
    const log2 = integer * 65536 + fraction;
    const loge = Math.floor((45426 * log2 + 32768) / 65536);
    return Math.floor((loge * 64 + 32768) / 65536);
  }

  // kissfft's 16-bit fixed point real FFT. The Int16Arrays do the int16 wraparound.
  class RealFFT {
    constructor(size) {
      const n = size / 2;
      this.n = n;
      this.twRe = new Int16Array(n);
      this.twIm = new Int16Array(n);
      for (let i = 0; i < n; i++) {
        const phase = (-2 * Math.PI * i) / n;
        this.twRe[i] = Math.floor(0.5 + 32767 * Math.cos(phase));
        this.twIm[i] = Math.floor(0.5 + 32767 * Math.sin(phase));
      }
      this.superRe = new Int16Array(n / 2);
      this.superIm = new Int16Array(n / 2);
      for (let i = 0; i < n / 2; i++) {
        const phase = -Math.PI * ((i + 1) / n + 0.5);
        this.superRe[i] = Math.floor(0.5 + 32767 * Math.cos(phase));
        this.superIm[i] = Math.floor(0.5 + 32767 * Math.sin(phase));
      }
      this.factors = [];
      for (let m = n; m > 1; ) {
        const p = m % 4 === 0 ? 4 : 2;
        m /= p;
        this.factors.push(p, m);
      }
      this.inRe = new Int16Array(n);
      this.inIm = new Int16Array(n);
      this.bufRe = new Int16Array(n);
      this.bufIm = new Int16Array(n);
      this.re = new Int16Array(n + 1);
      this.im = new Int16Array(n + 1);
    }

    run(samples) {
      const { n, inRe, inIm, bufRe: tr, bufIm: ti, re, im } = this;
      for (let i = 0; i < n; i++) {
        inRe[i] = samples[2 * i];
        inIm[i] = samples[2 * i + 1];
      }
      this.work(0, 0, 1, 0);

      const dcRe = half(tr[0]);
      const dcIm = half(ti[0]);
      re[0] = dcRe + dcIm;
      re[n] = dcRe - dcIm;
      im[0] = im[n] = 0;

      for (let k = 1; k <= n / 2; k++) {
        const aRe = half(tr[k]);
        const aIm = half(ti[k]);
        const bRe = half(tr[n - k]);
        const bIm = half(s16(-ti[n - k]));
        const sumRe = s16(aRe + bRe);
        const sumIm = s16(aIm + bIm);
        const dRe = s16(aRe - bRe);
        const dIm = s16(aIm - bIm);
        const wRe = this.superRe[k - 1];
        const wIm = this.superIm[k - 1];
        const tRe = s16((dRe * wRe - dIm * wIm + 16384) >> 15);
        const tIm = s16((dRe * wIm + dIm * wRe + 16384) >> 15);
        re[k] = (sumRe + tRe) >> 1;
        im[k] = (sumIm + tIm) >> 1;
        re[n - k] = (sumRe - tRe) >> 1;
        im[n - k] = (tIm - sumIm) >> 1;
      }
      return this;
    }

    work(out, from, stride, f) {
      const p = this.factors[f];
      const m = this.factors[f + 1];
      const end = out + p * m;
      if (m === 1) {
        for (let k = out; k < end; k++, from += stride) {
          this.bufRe[k] = this.inRe[from];
          this.bufIm[k] = this.inIm[from];
        }
      } else {
        for (let k = out; k < end; k += m, from += stride) this.work(k, from, stride * p, f + 2);
      }
      if (p === 4) this.radix4(out, stride, m);
      else this.radix2(out, stride, m);
    }

    radix2(out, stride, m) {
      const re = this.bufRe;
      const im = this.bufIm;
      for (let k = 0; k < m; k++) {
        const a = out + k;
        const b = a + m;
        re[a] = half(re[a]);
        im[a] = half(im[a]);
        re[b] = half(re[b]);
        im[b] = half(im[b]);
        const wRe = this.twRe[k * stride];
        const wIm = this.twIm[k * stride];
        const tRe = s16((re[b] * wRe - im[b] * wIm + 16384) >> 15);
        const tIm = s16((re[b] * wIm + im[b] * wRe + 16384) >> 15);
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
      }
    }

    radix4(out, stride, m) {
      const re = this.bufRe;
      const im = this.bufIm;
      const twRe = this.twRe;
      const twIm = this.twIm;
      for (let k = 0; k < m; k++) {
        const a = out + k;
        const b = a + m;
        const c = b + m;
        const d = c + m;
        for (let i = a; i <= d; i += m) {
          re[i] = quarter(re[i]);
          im[i] = quarter(im[i]);
        }
        const w1 = k * stride;
        const w2 = 2 * w1;
        const w3 = 3 * w1;
        const s0r = s16((re[b] * twRe[w1] - im[b] * twIm[w1] + 16384) >> 15);
        const s0i = s16((re[b] * twIm[w1] + im[b] * twRe[w1] + 16384) >> 15);
        const s1r = s16((re[c] * twRe[w2] - im[c] * twIm[w2] + 16384) >> 15);
        const s1i = s16((re[c] * twIm[w2] + im[c] * twRe[w2] + 16384) >> 15);
        const s2r = s16((re[d] * twRe[w3] - im[d] * twIm[w3] + 16384) >> 15);
        const s2i = s16((re[d] * twIm[w3] + im[d] * twRe[w3] + 16384) >> 15);
        const s5r = s16(re[a] - s1r);
        const s5i = s16(im[a] - s1i);
        re[a] += s1r;
        im[a] += s1i;
        const s3r = s16(s0r + s2r);
        const s3i = s16(s0i + s2i);
        const s4r = s16(s0r - s2r);
        const s4i = s16(s0i - s2i);
        re[c] = re[a] - s3r;
        im[c] = im[a] - s3i;
        re[a] += s3r;
        im[a] += s3i;
        re[b] = s5r + s4i;
        im[b] = s5i - s4r;
        re[d] = s5r - s4i;
        im[d] = s5i + s4r;
      }
    }
  }

  function half(x) {
    return s16((x * 16383 + 16384) >> 15);
  }

  function quarter(x) {
    return s16((x * 8191 + 16384) >> 15);
  }

  return { SAMPLE_RATE, STEP, Frontend, quantizeFeatures };
})();
