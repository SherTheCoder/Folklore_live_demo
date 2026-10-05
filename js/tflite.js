const Interpreter = (() => {
  const OPS = {
    0: 'ADD',
    2: 'CONCATENATION',
    3: 'CONV_2D',
    4: 'DEPTHWISE_CONV_2D',
    14: 'LOGISTIC',
    22: 'RESHAPE',
    45: 'STRIDED_SLICE',
    114: 'QUANTIZE',
    129: 'CALL_ONCE',
    142: 'VAR_HANDLE',
    143: 'READ_VARIABLE',
    144: 'ASSIGN_VARIABLE',
  };

  const NO_OPTIONS = { int: (field, fallback = 0) => fallback, string: () => '' };

  const INT32 = 2;
  const UINT8 = 3;
  const INT8 = 9;

  // Just enough flatbuffer reading for the TFLite schema.
  class Table {
    constructor(view, pos) {
      this.view = view;
      this.pos = pos;
      this.vtable = pos - view.getInt32(pos, true);
      this.vsize = view.getUint16(this.vtable, true);
    }

    offset(field) {
      const at = 4 + 2 * field;
      return at < this.vsize ? this.view.getUint16(this.vtable + at, true) : 0;
    }

    int(field, fallback = 0, size = 4) {
      const o = this.offset(field);
      if (!o) return fallback;
      const at = this.pos + o;
      return size === 1 ? this.view.getInt8(at) : this.view.getInt32(at, true);
    }

    ref(field) {
      const o = this.offset(field);
      return o ? this.pos + o + this.view.getUint32(this.pos + o, true) : 0;
    }

    table(field) {
      const at = this.ref(field);
      return at ? new Table(this.view, at) : null;
    }

    tables(field) {
      const at = this.ref(field);
      if (!at) return [];
      const count = this.view.getUint32(at, true);
      return Array.from({ length: count }, (_, i) => {
        const item = at + 4 + 4 * i;
        return new Table(this.view, item + this.view.getUint32(item, true));
      });
    }

    vector(field, width = 4, read = 'getInt32') {
      const at = this.ref(field);
      if (!at) return [];
      const count = this.view.getUint32(at, true);
      return Array.from({ length: count }, (_, i) => this.view[read](at + 4 + width * i, true));
    }

    bytes(field) {
      const at = this.ref(field);
      if (!at) return null;
      return new Uint8Array(this.view.buffer, at + 4, this.view.getUint32(at, true));
    }

    string(field) {
      const bytes = this.bytes(field);
      return bytes ? new TextDecoder().decode(bytes) : '';
    }
  }

  // Runs the int8 ops a streaming micro_wake_word model needs, with the same
  // arithmetic as TFLite Micro's reference kernels.
  class Interpreter {
    constructor(buffer) {
      const view = new DataView(buffer);
      const model = new Table(view, view.getUint32(0, true));
      this.buffers = model.tables(4).map((b) => b.bytes(0));
      this.opcodes = model.tables(1).map((c) => Math.max(c.int(0, 0, 1), c.int(3)));
      this.variables = new Map();
      this.subgraphs = model.tables(2).map((g) => this.load(g));

      const main = this.subgraphs[0];
      this.input = main.tensors[main.inputs[0]];
      this.output = main.tensors[main.outputs[0]];
    }

    invoke() {
      this.run(0);
      return this.output.data;
    }

    run(index) {
      for (const step of this.subgraphs[index].steps) step();
    }

    load(graph) {
      const tensors = graph.tables(0).map((t) => this.tensor(t));
      const subgraph = { tensors, inputs: graph.vector(1), outputs: graph.vector(2), steps: [] };
      for (const op of graph.tables(3)) {
        const name = OPS[this.opcodes[op.int(0)]];
        if (!name) throw new Error(`Unsupported op ${this.opcodes[op.int(0)]}`);
        const io = { inputs: op.vector(1).map((i) => tensors[i]), outputs: op.vector(2).map((i) => tensors[i]) };
        subgraph.steps.push(this[name](io, op.table(4) ?? NO_OPTIONS));
      }
      return subgraph;
    }

    tensor(t) {
      const shape = t.vector(0);
      const type = t.int(1, 0, 1);
      const raw = this.buffers[t.int(2)];
      const q = t.table(4);
      const size = shape.reduce((a, b) => a * b, 1);
      const tensor = {
        shape,
        type,
        scales: q ? q.vector(2, 4, 'getFloat32') : [],
        zeroPoints: q ? q.vector(3, 8) : [],
      };
      tensor.scale = tensor.scales[0] ?? 0;
      tensor.zeroPoint = tensor.zeroPoints[0] ?? 0;

      const Kind = type === INT32 ? Int32Array : type === UINT8 ? Uint8Array : type === INT8 ? Int8Array : null;
      if (raw && raw.length && Kind) {
        tensor.data = new Kind(raw.slice().buffer);
      } else if (Kind) {
        tensor.data = new Kind(size);
      }
      return tensor;
    }

    CALL_ONCE(io, options) {
      const graph = options.int(0);
      let done = false;
      return () => {
        if (done) return;
        done = true;
        this.run(graph);
      };
    }

    VAR_HANDLE({ outputs: [out] }, options) {
      out.handle = `${options.string(0)}/${options.string(1)}`;
      return () => {};
    }

    ASSIGN_VARIABLE({ inputs: [handle, value] }) {
      return () => {
        const stored = this.variables.get(handle.handle);
        if (stored && stored.length === value.data.length) stored.set(value.data);
        else this.variables.set(handle.handle, value.data.slice());
      };
    }

    READ_VARIABLE({ inputs: [handle], outputs: [out] }) {
      return () => out.data.set(this.variables.get(handle.handle));
    }

    RESHAPE({ inputs: [input], outputs: [out] }) {
      return () => out.data.set(input.data);
    }

    CONCATENATION({ inputs, outputs: [out] }, options) {
      const rank = out.shape.length;
      const axis = (options.int(0) + rank) % rank;
      const outer = product(out.shape.slice(0, axis));
      const chunks = inputs.map((t) => product(t.shape.slice(axis)));
      return () => {
        let at = 0;
        for (let o = 0; o < outer; o++) {
          inputs.forEach((t, i) => {
            out.data.set(t.data.subarray(o * chunks[i], (o + 1) * chunks[i]), at);
            at += chunks[i];
          });
        }
      };
    }

    STRIDED_SLICE({ inputs: [input, begin, end, strides], outputs: [out] }, options) {
      const beginMask = options.int(0);
      const endMask = options.int(1);
      const shrinkMask = options.int(4);
      const dims = input.shape;
      const ranges = dims.map((dim, i) => {
        const step = strides.data[i];
        const clamp = (v) => Math.max(step > 0 ? 0 : -1, Math.min(v < 0 ? v + dim : v, step > 0 ? dim : dim - 1));
        const from = beginMask & (1 << i) ? (step > 0 ? 0 : dim - 1) : clamp(begin.data[i]);
        let to = endMask & (1 << i) ? (step > 0 ? dim : -1) : clamp(end.data[i]);
        if (shrinkMask & (1 << i)) to = from + 1;
        const list = [];
        for (let v = from; step > 0 ? v < to : v > to; v += step) list.push(v);
        return list;
      });
      const index = new Int32Array(out.data.length);
      let n = 0;
      const walk = (axis, base) => {
        if (axis === dims.length) {
          index[n++] = base;
          return;
        }
        for (const v of ranges[axis]) walk(axis + 1, base * dims[axis] + v);
      };
      walk(0, 0);
      return () => {
        for (let i = 0; i < index.length; i++) out.data[i] = input.data[index[i]];
      };
    }

    CONV_2D({ inputs: [input, filter, bias], outputs: [out] }, options) {
      const [, inH, inW, inC] = input.shape;
      const [outC, kH, kW] = filter.shape;
      const [, outH, outW] = out.shape;
      const geometry = convGeometry(options.int(0, 0, 1), options.int(2, 1), options.int(1, 1), options.int(5, 1), options.int(4, 1), inH, inW, kH, kW, outH, outW);
      const quant = channelQuant(input, filter, out, outC, options.int(3, 0, 1));
      const offset = -input.zeroPoint;

      return () => {
        const x = input.data;
        const w = filter.data;
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            for (let oc = 0; oc < outC; oc++) {
              let acc = bias ? bias.data[oc] : 0;
              for (let ky = 0; ky < kH; ky++) {
                const iy = oy * geometry.strideH - geometry.padH + ky * geometry.dilationH;
                if (iy < 0 || iy >= inH) continue;
                for (let kx = 0; kx < kW; kx++) {
                  const ix = ox * geometry.strideW - geometry.padW + kx * geometry.dilationW;
                  if (ix < 0 || ix >= inW) continue;
                  const xi = (iy * inW + ix) * inC;
                  const wi = ((oc * kH + ky) * kW + kx) * inC;
                  for (let ic = 0; ic < inC; ic++) acc += w[wi + ic] * (x[xi + ic] + offset);
                }
              }
              out.data[(oy * outW + ox) * outC + oc] = quant(acc, oc);
            }
          }
        }
      };
    }

    DEPTHWISE_CONV_2D({ inputs: [input, filter, bias], outputs: [out] }, options) {
      const [, inH, inW, inC] = input.shape;
      const [, kH, kW, outC] = filter.shape;
      const [, outH, outW] = out.shape;
      const multiplier = outC / inC;
      const geometry = convGeometry(options.int(0, 0, 1), options.int(2, 1), options.int(1, 1), options.int(6, 1), options.int(5, 1), inH, inW, kH, kW, outH, outW);
      const quant = channelQuant(input, filter, out, outC, options.int(4, 0, 1));
      const offset = -input.zeroPoint;

      return () => {
        const x = input.data;
        const w = filter.data;
        for (let oy = 0; oy < outH; oy++) {
          for (let ox = 0; ox < outW; ox++) {
            for (let oc = 0; oc < outC; oc++) {
              const ic = Math.floor(oc / multiplier);
              let acc = bias ? bias.data[oc] : 0;
              for (let ky = 0; ky < kH; ky++) {
                const iy = oy * geometry.strideH - geometry.padH + ky * geometry.dilationH;
                if (iy < 0 || iy >= inH) continue;
                for (let kx = 0; kx < kW; kx++) {
                  const ix = ox * geometry.strideW - geometry.padW + kx * geometry.dilationW;
                  if (ix < 0 || ix >= inW) continue;
                  acc += w[(ky * kW + kx) * outC + oc] * (x[(iy * inW + ix) * inC + ic] + offset);
                }
              }
              out.data[(oy * outW + ox) * outC + oc] = quant(acc, oc);
            }
          }
        }
      };
    }

    ADD({ inputs: [a, b], outputs: [out] }, options) {
      const twiceMax = 2 * Math.max(a.scale, b.scale);
      const [aMul, aShift] = quantizeMultiplier(a.scale / twiceMax);
      const [bMul, bShift] = quantizeMultiplier(b.scale / twiceMax);
      const [outMul, outShift] = quantizeMultiplier(twiceMax / (2 ** 20 * out.scale));
      const [lo, hi] = activationRange(options.int(0, 0, 1), out);

      return () => {
        const x = a.data;
        const y = b.data;
        for (let i = 0; i < out.data.length; i++) {
          const sx = multiply((x[i % x.length] - a.zeroPoint) * 1048576, aMul, aShift);
          const sy = multiply((y[i % y.length] - b.zeroPoint) * 1048576, bMul, bShift);
          const v = multiply(sx + sy, outMul, outShift) + out.zeroPoint;
          out.data[i] = v < lo ? lo : v > hi ? hi : v;
        }
      };
    }

    LOGISTIC({ inputs: [input], outputs: [out] }) {
      const [mul, shift] = frexp(input.scale * 2 ** 27);
      const multiplier = roundAway(mul * 2 ** 31);
      const radius = Math.floor((15 * 2 ** 27) / 2 ** shift);
      const table = lookup((v) => {
        const x = v - input.zeroPoint;
        if (x <= -radius) return -128;
        if (x >= radius) return 127;
        const y = roundingShift(logistic(multiply(x, multiplier, shift)), 23) - 128;
        return Math.min(Math.max(y, -128), 127);
      });
      return () => applyTable(table, input.data, out.data);
    }

    QUANTIZE({ inputs: [input], outputs: [out] }) {
      const [mul, shift] = quantizeMultiplier(input.scale / out.scale);
      const [lo, hi] = out.type === UINT8 ? [0, 255] : [-128, 127];
      const table = lookup((v) => {
        const y = multiply(v - input.zeroPoint, mul, shift) + out.zeroPoint;
        return Math.min(Math.max(y, lo), hi);
      });
      return () => applyTable(table, input.data, out.data);
    }
  }

  function product(list) {
    return list.reduce((a, b) => a * b, 1);
  }

  function lookup(fn) {
    const table = new Int32Array(256);
    for (let v = -128; v < 128; v++) table[v & 255] = fn(v);
    return table;
  }

  function applyTable(table, input, output) {
    for (let i = 0; i < input.length; i++) output[i] = table[input[i] & 255];
  }

  function convGeometry(padding, strideH, strideW, dilationH, dilationW, inH, inW, kH, kW, outH, outW) {
    const pad = (size, stride, dilation, kernel, out) => {
      if (padding !== 0) return 0;
      const effective = (kernel - 1) * dilation + 1;
      return Math.max(Math.floor(((out - 1) * stride + effective - size) / 2), 0);
    };
    return {
      strideH,
      strideW,
      dilationH,
      dilationW,
      padH: pad(inH, strideH, dilationH, kH, outH),
      padW: pad(inW, strideW, dilationW, kW, outW),
    };
  }

  function channelQuant(input, filter, out, channels, activation) {
    const multipliers = new Int32Array(channels);
    const shifts = new Int32Array(channels);
    for (let c = 0; c < channels; c++) {
      const scale = filter.scales.length > 1 ? filter.scales[c] : filter.scale;
      [multipliers[c], shifts[c]] = quantizeMultiplier((input.scale * scale) / out.scale);
    }
    const [lo, hi] = activationRange(activation, out);
    const zero = out.zeroPoint;
    return (acc, c) => {
      const v = multiply(acc, multipliers[c], shifts[c]) + zero;
      return v < lo ? lo : v > hi ? hi : v;
    };
  }

  function activationRange(activation, tensor) {
    const q = (f) => tensor.zeroPoint + roundAway(Math.fround(f / tensor.scale));
    if (activation === 1) return [Math.max(-128, q(0)), 127];
    if (activation === 2) return [Math.max(-128, q(-1)), Math.min(127, q(1))];
    if (activation === 3) return [Math.max(-128, q(0)), Math.min(127, q(6))];
    return [-128, 127];
  }

  function roundAway(x) {
    return Math.sign(x) * Math.round(Math.abs(x));
  }

  function frexp(x) {
    if (x === 0) return [0, 0];
    let exp = Math.ceil(Math.log2(Math.abs(x)));
    let mant = x / 2 ** exp;
    while (Math.abs(mant) >= 1) {
      mant /= 2;
      exp++;
    }
    while (Math.abs(mant) < 0.5) {
      mant *= 2;
      exp--;
    }
    return [mant, exp];
  }

  function quantizeMultiplier(real) {
    if (real === 0) return [0, 0];
    let [mant, shift] = frexp(real);
    let fixed = roundAway(mant * 2 ** 31);
    if (fixed === 2 ** 31) {
      fixed /= 2;
      shift++;
    }
    if (shift < -31) return [0, 0];
    return [fixed, shift];
  }

  // gemmlowp's SaturatingRoundingDoublingHighMul. a * b can need 62 bits, so the
  // product is split in two to stay exact in doubles.
  function highMul(a, b) {
    if (a === -2147483648 && b === -2147483648) return 2147483647;
    const negative = a < 0 !== b < 0;
    const x = Math.abs(a);
    const y = Math.abs(b);
    const high = x * Math.floor(y / 65536);
    const low = x * (y % 65536) + (negative ? 1073741823 : 1073741824);
    const q = Math.floor(high / 32768);
    const m = q + Math.floor(((high - q * 32768) * 65536 + low) / 2147483648);
    return negative ? -m | 0 : m | 0;
  }

  function roundingShift(x, exponent) {
    const mask = (1 << exponent) - 1;
    const threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + ((x & mask) > threshold ? 1 : 0);
  }

  function multiply(x, multiplier, shift) {
    const left = shift > 0 ? shift : 0;
    return roundingShift(highMul(Math.imul(x, 1 << left), multiplier), left - shift);
  }

  // gemmlowp's fixed point logistic, which TFLite Micro uses for int8 sigmoid.
  function logistic(a) {
    if (a === 0) return 1 << 30;
    const positive = a > 0;
    const result = oneOverOnePlus(expNegative(positive ? -a : a));
    return positive ? result : (2147483647 - result) | 0;
  }

  function expNegative(a) {
    const quarter = 1 << 25;
    const modQuarter = ((a & (quarter - 1)) - quarter) | 0;
    let result = expQuarter(saturatingShift(modQuarter, 4));
    const remainder = (modQuarter - a) | 0;
    const multipliers = [1672461947, 1302514674, 790015084, 290630308, 39332535, 720401];
    multipliers.forEach((m, i) => {
      if (remainder & (1 << (25 + i))) result = highMul(result, m);
    });
    return result;
  }

  function expQuarter(a) {
    const constant = 1895147668;
    const third = 715827883;
    const x = (a + (1 << 28)) | 0;
    const x2 = highMul(x, x);
    const x3 = highMul(x2, x);
    const x4 = highMul(x2, x2);
    const poly = roundingShift((highMul((roundingShift(x4, 2) + x3) | 0, third) + x2) | 0, 1);
    return (constant + highMul(constant, (x + poly) | 0)) | 0;
  }

  function oneOverOnePlus(a) {
    const sum = a + 2147483647;
    const halfDenominator = Math.trunc((sum + (sum >= 0 ? 1 : -1)) / 2);
    let x = (1515870810 + highMul(halfDenominator, -1010580540)) | 0;
    for (let i = 0; i < 3; i++) {
      const rest = ((1 << 29) - highMul(halfDenominator, x)) | 0;
      x = (x + saturatingShift(highMul(x, rest), 2)) | 0;
    }
    return saturatingShift(x, 1);
  }

  function saturatingShift(x, exponent) {
    const limit = 2 ** (31 - exponent) - 1;
    if (x > limit) return 2147483647;
    if (x < -limit) return -2147483648;
    return x << exponent;
  }

  return Interpreter;
})();
