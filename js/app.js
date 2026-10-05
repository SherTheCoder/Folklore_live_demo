const CUTOFF = 0.65;
const HISTORY = 334;
const { SAMPLE_RATE } = AudioFrontend;

const $ = (id) => document.getElementById(id);
const micButton = $('mic');
const statusLine = $('status');
const tree = new KikkarTree($('tree'));
const model = decodeModel(MODEL_BASE64);

let live = null;

micButton.disabled = false;
micButton.textContent = 'Start listening';
micButton.addEventListener('click', () => (live ? stopListening() : startListening()));
new ResizeObserver(() => drawTrace()).observe($('trace'));

function decodeModel(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function say(text, error = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle('error', error);
}

// The AudioContext is created before the first await so Safari still counts it as part of the click.
async function startListening() {
  if (!navigator.mediaDevices?.getUserMedia) {
    say('This browser only allows the microphone on https or localhost pages.', true);
    return;
  }
  micButton.disabled = true;
  say('Waiting for the microphone…');
  let context = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    let source = connect(context, stream);
    if (!source) {
      context.close();
      context = new AudioContext();
      source = context.createMediaStreamSource(stream);
    }
    await loadWorklet(context);
    await context.resume();
    const node = new AudioWorkletNode(context, 'capture');
    source.connect(node).connect(context.destination);

    live = {
      stream,
      context,
      wakeword: new WakeWord(model, CUTOFF),
      levels: new Float32Array(HISTORY),
      marks: new Uint8Array(HISTORY),
      head: 0,
      micDb: -90,
      wakes: 0,
      started: performance.now(),
      pluck: null,
    };
    node.port.onmessage = (event) => hear(event.data);

    micButton.textContent = 'Stop';
    showWakes(0);
    const rate = context.sampleRate === SAMPLE_RATE ? '' : ` Your mic runs at ${context.sampleRate / 1000} kHz, so it's resampled to 16 kHz here.`;
    say(`Listening.${rate}`);
    requestAnimationFrame(draw);
  } catch (error) {
    context.close();
    say(micError(error), true);
  } finally {
    micButton.disabled = false;
  }
}

// Firefox can't connect a mic to a context running at another sample rate. It then
// gets a context at the mic's own rate, and the worklet does the resampling.
function connect(context, stream) {
  try {
    return context.createMediaStreamSource(stream);
  } catch {
    return null;
  }
}

// Chrome won't load a blob: worklet on a page opened from disk, but a data: URL works.
async function loadWorklet(context) {
  const code = `(${captureWorklet})();`;
  try {
    await context.audioWorklet.addModule(URL.createObjectURL(new Blob([code], { type: 'text/javascript' })));
  } catch {
    await context.audioWorklet.addModule(`data:text/javascript,${encodeURIComponent(code)}`);
  }
}

function micError(error) {
  if (error.name === 'NotAllowedError') return 'The microphone is blocked for this page. Allow it in the site settings and try again.';
  if (error.name === 'NotFoundError') return 'No microphone was found.';
  return `Could not start listening: ${error.message}`;
}

function stopListening() {
  const { stream, context, wakes, started } = live;
  stream.getTracks().forEach((track) => track.stop());
  context.close();
  live = null;
  micButton.textContent = 'Start listening';
  $('mic-bar').style.transform = 'scaleX(0)';
  tree.rest();

  const minutes = (performance.now() - started) / 60000;
  say(`Stopped after ${minutes.toFixed(1)} min with ${wakes} wake${wakes === 1 ? '' : 's'}.`);
}

function hear(samples) {
  let energy = 0;
  for (let i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
  const db = 20 * Math.log10(Math.max(Math.sqrt(energy / samples.length), 1) / 32768);
  live.micDb = Math.max(db, live.micDb - 4.5);

  live.wakeword.feed(samples, (result) => {
    live.levels[live.head] = result.level;
    live.marks[live.head] = result.woke ? 1 : 0;
    live.head = (live.head + 1) % HISTORY;
    if (result.woke) wake(result.level);
  });
}

function wake(level) {
  showWakes(++live.wakes);
  say(`Woke at ${new Date().toLocaleTimeString()} with a score of ${level.toFixed(2)}.`);
  tree.wake();

  if ($('sound').checked) {
    live.pluck ??= tumbi(live.context);
    const voice = live.context.createBufferSource();
    voice.buffer = live.pluck;
    voice.connect(live.context.destination);
    voice.start();
  }
}

// A Karplus-Strong pluck, close enough to a tumbi.
function tumbi(context) {
  const rate = context.sampleRate;
  const buffer = context.createBuffer(1, Math.floor(rate * 0.8), rate);
  const data = buffer.getChannelData(0);
  const string = Float32Array.from({ length: Math.round(rate / 660) }, () => Math.random() * 2 - 1);
  for (let i = 0, at = 0; i < data.length; i++) {
    const next = (at + 1) % string.length;
    data[i] = string[at] * 0.4;
    string[at] = 0.497 * (string[at] + string[next]);
    at = next;
  }
  return buffer;
}

function showWakes(count) {
  $('wakes').textContent = `${count} wake${count === 1 ? '' : 's'}`;
}

function draw() {
  if (!live) return;
  const latest = live.levels[(live.head + HISTORY - 1) % HISTORY];
  tree.level = latest / CUTOFF;
  $('mic-bar').style.transform = `scaleX(${Math.min(Math.max((live.micDb + 70) / 70, 0), 1)})`;
  drawTrace();
  requestAnimationFrame(draw);
}

function drawTrace() {
  const canvas = $('trace');
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  if (!width || !height) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  const pad = 8 * ratio;
  const y = (v) => height - pad - v * (height - 2 * pad);
  const x = (i) => (i / (HISTORY - 1)) * width;
  const at = (i) => (live ? (live.head + i) % HISTORY : 0);
  const level = (i) => (live ? live.levels[at(i)] : 0);
  ctx.clearRect(0, 0, width, height);

  ctx.lineWidth = ratio;
  ctx.strokeStyle = 'rgba(224, 106, 87, 0.9)';
  ctx.setLineDash([6 * ratio, 5 * ratio]);
  ctx.beginPath();
  ctx.moveTo(0, y(CUTOFF));
  ctx.lineTo(width, y(CUTOFF));
  ctx.stroke();
  ctx.setLineDash([]);

  const fill = ctx.createLinearGradient(0, y(1), 0, y(0));
  fill.addColorStop(0, 'rgba(242, 182, 60, 0.35)');
  fill.addColorStop(1, 'rgba(242, 182, 60, 0)');
  ctx.beginPath();
  ctx.moveTo(0, y(0));
  for (let i = 0; i < HISTORY; i++) ctx.lineTo(x(i), y(level(i)));
  ctx.lineTo(width, y(0));
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.lineWidth = 2 * ratio;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#f2b63c';
  ctx.beginPath();
  for (let i = 0; i < HISTORY; i++) ctx[i ? 'lineTo' : 'moveTo'](x(i), y(level(i)));
  ctx.stroke();

  if (!live) return;
  ctx.fillStyle = '#ffe08a';
  for (let i = 0; i < HISTORY; i++) {
    if (!live.marks[at(i)]) continue;
    ctx.beginPath();
    ctx.arc(x(i), y(level(i)), 4.5 * ratio, 0, 2 * Math.PI);
    ctx.fill();
  }
}
