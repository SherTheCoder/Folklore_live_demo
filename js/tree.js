const KikkarTree = (() => {
  const WIDTH = 1000;
  const HEIGHT = 400;
  const GROUND = 386;
  const STEPS = 9;
  const VARIANTS = 3;

  function seeded(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const smooth = (a, b, v) => {
    const t = clamp((v - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
  };

  // The trunk and leaves are painted once per resize. Flowers, glow and pollen are
  // redrawn every frame; bloom runs from 0 (buds) to 1 (fully open).
  class KikkarTree {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.target = 0;
      this.bloom = 0;
      this.flash = 0;
      this.idle = true;
      this.pollen = [];
      this.visible = true;
      this.last = 0;

      const rand = seeded(2026);
      this.shape = grow(rand);
      this.flowers = placeFlowers(this.shape.nodes, rand);
      const xs = this.shape.nodes.map((n) => n.x);
      const ys = this.shape.nodes.map((n) => n.y);
      this.middle = (Math.min(...xs) + Math.max(...xs)) / 2;
      this.crown = (Math.min(...ys) + Math.max(...ys)) / 2;
      this.sprites = Array.from({ length: VARIANTS }, (_, v) => flowerSprites(seeded(77 + v)));
      this.glow = glowSprite();

      new ResizeObserver(() => this.resize()).observe(canvas);
      new IntersectionObserver(([entry]) => {
        this.visible = entry.isIntersecting;
        if (this.visible) this.loop();
      }).observe(canvas);
    }

    set level(value) {
      this.idle = false;
      this.target = clamp(value, 0, 1);
    }

    rest() {
      this.idle = true;
    }

    wake() {
      this.flash = 1;
      if (this.calm) return;
      for (const flower of this.flowers) {
        for (let i = 0; i < 6; i++) this.puff(flower, 2.2 + Math.random() * 2.5);
      }
    }

    resize() {
      const ratio = Math.min(devicePixelRatio || 1, 2);
      const width = Math.round(this.canvas.clientWidth * ratio);
      const height = Math.round(this.canvas.clientHeight * ratio);
      if (!width || !height) return;
      this.canvas.width = width;
      this.canvas.height = height;
      const scale = Math.min(height / HEIGHT, width / (WIDTH * 0.72));
      this.view = { scale, x: width / 2 - this.middle * scale, y: height - HEIGHT * scale };
      this.backdrop = this.paintBackdrop(width, height);
      this.loop();
    }

    loop() {
      if (this.running || !this.visible || !this.view) return;
      this.running = true;
      requestAnimationFrame((time) => this.frame(time));
    }

    frame(time) {
      this.running = false;
      if (!this.visible) return;
      const dt = Math.min((time - (this.last || time)) / 1000, 0.1);
      this.last = time;
      this.update(dt, time / 1000);
      this.draw(time / 1000);
      this.loop();
    }

    update(dt, t) {
      const goal = this.idle ? 0.12 + (this.calm ? 0 : 0.06 * Math.sin(t * 1.3)) : this.target;
      const rate = goal > this.bloom ? 9 : 2.5;
      this.bloom += (goal - this.bloom) * Math.min(1, rate * dt);
      this.flash = Math.max(0, this.flash - dt / 1.8);
      if (this.calm) return;

      if (Math.random() < dt * (1.5 + 10 * this.bloom)) {
        this.puff(this.flowers[Math.floor(Math.random() * this.flowers.length)], 0.4);
      }
      for (const p of this.pollen) {
        p.vx += -p.vy * p.curl * dt;
        p.vy += p.curl * p.vx * dt - 6 * dt;
        p.vx *= 1 - 0.9 * dt;
        p.vy *= 1 - 0.9 * dt;
        p.px = p.x;
        p.py = p.y;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.life -= dt;
      }
      this.pollen = this.pollen.filter((p) => p.life > 0);
    }

    puff(flower, speed) {
      const angle = Math.random() * Math.PI * 2;
      const v = speed * (20 + Math.random() * 30);
      this.pollen.push({
        x: flower.x,
        y: flower.y,
        px: flower.x,
        py: flower.y,
        vx: Math.cos(angle) * v,
        vy: Math.sin(angle) * v - 10,
        curl: (Math.random() - 0.5) * 3,
        life: 1.2 + Math.random() * 2.2,
        size: 0.8 + Math.random() * 1.6,
      });
    }

    draw(t) {
      const { ctx, view } = this;
      const open = Math.max(this.bloom, this.flash);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
      ctx.drawImage(this.backdrop, 0, 0);

      ctx.setTransform(view.scale, 0, 0, view.scale, view.x, view.y);
      ctx.globalCompositeOperation = 'lighter';
      const halo = ctx.createRadialGradient(this.middle, this.crown, 0, this.middle, this.crown, 380);
      halo.addColorStop(0, `rgba(255, 170, 60, ${0.08 + 0.22 * open})`);
      halo.addColorStop(1, 'rgba(255, 140, 40, 0)');
      ctx.fillStyle = halo;
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      for (const f of this.flowers) {
        const b = Math.max(smooth(f.wakes, f.wakes + 0.4, this.bloom), this.flash);
        const sway = this.calm ? 0 : Math.sin(t * 1.1 + f.phase) * 1.2;
        const size = f.r * (1 + 0.25 * this.flash);
        ctx.globalAlpha = 0.25 + 0.6 * b;
        const g = size * 3.4;
        ctx.drawImage(this.glow, f.x - g, f.y + sway - g, g * 2, g * 2);
      }

      ctx.globalCompositeOperation = 'source-over';
      for (const f of this.flowers) {
        const b = Math.max(smooth(f.wakes, f.wakes + 0.4, this.bloom), this.flash);
        const sway = this.calm ? 0 : Math.sin(t * 1.1 + f.phase) * 1.2;
        const size = f.r * (1 + 0.25 * this.flash);
        const sprite = this.sprites[f.variant][Math.round(b * (STEPS - 1))];
        ctx.globalAlpha = 1;
        ctx.save();
        ctx.translate(f.x, f.y + sway);
        ctx.rotate(f.turn);
        ctx.drawImage(sprite, -size * 1.2, -size * 1.2, size * 2.4, size * 2.4);
        ctx.restore();
      }

      ctx.globalCompositeOperation = 'lighter';
      ctx.lineCap = 'round';
      for (const p of this.pollen) {
        const fade = Math.min(1, p.life / 0.8);
        ctx.globalAlpha = fade * 0.9;
        ctx.strokeStyle = '#ffc24a';
        ctx.lineWidth = p.size * 0.6;
        ctx.beginPath();
        ctx.moveTo(p.px, p.py);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        ctx.fillStyle = '#ffe9a8';
        ctx.fillRect(p.x - p.size / 2, p.y - p.size / 2, p.size, p.size);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
    }

    paintBackdrop(width, height) {
      const layer = document.createElement('canvas');
      layer.width = width;
      layer.height = height;
      const ctx = layer.getContext('2d');
      const sky = ctx.createLinearGradient(0, 0, 0, height);
      sky.addColorStop(0, '#0a0605');
      sky.addColorStop(1, '#120a06');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, width, height);

      const rand = seeded(5);
      for (let i = 0; i < 90; i++) {
        ctx.globalAlpha = 0.15 + rand() * 0.45;
        ctx.fillStyle = rand() < 0.3 ? '#ffd98a' : '#f3e7cf';
        const r = (0.4 + rand() * 0.9) * Math.max(1, width / 900);
        ctx.beginPath();
        ctx.arc(rand() * width, rand() * height * 0.75, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      const { scale, x, y } = this.view;
      ctx.setTransform(scale, 0, 0, scale, x, y);
      paintLimbs(ctx, this.shape.limbs);
      paintLeaves(ctx, this.shape.nodes, seeded(31));

      ctx.fillStyle = '#0d0806';
      ctx.beginPath();
      ctx.moveTo(-600, HEIGHT + 40);
      ctx.lineTo(-600, GROUND);
      ctx.bezierCurveTo(150, GROUND - 14, 850, GROUND - 14, 1600, GROUND);
      ctx.lineTo(1600, HEIGHT + 40);
      ctx.fill();
      ctx.strokeStyle = 'rgba(226, 161, 27, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(-600, GROUND);
      ctx.bezierCurveTo(150, GROUND - 14, 850, GROUND - 14, 1600, GROUND);
      ctx.stroke();
      return layer;
    }
  }

  // A kikkar forks low into two long limbs, then spreads into a wide, flat crown.
  function grow(rand) {
    const limbs = [];
    const nodes = [];
    const branch = (x, y, angle, length, width, depth) => {
      const bend = angle + (rand() - 0.5) * 0.7;
      const ex = x + Math.cos(angle) * length;
      const ey = y - Math.sin(angle) * length;
      const end = depth === 0 ? width * 0.5 : width * 0.66;
      limbs.push({ x, y, cx: x + Math.cos(bend) * length * 0.55, cy: y - Math.sin(bend) * length * 0.55, ex, ey, from: width, to: end });
      if (depth <= 2) nodes.push({ x: ex, y: ey, angle, tip: depth === 0 });
      if (depth === 0) return;
      const count = depth > 4 ? 2 : rand() < 0.5 ? 3 : 2;
      for (let i = 0; i < count; i++) {
        const side = i / (count - 1) - 0.5;
        let a = angle + side * (depth === 6 ? 1.5 : 1 + rand() * 0.6) + (rand() - 0.5) * 0.3;
        a += (a - Math.PI / 2) * (depth > 2 ? 0.45 : 0.05);
        a = clamp(a, 0.15, Math.PI - 0.15);
        const shrink = depth > 4 ? 0.9 : 0.68 + rand() * 0.14;
        branch(ex, ey, a, length * shrink, end, depth - 1);
      }
    };
    branch(500, GROUND + 2, Math.PI / 2 + 0.03, 87, 46, 6);
    return { limbs, nodes };
  }

  function paintLimbs(ctx, limbs) {
    ctx.lineCap = 'round';
    const pieces = 8;
    const point = (l, t) => [
      (1 - t) * (1 - t) * l.x + 2 * (1 - t) * t * l.cx + t * t * l.ex,
      (1 - t) * (1 - t) * l.y + 2 * (1 - t) * t * l.cy + t * t * l.ey,
    ];
    for (const pass of [
      { color: '#3a2615', size: 1, shift: 0, cap: 'round' },
      { color: 'rgba(196, 140, 82, 0.4)', size: 0.22, shift: -0.28, cap: 'butt' },
    ]) {
      ctx.strokeStyle = pass.color;
      ctx.lineCap = pass.cap;
      for (const l of limbs) {
        for (let i = 0; i < pieces; i++) {
          const [x0, y0] = point(l, i / pieces);
          const [x1, y1] = point(l, (i + 1) / pieces);
          const width = l.from + (l.to - l.from) * ((i + 0.5) / pieces);
          ctx.lineWidth = Math.max(width * pass.size, 0.7);
          ctx.beginPath();
          ctx.moveTo(x0 + width * pass.shift, y0);
          ctx.lineTo(x1 + width * pass.shift, y1);
          ctx.stroke();
        }
      }
    }
  }

  function paintLeaves(ctx, nodes, rand) {
    for (const node of nodes) {
      const shade = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 46);
      shade.addColorStop(0, 'rgba(24, 34, 12, 0.85)');
      shade.addColorStop(1, 'rgba(24, 34, 12, 0)');
      ctx.fillStyle = shade;
      ctx.fillRect(node.x - 46, node.y - 46, 92, 92);
    }
    ctx.lineCap = 'round';
    for (const [colors, share] of [
      [['#2c3f15', '#33481a'], 1],
      [['#4d6726', '#5b7a2e', '#6b8a36'], 0.75],
    ]) {
      for (const node of nodes) {
        if (rand() > share) continue;
        const sprays = node.tip ? 3 + Math.floor(rand() * 2) : 2;
        for (let s = 0; s < sprays; s++) {
          spray(ctx, node.x, node.y, node.angle + (rand() - 0.5) * 2.6, 28 + rand() * 30, colors[Math.floor(rand() * colors.length)], rand);
        }
      }
    }
  }

  function spray(ctx, x, y, angle, length, color, rand) {
    const dx = Math.cos(angle);
    const dy = -Math.sin(angle);
    const pairs = 5 + Math.floor(rand() * 3);
    ctx.strokeStyle = '#3b2a17';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dx * length, y + dy * length);
    ctx.stroke();
    for (let p = 1; p <= pairs; p++) {
      const at = (p / (pairs + 0.6)) * length;
      const bx = x + dx * at;
      const by = y + dy * at;
      const pinna = 14 * (1 - (p / (pairs + 1)) * 0.5);
      for (const side of [-1, 1]) {
        const a = angle + side * 1.05;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3.4;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + Math.cos(a) * pinna, by - Math.sin(a) * pinna);
        ctx.stroke();
        ctx.strokeStyle = 'rgba(18, 26, 6, 0.75)';
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
    }
  }

  function placeFlowers(nodes, rand) {
    const flowers = [];
    const shuffled = [...nodes].sort(() => rand() - 0.5);
    for (const node of shuffled) {
      const x = node.x + Math.cos(node.angle) * 8 + (rand() - 0.5) * 16;
      const y = node.y - Math.sin(node.angle) * 8 + (rand() - 0.5) * 12;
      if (flowers.some((f) => Math.hypot(f.x - x, f.y - y) < 44)) continue;
      flowers.push({
        x,
        y,
        r: 15 + rand() * 7,
        turn: rand() * Math.PI * 2,
        phase: rand() * Math.PI * 2,
        variant: Math.floor(rand() * VARIANTS),
        wakes: rand() * 0.6,
      });
    }
    return flowers;
  }

  function flowerSprites(rand) {
    const size = 96;
    const r = size / 2.4;
    const filaments = Array.from({ length: 54 }, () => ({
      angle: rand() * Math.PI * 2,
      reach: 0.55 + rand() * 0.45,
      depth: rand(),
    })).sort((a, b) => a.depth - b.depth);

    return Array.from({ length: STEPS }, (_, step) => {
      const open = step / (STEPS - 1);
      const sprite = document.createElement('canvas');
      sprite.width = sprite.height = size;
      const ctx = sprite.getContext('2d');
      ctx.translate(size / 2, size / 2);
      ctx.lineCap = 'round';

      for (const f of filaments) {
        const length = r * f.reach * (0.32 + 0.68 * open);
        const x = Math.cos(f.angle) * length;
        const y = Math.sin(f.angle) * length;
        const light = 0.55 + 0.45 * f.depth;
        const stem = ctx.createLinearGradient(0, 0, x, y);
        stem.addColorStop(0, `rgba(214, 112, 20, ${light})`);
        stem.addColorStop(1, `rgba(255, 214, 110, ${light})`);
        ctx.strokeStyle = stem;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.fillStyle = `rgba(255, ${190 + Math.round(50 * f.depth)}, 70, ${0.7 + 0.3 * light})`;
        ctx.beginPath();
        ctx.arc(x, y, 2.6 + 0.8 * (1 - open), 0, Math.PI * 2);
        ctx.fill();
      }

      const core = ctx.createRadialGradient(0, 0, 0, 0, 0, r * 0.3);
      core.addColorStop(0, '#ffb43c');
      core.addColorStop(1, 'rgba(226, 110, 20, 0)');
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.3, 0, Math.PI * 2);
      ctx.fill();
      return sprite;
    });
  }

  function glowSprite() {
    const glow = document.createElement('canvas');
    glow.width = glow.height = 128;
    const ctx = glow.getContext('2d');
    const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255, 196, 80, 0.55)');
    g.addColorStop(0.35, 'rgba(255, 150, 40, 0.18)');
    g.addColorStop(1, 'rgba(255, 120, 20, 0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 128, 128);
    return glow;
  }

  return KikkarTree;
})();
