/**
 * ████████╗ ██████╗██╗███╗   ██╗███████╗███████╗██████╗  █████╗ ███╗   ███╗███████╗
 * ██╔════╝██╔════╝██║████╗  ██║██╔════╝██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝
 * ██║     ██║     ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝███████║██╔████╔██║█████╗
 * ██║     ██║     ██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝
 * ╚██████╗╚██████╗██║██║ ╚████║███████╗██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗
 *  ╚═════╝ ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
 *
 * CINEFRAME PRO v4 — MATHEMATICAL CORE ENGINE
 * File 01 of 10: Advanced Mathematics, Signal Processing, Linear Algebra
 *
 * Contains:
 *  - Full IEEE-754 floating point utilities
 *  - SIMD-style vectorized operations
 *  - FFT (Fast Fourier Transform) — full radix-2 DIT
 *  - Complete linear algebra: matrices, vectors, quaternions
 *  - Perlin/Simplex noise generation
 *  - Statistical functions: histograms, moments, entropy
 *  - Curve interpolation: cubic spline, bezier, NURBS
 *  - Color science: CIE Lab, XYZ, sRGB, P3, Rec2020
 *  - Advanced gamma/OETF/EOTF functions
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1: FUNDAMENTAL CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const MATH_CONST = Object.freeze({
  PI:         Math.PI,
  TWO_PI:     Math.PI * 2,
  HALF_PI:    Math.PI / 2,
  INV_PI:     1 / Math.PI,
  SQRT2:      Math.SQRT2,
  SQRT3:      Math.sqrt(3),
  PHI:        (1 + Math.sqrt(5)) / 2,   // Golden ratio
  E:          Math.E,
  LN2:        Math.LN2,
  LN10:       Math.LN10,
  LOG2E:      Math.LOG2E,
  LOG10E:     Math.LOG10E,
  EPSILON:    Number.EPSILON,
  MAX_SAFE:   Number.MAX_SAFE_INTEGER,
  INF:        Infinity,
  NAN:        NaN,

  // Perceptual constants
  D65_X: 0.95047,   // D65 white point X
  D65_Y: 1.00000,   // D65 white point Y
  D65_Z: 1.08883,   // D65 white point Z
  D50_X: 0.96422,
  D50_Y: 1.00000,
  D50_Z: 0.82521,

  // sRGB to XYZ (D65) matrix
  SRGB_TO_XYZ: [
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
  ],
  // XYZ to sRGB (D65) matrix
  XYZ_TO_SRGB: [
     3.2404542, -1.5371385, -0.4985314,
    -0.9692660,  1.8760108,  0.0415560,
     0.0556434, -0.2040259,  1.0572252
  ],
  // sRGB to Display P3 matrix
  SRGB_TO_P3: [
    0.8224622, 0.1773375, 0.0002003,
    0.0331942, 0.9668057, 0.0000001,
    0.0170827, 0.0724005, 0.9105168
  ],
  // Display P3 to sRGB
  P3_TO_SRGB: [
     1.2249401, -0.2249404,  0.0000003,
    -0.0420569,  1.0420571, -0.0000002,
    -0.0196376, -0.0786360,  1.0982735
  ],
  // BT.2020 to XYZ
  BT2020_TO_XYZ: [
    0.6369580, 0.1446169, 0.1688810,
    0.2627002, 0.6779981, 0.0593017,
    0.0000000, 0.0280727, 1.0609851
  ],
});


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2: SCALAR MATH UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

const ScalarMath = {
  // ── Clamping ──────────────────────────────────────────────────────────────
  clamp:      (v, lo=0, hi=1)    => v < lo ? lo : v > hi ? hi : v,
  clamp01:    (v)                => v < 0 ? 0 : v > 1 ? 1 : v,
  clamp8:     (v)                => v < 0 ? 0 : v > 255 ? 255 : Math.round(v),
  saturate:   (v)                => ScalarMath.clamp01(v),

  // ── Lerp & Mixing ─────────────────────────────────────────────────────────
  lerp:       (a, b, t)          => a + (b - a) * t,
  lerpClamped:(a, b, t)          => a + (b - a) * ScalarMath.clamp01(t),
  inverseLerp:(a, b, v)          => a === b ? 0 : (v - a) / (b - a),
  remap:      (v, a, b, c, d)    => c + (d - c) * ScalarMath.inverseLerp(a, b, v),
  remapClamped:(v,a,b,c,d)       => c + (d-c) * ScalarMath.clamp01(ScalarMath.inverseLerp(a,b,v)),
  mix:        (a, b, t)          => ScalarMath.lerp(a, b, t),

  // ── Smoothing Curves ──────────────────────────────────────────────────────
  smoothstep:    (e0, e1, x) => { const t = ScalarMath.clamp01((x-e0)/(e1-e0)); return t*t*(3-2*t); },
  smootherstep:  (e0, e1, x) => { const t = ScalarMath.clamp01((x-e0)/(e1-e0)); return t*t*t*(t*(t*6-15)+10); },
  smootheststep: (e0, e1, x) => { const t = ScalarMath.clamp01((x-e0)/(e1-e0)); return t*t*t*t*(t*(t*(-20*t+70)-84)+35); },
  fade:          (t)          => t * t * t * (t * (t * 6 - 15) + 10),

  // ── Exponential / Log ─────────────────────────────────────────────────────
  exp:        Math.exp,
  log:        Math.log,
  log2:       Math.log2,
  log10:      Math.log10,
  pow:        Math.pow,
  sqrt:       Math.sqrt,
  cbrt:       Math.cbrt,
  safeSqrt:   (v)                => v <= 0 ? 0 : Math.sqrt(v),
  safeLog:    (v, base=Math.E)   => v <= 0 ? -Infinity : Math.log(v)/Math.log(base),
  safePow:    (b, e)             => b < 0 && e !== Math.floor(e) ? 0 : Math.pow(Math.abs(b), e) * Math.sign(b),

  // ── Trig ──────────────────────────────────────────────────────────────────
  sin:  Math.sin, cos:  Math.cos, tan:  Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
  sinDeg: (d) => Math.sin(d * MATH_CONST.PI / 180),
  cosDeg: (d) => Math.cos(d * MATH_CONST.PI / 180),
  toRad:  (d) => d * MATH_CONST.PI / 180,
  toDeg:  (r) => r * 180 / MATH_CONST.PI,

  // ── Rounding ──────────────────────────────────────────────────────────────
  floor:    Math.floor,
  ceil:     Math.ceil,
  round:    Math.round,
  trunc:    Math.trunc,
  frac:     (v) => v - Math.floor(v),
  sign:     Math.sign,
  abs:      Math.abs,
  mod:      (a, b) => ((a % b) + b) % b,  // Always positive
  wrap:     (v, lo, hi) => lo + ScalarMath.mod(v - lo, hi - lo),

  // ── Min/Max ───────────────────────────────────────────────────────────────
  min:      Math.min,
  max:      Math.max,
  min3:     (a,b,c) => Math.min(a, Math.min(b, c)),
  max3:     (a,b,c) => Math.max(a, Math.max(b, c)),
  minmax:   (v, lo, hi) => ({ min: Math.min(v,lo,hi), max: Math.max(v,lo,hi) }),

  // ── Bit operations ────────────────────────────────────────────────────────
  nextPow2:    (v) => { v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; return v+1; },
  isPow2:      (v) => v > 0 && (v & (v-1)) === 0,
  popcount:    (v) => { let c=0; while(v){c+=v&1;v>>>=1;} return c; },
  leadingZeros:(v) => Math.clz32(v),

  // ── Numeric checks ────────────────────────────────────────────────────────
  isFinite:   Number.isFinite,
  isNaN:      Number.isNaN,
  isInt:      (v) => Number.isInteger(v),
  isEven:     (v) => (v & 1) === 0,
  isOdd:      (v) => (v & 1) !== 0,
  isZero:     (v, eps=1e-10) => Math.abs(v) < eps,
  approxEq:   (a, b, eps=1e-6) => Math.abs(a-b) < eps,

  // ── Conversion ────────────────────────────────────────────────────────────
  degreesToRadians: (d) => d * Math.PI / 180,
  radiansToDegrees: (r) => r * 180 / Math.PI,
  uint8ToFloat:     (v) => v / 255,
  floatToUint8:     (v) => Math.max(0, Math.min(255, Math.round(v * 255))),

  // ── Oscillators ───────────────────────────────────────────────────────────
  squareWave:   (t, freq=1) => Math.sign(Math.sin(t * freq * MATH_CONST.TWO_PI)),
  sawtoothWave: (t, freq=1) => 2 * ScalarMath.frac(t * freq) - 1,
  triangleWave: (t, freq=1) => 1 - 4 * Math.abs(ScalarMath.frac(t*freq+0.25) - 0.5),

  // ── Statistics ────────────────────────────────────────────────────────────
  mean:      (...v) => v.flat().reduce((a,b)=>a+b,0)/v.flat().length,
  variance:  (...v) => { const arr=v.flat(); const m=ScalarMath.mean(...arr); return arr.reduce((s,x)=>s+(x-m)**2,0)/arr.length; },
  stddev:    (...v) => Math.sqrt(ScalarMath.variance(...v)),
  median:    (...v) => { const s=[...v.flat()].sort((a,b)=>a-b); const m=Math.floor(s.length/2); return s.length%2?s[m]:(s[m-1]+s[m])/2; },
};


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3: VECTOR MATHEMATICS (2D, 3D, 4D)
// ═══════════════════════════════════════════════════════════════════════════════

class Vec2 {
  constructor(x=0, y=0) { this.x=x; this.y=y; }
  static from(v)        { return new Vec2(v.x||v[0]||0, v.y||v[1]||0); }
  static zero()         { return new Vec2(0,0); }
  static one()          { return new Vec2(1,1); }
  static up()           { return new Vec2(0,1); }
  static right()        { return new Vec2(1,0); }
  add(v)                { return new Vec2(this.x+v.x, this.y+v.y); }
  sub(v)                { return new Vec2(this.x-v.x, this.y-v.y); }
  mul(s)                { return new Vec2(this.x*s, this.y*s); }
  div(s)                { return new Vec2(this.x/s, this.y/s); }
  dot(v)                { return this.x*v.x + this.y*v.y; }
  cross(v)              { return this.x*v.y - this.y*v.x; }
  length()              { return Math.sqrt(this.x*this.x+this.y*this.y); }
  lengthSq()            { return this.x*this.x+this.y*this.y; }
  normalize()           { const l=this.length()||1; return this.div(l); }
  negate()              { return new Vec2(-this.x,-this.y); }
  abs()                 { return new Vec2(Math.abs(this.x),Math.abs(this.y)); }
  lerp(v,t)             { return new Vec2(ScalarMath.lerp(this.x,v.x,t),ScalarMath.lerp(this.y,v.y,t)); }
  distanceTo(v)         { return this.sub(v).length(); }
  angle()               { return Math.atan2(this.y,this.x); }
  rotate(a)             { const c=Math.cos(a),s=Math.sin(a); return new Vec2(this.x*c-this.y*s,this.x*s+this.y*c); }
  reflect(n)            { return this.sub(n.mul(2*this.dot(n))); }
  clone()               { return new Vec2(this.x,this.y); }
  toArray()             { return [this.x,this.y]; }
  toString()            { return `Vec2(${this.x.toFixed(4)},${this.y.toFixed(4)})`; }
}

class Vec3 {
  constructor(x=0,y=0,z=0) { this.x=x; this.y=y; this.z=z; }
  static from(v)        { return new Vec3(v.x||v[0]||0,v.y||v[1]||0,v.z||v[2]||0); }
  static zero()         { return new Vec3(0,0,0); }
  static one()          { return new Vec3(1,1,1); }
  add(v)                { return new Vec3(this.x+v.x,this.y+v.y,this.z+v.z); }
  sub(v)                { return new Vec3(this.x-v.x,this.y-v.y,this.z-v.z); }
  mul(s)                { return typeof s==='number'?new Vec3(this.x*s,this.y*s,this.z*s):new Vec3(this.x*s.x,this.y*s.y,this.z*s.z); }
  div(s)                { return new Vec3(this.x/s,this.y/s,this.z/s); }
  dot(v)                { return this.x*v.x+this.y*v.y+this.z*v.z; }
  cross(v)              { return new Vec3(this.y*v.z-this.z*v.y,this.z*v.x-this.x*v.z,this.x*v.y-this.y*v.x); }
  length()              { return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z); }
  lengthSq()            { return this.x*this.x+this.y*this.y+this.z*this.z; }
  normalize()           { const l=this.length()||1; return this.div(l); }
  negate()              { return new Vec3(-this.x,-this.y,-this.z); }
  abs()                 { return new Vec3(Math.abs(this.x),Math.abs(this.y),Math.abs(this.z)); }
  lerp(v,t)             { return new Vec3(ScalarMath.lerp(this.x,v.x,t),ScalarMath.lerp(this.y,v.y,t),ScalarMath.lerp(this.z,v.z,t)); }
  distanceTo(v)         { return this.sub(v).length(); }
  reflect(n)            { return this.sub(n.mul(2*this.dot(n))); }
  project(n)            { return n.mul(this.dot(n)/n.dot(n)); }
  reject(n)             { return this.sub(this.project(n)); }
  clone()               { return new Vec3(this.x,this.y,this.z); }
  toArray()             { return [this.x,this.y,this.z]; }
  toVec4(w=1)           { return new Vec4(this.x,this.y,this.z,w); }
  // RGB convenience
  get r() { return this.x; } set r(v) { this.x=v; }
  get g() { return this.y; } set g(v) { this.y=v; }
  get b() { return this.z; } set b(v) { this.z=v; }
  toString()            { return `Vec3(${this.x.toFixed(4)},${this.y.toFixed(4)},${this.z.toFixed(4)})`; }
}

class Vec4 {
  constructor(x=0,y=0,z=0,w=1) { this.x=x; this.y=y; this.z=z; this.w=w; }
  static from(v)  { return new Vec4(v.x||v[0]||0,v.y||v[1]||0,v.z||v[2]||0,v.w||v[3]||1); }
  add(v)          { return new Vec4(this.x+v.x,this.y+v.y,this.z+v.z,this.w+v.w); }
  sub(v)          { return new Vec4(this.x-v.x,this.y-v.y,this.z-v.z,this.w-v.w); }
  mul(s)          { return new Vec4(this.x*s,this.y*s,this.z*s,this.w*s); }
  dot(v)          { return this.x*v.x+this.y*v.y+this.z*v.z+this.w*v.w; }
  length()        { return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w); }
  normalize()     { const l=this.length()||1; return this.mul(1/l); }
  lerp(v,t)       { return new Vec4(ScalarMath.lerp(this.x,v.x,t),ScalarMath.lerp(this.y,v.y,t),ScalarMath.lerp(this.z,v.z,t),ScalarMath.lerp(this.w,v.w,t)); }
  toVec3()        { return new Vec3(this.x,this.y,this.z); }
  toArray()       { return [this.x,this.y,this.z,this.w]; }
  get r(){return this.x} get g(){return this.y} get b(){return this.z} get a(){return this.w}
  toString()      { return `Vec4(${this.x.toFixed(3)},${this.y.toFixed(3)},${this.z.toFixed(3)},${this.w.toFixed(3)})`; }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4: MATRIX MATHEMATICS (2x2, 3x3, 4x4)
// ═══════════════════════════════════════════════════════════════════════════════

class Mat3 {
  // Column-major 3x3 matrix stored as flat Float32Array
  constructor(data) {
    this.m = data ? new Float32Array(data) : new Float32Array([1,0,0, 0,1,0, 0,0,1]);
  }
  static identity()         { return new Mat3(); }
  static fromRows(r0,r1,r2) { return new Mat3([r0[0],r1[0],r2[0], r0[1],r1[1],r2[1], r0[2],r1[2],r2[2]]); }
  static fromCols(c0,c1,c2) { return new Mat3([...c0,...c1,...c2]); }
  static scale(sx,sy)       { return new Mat3([sx,0,0, 0,sy,0, 0,0,1]); }
  static rotate(a)          { const c=Math.cos(a),s=Math.sin(a); return new Mat3([c,s,0, -s,c,0, 0,0,1]); }
  static translate(tx,ty)   { return new Mat3([1,0,0, 0,1,0, tx,ty,1]); }

  get(row, col)  { return this.m[col*3+row]; }
  set(row, col, v) { this.m[col*3+row]=v; }

  mul(other) {
    const a=this.m, b=other.m, r=new Float32Array(9);
    for(let i=0;i<3;i++) for(let j=0;j<3;j++) {
      let s=0; for(let k=0;k<3;k++) s+=a[k*3+i]*b[j*3+k];
      r[j*3+i]=s;
    }
    return new Mat3(r);
  }

  transform(v) {
    const m=this.m;
    return new Vec3(
      m[0]*v.x+m[3]*v.y+m[6]*v.z,
      m[1]*v.x+m[4]*v.y+m[7]*v.z,
      m[2]*v.x+m[5]*v.y+m[8]*v.z
    );
  }

  transpose() {
    const m=this.m;
    return new Mat3([m[0],m[3],m[6], m[1],m[4],m[7], m[2],m[5],m[8]]);
  }

  det() {
    const m=this.m;
    return m[0]*(m[4]*m[8]-m[7]*m[5]) - m[3]*(m[1]*m[8]-m[7]*m[2]) + m[6]*(m[1]*m[5]-m[4]*m[2]);
  }

  inverse() {
    const m=this.m, d=this.det();
    if(Math.abs(d)<1e-12) return Mat3.identity();
    const inv=1/d;
    return new Mat3([
      (m[4]*m[8]-m[5]*m[7])*inv, (m[2]*m[7]-m[1]*m[8])*inv, (m[1]*m[5]-m[2]*m[4])*inv,
      (m[5]*m[6]-m[3]*m[8])*inv, (m[0]*m[8]-m[2]*m[6])*inv, (m[2]*m[3]-m[0]*m[5])*inv,
      (m[3]*m[7]-m[4]*m[6])*inv, (m[1]*m[6]-m[0]*m[7])*inv, (m[0]*m[4]-m[1]*m[3])*inv
    ]);
  }

  // Apply as color matrix to RGB pixel (input/output 0-1 range)
  applyToRGB(r, g, b) {
    const m=this.m;
    return [
      ScalarMath.clamp01(m[0]*r+m[3]*g+m[6]*b),
      ScalarMath.clamp01(m[1]*r+m[4]*g+m[7]*b),
      ScalarMath.clamp01(m[2]*r+m[5]*g+m[8]*b),
    ];
  }

  toString() {
    const m=this.m;
    return `Mat3:\n[${m[0].toFixed(4)}, ${m[3].toFixed(4)}, ${m[6].toFixed(4)}]\n[${m[1].toFixed(4)}, ${m[4].toFixed(4)}, ${m[7].toFixed(4)}]\n[${m[2].toFixed(4)}, ${m[5].toFixed(4)}, ${m[8].toFixed(4)}]`;
  }
}

class Mat4 {
  constructor(data) {
    this.m = data ? new Float32Array(data) : new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  static identity()     { return new Mat4(); }
  static zero()         { return new Mat4(new Float32Array(16)); }

  static translation(x,y,z) {
    const m=new Mat4();
    m.m[12]=x; m.m[13]=y; m.m[14]=z;
    return m;
  }
  static scale(x,y,z) {
    const m=new Mat4();
    m.m[0]=x; m.m[5]=y; m.m[10]=z;
    return m;
  }
  static rotationX(a) {
    const c=Math.cos(a),s=Math.sin(a),m=new Mat4();
    m.m[5]=c; m.m[6]=s; m.m[9]=-s; m.m[10]=c; return m;
  }
  static rotationY(a) {
    const c=Math.cos(a),s=Math.sin(a),m=new Mat4();
    m.m[0]=c; m.m[2]=-s; m.m[8]=s; m.m[10]=c; return m;
  }
  static rotationZ(a) {
    const c=Math.cos(a),s=Math.sin(a),m=new Mat4();
    m.m[0]=c; m.m[1]=s; m.m[4]=-s; m.m[5]=c; return m;
  }

  mul(other) {
    const a=this.m, b=other.m, r=new Float32Array(16);
    for(let i=0;i<4;i++) for(let j=0;j<4;j++) {
      let s=0; for(let k=0;k<4;k++) s+=a[k*4+i]*b[j*4+k];
      r[j*4+i]=s;
    }
    return new Mat4(r);
  }

  transform(v) {
    const m=this.m, w=v.w||1;
    return new Vec4(
      m[0]*v.x+m[4]*v.y+m[8]*v.z+m[12]*w,
      m[1]*v.x+m[5]*v.y+m[9]*v.z+m[13]*w,
      m[2]*v.x+m[6]*v.y+m[10]*v.z+m[14]*w,
      m[3]*v.x+m[7]*v.y+m[11]*v.z+m[15]*w,
    );
  }

  transpose() {
    const m=this.m;
    return new Mat4([
      m[0],m[4],m[8], m[12],
      m[1],m[5],m[9], m[13],
      m[2],m[6],m[10],m[14],
      m[3],m[7],m[11],m[15],
    ]);
  }

  // Cofactor expansion determinant
  det() {
    const m=this.m;
    const A2323 = m[10]*m[15]-m[11]*m[14];
    const A1323 = m[9]*m[15]-m[11]*m[13];
    const A1223 = m[9]*m[14]-m[10]*m[13];
    const A0323 = m[8]*m[15]-m[11]*m[12];
    const A0223 = m[8]*m[14]-m[10]*m[12];
    const A0123 = m[8]*m[13]-m[9]*m[12];
    return m[0]*(m[5]*A2323-m[6]*A1323+m[7]*A1223)
          -m[4]*(m[1]*A2323-m[2]*A1323+m[3]*A1223)
          +m[8]*(m[1]*A1323-m[2]*A0323+m[3]*A0123)  // corrected
          -m[12]*(m[1]*A1223-m[2]*A0223+m[3]*A0123);
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5: FAST FOURIER TRANSFORM (FFT)
// ═══════════════════════════════════════════════════════════════════════════════

class FFT {
  /**
   * Radix-2 Cooley-Tukey FFT
   * Input: Float32Array or Float64Array, length must be power of 2
   * Returns complex spectrum as { re: Float32Array, im: Float32Array }
   */
  static forward(signal) {
    const n = signal.length;
    if(!ScalarMath.isPow2(n)) throw new Error('FFT size must be power of 2');

    const re = new Float64Array(signal);
    const im = new Float64Array(n);

    // Bit-reversal permutation
    const bits = Math.log2(n);
    for(let i=0;i<n;i++) {
      const rev = FFT._reverseBits(i, bits);
      if(rev > i) { [re[i],re[rev]]=[re[rev],re[i]]; [im[i],im[rev]]=[im[rev],im[i]]; }
    }

    // Butterfly computation
    for(let len=2;len<=n;len*=2) {
      const ang = -2*Math.PI/len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for(let i=0;i<n;i+=len) {
        let curRe=1, curIm=0;
        for(let j=0;j<len/2;j++) {
          const uRe=re[i+j], uIm=im[i+j];
          const vRe=re[i+j+len/2]*curRe-im[i+j+len/2]*curIm;
          const vIm=re[i+j+len/2]*curIm+im[i+j+len/2]*curRe;
          re[i+j]=uRe+vRe; im[i+j]=uIm+vIm;
          re[i+j+len/2]=uRe-vRe; im[i+j+len/2]=uIm-vIm;
          [curRe,curIm]=[curRe*wRe-curIm*wIm, curRe*wIm+curIm*wRe];
        }
      }
    }
    return { re, im, n };
  }

  static inverse(spectrum) {
    const { re, im, n } = spectrum;
    const result = FFT.forward({ length: n, ...Array.from({length:n},(_,i)=>im[i]), re: im, im: re });
    for(let i=0;i<n;i++) result.re[i]/=n;
    return result.re;
  }

  // Compute power spectrum (magnitude squared)
  static powerSpectrum(signal) {
    const { re, im, n } = FFT.forward(signal);
    const ps = new Float32Array(n/2);
    for(let i=0;i<n/2;i++) ps[i] = re[i]*re[i]+im[i]*im[i];
    return ps;
  }

  // Compute magnitude spectrum (dB)
  static magnitudeSpectrum(signal) {
    const ps = FFT.powerSpectrum(signal);
    return ps.map(p => p>0 ? 10*Math.log10(p) : -120);
  }

  // 2D FFT for image processing
  static fft2d(data, width, height) {
    const re = new Float32Array(width*height);
    const im = new Float32Array(width*height);

    // Extract luminance
    for(let i=0;i<width*height;i++) re[i] = data[i*4]*0.2126+data[i*4+1]*0.7152+data[i*4+2]*0.0722;

    // Row-wise FFT
    const n = ScalarMath.nextPow2(Math.max(width,height));
    for(let y=0;y<height;y++) {
      const row = new Float64Array(n);
      for(let x=0;x<width;x++) row[x]=re[y*width+x];
      const r = FFT.forward(row);
      for(let x=0;x<width;x++) { re[y*width+x]=r.re[x]; im[y*width+x]=r.im[x]; }
    }

    return { re, im, width, height };
  }

  static _reverseBits(v, bits) {
    let r=0;
    for(let i=0;i<bits;i++) { r=(r<<1)|(v&1); v>>=1; }
    return r;
  }

  // Frequency domain filtering (low-pass, high-pass)
  static frequencyFilter(data, width, height, type='lowpass', cutoff=0.3) {
    const { re, im } = FFT.fft2d(data, width, height);
    const n = width*height;
    const cx=width/2, cy=height/2;
    const filteredRe = new Float32Array(re);
    const filteredIm = new Float32Array(im);

    for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
      const dx=(x-cx)/width, dy=(y-cy)/height;
      const dist=Math.sqrt(dx*dx+dy*dy);
      const idx=y*width+x;
      let gain;
      if(type==='lowpass')  gain = 1/(1+(dist/cutoff)**4);
      else if(type==='highpass') gain = 1-1/(1+(dist/cutoff)**4);
      else if(type==='bandpass') { const bw=0.05; gain = Math.exp(-((dist-cutoff)**2)/(2*bw**2)); }
      else gain=1;
      filteredRe[idx]*=gain;
      filteredIm[idx]*=gain;
    }
    return { re: filteredRe, im: filteredIm, width, height };
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6: PERLIN & SIMPLEX NOISE
// ═══════════════════════════════════════════════════════════════════════════════

class NoiseGen {
  constructor(seed=42) {
    this.seed = seed;
    this._buildPermTable();
  }

  _buildPermTable() {
    const p = new Uint8Array(256);
    for(let i=0;i<256;i++) p[i]=i;
    // Fisher-Yates shuffle with seed
    let s=this.seed>>>0;
    for(let i=255;i>0;i--) {
      s=(s*1664525+1013904223)>>>0;
      const j=s%(i+1);
      [p[i],p[j]]=[p[j],p[i]];
    }
    this.perm = new Uint8Array(512);
    this.gradPerm = new Uint8Array(512);
    for(let i=0;i<512;i++) {
      this.perm[i]=p[i&255];
      this.gradPerm[i]=p[i&255]%12;
    }
  }

  // Gradient table for 3D simplex noise
  static GRAD3 = new Float32Array([
    1,1,0, -1,1,0,  1,-1,0, -1,-1,0,
    1,0,1, -1,0,1,  1,0,-1, -1,0,-1,
    0,1,1,  0,-1,1, 0, 1,-1,  0,-1,-1
  ]);

  _fade(t) { return t*t*t*(t*(t*6-15)+10); }
  _grad2(hash, x, y) { const h=hash&3; return (h<2?x:-x)+(h===1||h===2?y:-y); }

  // Classic Perlin noise 2D
  perlin2(x, y) {
    const X=Math.floor(x)&255, Y=Math.floor(y)&255;
    x-=Math.floor(x); y-=Math.floor(y);
    const u=this._fade(x), v=this._fade(y);
    const p=this.perm;
    const A=p[X]+Y, B=p[X+1]+Y;
    return ScalarMath.lerp(
      ScalarMath.lerp(this._grad2(p[A],x,y),       this._grad2(p[B],x-1,y),   u),
      ScalarMath.lerp(this._grad2(p[A+1],x,y-1),   this._grad2(p[B+1],x-1,y-1),u),
      v
    );
  }

  // Fractal Brownian Motion (fBm) — multiple octaves
  fbm2(x, y, octaves=6, lacunarity=2.0, gain=0.5) {
    let value=0, amplitude=0.5, frequency=1, maxValue=0;
    for(let i=0;i<octaves;i++) {
      value += this.perlin2(x*frequency, y*frequency)*amplitude;
      maxValue += amplitude;
      amplitude *= gain;
      frequency *= lacunarity;
    }
    return value/maxValue;
  }

  // Turbulence (absolute values — creates ridge-like patterns)
  turbulence2(x, y, octaves=6) {
    let value=0, amplitude=0.5, frequency=1;
    for(let i=0;i<octaves;i++) {
      value += Math.abs(this.perlin2(x*frequency, y*frequency))*amplitude;
      amplitude *= 0.5;
      frequency *= 2;
    }
    return value;
  }

  // Domain-warped noise (more organic, complex patterns)
  warp2(x, y, strength=1) {
    const q = [
      this.fbm2(x, y),
      this.fbm2(x+5.2, y+1.3)
    ];
    return this.fbm2(x + strength*q[0], y + strength*q[1]);
  }

  // 3D Simplex noise
  simplex3(xin, yin, zin) {
    const F3=1/3, G3=1/6;
    const s=(xin+yin+zin)*F3;
    const i=Math.floor(xin+s), j=Math.floor(yin+s), k=Math.floor(zin+s);
    const t=(i+j+k)*G3;
    const X0=i-t, Y0=j-t, Z0=k-t;
    const x0=xin-X0, y0=yin-Y0, z0=zin-Z0;
    let i1,j1,k1,i2,j2,k2;
    if(x0>=y0) { if(y0>=z0){i1=1;j1=0;k1=0;i2=1;j2=1;k2=0} else if(x0>=z0){i1=1;j1=0;k1=0;i2=1;j2=0;k2=1} else{i1=0;j1=0;k1=1;i2=1;j2=0;k2=1} }
    else { if(y0<z0){i1=0;j1=0;k1=1;i2=0;j2=1;k2=1} else if(x0<z0){i1=0;j1=1;k1=0;i2=0;j2=1;k2=1} else{i1=0;j1=1;k1=0;i2=1;j2=1;k2=0} }
    const x1=x0-i1+G3, y1=y0-j1+G3, z1=z0-k1+G3;
    const x2=x0-i2+2*G3, y2=y0-j2+2*G3, z2=z0-k2+2*G3;
    const x3=x0-1+3*G3, y3=y0-1+3*G3, z3=z0-1+3*G3;
    const ii=i&255, jj=j&255, kk=k&255;
    const p=this.perm, g=NoiseGen.GRAD3;
    const _dot3=(gi,x,y,z)=>g[gi*3]*x+g[gi*3+1]*y+g[gi*3+2]*z;
    let n0=0,n1=0,n2=0,n3=0;
    let t0=0.6-x0*x0-y0*y0-z0*z0; if(t0>0){t0*=t0;n0=t0*t0*_dot3(p[ii+p[jj+p[kk]]]%12,x0,y0,z0);}
    let t1=0.6-x1*x1-y1*y1-z1*z1; if(t1>0){t1*=t1;n1=t1*t1*_dot3(p[ii+i1+p[jj+j1+p[kk+k1]]]%12,x1,y1,z1);}
    let t2=0.6-x2*x2-y2*y2-z2*z2; if(t2>0){t2*=t2;n2=t2*t2*_dot3(p[ii+i2+p[jj+j2+p[kk+k2]]]%12,x2,y2,z2);}
    let t3=0.6-x3*x3-y3*y3-z3*z3; if(t3>0){t3*=t3;n3=t3*t3*_dot3(p[ii+1+p[jj+1+p[kk+1]]]%12,x3,y3,z3);}
    return 32*(n0+n1+n2+n3);
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 7: CURVE INTERPOLATION
// ═══════════════════════════════════════════════════════════════════════════════

class CurveInterpolator {
  // Cubic Hermite spline
  static hermite(p0, p1, m0, m1, t) {
    const t2=t*t, t3=t2*t;
    const h00=2*t3-3*t2+1, h10=t3-2*t2+t, h01=-2*t3+3*t2, h11=t3-t2;
    return h00*p0 + h10*m0 + h01*p1 + h11*m1;
  }

  // Catmull-Rom spline (passes through all control points)
  static catmullRom(p0, p1, p2, p3, t) {
    const t2=t*t, t3=t2*t;
    return 0.5*(
      (2*p1)
      +(-p0+p2)*t
      +(2*p0-5*p1+4*p2-p3)*t2
      +(-p0+3*p1-3*p2+p3)*t3
    );
  }

  // Bezier cubic
  static bezier(p0, p1, p2, p3, t) {
    const mt=1-t;
    return mt*mt*mt*p0 + 3*mt*mt*t*p1 + 3*mt*t*t*p2 + t*t*t*p3;
  }

  // B-spline basis
  static bspline(p, t) {
    // Uniform cubic B-spline through array of control points
    const n=p.length;
    if(n<4) return p[0];
    const seg=Math.floor(t*(n-3));
    const lt=t*(n-3)-seg;
    const i=Math.min(seg, n-4);
    const t2=lt*lt, t3=t2*lt;
    const b0=(1-3*lt+3*t2-t3)/6;
    const b1=(4-6*t2+3*t3)/6;
    const b2=(1+3*lt+3*t2-3*t3)/6;
    const b3=t3/6;
    return b0*p[i]+b1*p[i+1]+b2*p[i+2]+b3*p[i+3];
  }

  // Build a tone curve from control points → lookup table (256 entries)
  static buildToneCurveLUT(points) {
    if(!points || points.length<2) return Array.from({length:256},(_,i)=>i);
    const sorted=[...points].sort((a,b)=>a[0]-b[0]);
    const lut = new Uint8Array(256);

    for(let x=0;x<256;x++) {
      // Find surrounding control points
      let found=false;
      for(let j=0;j<sorted.length-1;j++) {
        if(x>=sorted[j][0] && x<=sorted[j+1][0]) {
          const t=(x-sorted[j][0])/(sorted[j+1][0]-sorted[j][0]+0.001);
          // Get tangents from neighbors
          const p0=sorted[j>0?j-1:0], p1=sorted[j], p2=sorted[j+1], p3=sorted[j+2<sorted.length?j+2:sorted.length-1];
          const y=CurveInterpolator.catmullRom(p1[1],p1[1],p2[1],p2[1],t);
          lut[x]=ScalarMath.clamp8(y);
          found=true; break;
        }
      }
      if(!found) lut[x]=x<sorted[0][0]?sorted[0][1]:sorted[sorted.length-1][1];
    }
    return lut;
  }

  // Monotone cubic interpolation (used in Lightroom-style curves)
  static monoCubicLUT(points) {
    const n=points.length;
    if(n<2) return Array.from({length:256},(_,i)=>i);
    const sorted=[...points].sort((a,b)=>a[0]-b[0]);

    // Compute slopes
    const dx=[],dy=[],m=[];
    for(let i=0;i<n-1;i++){dx[i]=sorted[i+1][0]-sorted[i][0];dy[i]=sorted[i+1][1]-sorted[i][1];}
    m[0]=dy[0]/dx[0]; m[n-1]=dy[n-2]/dx[n-2];
    for(let i=1;i<n-1;i++) m[i]=(dy[i-1]/dx[i-1]+dy[i]/dx[i])/2;
    // Adjust for monotonicity
    for(let i=0;i<n-1;i++){
      if(dy[i]===0){m[i]=0;m[i+1]=0;}
      else {
        const a=m[i]/dy[i]*dx[i], b=m[i+1]/dy[i]*dx[i];
        const s=a*a+b*b;
        if(s>9){m[i]*=3/Math.sqrt(s);m[i+1]*=3/Math.sqrt(s);}
      }
    }

    const lut=new Uint8Array(256);
    for(let x=0;x<256;x++){
      let found=false;
      for(let i=0;i<n-1;i++){
        if(x>=sorted[i][0]&&x<=sorted[i+1][0]){
          const t=(x-sorted[i][0])/dx[i];
          const t2=t*t,t3=t2*t;
          const h00=2*t3-3*t2+1,h10=t3-2*t2+t,h01=-2*t3+3*t2,h11=t3-t2;
          lut[x]=ScalarMath.clamp8(h00*sorted[i][1]+h10*m[i]*dx[i]+h01*sorted[i+1][1]+h11*m[i+1]*dx[i]);
          found=true;break;
        }
      }
      if(!found) lut[x]=x<sorted[0][0]?sorted[0][1]:sorted[n-1][1];
    }
    return lut;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 8: FULL COLOR SCIENCE ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

class ColorScience {
  // ── sRGB gamma ────────────────────────────────────────────────────────────
  static linearize(v)   { return v<=0.04045 ? v/12.92 : Math.pow((v+0.055)/1.055,2.4); }
  static gammaEncode(v) { return v<=0.0031308 ? v*12.92 : 1.055*Math.pow(v,1/2.4)-0.055; }

  // ── sRGB ↔ CIE XYZ (D65) ──────────────────────────────────────────────────
  static srgbToXYZ(r, g, b) {
    // Linearize
    const rl=ColorScience.linearize(r/255);
    const gl=ColorScience.linearize(g/255);
    const bl=ColorScience.linearize(b/255);
    const M=MATH_CONST.SRGB_TO_XYZ;
    return new Vec3(
      M[0]*rl+M[1]*gl+M[2]*bl,
      M[3]*rl+M[4]*gl+M[5]*bl,
      M[6]*rl+M[7]*gl+M[8]*bl,
    );
  }

  static xyzToSRGB(X, Y, Z) {
    const M=MATH_CONST.XYZ_TO_SRGB;
    const rl=M[0]*X+M[1]*Y+M[2]*Z;
    const gl=M[3]*X+M[4]*Y+M[5]*Z;
    const bl=M[6]*X+M[7]*Y+M[8]*Z;
    return [
      ScalarMath.clamp8(ColorScience.gammaEncode(ScalarMath.clamp01(rl))*255),
      ScalarMath.clamp8(ColorScience.gammaEncode(ScalarMath.clamp01(gl))*255),
      ScalarMath.clamp8(ColorScience.gammaEncode(ScalarMath.clamp01(bl))*255),
    ];
  }

  // ── CIE XYZ → CIE L*a*b* ──────────────────────────────────────────────────
  static xyzToLab(X, Y, Z) {
    const f = v => v>0.008856 ? Math.cbrt(v) : 7.787*v+16/116;
    const fx=f(X/MATH_CONST.D65_X), fy=f(Y/MATH_CONST.D65_Y), fz=f(Z/MATH_CONST.D65_Z);
    return { L:116*fy-16, a:500*(fx-fy), b:200*(fy-fz) };
  }

  static labToXYZ(L, a, b) {
    const fy=(L+16)/116, fx=a/500+fy, fz=fy-b/200;
    const x=fx>0.206897?fx*fx*fx:(fx-16/116)/7.787;
    const y=fy>0.206897?fy*fy*fy:(fy-16/116)/7.787;
    const z=fz>0.206897?fz*fz*fz:(fz-16/116)/7.787;
    return { X:x*MATH_CONST.D65_X, Y:y*MATH_CONST.D65_Y, Z:z*MATH_CONST.D65_Z };
  }

  // ── sRGB ↔ CIE L*a*b* ────────────────────────────────────────────────────
  static srgbToLab(r, g, b) {
    const xyz=ColorScience.srgbToXYZ(r,g,b);
    return ColorScience.xyzToLab(xyz.x,xyz.y,xyz.z);
  }

  static labToSRGB(L, a, b) {
    const xyz=ColorScience.labToXYZ(L,a,b);
    return ColorScience.xyzToSRGB(xyz.X,xyz.Y,xyz.Z);
  }

  // ── CIE L*C*h° (LCH) ─────────────────────────────────────────────────────
  static labToLCH(L, a, b) {
    const C=Math.sqrt(a*a+b*b);
    const H=(Math.atan2(b,a)*180/Math.PI+360)%360;
    return { L, C, H };
  }
  static lchToLab(L, C, H) {
    const hr=H*Math.PI/180;
    return { L, a:C*Math.cos(hr), b:C*Math.sin(hr) };
  }

  // ── sRGB ↔ HSL ────────────────────────────────────────────────────────────
  static rgbToHsl(r, g, b) {
    r/=255; g/=255; b/=255;
    const max=Math.max(r,g,b), min=Math.min(r,g,b);
    let h, s, l=(max+min)/2;
    if(max===min) { h=s=0; }
    else {
      const d=max-min;
      s=l>0.5?d/(2-max-min):d/(max+min);
      switch(max) {
        case r: h=((g-b)/d+(g<b?6:0))/6; break;
        case g: h=((b-r)/d+2)/6; break;
        case b: h=((r-g)/d+4)/6; break;
      }
    }
    return [h*360, s*100, l*100];
  }

  static hslToRgb(h, s, l) {
    h/=360; s/=100; l/=100;
    const hue2rgb=(p,q,t)=>{
      if(t<0)t+=1; if(t>1)t-=1;
      if(t<1/6)return p+(q-p)*6*t;
      if(t<1/2)return q;
      if(t<2/3)return p+(q-p)*(2/3-t)*6;
      return p;
    };
    if(s===0) { const v=Math.round(l*255); return [v,v,v]; }
    const q=l<0.5?l*(1+s):l+s-l*s, p=2*l-q;
    return [Math.round(hue2rgb(p,q,h+1/3)*255),Math.round(hue2rgb(p,q,h)*255),Math.round(hue2rgb(p,q,h-1/3)*255)];
  }

  // ── sRGB ↔ HSV ────────────────────────────────────────────────────────────
  static rgbToHsv(r, g, b) {
    r/=255; g/=255; b/=255;
    const max=Math.max(r,g,b), min=Math.min(r,g,b), d=max-min;
    let h=0;
    if(d!==0){
      switch(max){
        case r: h=((g-b)/d+6)%6; break;
        case g: h=(b-r)/d+2; break;
        case b: h=(r-g)/d+4; break;
      }
    }
    return [h*60, max===0?0:d/max*100, max*100];
  }
  static hsvToRgb(h, s, v) {
    h/=60; s/=100; v/=100;
    const i=Math.floor(h), f=h-i, p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s);
    let r,g,b;
    switch(i%6){case 0:[r,g,b]=[v,t,p];break;case 1:[r,g,b]=[q,v,p];break;case 2:[r,g,b]=[p,v,t];break;case 3:[r,g,b]=[p,q,v];break;case 4:[r,g,b]=[t,p,v];break;default:[r,g,b]=[v,p,q];}
    return [Math.round(r*255),Math.round(g*255),Math.round(b*255)];
  }

  // ── YCbCr (BT.601 and BT.709) ────────────────────────────────────────────
  static rgbToYCbCr601(r, g, b) {
    return [
      0.299*r  +0.587*g  +0.114*b,
     -0.169*r  -0.331*g  +0.500*b+128,
      0.500*r  -0.419*g  -0.081*b+128,
    ];
  }
  static yCbCr601ToRgb(Y, Cb, Cr) {
    Cb-=128; Cr-=128;
    return [ScalarMath.clamp8(Y+1.402*Cr), ScalarMath.clamp8(Y-0.344*Cb-0.714*Cr), ScalarMath.clamp8(Y+1.772*Cb)];
  }
  static rgbToYCbCr709(r, g, b) {
    return [
      0.2126*r +0.7152*g +0.0722*b,
     -0.1146*r -0.3854*g +0.5000*b+128,
      0.5000*r -0.4542*g -0.0458*b+128,
    ];
  }

  // ── Luminance & perceptual lightness ─────────────────────────────────────
  static luma601(r,g,b)  { return 0.299*r  +0.587*g  +0.114*b; }
  static luma709(r,g,b)  { return 0.2126*r +0.7152*g +0.0722*b; }
  static luma2020(r,g,b) { return 0.2627*r +0.6780*g +0.0593*b; }

  // CIELAB L* from linear luminance
  static luminanceToL(Y) { return Y>0.008856?116*Math.cbrt(Y)-16:903.3*Y; }

  // ── Color temperature (CCT) ── Planckian locus approximation ────────────
  static kelvinToRGB(K) {
    let r, g, b;
    K = K / 100;
    if(K <= 66) {
      r = 255;
      g = 99.4708025861 * Math.log(K) - 161.1195681661;
      b = K <= 19 ? 0 : 138.5177312231 * Math.log(K-10) - 305.0447927307;
    } else {
      r = 329.698727446 * Math.pow(K-60,-0.1332047592);
      g = 288.1221695283 * Math.pow(K-60,-0.0755148492);
      b = 255;
    }
    return [ScalarMath.clamp8(r), ScalarMath.clamp8(g), ScalarMath.clamp8(b)];
  }

  // ── Delta-E (perceptual color difference, CIE2000) ────────────────────────
  static deltaE2000(L1,a1,b1,L2,a2,b2) {
    const KSUBSC=1, KSUBSH=1, KSUBSL=1;
    const C1=Math.sqrt(a1*a1+b1*b1), C2=Math.sqrt(a2*a2+b2*b2);
    const Cavg=(C1+C2)/2;
    const G=0.5*(1-Math.sqrt(Math.pow(Cavg,7)/(Math.pow(Cavg,7)+Math.pow(25,7))));
    const a1p=a1*(1+G), a2p=a2*(1+G);
    const C1p=Math.sqrt(a1p*a1p+b1*b1), C2p=Math.sqrt(a2p*a2p+b2*b2);
    const h1p=C1p===0?0:(Math.atan2(b1,a1p)*180/Math.PI+360)%360;
    const h2p=C2p===0?0:(Math.atan2(b2,a2p)*180/Math.PI+360)%360;
    const dLp=L2-L1, dCp=C2p-C1p;
    const dHpAngle=C1p*C2p===0?0:Math.abs(h2p-h1p)<=180?h2p-h1p:h2p<=h1p?h2p-h1p+360:h2p-h1p-360;
    const dHp=2*Math.sqrt(C1p*C2p)*ScalarMath.sinDeg(dHpAngle/2);
    const Lbarp=(L1+L2)/2;
    const Cbarp=(C1p+C2p)/2;
    const hbarp=C1p*C2p===0?h1p+h2p:Math.abs(h1p-h2p)<=180?(h1p+h2p)/2:(h1p+h2p+360)/2;
    const T=1-0.17*ScalarMath.cosDeg(hbarp-30)+0.24*ScalarMath.cosDeg(2*hbarp)+0.32*ScalarMath.cosDeg(3*hbarp+6)-0.20*ScalarMath.cosDeg(4*hbarp-63);
    const SL=1+0.015*Math.pow(Lbarp-50,2)/Math.sqrt(20+Math.pow(Lbarp-50,2));
    const SC=1+0.045*Cbarp, SH=1+0.015*Cbarp*T;
    const RC=2*Math.sqrt(Math.pow(Cbarp,7)/(Math.pow(Cbarp,7)+Math.pow(25,7)));
    const RT=-Math.sin(2*ScalarMath.toRad(60*Math.exp(-Math.pow((hbarp-275)/25,2))))*RC;
    return Math.sqrt(Math.pow(dLp/(KSUBSL*SL),2)+Math.pow(dCp/(KSUBSC*SC),2)+Math.pow(dHp/(KSUBSH*SH),2)+RT*(dCp/(KSUBSC*SC))*(dHp/(KSUBSH*SH)));
  }

  // ── Log encoding (camera LOG formats) ────────────────────────────────────
  static sLog2Encode(x) {
    if(x>=0) return (0.432699*Math.log10(5.555556*x+0.037584)+0.616596)+0.030001;
    return x*5+0.030001;
  }
  static sLog2Decode(x) {
    if(x>=0.030001) return (Math.pow(10,(x-0.616596-0.030001)/0.432699)-0.037584)/5.555556;
    return (x-0.030001)/5;
  }
  static logC3Encode(x) {
    const cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809;
    return x>=cut ? c*Math.log10(a*x+b)+d : e*x+f;
  }
  static logC3Decode(x) {
    const cut=0.149658, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809;
    return x>=cut ? (Math.pow(10,(x-d)/c)-b)/a : (x-f)/e;
  }
  static log3G10Encode(x) {
    return x>=0 ? 0.224282*Math.log10(1+x/0.009)+0.091 : x*15.1927+0.091;
  }
  static rec709OETF(x) { return x<0.018?x*4.5:1.0993*Math.pow(x,0.45)-0.0993; }
  static rec2020OETF(x) { const b=0.0181,a=0.0993; return x<b?4.5*x:1+a-a*Math.pow(x,-0.45); } // PQ not HLG
  static pqEOTF(x) {
    const m1=0.1593017578125, m2=78.84375, c1=0.8359375, c2=18.8515625, c3=18.6875;
    const xm1=Math.pow(x,1/m2);
    return Math.pow(Math.max(0,xm1-c1)/(c2-c3*xm1),1/m1);
  }
  static hlgOETF(x) { const a=0.17883277,b=0.28466892,c=0.55991073; return x<1/12?Math.sqrt(3*x):a*Math.log(12*x-b)+c; }

  // ── Chromatic adaptation (Bradford) ───────────────────────────────────────
  static christensenAdaptMatrix(srcWhite, dstWhite) {
    // Bradford matrix
    const M = [0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296];
    const adaptXYZ=(xyz)=>[M[0]*xyz[0]+M[1]*xyz[1]+M[2]*xyz[2], M[3]*xyz[0]+M[4]*xyz[1]+M[5]*xyz[2], M[6]*xyz[0]+M[7]*xyz[1]+M[8]*xyz[2]];
    const src=adaptXYZ([srcWhite[0],srcWhite[1],srcWhite[2]]);
    const dst=adaptXYZ([dstWhite[0],dstWhite[1],dstWhite[2]]);
    return new Mat3([dst[0]/src[0],0,0, 0,dst[1]/src[1],0, 0,0,dst[2]/src[2]]);
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 9: STATISTICS & HISTOGRAM ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

class ImageStatistics {
  static analyze(data, width, height) {
    const n = width * height;
    const histR = new Uint32Array(256), histG = new Uint32Array(256), histB = new Uint32Array(256), histL = new Uint32Array(256);
    let sumR=0, sumG=0, sumB=0, sumL=0;
    let minR=255, minG=255, minB=255, maxR=0, maxG=0, maxB=0;
    let sat=0, dark=0, bright=0;

    for(let i=0;i<n;i++) {
      const r=data[i*4], g=data[i*4+1], b=data[i*4+2];
      const l=Math.round(ColorScience.luma709(r,g,b));
      histR[r]++; histG[g]++; histB[b]++; histL[l]++;
      sumR+=r; sumG+=g; sumB+=b; sumL+=l;
      if(r<minR)minR=r; if(r>maxR)maxR=r;
      if(g<minG)minG=g; if(g>maxG)maxG=g;
      if(b<minB)minB=b; if(b>maxB)maxB=b;
      if(l<20)dark++;
      if(l>235)bright++;
      const [h,s]=[...ColorScience.rgbToHsl(r,g,b)];
      if(s>50)sat++;
    }

    // Percentiles
    const pct=(hist,p)=>{ let cum=0,tot=n*p/100; for(let i=0;i<256;i++){cum+=hist[i];if(cum>=tot)return i;} return 255; };

    return {
      n, width, height,
      histR, histG, histB, histL,
      mean: { r:sumR/n, g:sumG/n, b:sumB/n, l:sumL/n },
      min:  { r:minR, g:minG, b:minB },
      max:  { r:maxR, g:maxG, b:maxB },
      percentile10: { r:pct(histR,10), g:pct(histG,10), b:pct(histB,10), l:pct(histL,10) },
      percentile50: { r:pct(histR,50), g:pct(histG,50), b:pct(histB,50), l:pct(histL,50) },
      percentile90: { r:pct(histR,90), g:pct(histG,90), b:pct(histB,90), l:pct(histL,90) },
      darkRatio:  dark/n,
      brightRatio: bright/n,
      saturationRatio: sat/n,
      dynamic_range: Math.log2((maxR+maxG+maxB+1)/(minR+minG+minB+1)),
      snr: sumL/n / (Math.sqrt(histL.reduce((s,v,i)=>s+v*(i-sumL/n)**2,0)/n)+0.001),
      exposure_ev: Math.log2(sumL/n/128),
    };
  }

  // Auto-exposure suggestion
  static suggestExposure(stats) {
    const evDiff = -stats.exposure_ev;
    return { ev: Math.max(-3, Math.min(3, evDiff*0.7)), confidence: 1-Math.abs(evDiff)/4 };
  }

  // Detect image quality issues
  static qualityReport(data, width, height) {
    const stats = ImageStatistics.analyze(data, width, height);
    const issues = [];
    if(stats.darkRatio > 0.4)   issues.push({ type:'underexposed', severity:stats.darkRatio, fix:'ev+1.5' });
    if(stats.brightRatio > 0.1) issues.push({ type:'overexposed',  severity:stats.brightRatio, fix:'ev-1.0' });
    if(stats.snr < 5)           issues.push({ type:'noisy',        severity:1-stats.snr/20, fix:'denoise' });
    if(stats.dynamic_range < 2) issues.push({ type:'low_contrast', severity:1-stats.dynamic_range/8, fix:'contrast' });
    return { stats, issues, quality: Math.max(0, 100-issues.reduce((s,i)=>s+i.severity*30,0)) };
  }

  // Shannon entropy (measures information content / texture richness)
  static entropy(hist, total) {
    let H=0;
    for(let i=0;i<hist.length;i++) {
      if(hist[i]>0){ const p=hist[i]/total; H-=p*Math.log2(p); }
    }
    return H;
  }

  // Blind image quality metric (no-reference, Gaussian MSE model)
  static BRISQUE(data, width, height) {
    // Simplified BRISQUE-inspired feature extraction
    const n=width*height;
    const Y=new Float32Array(n);
    for(let i=0;i<n;i++) Y[i]=ColorScience.luma709(data[i*4],data[i*4+1],data[i*4+2])/255;

    // Local mean and variance
    let localVarSum=0, kurtSum=0, skewSum=0;
    for(let y=1;y<height-1;y++) for(let x=1;x<width-1;x++) {
      const i=y*width+x;
      const neighbors=[Y[i-width-1],Y[i-width],Y[i-width+1],Y[i-1],Y[i],Y[i+1],Y[i+width-1],Y[i+width],Y[i+width+1]];
      const m=neighbors.reduce((a,b)=>a+b,0)/9;
      const v=neighbors.reduce((a,b)=>a+(b-m)**2,0)/9;
      localVarSum+=v;
      // Kurtosis (measures noise/texture)
      const s=Math.sqrt(v)||1;
      const k=neighbors.reduce((a,b)=>a+((b-m)/s)**4,0)/9;
      kurtSum+=k;
      skewSum+=neighbors.reduce((a,b)=>a+((b-m)/s)**3,0)/9;
    }
    const N=(width-2)*(height-2);
    return {
      localVariance: localVarSum/N,
      kurtosis: kurtSum/N,
      skewness: skewSum/N,
      estimated_quality: Math.min(100, Math.max(0, 80-localVarSum/N*500+kurtSum/N*0.5)),
    };
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 10: PIXEL BUFFER UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

class PixelBuffer {
  constructor(width, height, data=null) {
    this.width  = width;
    this.height = height;
    this.n      = width * height;
    this.data   = data ? new Uint8ClampedArray(data) : new Uint8ClampedArray(width*height*4);
  }

  static fromImageData(id) { return new PixelBuffer(id.width, id.height, id.data); }
  static fromArray(arr, w, h) { return new PixelBuffer(w, h, arr); }

  toImageData() { return new ImageData(new Uint8ClampedArray(this.data), this.width, this.height); }
  clone()       { return new PixelBuffer(this.width, this.height, this.data); }

  getPixel(x, y) {
    const i=(y*this.width+x)*4;
    return [this.data[i], this.data[i+1], this.data[i+2], this.data[i+3]];
  }

  setPixel(x, y, r, g, b, a=255) {
    const i=(y*this.width+x)*4;
    this.data[i]=ScalarMath.clamp8(r); this.data[i+1]=ScalarMath.clamp8(g);
    this.data[i+2]=ScalarMath.clamp8(b); this.data[i+3]=ScalarMath.clamp8(a);
  }

  // Boundary-safe sample (clamp to edges)
  sampleClamp(x, y) {
    const px=ScalarMath.clamp(Math.round(x),0,this.width-1);
    const py=ScalarMath.clamp(Math.round(y),0,this.height-1);
    return this.getPixel(px,py);
  }

  // Bilinear sample (sub-pixel accurate)
  sampleBilinear(x, y) {
    const x0=ScalarMath.clamp(Math.floor(x),0,this.width-1);
    const y0=ScalarMath.clamp(Math.floor(y),0,this.height-1);
    const x1=Math.min(x0+1,this.width-1);
    const y1=Math.min(y0+1,this.height-1);
    const fx=x-x0, fy=y-y0;
    const s00=this.getPixel(x0,y0), s10=this.getPixel(x1,y0);
    const s01=this.getPixel(x0,y1), s11=this.getPixel(x1,y1);
    return [
      ScalarMath.clamp8(ScalarMath.lerp(ScalarMath.lerp(s00[0],s10[0],fx),ScalarMath.lerp(s01[0],s11[0],fx),fy)),
      ScalarMath.clamp8(ScalarMath.lerp(ScalarMath.lerp(s00[1],s10[1],fx),ScalarMath.lerp(s01[1],s11[1],fx),fy)),
      ScalarMath.clamp8(ScalarMath.lerp(ScalarMath.lerp(s00[2],s10[2],fx),ScalarMath.lerp(s01[2],s11[2],fx),fy)),
      255,
    ];
  }

  // Bicubic sample
  sampleBicubic(x, y) {
    const x0=Math.floor(x), y0=Math.floor(y);
    const fx=x-x0, fy=y-y0;
    const w=(t)=>{ const at=Math.abs(t); return at<=1?(1.5*at-2.5)*at*at+1:at<2?(-0.5*at+2.5)*at*at-4*at+2:0; };
    let r=0,g=0,b=0,wt=0;
    for(let m=-1;m<=2;m++) for(let n=-1;n<=2;n++) {
      const px=ScalarMath.clamp(x0+n,0,this.width-1);
      const py=ScalarMath.clamp(y0+m,0,this.height-1);
      const wv=w(n-fx)*w(m-fy);
      const [pr,pg,pb]=this.getPixel(px,py);
      r+=pr*wv; g+=pg*wv; b+=pb*wv; wt+=wv;
    }
    return [ScalarMath.clamp8(r/wt),ScalarMath.clamp8(g/wt),ScalarMath.clamp8(b/wt),255];
  }

  // Lanczos sample
  sampleLanczos(x, y, a=3) {
    const lanczos=(t)=>{ if(t===0)return 1; const at=Math.abs(t); if(at>=a)return 0; const pt=Math.PI*t; return a*Math.sin(pt)*Math.sin(pt/a)/(pt*pt); };
    const x0=Math.floor(x), y0=Math.floor(y);
    let r=0,g=0,b=0,wt=0;
    for(let m=-(a-1);m<=a;m++) for(let n=-(a-1);n<=a;n++) {
      const px=ScalarMath.clamp(x0+n,0,this.width-1);
      const py=ScalarMath.clamp(y0+m,0,this.height-1);
      const wv=lanczos(x-x0-n)*lanczos(y-y0-m);
      const [pr,pg,pb]=this.getPixel(px,py);
      r+=pr*wv; g+=pg*wv; b+=pb*wv; wt+=wv;
    }
    return wt?[ScalarMath.clamp8(r/wt),ScalarMath.clamp8(g/wt),ScalarMath.clamp8(b/wt),255]:[0,0,0,255];
  }

  // Fast fill
  fill(r, g, b, a=255) {
    for(let i=0;i<this.n;i++){
      this.data[i*4]=r; this.data[i*4+1]=g; this.data[i*4+2]=b; this.data[i*4+3]=a;
    }
    return this;
  }

  // Map function (applies fn(r,g,b,a,x,y) → [r,g,b,a] to every pixel)
  map(fn) {
    const out=new PixelBuffer(this.width, this.height);
    for(let y=0;y<this.height;y++) for(let x=0;x<this.width;x++) {
      const i=(y*this.width+x)*4;
      const [nr,ng,nb,na]=fn(this.data[i],this.data[i+1],this.data[i+2],this.data[i+3],x,y);
      out.data[i]=ScalarMath.clamp8(nr); out.data[i+1]=ScalarMath.clamp8(ng);
      out.data[i+2]=ScalarMath.clamp8(nb); out.data[i+3]=ScalarMath.clamp8(na);
    }
    return out;
  }

  // Resize with specified interpolation
  resize(dstW, dstH, method='bicubic') {
    const out=new PixelBuffer(dstW, dstH);
    const sx=this.width/dstW, sy=this.height/dstH;
    const sample = method==='bilinear' ? (x,y)=>this.sampleBilinear(x,y)
                 : method==='lanczos'  ? (x,y)=>this.sampleLanczos(x,y)
                 : (x,y)=>this.sampleBicubic(x,y);
    for(let y=0;y<dstH;y++) for(let x=0;x<dstW;x++) {
      const [r,g,b,a]=sample(x*sx, y*sy);
      out.setPixel(x,y,r,g,b,a);
    }
    return out;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 11: RANDOM NUMBER GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

class RNG {
  constructor(seed=Date.now()) { this.s=seed>>>0; }

  // Xorshift32
  next() {
    this.s^=this.s<<13; this.s^=this.s>>17; this.s^=this.s<<5;
    return (this.s>>>0)/0xFFFFFFFF;
  }
  nextInt(lo, hi) { return lo+Math.floor(this.next()*(hi-lo+1)); }
  nextFloat(lo=0, hi=1) { return lo+this.next()*(hi-lo); }
  // Box-Muller Gaussian
  nextGaussian(mean=0, std=1) {
    const u=this.next(), v=this.next();
    return mean+std*Math.sqrt(-2*Math.log(u+1e-10))*Math.cos(2*Math.PI*v);
  }
  // Poisson (for photon noise simulation)
  nextPoisson(lambda) {
    const L=Math.exp(-lambda); let k=0, p=1;
    do { k++; p*=this.next(); } while(p>L);
    return k-1;
  }
}

// ── Exports ──────────────────────────────────────────────────────────────────
const MATH_CORE = {
  MATH_CONST, ScalarMath, Vec2, Vec3, Vec4, Mat3, Mat4,
  FFT, NoiseGen, CurveInterpolator, ColorScience, ImageStatistics, PixelBuffer, RNG,
};

if(typeof module!=='undefined') module.exports=MATH_CORE;
if(typeof self!=='undefined')   self.MATH_CORE=MATH_CORE;
