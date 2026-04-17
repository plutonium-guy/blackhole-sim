use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{
    HtmlCanvasElement, HtmlInputElement, MouseEvent, WebGl2RenderingContext as GL, WebGlProgram,
    WebGlShader, WheelEvent,
};

// ── Black hole ray tracer (Kerr metric) ─────────────────────────

const VERTEX_SHADER: &str = r#"#version 300 es
in vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"#;

const FRAGMENT_SHADER: &str = r#"#version 300 es
precision highp float;

// Camera & resolution
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_camera_dist;
uniform float u_camera_theta;
uniform float u_camera_phi;
uniform float u_fov;

// Black hole parameters (Kerr metric)
uniform float u_bh_mass;        // M: determines Schwarzschild radius Rs = 2M
uniform float u_bh_spin;        // a: spin parameter (0 = Schwarzschild, <1 = Kerr)

// Accretion disk
uniform float u_disk_inner;     // inner edge (ISCO-like)
uniform float u_disk_outer;     // outer edge
uniform float u_disk_temp;      // temperature scaling
uniform float u_disk_opacity;   // opacity multiplier

// Physics toggles
uniform float u_doppler;        // Doppler beaming strength (0-2)
uniform float u_redshift;       // gravitational redshift strength (0-2)
uniform float u_bloom;          // glow intensity near horizon

// Integration
uniform float u_step_size;      // geodesic step size
uniform int u_max_steps;        // max ray march steps

out vec4 fragColor;

const float PI = 3.14159265359;

// ── Kerr metric quantities ──────────────────────────────────────
// Boyer-Lindquist: Σ = r² + a²cos²θ,  Δ = r² - 2Mr + a²
// Event horizons: r± = M ± √(M² - a²)
// ISCO depends on spin

float kerr_horizon(float M, float a) {
    return M + sqrt(max(M * M - a * a, 0.0));
}

// Frame dragging angular velocity: ω = 2Mar / (Σ(r² + a²) + 2Ma²r sin²θ)
// Simplified for ray deflection
vec3 frame_drag_accel(vec3 pos, vec3 vel, float M, float a) {
    float r = length(pos);
    if (r < 0.01) return vec3(0.0);

    float r2 = r * r;
    // Approximate frame dragging as cross-product torque
    // Lense-Thirring: Ω_LT ≈ 2Ma/(r³) ẑ  (far from hole)
    float omega = 2.0 * M * a / (r2 * r);

    // Spin axis is y
    vec3 spin_axis = vec3(0.0, 1.0, 0.0);
    vec3 drag_vel = omega * cross(spin_axis, pos);

    // Force proportional to velocity difference from frame-dragged frame
    return (drag_vel - vel * omega) * 0.5;
}

// ── Accretion disk color (physically motivated) ─────────────────

vec3 diskColor(float r, float phi, float time, float M, float a,
               float temp_scale, float doppler_str, float redshift_str) {
    // Keplerian orbital velocity (prograde, Kerr)
    float r_s = 2.0 * M;
    float r32 = pow(r, 1.5);
    float v_orb = sqrt(M) / (r32 + a * sqrt(M));

    // Doppler beaming: D = 1/γ(1 - v·n̂)
    // Simplified: approaching side (sin φ > 0) is boosted
    float doppler = 1.0 / (1.0 + v_orb * sin(phi) * doppler_str);
    doppler = pow(doppler, 3.0);

    // Temperature profile: T ∝ r^(-3/4) * f(r, a) from Novikov-Thorne
    // f(r) includes ISCO correction
    float isco = u_disk_inner;
    float f_nt = 1.0 - sqrt(isco / r);
    f_nt = max(f_nt, 0.0);
    float temp = pow(isco / r, 0.75) * f_nt * doppler * temp_scale;

    // Gravitational redshift: g = √(1 - rs/r) for Schwarzschild
    // For Kerr: more complex, approximate
    float g_redshift = sqrt(max(1.0 - r_s / r + a * a / (r * r), 0.01));
    temp *= mix(1.0, g_redshift, redshift_str);

    // Blackbody color mapping
    vec3 hot  = vec3(0.95, 0.90, 1.0);   // blue-white
    vec3 warm = vec3(1.0, 0.55, 0.15);   // orange
    vec3 cool = vec3(0.7, 0.1, 0.02);    // deep red

    vec3 col;
    if (temp > 0.8) col = mix(warm, hot, clamp((temp - 0.8) / 0.4, 0.0, 1.0));
    else if (temp > 0.3) col = mix(cool, warm, (temp - 0.3) / 0.5);
    else col = cool * (temp / 0.3);

    // Turbulence/spiral structure
    float spiral = phi + log(r) * 3.0 - time * 0.3;
    float turb = 0.75 + 0.25 * sin(spiral * 4.0);
    turb *= 0.85 + 0.15 * sin(phi * 11.0 + r * 1.5 - time * 0.7);

    return col * 3.0 * temp * turb;
}

// ── Main ray tracer ─────────────────────────────────────────────

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);

    float M = u_bh_mass;
    float a = u_bh_spin * M;  // a = spin * M (dimensionless spin → physical)
    float r_h = kerr_horizon(M, a);

    // Camera
    float camDist = u_camera_dist;
    vec3 camPos = camDist * vec3(
        sin(u_camera_theta) * cos(u_camera_phi),
        cos(u_camera_theta),
        sin(u_camera_theta) * sin(u_camera_phi)
    );

    vec3 forward = normalize(-camPos);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(forward, worldUp));
    vec3 up = cross(right, forward);

    vec3 rd = normalize(forward * u_fov + right * uv.x + up * uv.y);

    // ── Geodesic integration ────────────────────────────────
    vec3 pos = camPos;
    vec3 vel = rd;
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    float step = u_step_size;

    for (int i = 0; i < 600; i++) {
        if (i >= u_max_steps) break;

        float r = length(pos);

        // Inside event horizon
        if (r < r_h * 0.5) {
            alpha = 1.0;
            break;
        }

        // Escaped
        if (r > 60.0) break;

        float r2 = r * r;
        float r3 = r2 * r;

        // ── Null geodesic (Schwarzschild) ───────────────
        // For photons, the effective acceleration in pseudo-Newtonian form:
        //   d²x/dλ² = -3GM/(r⁵) h² x̂
        // where h² = |x × v|² (specific angular momentum squared).
        // This is the ONLY term — no separate Newtonian gravity for light.
        // Gives correct photon sphere at r = 3M = 1.5rs.
        // Deflection angle = 4GM/b (Einstein's prediction).
        float h2 = dot(cross(pos, vel), cross(pos, vel));
        vec3 accel = -1.5 * (2.0 * M) * h2 / (r3 * r2) * pos;

        // ── Kerr frame dragging ─────────────────────────
        if (abs(a) > 0.001) {
            accel += frame_drag_accel(pos, vel, M, a);
        }

        // Adaptive step size: smaller near horizon
        float adaptive = step * clamp(r / (4.0 * r_h), 0.3, 2.0);

        vel += accel * adaptive;
        pos += vel * adaptive;

        // ── Accretion disk crossing ─────────────────────
        float prev_y = pos.y - vel.y * adaptive;
        if (prev_y * pos.y < 0.0 && alpha < 0.95) {
            float t = -prev_y / (pos.y - prev_y);
            vec3 hitPos = (pos - vel * adaptive) + vel * adaptive * t;
            float hitR = length(hitPos);

            if (hitR > u_disk_inner && hitR < u_disk_outer) {
                float hitPhi = atan(hitPos.z, hitPos.x);

                vec3 dCol = diskColor(hitR, hitPhi, u_time, M, a,
                                      u_disk_temp, u_doppler, u_redshift);

                float diskAlpha = smoothstep(u_disk_outer, u_disk_outer * 0.7, hitR)
                                * smoothstep(u_disk_inner, u_disk_inner * 1.3, hitR);
                diskAlpha *= u_disk_opacity;

                color = mix(color, dCol, diskAlpha * (1.0 - alpha));
                alpha += diskAlpha * (1.0 - alpha);
            }
        }
    }

    // ── Background stars ────────────────────────────────────
    if (alpha < 1.0) {
        vec3 dir = normalize(vel);
        vec2 cell = floor(dir.xy * 200.0 + dir.z * 137.0);
        vec2 f = fract(dir.xy * 200.0 + dir.z * 137.0);
        float h = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
        float h2 = fract(sin(dot(cell, vec2(269.5, 183.3))) * 43758.5453);

        if (h > 0.985) {
            float d = length(f - vec2(h, h2));
            float brightness = smoothstep(0.05, 0.0, d) * (h - 0.985) * 66.0;
            vec3 sCol = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.9, 0.7), h2);
            color = mix(color, sCol * brightness, 1.0 - alpha);
        }
    }

    // ── Photon sphere / bloom glow ──────────────────────────
    vec3 toCenter = -camPos;
    float angDist = length(cross(rd, normalize(toCenter)));
    float glow = 0.02 / (angDist * angDist + 0.001);
    glow *= smoothstep(5.0, 1.5, angDist * u_camera_dist);
    color += vec3(0.3, 0.15, 0.05) * glow * u_bloom;

    // Ergosphere hint (Kerr): ring of light near equator
    if (abs(a) > 0.01) {
        float ergo_r = M + sqrt(max(M * M - a * a * camPos.y * camPos.y / (camDist * camDist), 0.0));
        float ergo_ang = ergo_r / u_camera_dist;
        float ergo_glow = 0.005 / (abs(angDist - ergo_ang) + 0.003);
        ergo_glow *= smoothstep(0.0, 0.5, abs(a) / M);
        color += vec3(0.1, 0.15, 0.4) * ergo_glow * u_bloom * 0.3;
    }

    // ── Tone mapping + gamma ────────────────────────────────
    color = color / (1.0 + color);
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}
"#;

// ── Particle shaders (spaghettification + redshift) ─────────────

const PARTICLE_VERT: &str = r#"#version 300 es
uniform mat4 u_view_proj;
uniform float u_point_scale;
in vec3 a_pos;
in vec4 a_color;       // rgb + alpha
in float a_stretch;    // tidal stretch factor (0=sphere, >1=spaghettified)
in float a_redshift;   // gravitational redshift factor
in float a_is_photon;  // 1.0 = photon, 0.0 = massive particle

out vec4 v_color;
out float v_stretch;
out float v_is_photon;

void main() {
    vec4 clip = u_view_proj * vec4(a_pos, 1.0);
    gl_Position = clip;

    // Photons smaller, massive particles bigger when stretched
    float base_size = a_is_photon > 0.5 ? u_point_scale * 0.4 : u_point_scale;
    float stretch_size = base_size * (1.0 + a_stretch * 0.5);
    gl_PointSize = clamp(stretch_size / clip.w, 2.0, 60.0);

    // Apply gravitational redshift to color
    // Redshift: frequency decreases → color shifts R, brightness drops
    vec3 col = a_color.rgb;
    float z = a_redshift; // 0=no shift, 1=infinite redshift
    // Shift hue toward red and dim
    col = mix(col, col * vec3(1.2, 0.5, 0.3), z * 0.7);
    col *= (1.0 - z * 0.6); // dim near horizon

    v_color = vec4(col, a_color.a);
    v_stretch = a_stretch;
    v_is_photon = a_is_photon;
}
"#;

const PARTICLE_FRAG: &str = r#"#version 300 es
precision highp float;
in vec4 v_color;
in float v_stretch;
in float v_is_photon;
out vec4 fragColor;

void main() {
    vec2 c = gl_PointCoord - 0.5;

    // Spaghettification: stretch along y (radial direction in screen space)
    // As tidal force increases, particle elongates
    float sx = 1.0;
    float sy = 1.0 / (1.0 + v_stretch * 2.0); // squash x, stretch y
    vec2 stretched = vec2(c.x * (1.0 + v_stretch), c.y * sy);
    float d = length(stretched);

    float a;
    if (v_is_photon > 0.5) {
        // Photons: sharp bright core with wide glow
        a = exp(-d * d * 20.0);
        float glow = exp(-d * 3.0) * 0.8;
        fragColor = vec4(v_color.rgb * (a + glow) * 2.0, (a + glow * 0.5));
    } else {
        // Massive particles: soft circle with spaghetti stretch
        a = smoothstep(0.5, 0.15, d);
        float glow = exp(-d * 4.0) * 0.5;

        // Hot glow when highly stretched (tidal heating)
        vec3 heat = mix(vec3(0.0), vec3(1.0, 0.6, 0.2), smoothstep(0.5, 3.0, v_stretch));

        fragColor = vec4((v_color.rgb + heat) * (a + glow), v_color.a * a);
    }
}
"#;

const TRAIL_VERT: &str = r#"#version 300 es
uniform mat4 u_view_proj;
uniform float u_bh_mass;
in vec3 a_pos;
in float a_age;
out float v_age;
out float v_redshift;

void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
    v_age = a_age;
    // Gravitational redshift based on distance from BH
    float r = length(a_pos);
    float rs = 2.0 * u_bh_mass;
    v_redshift = 1.0 - sqrt(max(1.0 - rs / max(r, rs * 0.6), 0.0));
}
"#;

const TRAIL_FRAG: &str = r#"#version 300 es
precision highp float;
uniform vec3 u_trail_color;
uniform float u_is_photon_trail;
in float v_age;
in float v_redshift;
out vec4 fragColor;

void main() {
    float alpha = (1.0 - v_age) * 0.6;

    // Apply gravitational redshift to trail color
    vec3 col = u_trail_color;
    col = mix(col, col * vec3(1.3, 0.4, 0.2), v_redshift * 0.8);
    col *= (1.0 - v_redshift * 0.5);

    // Photon trails: brighter, thinner fade
    if (u_is_photon_trail > 0.5) {
        alpha *= 1.5;
        col *= 1.5;
    }

    fragColor = vec4(col * (1.0 - v_age * 0.5), alpha);
}
"#;

// ── Physics ─────────────────────────────────────────────────────

const MAX_PARTICLES: usize = 50;
const TRAIL_LENGTH: usize = 200;
const SUBSTEPS: usize = 8;

#[derive(Clone)]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    trail: Vec<[f32; 3]>,
    alive: bool,
    color: [f32; 3],
    tidal_stretch: f32,
    redshift: f32,       // gravitational redshift: 0=none, 1=infinite
    time_dilation: f32,  // proper time factor: √(1 - rs/r)
    is_photon: bool,     // true = null geodesic (light), false = massive particle
    age: f32,
}

impl Particle {
    fn new(pos: [f32; 3], vel: [f32; 3], color: [f32; 3], is_photon: bool) -> Self {
        Self {
            pos,
            vel,
            trail: vec![pos],
            alive: true,
            color,
            tidal_stretch: 0.0,
            redshift: 0.0,
            time_dilation: 1.0,
            is_photon,
            age: 0.0,
        }
    }

    fn step(&mut self, dt: f32, bh_mass: f32, bh_spin: f32) {
        let gm = bh_mass;
        let a = bh_spin * bh_mass;
        let r_h = bh_mass + (bh_mass * bh_mass - a * a).max(0.0).sqrt();
        let rs = 2.0 * bh_mass;

        let r = vec_len(&self.pos);
        self.time_dilation = (1.0 - rs / r.max(rs * 0.6)).max(0.0).sqrt();
        self.redshift = 1.0 - self.time_dilation;

        if self.is_photon {
            self.step_photon(bh_mass, bh_spin, r_h, rs, a);
        } else {
            self.step_massive(dt, gm, a, r_h, rs);
        }

        self.age += dt;
    }

    /// Photon: trace using affine parameter steps (same as GPU shader).
    /// Photons complete many steps per frame because light is fast.
    /// Records trail points along the way for visualization.
    fn step_photon(&mut self, _bh_mass: f32, _bh_spin: f32, r_h: f32, rs: f32, a: f32) {
        let gm = rs * 0.5;
        // Use same step size and count as GPU shader for consistency
        let step_size: f32 = 0.08;
        let steps_per_frame: usize = 20; // 20 steps per frame = visible travel speed

        for _ in 0..steps_per_frame {
            let r = vec_len(&self.pos);

            if r < r_h * 0.5 {
                self.alive = false;
                return;
            }
            if r > 60.0 {
                self.alive = false;
                return;
            }

            let r2 = r * r;
            let r3 = r2 * r;

            // ── Null geodesic (matches GPU shader) ──
            // For photons: accel = -3GM·h²/(r⁵) x̂
            // This is the ONLY term for null geodesics.
            // Photon sphere: r = 3M = 1.5rs ✓
            let hx = self.pos[1] * self.vel[2] - self.pos[2] * self.vel[1];
            let hy = self.pos[2] * self.vel[0] - self.pos[0] * self.vel[2];
            let hz = self.pos[0] * self.vel[1] - self.pos[1] * self.vel[0];
            let h2 = hx * hx + hy * hy + hz * hz;

            let coeff = -1.5 * rs * h2 / (r3 * r2);
            let mut ax = coeff * self.pos[0];
            let mut ay = coeff * self.pos[1];
            let mut az = coeff * self.pos[2];

            // Kerr frame dragging
            if a.abs() > 0.001 {
                let accel_fd = frame_drag_accel_cpu(&self.pos, &self.vel, gm, a);
                ax += accel_fd[0];
                ay += accel_fd[1];
                az += accel_fd[2];
            }

            // Adaptive step: smaller near horizon (same as shader)
            let adaptive = step_size * (r / (4.0 * r_h)).clamp(0.3, 2.0);

            self.vel[0] += ax * adaptive;
            self.vel[1] += ay * adaptive;
            self.vel[2] += az * adaptive;
            self.pos[0] += self.vel[0] * adaptive;
            self.pos[1] += self.vel[1] * adaptive;
            self.pos[2] += self.vel[2] * adaptive;

            // Tidal stretch
            self.tidal_stretch = (gm / r3).min(5.0);

            // Record trail every few steps
            if self.trail.len() < TRAIL_LENGTH {
                self.trail.push(self.pos);
            }
        }

        // Trim trail
        while self.trail.len() > TRAIL_LENGTH {
            self.trail.remove(0);
        }
    }

    /// Massive particle: integrate using frame dt with time dilation.
    fn step_massive(&mut self, dt: f32, gm: f32, a: f32, r_h: f32, _rs: f32) {
        let effective_dt = dt * self.time_dilation.max(0.05);
        let sub_dt = effective_dt / SUBSTEPS as f32;

        for _ in 0..SUBSTEPS {
            let r = vec_len(&self.pos);

            if r < r_h * 0.6 {
                self.alive = false;
                return;
            }
            if r > 100.0 {
                self.alive = false;
                return;
            }

            let r2 = r * r;
            let r3 = r2 * r;

            // Newtonian gravity: a = -GM/r³ x
            let mut ax = -gm * self.pos[0] / r3;
            let mut ay = -gm * self.pos[1] / r3;
            let mut az = -gm * self.pos[2] / r3;

            // GR precession: -3GM L²/(r⁵) x
            let lx = self.pos[1] * self.vel[2] - self.pos[2] * self.vel[1];
            let ly = self.pos[2] * self.vel[0] - self.pos[0] * self.vel[2];
            let lz = self.pos[0] * self.vel[1] - self.pos[1] * self.vel[0];
            let l2 = lx * lx + ly * ly + lz * lz;

            let gr_factor = -3.0 * gm * l2 / (r3 * r2);
            ax += gr_factor * self.pos[0];
            ay += gr_factor * self.pos[1];
            az += gr_factor * self.pos[2];

            // Kerr frame dragging (Lense-Thirring, matches GPU shader)
            if a.abs() > 0.001 {
                let fd = frame_drag_accel_cpu(&self.pos, &self.vel, gm, a);
                ax += fd[0];
                ay += fd[1];
                az += fd[2];
            }

            self.vel[0] += ax * sub_dt;
            self.vel[1] += ay * sub_dt;
            self.vel[2] += az * sub_dt;
            self.pos[0] += self.vel[0] * sub_dt;
            self.pos[1] += self.vel[1] * sub_dt;
            self.pos[2] += self.vel[2] * sub_dt;

            // Tidal force: F_tidal ~ GM/r³ (spaghettification)
            self.tidal_stretch = (gm / r3).min(5.0);
        }

        self.trail.push(self.pos);
        if self.trail.len() > TRAIL_LENGTH {
            self.trail.remove(0);
        }
    }
}

fn vec_len(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Frame dragging acceleration matching GPU shader's frame_drag_accel()
/// GPU: cross((0,1,0), pos) = (pos.z, 0, -pos.x)
/// drag_vel = omega * cross(spin_axis, pos)
/// return (drag_vel - vel * omega) * 0.5
fn frame_drag_accel_cpu(pos: &[f32; 3], vel: &[f32; 3], gm: f32, a: f32) -> [f32; 3] {
    let r = vec_len(pos);
    if r < 0.01 {
        return [0.0; 3];
    }
    let r3 = r * r * r;
    let omega = 2.0 * gm * a / r3;
    // cross(ŷ, pos) = (pos[2], 0, -pos[0])
    let drag_vx = omega * pos[2];
    let drag_vy = 0.0;
    let drag_vz = omega * (-pos[0]);
    [
        (drag_vx - vel[0] * omega) * 0.5,
        (drag_vy - vel[1] * omega) * 0.5,
        (drag_vz - vel[2] * omega) * 0.5,
    ]
}

// ── Hyperparameters (read from DOM sliders) ─────────────────────

struct Params {
    bh_mass: f32,
    bh_spin: f32,
    disk_inner: f32,
    disk_outer: f32,
    disk_temp: f32,
    disk_opacity: f32,
    doppler: f32,
    redshift: f32,
    bloom: f32,
    step_size: f32,
    max_steps: i32,
    fov: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            bh_mass: 0.5,
            bh_spin: 0.0,
            disk_inner: 3.0,
            disk_outer: 12.0,
            disk_temp: 1.0,
            disk_opacity: 0.9,
            doppler: 1.0,
            redshift: 1.0,
            bloom: 0.1,
            step_size: 0.08,
            max_steps: 300,
            fov: 1.2,
        }
    }
}

fn read_slider(document: &web_sys::Document, id: &str, default: f32) -> f32 {
    document
        .get_element_by_id(id)
        .and_then(|el| el.dyn_into::<HtmlInputElement>().ok())
        .and_then(|input| input.value().parse::<f32>().ok())
        .unwrap_or(default)
}

fn read_params(document: &web_sys::Document) -> Params {
    let d = Params::default();
    Params {
        bh_mass: read_slider(document, "sl-mass", d.bh_mass),
        bh_spin: read_slider(document, "sl-spin", d.bh_spin),
        disk_inner: read_slider(document, "sl-disk-inner", d.disk_inner),
        disk_outer: read_slider(document, "sl-disk-outer", d.disk_outer),
        disk_temp: read_slider(document, "sl-disk-temp", d.disk_temp),
        disk_opacity: read_slider(document, "sl-disk-opacity", d.disk_opacity),
        doppler: read_slider(document, "sl-doppler", d.doppler),
        redshift: read_slider(document, "sl-redshift", d.redshift),
        bloom: read_slider(document, "sl-bloom", d.bloom),
        step_size: read_slider(document, "sl-step-size", d.step_size),
        max_steps: read_slider(document, "sl-max-steps", d.max_steps as f32) as i32,
        fov: read_slider(document, "sl-fov", d.fov),
    }
}

// ── State ───────────────────────────────────────────────────────

struct State {
    gl: GL,
    bg_program: WebGlProgram,
    particle_program: WebGlProgram,
    trail_program: WebGlProgram,
    time: f64,
    last_time: f64,
    camera_dist: f32,
    camera_theta: f32,
    camera_phi: f32,
    mouse_down: bool,
    last_mouse: (f32, f32),
    mouse_dragged: bool,
    particles: Vec<Particle>,
    canvas_width: u32,
    canvas_height: u32,
    spawn_color_idx: usize,
}

const SPAWN_COLORS: [[f32; 3]; 6] = [
    [0.2, 0.8, 1.0],
    [1.0, 0.4, 0.2],
    [0.3, 1.0, 0.3],
    [1.0, 0.2, 0.8],
    [1.0, 1.0, 0.2],
    [0.5, 0.3, 1.0],
];

impl State {
    fn camera_pos(&self) -> [f32; 3] {
        [
            self.camera_dist * self.camera_theta.sin() * self.camera_phi.cos(),
            self.camera_dist * self.camera_theta.cos(),
            self.camera_dist * self.camera_theta.sin() * self.camera_phi.sin(),
        ]
    }

    fn view_proj_matrix(&self, fov: f32) -> [f32; 16] {
        let cam = self.camera_pos();
        let fwd = normalize(&[-cam[0], -cam[1], -cam[2]]);
        let world_up = [0.0f32, 1.0, 0.0];
        let right = normalize(&cross(&fwd, &world_up));
        let up = cross(&right, &fwd);

        let view: [f32; 16] = [
            right[0], up[0], -fwd[0], 0.0,
            right[1], up[1], -fwd[1], 0.0,
            right[2], up[2], -fwd[2], 0.0,
            -dot(&right, &cam), -dot(&up, &cam), dot(&fwd, &cam), 1.0,
        ];

        let aspect = self.canvas_width as f32 / self.canvas_height.max(1) as f32;
        let near = 0.1f32;
        let far = 200.0f32;
        let f = 1.0 / (fov * 0.5).tan();

        let proj: [f32; 16] = [
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) / (near - far), -1.0,
            0.0, 0.0, 2.0 * far * near / (near - far), 0.0,
        ];

        mat4_mul(&proj, &view)
    }

    fn spawn_particle(&mut self, screen_x: f32, screen_y: f32, bh_mass: f32, is_photon: bool, fov: f32) {
        if self.particles.len() >= MAX_PARTICLES {
            if let Some(idx) = self.particles.iter().position(|p| !p.alive) {
                self.particles.remove(idx);
            } else {
                self.particles.remove(0);
            }
        }

        let cam = self.camera_pos();
        let fwd = normalize(&[-cam[0], -cam[1], -cam[2]]);
        let world_up = [0.0f32, 1.0, 0.0];
        let right = normalize(&cross(&fwd, &world_up));
        let up = cross(&right, &fwd);

        let w = self.canvas_width as f32;
        let h = self.canvas_height as f32;
        let min_dim = w.min(h);
        let ndc_x = (screen_x - w * 0.5) / (min_dim * 0.5);
        let ndc_y = -(screen_y - h * 0.5) / (min_dim * 0.5);

        if is_photon {
            // Photon: spawn from camera, shoot in the ray direction
            // matching exactly how the GPU ray tracer casts rays.
            // This means the photon follows the same path as the pixel ray.
            let pos = [
                cam[0],
                cam[1],
                cam[2],
            ];
            // Ray direction: same as fragment shader camera model
            // rd = normalize(forward * fov + right * uv.x + up * uv.y)
            let fov_val = fov;
            let dir = [
                fwd[0] * fov_val + right[0] * ndc_x + up[0] * ndc_y,
                fwd[1] * fov_val + right[1] * ndc_x + up[1] * ndc_y,
                fwd[2] * fov_val + right[2] * ndc_x + up[2] * ndc_y,
            ];
            let vel = normalize(&dir); // photons: |v| = 1 (c = 1 in natural units)

            // Photon color: white-yellow
            let color = [1.0f32, 0.95, 0.7];
            self.particles.push(Particle::new(pos, vel, color, true));
        } else {
            // Massive particle
            let spawn_dist = 15.0;
            let pos = [
                cam[0] + fwd[0] * spawn_dist + right[0] * ndc_x * spawn_dist * 0.5
                    + up[0] * ndc_y * spawn_dist * 0.5,
                cam[1] + fwd[1] * spawn_dist + right[1] * ndc_x * spawn_dist * 0.5
                    + up[1] * ndc_y * spawn_dist * 0.5,
                cam[2] + fwd[2] * spawn_dist + right[2] * ndc_x * spawn_dist * 0.5
                    + up[2] * ndc_y * spawn_dist * 0.5,
            ];

            let r = vec_len(&pos);
            let to_center = normalize(&[-pos[0], -pos[1], -pos[2]]);
            let v_orbit = (bh_mass / r).sqrt();
            let tangent = normalize(&cross(&to_center, &[0.0, 1.0, 0.0]));

            let vel = [
                tangent[0] * v_orbit * 0.8,
                tangent[1] * v_orbit * 0.8,
                tangent[2] * v_orbit * 0.8,
            ];

            let color = SPAWN_COLORS[self.spawn_color_idx % SPAWN_COLORS.len()];
            self.spawn_color_idx += 1;
            self.particles.push(Particle::new(pos, vel, color, false));
        }
    }
}

// ── Linear algebra ──────────────────────────────────────────────

fn normalize(v: &[f32; 3]) -> [f32; 3] {
    let l = vec_len(v);
    if l < 1e-10 { return [0.0; 3]; }
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            out[j * 4 + i] = (0..4).map(|k| a[k * 4 + i] * b[j * 4 + k]).sum();
        }
    }
    out
}

// ── WebGL helpers ───────────────────────────────────────────────

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl.create_shader(shader_type).ok_or("cannot create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);
    if gl.get_shader_parameter(&shader, GL::COMPILE_STATUS).as_bool().unwrap_or(false) {
        Ok(shader)
    } else {
        let log = gl.get_shader_info_log(&shader).unwrap_or_default();
        gl.delete_shader(Some(&shader));
        Err(log)
    }
}

fn create_program(gl: &GL, vert: &str, frag: &str) -> Result<WebGlProgram, String> {
    let vs = compile_shader(gl, GL::VERTEX_SHADER, vert)?;
    let fs = compile_shader(gl, GL::FRAGMENT_SHADER, frag)?;
    let program = gl.create_program().ok_or("cannot create program")?;
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);
    if gl.get_program_parameter(&program, GL::LINK_STATUS).as_bool().unwrap_or(false) {
        Ok(program)
    } else {
        let log = gl.get_program_info_log(&program).unwrap_or_default();
        Err(log)
    }
}

// ── Entry point ─────────────────────────────────────────────────

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()?;

    let w = window.inner_width()?.as_f64().unwrap() as u32;
    let h = window.inner_height()?.as_f64().unwrap() as u32;
    canvas.set_width(w);
    canvas.set_height(h);

    let gl: GL = canvas.get_context("webgl2")?.unwrap().dyn_into::<GL>()?;

    let bg_program = create_program(&gl, VERTEX_SHADER, FRAGMENT_SHADER)
        .map_err(|e| JsValue::from_str(&e))?;
    let particle_program = create_program(&gl, PARTICLE_VERT, PARTICLE_FRAG)
        .map_err(|e| JsValue::from_str(&format!("particle: {e}")))?;
    let trail_program = create_program(&gl, TRAIL_VERT, TRAIL_FRAG)
        .map_err(|e| JsValue::from_str(&format!("trail: {e}")))?;

    let vertices: [f32; 12] = [
        -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    ];
    let bg_buffer = gl.create_buffer().unwrap();
    gl.bind_buffer(GL::ARRAY_BUFFER, Some(&bg_buffer));
    unsafe {
        let arr = js_sys::Float32Array::view(&vertices);
        gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &arr, GL::STATIC_DRAW);
    }

    let particle_buffer = gl.create_buffer().unwrap();
    let trail_buffer = gl.create_buffer().unwrap();

    let state = Rc::new(RefCell::new(State {
        gl,
        bg_program,
        particle_program,
        trail_program,
        time: 0.0,
        last_time: 0.0,
        camera_dist: 25.0,
        camera_theta: 1.2,
        camera_phi: 0.0,
        mouse_down: false,
        last_mouse: (0.0, 0.0),
        mouse_dragged: false,
        particles: Vec::new(),
        canvas_width: w,
        canvas_height: h,
        spawn_color_idx: 0,
    }));

    // ── Mouse events ────────────────────────────────────────
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = s.borrow_mut();
            st.mouse_down = true;
            st.mouse_dragged = false;
            st.last_mouse = (e.client_x() as f32, e.client_y() as f32);
        });
        canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = s.borrow_mut();
            if st.mouse_down && !st.mouse_dragged {
                let doc = web_sys::window().unwrap().document().unwrap();
                let params = read_params(&doc);
                let is_photon = e.shift_key(); // Shift+click = photon
                st.spawn_particle(e.client_x() as f32, e.client_y() as f32, params.bh_mass, is_photon, params.fov);
            }
            st.mouse_down = false;
        });
        canvas.add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            let mut st = s.borrow_mut();
            if st.mouse_down {
                let x = e.client_x() as f32;
                let y = e.client_y() as f32;
                let dx = x - st.last_mouse.0;
                let dy = y - st.last_mouse.1;
                if dx.abs() > 2.0 || dy.abs() > 2.0 {
                    st.mouse_dragged = true;
                }
                if st.mouse_dragged {
                    st.camera_phi += dx * 0.005;
                    st.camera_theta = (st.camera_theta + dy * 0.005).clamp(0.1, PI - 0.1);
                }
                st.last_mouse = (x, y);
            }
        });
        canvas.add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }
    {
        let s = state.clone();
        let cb = Closure::<dyn FnMut(WheelEvent)>::new(move |e: WheelEvent| {
            e.prevent_default();
            let mut st = s.borrow_mut();
            st.camera_dist = (st.camera_dist + e.delta_y() as f32 * 0.02).clamp(5.0, 80.0);
        });
        canvas.add_event_listener_with_callback_and_add_event_listener_options(
            "wheel",
            cb.as_ref().unchecked_ref(),
            web_sys::AddEventListenerOptions::new().passive(false),
        )?;
        cb.forget();
    }
    {
        let cb = Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| {
            e.prevent_default();
        });
        canvas.add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())?;
        cb.forget();
    }

    // ── Animation loop ──────────────────────────────────────
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let start_time = window.performance().unwrap().now();

    let bg_buf = bg_buffer;
    let p_buf = particle_buffer;
    let t_buf = trail_buffer;

    *g.borrow_mut() = Some(Closure::new(move |timestamp: f64| {
        let mut st = state.borrow_mut();
        let now = (timestamp - start_time) / 1000.0;
        let dt = ((now - st.last_time) as f32).min(0.05);
        st.last_time = now;
        st.time = now;

        // Read hyperparameters from sliders
        let doc = web_sys::window().unwrap().document().unwrap();
        let params = read_params(&doc);

        // Physics update (particles use current BH params)
        for p in st.particles.iter_mut() {
            if p.alive {
                p.step(dt, params.bh_mass, params.bh_spin);
            }
        }
        st.particles.retain(|p| p.alive || p.age < 5.0);

        // Resize
        {
            let gl = &st.gl;
            let canvas: HtmlCanvasElement = gl.canvas().unwrap().dyn_into().unwrap();
            let w = canvas.client_width() as u32;
            let h = canvas.client_height() as u32;
            if canvas.width() != w || canvas.height() != h {
                canvas.set_width(w);
                canvas.set_height(h);
                gl.viewport(0, 0, w as i32, h as i32);
            }
            st.canvas_width = w;
            st.canvas_height = h;
        }

        let gl = &st.gl;
        let w = st.canvas_width;
        let h = st.canvas_height;

        // ── Draw black hole (GPU ray tracer) ────────────────
        gl.disable(GL::BLEND);
        gl.disable(GL::DEPTH_TEST);
        gl.use_program(Some(&st.bg_program));

        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&bg_buf));
        let pos_loc = gl.get_attrib_location(&st.bg_program, "a_position") as u32;
        gl.enable_vertex_attrib_array(pos_loc);
        gl.vertex_attrib_pointer_with_i32(pos_loc, 2, GL::FLOAT, false, 0, 0);

        let loc = |name: &str| gl.get_uniform_location(&st.bg_program, name);

        // Camera uniforms
        gl.uniform2f(loc("u_resolution").as_ref(), w as f32, h as f32);
        gl.uniform1f(loc("u_time").as_ref(), st.time as f32);
        gl.uniform1f(loc("u_camera_dist").as_ref(), st.camera_dist);
        gl.uniform1f(loc("u_camera_theta").as_ref(), st.camera_theta);
        gl.uniform1f(loc("u_camera_phi").as_ref(), st.camera_phi);
        gl.uniform1f(loc("u_fov").as_ref(), params.fov);

        // Black hole uniforms
        gl.uniform1f(loc("u_bh_mass").as_ref(), params.bh_mass);
        gl.uniform1f(loc("u_bh_spin").as_ref(), params.bh_spin);

        // Disk uniforms
        gl.uniform1f(loc("u_disk_inner").as_ref(), params.disk_inner);
        gl.uniform1f(loc("u_disk_outer").as_ref(), params.disk_outer);
        gl.uniform1f(loc("u_disk_temp").as_ref(), params.disk_temp);
        gl.uniform1f(loc("u_disk_opacity").as_ref(), params.disk_opacity);

        // Physics uniforms
        gl.uniform1f(loc("u_doppler").as_ref(), params.doppler);
        gl.uniform1f(loc("u_redshift").as_ref(), params.redshift);
        gl.uniform1f(loc("u_bloom").as_ref(), params.bloom);

        // Integration uniforms
        gl.uniform1f(loc("u_step_size").as_ref(), params.step_size);
        gl.uniform1i(loc("u_max_steps").as_ref(), params.max_steps);

        gl.draw_arrays(GL::TRIANGLES, 0, 6);

        // ── Draw trails (with gravitational redshift) ────────
        let vp = st.view_proj_matrix(params.fov);
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE);

        let mut trail_data: Vec<f32> = Vec::new();
        let mut trail_counts: Vec<(usize, [f32; 3], bool)> = Vec::new(); // (count, color, is_photon)

        for p in &st.particles {
            if p.trail.len() < 2 { continue; }
            let len = p.trail.len();
            for (i, pos) in p.trail.iter().enumerate() {
                trail_data.push(pos[0]);
                trail_data.push(pos[1]);
                trail_data.push(pos[2]);
                trail_data.push(i as f32 / len as f32);
            }
            trail_counts.push((p.trail.len(), p.color, p.is_photon));
        }

        if !trail_data.is_empty() {
            gl.use_program(Some(&st.trail_program));
            gl.bind_buffer(GL::ARRAY_BUFFER, Some(&t_buf));
            unsafe {
                let arr = js_sys::Float32Array::view(&trail_data);
                gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &arr, GL::DYNAMIC_DRAW);
            }

            let vp_loc = gl.get_uniform_location(&st.trail_program, "u_view_proj");
            gl.uniform_matrix4fv_with_f32_array(vp_loc.as_ref(), false, &vp);

            let mass_loc = gl.get_uniform_location(&st.trail_program, "u_bh_mass");
            gl.uniform1f(mass_loc.as_ref(), params.bh_mass);

            let pos_loc = gl.get_attrib_location(&st.trail_program, "a_pos") as u32;
            let age_loc = gl.get_attrib_location(&st.trail_program, "a_age") as u32;
            gl.enable_vertex_attrib_array(pos_loc);
            gl.enable_vertex_attrib_array(age_loc);
            gl.vertex_attrib_pointer_with_i32(pos_loc, 3, GL::FLOAT, false, 16, 0);
            gl.vertex_attrib_pointer_with_i32(age_loc, 1, GL::FLOAT, false, 16, 12);

            let color_loc = gl.get_uniform_location(&st.trail_program, "u_trail_color");
            let photon_loc = gl.get_uniform_location(&st.trail_program, "u_is_photon_trail");

            let mut offset = 0i32;
            for (count, color, is_photon) in &trail_counts {
                gl.uniform3f(color_loc.as_ref(), color[0], color[1], color[2]);
                gl.uniform1f(photon_loc.as_ref(), if *is_photon { 1.0 } else { 0.0 });
                gl.draw_arrays(GL::LINE_STRIP, offset, *count as i32);
                offset += *count as i32;
            }
            gl.disable_vertex_attrib_array(age_loc);
        }

        // ── Draw particles (spaghettification + redshift) ───
        // Vertex layout: pos(3) + color(4) + stretch(1) + redshift(1) + is_photon(1) = 10 floats = 40 bytes
        let mut p_data: Vec<f32> = Vec::new();
        let mut p_count = 0i32;

        for p in &st.particles {
            if !p.alive { continue; }
            // Position
            p_data.push(p.pos[0]);
            p_data.push(p.pos[1]);
            p_data.push(p.pos[2]);
            // Color (with tidal heating glow for massive particles)
            let heat = if !p.is_photon {
                (1.0 + p.tidal_stretch * 1.5).min(4.0)
            } else {
                1.0
            };
            p_data.push(p.color[0] * heat);
            p_data.push(p.color[1] * heat);
            p_data.push(p.color[2] * heat);
            p_data.push(1.0);
            // Spaghettification stretch
            p_data.push(p.tidal_stretch);
            // Gravitational redshift
            p_data.push(p.redshift);
            // Is photon
            p_data.push(if p.is_photon { 1.0 } else { 0.0 });
            p_count += 1;
        }

        if p_count > 0 {
            gl.use_program(Some(&st.particle_program));
            gl.bind_buffer(GL::ARRAY_BUFFER, Some(&p_buf));
            unsafe {
                let arr = js_sys::Float32Array::view(&p_data);
                gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &arr, GL::DYNAMIC_DRAW);
            }

            let vp_loc = gl.get_uniform_location(&st.particle_program, "u_view_proj");
            gl.uniform_matrix4fv_with_f32_array(vp_loc.as_ref(), false, &vp);

            let scale_loc = gl.get_uniform_location(&st.particle_program, "u_point_scale");
            gl.uniform1f(scale_loc.as_ref(), h as f32 * 0.15);

            // Stride: 10 floats = 40 bytes
            let stride = 40;
            let pos_loc = gl.get_attrib_location(&st.particle_program, "a_pos") as u32;
            let col_loc = gl.get_attrib_location(&st.particle_program, "a_color") as u32;
            let stretch_loc = gl.get_attrib_location(&st.particle_program, "a_stretch") as u32;
            let redshift_loc = gl.get_attrib_location(&st.particle_program, "a_redshift") as u32;
            let photon_loc = gl.get_attrib_location(&st.particle_program, "a_is_photon") as u32;

            gl.enable_vertex_attrib_array(pos_loc);
            gl.enable_vertex_attrib_array(col_loc);
            gl.enable_vertex_attrib_array(stretch_loc);
            gl.enable_vertex_attrib_array(redshift_loc);
            gl.enable_vertex_attrib_array(photon_loc);

            gl.vertex_attrib_pointer_with_i32(pos_loc, 3, GL::FLOAT, false, stride, 0);
            gl.vertex_attrib_pointer_with_i32(col_loc, 4, GL::FLOAT, false, stride, 12);
            gl.vertex_attrib_pointer_with_i32(stretch_loc, 1, GL::FLOAT, false, stride, 28);
            gl.vertex_attrib_pointer_with_i32(redshift_loc, 1, GL::FLOAT, false, stride, 32);
            gl.vertex_attrib_pointer_with_i32(photon_loc, 1, GL::FLOAT, false, stride, 36);

            gl.draw_arrays(GL::POINTS, 0, p_count);

            gl.disable_vertex_attrib_array(col_loc);
            gl.disable_vertex_attrib_array(stretch_loc);
            gl.disable_vertex_attrib_array(redshift_loc);
            gl.disable_vertex_attrib_array(photon_loc);
        }

        drop(st);
        web_sys::window()
            .unwrap()
            .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }));

    window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    Ok(())
}
