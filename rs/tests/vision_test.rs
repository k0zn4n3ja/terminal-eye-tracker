//! Pure-math tests for `tmux_eyes::vision`.
//!
//! Mirrors `tests/test_vision.py`. Does NOT instantiate `FaceTracker` -
//! that pulls ORT and downloads a model, far too heavy for a unit test.

use approx::assert_relative_eq;
use tmux_eyes::vision::{compute_head_yaw, compute_iris_ratio};

// ===========================================================================
// compute_iris_ratio
// ===========================================================================

#[test]
fn iris_centered_both_eyes_ratio_half() {
    let r = compute_iris_ratio(50.0, 0.0, 100.0, 50.0, 0.0, 100.0);
    assert_relative_eq!(r, 0.5, epsilon = 1e-6);
}

#[test]
fn iris_near_inner_corner_both_eyes_below_threshold() {
    let r = compute_iris_ratio(25.0, 0.0, 100.0, 25.0, 0.0, 100.0);
    assert!(r < 0.42, "expected r < 0.42, got {}", r);
}

#[test]
fn iris_near_outer_corner_both_eyes_above_threshold() {
    let r = compute_iris_ratio(75.0, 0.0, 100.0, 75.0, 0.0, 100.0);
    assert!(r > 0.58, "expected r > 0.58, got {}", r);
}

#[test]
fn iris_asymmetric_left_looking_left_right_centered() {
    let r = compute_iris_ratio(25.0, 0.0, 100.0, 50.0, 0.0, 100.0);
    let expected = (0.25 + 0.50) / 2.0; // 0.375
    assert_relative_eq!(r, expected, epsilon = 1e-3);
}

#[test]
fn iris_unit_agnostic_pixel_vs_normalized() {
    let r_px = compute_iris_ratio(50.0, 0.0, 100.0, 50.0, 0.0, 100.0);
    let r_norm = compute_iris_ratio(0.5, 0.0, 1.0, 0.5, 0.0, 1.0);
    assert_relative_eq!(r_px, r_norm, epsilon = 1e-9);
}

#[test]
fn iris_degenerate_zero_span_returns_half_fallback() {
    // Left eye has zero span (inner == outer); right eye centred.
    let r = compute_iris_ratio(10.0, 10.0, 10.0, 50.0, 0.0, 100.0);
    // Left -> 0.5 fallback, right -> 0.5, average -> 0.5.
    assert_relative_eq!(r, 0.5, epsilon = 1e-6);
}

// ===========================================================================
// compute_head_yaw
// ===========================================================================
//
// Strategy: project the canonical 6-point 3D model with a known yaw
// rotation through a pinhole camera (focal = image width, principal point
// at image centre, distance 500 mm). compute_head_yaw must recover sign
// and rough magnitude of that rotation.

const IMAGE_W: u32 = 640;
const IMAGE_H: u32 = 480;
const IMAGE_SIZE: (u32, u32) = (IMAGE_W, IMAGE_H);

const MODEL_POINTS_3D: [[f32; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
];

/// Project the 3-D model with a pure yaw rotation onto a 640x480 image.
fn project_landmarks(yaw_deg: f32) -> [(f32, f32); 6] {
    let focal = IMAGE_W as f32;
    let cx = IMAGE_W as f32 / 2.0;
    let cy = IMAGE_H as f32 / 2.0;

    let yaw_rad = yaw_deg.to_radians();
    let (sy, cy_r) = (yaw_rad.sin(), yaw_rad.cos());

    // R_y rotation matrix:
    //   [ cos  0  sin ]
    //   [  0   1   0  ]
    //   [-sin  0  cos ]
    let translate_z = 500.0_f32;

    let mut out = [(0.0_f32, 0.0_f32); 6];
    for (i, p) in MODEL_POINTS_3D.iter().enumerate() {
        let (x0, y0, z0) = (p[0], p[1], p[2]);
        let xr = cy_r * x0 + sy * z0;
        let yr = y0;
        let zr = -sy * x0 + cy_r * z0;
        let zc = zr + translate_z;
        let u = focal * xr / zc + cx;
        let v = focal * yr / zc + cy;
        out[i] = (u, v);
    }
    out
}

#[test]
fn yaw_symmetric_face_near_zero() {
    let lm = project_landmarks(0.0);
    let yaw = compute_head_yaw(&lm, IMAGE_SIZE);
    assert!(yaw.abs() < 5.0, "expected |yaw| < 5 deg, got {:.2}", yaw);
}

#[test]
fn yaw_face_rotated_right_positive() {
    let lm = project_landmarks(15.0);
    let yaw = compute_head_yaw(&lm, IMAGE_SIZE);
    assert!(yaw > 5.0, "expected yaw > 5 deg, got {:.2}", yaw);
}

#[test]
fn yaw_face_rotated_left_negative() {
    let lm = project_landmarks(-15.0);
    let yaw = compute_head_yaw(&lm, IMAGE_SIZE);
    assert!(yaw < -5.0, "expected yaw < -5 deg, got {:.2}", yaw);
}

#[test]
fn yaw_sign_convention_right_positive_left_negative() {
    let yaw_r = compute_head_yaw(&project_landmarks(20.0), IMAGE_SIZE);
    let yaw_l = compute_head_yaw(&project_landmarks(-20.0), IMAGE_SIZE);
    assert!(yaw_r > 0.0, "right gaze should be positive yaw, got {:.2}", yaw_r);
    assert!(yaw_l < 0.0, "left gaze should be negative yaw, got {:.2}", yaw_l);
}

#[test]
fn yaw_magnitude_symmetric_within_5_deg() {
    let yaw_r = compute_head_yaw(&project_landmarks(15.0), IMAGE_SIZE);
    let yaw_l = compute_head_yaw(&project_landmarks(-15.0), IMAGE_SIZE);
    let diff = (yaw_r.abs() - yaw_l.abs()).abs();
    assert!(diff < 5.0, "asymmetric magnitudes: r={:.2}, l={:.2}", yaw_r, yaw_l);
}
