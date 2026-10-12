use std::sync::Arc;

use rand::RngExt;
use rustfft::{Fft, num_complex::Complex};

use super::Kernel;
use crate::architectures::{VectorSymbolicArchitecture, named_rng};

/// The dense (bipolar or real-valued) representation of a vector.
pub(crate) fn to_dense<V: VectorSymbolicArchitecture>(
    storage: &V::Storage,
    size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; size];
    V::accumulate_dense(storage, &mut out, 1.0);
    out
}

/// Level `index` out of `count` levels of a pool.
///
/// Each dimension gets a fixed random threshold `t_j ~ U(0, 1)`; level `k` takes the element of the last level where
/// `t_j < k / (count - 1)` and the element of the first level otherwise. Hence, the similarity between two levels
/// decreases linearly with their distance and the first and last level are quasi-orthogonal.
pub(crate) fn level<V: VectorSymbolicArchitecture>(
    vsa: &V,
    size: usize,
    pool: &str,
    index: usize,
    count: usize,
) -> V::Storage {
    debug_assert!(index < count, "level out of range");
    let first = vsa.random_named(size, &format!("{pool}#0"));
    if index == 0 || count < 2 {
        return first;
    }
    let last = vsa.random_named(size, &format!("{pool}#{}", count - 1));
    if index == count - 1 {
        return last;
    }

    let fraction = index as f32 / (count - 1) as f32;
    let mut thresholds = named_rng(vsa.seed(), &format!("{pool}#thresholds"));
    let mixed: Vec<f32> = to_dense::<V>(&first, size)
        .into_iter()
        .zip(to_dense::<V>(&last, size))
        .map(|(first, last)| {
            if thresholds.random::<f32>() < fraction {
                last
            } else {
                first
            }
        })
        .collect();
    vsa.normalize_dense(&mixed)
}

/// Fractional power encoding `B^x = IFFT(exp(i·φ·x))` with conjugate-symmetric random phases `φ`, such that
/// `⟨B^x, B^y⟩ = 1/d · Σ_j cos(φ_j (x - y))`, which approximates the kernel of the phase distribution.
#[derive(Clone)]
pub(crate) struct FractionalPower {
    phases: Vec<f64>,
    inverse_fft: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for FractionalPower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractionalPower")
            .field("size", &self.phases.len())
            .finish_non_exhaustive()
    }
}

impl FractionalPower {
    pub fn new(seed: u64, base: &str, kernel: Kernel, inverse_fft: Arc<dyn Fft<f64>>) -> Self {
        let size = inverse_fft.len();
        let mut rng = named_rng(seed, base);
        let mut phases = vec![0.0; size];
        // The DC component and (for even sizes) the Nyquist component stay at phase 0, keeping `B^x` real for any `x`.
        for j in 1..=(size.saturating_sub(1) / 2) {
            let phase = match kernel {
                Kernel::Sinc => rng.random_range(-std::f64::consts::PI..std::f64::consts::PI),
                Kernel::Gaussian => {
                    // Box-Muller transform.
                    let u1: f64 = 1.0 - rng.random::<f64>();
                    let u2: f64 = rng.random::<f64>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                }
            };
            phases[j] = phase;
            phases[size - j] = -phase;
        }
        Self {
            phases,
            inverse_fft,
        }
    }

    /// The dense, unit-norm representation of `B^x`.
    pub fn dense(&self, x: f64) -> Vec<f32> {
        let size = self.phases.len() as f64;
        let mut buffer: Vec<Complex<f64>> = self
            .phases
            .iter()
            .map(|phase| Complex::from_polar(1.0, phase * x))
            .collect();
        self.inverse_fft.process(&mut buffer);
        // Parseval: the unnormalized inverse transform of d unit magnitudes has norm d; scaling by 1/d yields unit norm.
        buffer
            .into_iter()
            .map(|value| (value.re / size) as f32)
            .collect()
    }
}
