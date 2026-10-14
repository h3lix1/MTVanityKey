//! The one-line progress report of a running search.

use std::time::Duration;

pub struct ProgressSnapshot<'a> {
    /// What is being searched for ("prefix" / "pattern") and its text.
    kind: &'static str,
    label: &'a str,
    /// Per-attempt success probability.
    probability: f64,
    attempts: u64,
    cpu_attempts: u64,
    gpu_attempts: u64,
    elapsed: Duration,
    max_attempts: Option<u64>,
}

impl<'a> ProgressSnapshot<'a> {
    /// A snapshot of a search with a fixed per-attempt hit probability
    /// (`kind` is "pattern" or "prefix", `label` its text).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: &'static str,
        label: &'a str,
        probability: f64,
        attempts: u64,
        cpu_attempts: u64,
        gpu_attempts: u64,
        elapsed: Duration,
        max_attempts: Option<u64>,
    ) -> Self {
        Self {
            kind,
            label,
            probability,
            attempts,
            cpu_attempts,
            gpu_attempts,
            elapsed,
            max_attempts,
        }
    }

    pub fn render(&self) -> String {
        let elapsed_secs = self.elapsed.as_secs_f64().max(0.001);
        let total_rate = self.attempts as f64 / elapsed_secs;
        let cpu_rate = self.cpu_attempts as f64 / elapsed_secs;
        let gpu_rate = self.gpu_attempts as f64 / elapsed_secs;
        let probability = self.probability;
        let chance = 1.0 - (-(self.attempts as f64) * probability).exp();
        // ETA seconds can astronomically exceed what Duration represents for
        // long prefixes (e.g. 16^-64 probability), so format from f64 and
        // saturate into years instead of Duration::from_secs_f64 (panics).
        let eta = if total_rate > 0.0 {
            let p50 = std::f64::consts::LN_2 / (probability * total_rate);
            let p95 = -0.05f64.ln() / (probability * total_rate);
            format!("p50 {} p95 {}", format_eta_secs(p50), format_eta_secs(p95))
        } else {
            "warming up".to_string()
        };

        let limit = self.max_attempts.map_or_else(String::new, |max| {
            format!(
                " | limit {:.1}%",
                (self.attempts as f64 / max as f64 * 100.0).min(100.0)
            )
        });

        format!(
            "{} {} | {} attempts | {:.0}/s total ({:.0}/s cpu, {:.0}/s gpu) | elapsed {} | chance {:.1}% | ETA {}{}",
            self.kind,
            self.label,
            self.attempts,
            total_rate,
            cpu_rate,
            gpu_rate,
            format_duration(self.elapsed),
            chance * 100.0,
            eta,
            limit
        )
    }
}

/// Format an ETA given in (possibly astronomical) seconds. Durations beyond
/// 100 years are rendered as years so arbitrarily long prefixes cannot
/// overflow Duration.
fn format_eta_secs(secs: f64) -> String {
    const YEAR_SECS: f64 = 31_557_600.0; // Julian year
    if !secs.is_finite() {
        return "never".to_string();
    }
    if secs >= 100.0 * YEAR_SECS {
        let years = secs / YEAR_SECS;
        return if years >= 1e6 {
            format!("{years:.1e}y")
        } else {
            format!("{years:.0}y")
        };
    }
    format_duration(Duration::from_secs_f64(secs.max(0.0)))
}

pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        let minutes = (secs / 60.0).floor();
        let seconds = secs % 60.0;
        format!("{minutes:.0}m {seconds:.0}s")
    } else {
        let hours = (secs / 3600.0).floor();
        let minutes = ((secs % 3600.0) / 60.0).floor();
        format!("{hours:.0}h {minutes:.0}m")
    }
}
