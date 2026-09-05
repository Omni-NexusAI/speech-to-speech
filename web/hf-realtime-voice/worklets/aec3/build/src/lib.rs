use std::cell::RefCell;

use sonora::config::EchoCanceller;
use sonora::{AudioProcessing, Config, StreamConfig};

const ABI_VERSION: i32 = 1;
const MAX_DELAY_MS: i32 = 500;

struct Session {
    apm: AudioProcessing,
    sample_rate_hz: u32,
    frame_samples: usize,
    render: Box<[f32]>,
    render_out: Box<[f32]>,
    capture: Box<[f32]>,
    output: Box<[f32]>,
    strict: bool,
    render_rms: f32,
    capture_rms: f32,
    output_rms: f32,
    double_talk: bool,
}

impl Session {
    fn new(sample_rate_hz: u32) -> Option<Self> {
        if !(8_000..=96_000).contains(&sample_rate_hz) || sample_rate_hz % 100 != 0 {
            return None;
        }
        let stream = StreamConfig::new(sample_rate_hz, 1);
        let frame_samples = stream.num_frames();
        let config = Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Default::default()
        };
        let apm = AudioProcessing::builder()
            .config(config)
            .capture_config(stream)
            .render_config(stream)
            .echo_detector(true)
            .build();
        Some(Self {
            apm,
            sample_rate_hz,
            frame_samples,
            render: vec![0.0; frame_samples].into_boxed_slice(),
            render_out: vec![0.0; frame_samples].into_boxed_slice(),
            capture: vec![0.0; frame_samples].into_boxed_slice(),
            output: vec![0.0; frame_samples].into_boxed_slice(),
            strict: false,
            render_rms: 0.0,
            capture_rms: 0.0,
            output_rms: 0.0,
            double_talk: false,
        })
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let energy = samples.iter().map(|sample| sample * sample).sum::<f32>();
        (energy / samples.len() as f32).sqrt()
    }

    fn process_render(&mut self) -> i32 {
        self.render_rms = Self::rms(&self.render);
        match self
            .apm
            .process_render_f32(&[&self.render], &mut [&mut self.render_out])
        {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }

    fn process_capture(&mut self, delay_ms: i32) -> i32 {
        let _ = self
            .apm
            .set_stream_delay_ms(delay_ms.clamp(0, MAX_DELAY_MS));
        self.capture_rms = Self::rms(&self.capture);
        let status = match self
            .apm
            .process_capture_f32(&[&self.capture], &mut [&mut self.output])
        {
            Ok(()) => 0,
            Err(_) => -1,
        };
        self.output_rms = Self::rms(&self.output);

        // AEC3's internal near-end detector protects double-talk during
        // suppression. Sonora does not expose that internal boolean, so this
        // is diagnostics-only evidence from AEC3's processed output. It never
        // gates or rewrites Adaptive audio.
        let echo_likelihood = self
            .apm
            .statistics()
            .residual_echo_likelihood
            .unwrap_or(1.0);
        self.double_talk = self.render_rms >= 0.001
            && self.capture_rms >= 0.004
            && self.output_rms >= 0.004
            && echo_likelihood < 0.8;
        status
    }
}

thread_local! {
    static SESSIONS: RefCell<Vec<Option<Session>>> = const { RefCell::new(Vec::new()) };
}

fn with_session<R>(handle: i32, fallback: R, callback: impl FnOnce(&Session) -> R) -> R {
    if handle <= 0 {
        return fallback;
    }
    SESSIONS.with_borrow(|sessions| {
        sessions
            .get((handle - 1) as usize)
            .and_then(Option::as_ref)
            .map(callback)
            .unwrap_or(fallback)
    })
}

fn with_session_mut<R>(handle: i32, fallback: R, callback: impl FnOnce(&mut Session) -> R) -> R {
    if handle <= 0 {
        return fallback;
    }
    SESSIONS.with_borrow_mut(|sessions| {
        sessions
            .get_mut((handle - 1) as usize)
            .and_then(Option::as_mut)
            .map(callback)
            .unwrap_or(fallback)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_abi_version() -> i32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_create(sample_rate_hz: i32, channels: i32) -> i32 {
    if channels != 1 || sample_rate_hz <= 0 {
        return 0;
    }
    let Some(session) = Session::new(sample_rate_hz as u32) else {
        return 0;
    };
    SESSIONS.with_borrow_mut(|sessions| {
        if let Some((index, slot)) = sessions
            .iter_mut()
            .enumerate()
            .find(|(_, value)| value.is_none())
        {
            *slot = Some(session);
            (index + 1) as i32
        } else {
            sessions.push(Some(session));
            sessions.len() as i32
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_destroy(handle: i32) {
    if handle <= 0 {
        return;
    }
    SESSIONS.with_borrow_mut(|sessions| {
        if let Some(slot) = sessions.get_mut((handle - 1) as usize) {
            *slot = None;
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_reset(handle: i32) -> i32 {
    with_session_mut(handle, -1, |session| {
        let Some(replacement) = Session::new(session.sample_rate_hz) else {
            return -1;
        };
        *session = replacement;
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_frame_samples(handle: i32) -> i32 {
    with_session(handle, 0, |session| session.frame_samples as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_render_ptr(handle: i32) -> *mut f32 {
    with_session_mut(handle, std::ptr::null_mut(), |session| session.render.as_mut_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_capture_ptr(handle: i32) -> *mut f32 {
    with_session_mut(handle, std::ptr::null_mut(), |session| session.capture.as_mut_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_output_ptr(handle: i32) -> *mut f32 {
    with_session_mut(handle, std::ptr::null_mut(), |session| session.output.as_mut_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_set_suppression(handle: i32, strict: i32) -> i32 {
    with_session_mut(handle, -1, |session| {
        session.strict = strict != 0;
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_process_render(handle: i32, _timestamp_ms: f64) -> i32 {
    with_session_mut(handle, -1, Session::process_render)
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_process_capture(handle: i32, delay_ms: i32, _timestamp_ms: f64) -> i32 {
    with_session_mut(handle, -1, |session| session.process_capture(delay_ms))
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_delay_ms(handle: i32) -> i32 {
    with_session(handle, -1, |session| {
        session.apm.statistics().delay_ms.unwrap_or(-1)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_erle_db(handle: i32) -> f64 {
    with_session(handle, f64::NAN, |session| {
        session
            .apm
            .statistics()
            .echo_return_loss_enhancement
            .unwrap_or(f64::NAN)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_residual_echo_likelihood(handle: i32) -> f64 {
    with_session(handle, f64::NAN, |session| {
        session
            .apm
            .statistics()
            .residual_echo_likelihood
            .unwrap_or(f64::NAN)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_double_talk(handle: i32) -> i32 {
    with_session(handle, -1, |session| i32::from(session.double_talk))
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_render_rms(handle: i32) -> f32 {
    with_session(handle, 0.0, |session| session.render_rms)
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_capture_rms(handle: i32) -> f32 {
    with_session(handle, 0.0, |session| session.capture_rms)
}

#[unsafe(no_mangle)]
pub extern "C" fn aec3_get_output_rms(handle: i32) -> f32 {
    with_session(handle, 0.0, |session| session.output_rms)
}
