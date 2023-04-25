// Ref: https://github.com/rustformers/llama-rs/blob/1b20306d/ggml/src/lib.rs

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::convert::TryFrom;

use inference::{InferenceError, InferenceSession};
use vocabulary::{TokenId, Vocabulary};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub mod computation_graph;
pub mod context;
pub mod element_type;
pub mod inference;
pub mod loader;
pub mod tensor;
pub mod vocabulary;

pub trait Model {
    fn context_size(&self) -> &usize;
    fn vocabulary(&self) -> &Vocabulary;
    fn start_session(&self) -> InferenceSession;
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input: &[TokenId],
    ) -> Result<(), InferenceError>;
}

/// A buffer of memory that can be used as a scratch buffer for a [Context].
///
/// See [Context::use_scratch].
pub struct Buffer {
    data: Box<[u8]>,
}

impl Buffer {
    /// Creates a new buffer of the specified size.
    pub fn new(size: usize) -> Self {
        let mut data: Vec<u8> = Vec::with_capacity(size);

        // SAFETY: The contents are intentionally uninitialized, as they will be passed to
        // the ggml C API which will fill them with data.
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(size);
        }

        Buffer {
            data: data.into_boxed_slice(),
        }
    }
}

#[macro_export]
macro_rules! mulf {
    ($term:expr, $($terms:expr),*) => {
        usize::try_from((($term as f64) $(* ($terms as f64))*) as u64).unwrap()
    };
}

fn usize_to_i32(val: usize) -> i32 {
    i32::try_from(val).unwrap()
}

fn usize_to_i64(val: usize) -> i64 {
    i64::try_from(val).unwrap()
}
