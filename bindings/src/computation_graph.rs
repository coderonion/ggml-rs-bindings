use crate::{tensor::Tensor, usize_to_i32};

/// A `ggml` computation graph. Keeps track of all state during computation.
pub struct ComputationGraph {
    pub(crate) inner: crate::ggml_cgraph,
}

impl ComputationGraph {
    /// Build this computational graph in the forward direction in preparation for computation.
    pub fn build_forward_expand(&mut self, tensor: &Tensor) {
        unsafe { crate::ggml_build_forward_expand(&mut self.inner, tensor.ptr.as_ptr()) }
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self {
            inner: crate::ggml_cgraph {
                n_threads: usize_to_i32(num_cpus::get()),
                // SAFETY: This should be safe to zero. The original C++ impl
                // just leaves it uninitialized
                ..unsafe { std::mem::zeroed::<crate::ggml_cgraph>() }
            },
        }
    }
}
