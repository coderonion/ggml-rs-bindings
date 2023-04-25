use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use crate::{
    computation_graph::ComputationGraph, element_type::ElementType, tensor::Tensor, usize_to_i32,
    usize_to_i64, Buffer,
};

/// Acts as a RAII-guard over a `ggml_context`, allocating via
/// `ggml_init` and dropping via `ggml_free`.
#[derive(Debug, Clone)]
pub struct Context {
    /// An `Arc` is used to model the relation between the context and the
    /// allocated tensors. Tensors are owned by the object, so a [`Tensor`]
    /// contains a `Weak` reference underneath and doesn't let you do anything
    /// with it if the underlying context has been deallocated.
    ptr: Arc<NonNull<crate::ggml_context>>,
}

impl Context {
    /// Creates a new [Context] with the specified `mem_size` as a working area.
    pub fn init(mem_size: usize, no_alloc: bool) -> Self {
        let raw = unsafe {
            crate::ggml_init(crate::ggml_init_params {
                mem_size,
                no_alloc,
                // Null here means we want ggml to own this memory. We don't
                // support passing an owned buffer from the Rust side.
                mem_buffer: std::ptr::null_mut(),
            })
        };
        Self {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    /// Wraps a raw tensor with a weak pointer to the context.
    fn new_tensor_raw(&self, raw: *mut crate::ggml_tensor) -> Tensor {
        Tensor {
            ptr: NonNull::new(raw).expect("Should not be null"),
            ctx: Arc::downgrade(&self.ptr),
            buf: Default::default(),
        }
    }

    /// Creates a new 1D tensor.
    pub fn new_tensor_1d(&self, typ: ElementType, ne0: usize) -> Tensor {
        let raw =
            unsafe { crate::ggml_new_tensor_1d(self.ptr.as_ptr(), typ.into(), usize_to_i64(ne0)) };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 2D tensor.
    pub fn new_tensor_2d(&self, typ: ElementType, ne0: usize, ne1: usize) -> Tensor {
        let raw = unsafe {
            crate::ggml_new_tensor_2d(
                self.ptr.as_ptr(),
                typ.into(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
            )
        };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 3D tensor.
    pub fn new_tensor_3d(&self, typ: ElementType, ne0: usize, ne1: usize, ne2: usize) -> Tensor {
        let raw = unsafe {
            crate::ggml_new_tensor_3d(
                self.ptr.as_ptr(),
                typ.into(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
            )
        };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 1D tensor with the specified value.
    pub fn new_f32(&self, x: f32) -> Tensor {
        let raw = unsafe { crate::ggml_new_f32(self.ptr.as_ptr(), x) };
        self.new_tensor_raw(raw)
    }

    /// Unknown, aside from the obvious. It's transposing something!
    pub fn op_transpose(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_transpose(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Unknown.
    pub fn op_get_rows(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { crate::ggml_get_rows(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the values of `a`, but normalized.
    pub fn op_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the values of `a`, but normalized using RMSNorm.
    pub fn op_rms_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_rms_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the multiplication of `a` and `b`.
    pub fn op_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_mul(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Unknown.
    pub fn op_repeat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { crate::ggml_repeat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the multiplication of `a` and `b` as if they were matrices.
    ///
    /// `a`: m rows, n columns
    ///
    /// `b`: p rows, n columns (i.e. we transpose it internally)
    ///
    /// Result is m columns, p rows
    pub fn op_mul_mat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { crate::ggml_mul_mat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the addition of `a` and `b`.
    pub fn op_add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_add(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) activation function applied to `a`.
    pub fn op_silu(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_silu(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, scales `a` by the 1D tensor `b`.
    pub fn op_scale(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { crate::ggml_scale(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, sets the elements above the diagonal to -INF.
    pub fn op_diag_mask_inf(&self, a: &Tensor, n_past: usize) -> Tensor {
        let tensor = unsafe {
            crate::ggml_diag_mask_inf(self.ptr.as_ptr(), a.ptr.as_ptr(), usize_to_i32(n_past))
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place, applies the [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) to `a`.
    pub fn op_soft_max(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_soft_max(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with result of mapping `fun` with `a`.
    ///
    /// `cnt` is the number of `f32` elements to be mapped.
    /// `src` is source for elements to be mapped.
    /// `dst` is the destination for mapped elements.
    ///
    /// # Safety
    ///
    /// This is marked unsafe since we're passing pointers into C code, and not
    /// only vanilla pointers but a pointer to a function. For obvious reasons, it's
    /// important not to do anything crazy like mutate any of these values concurrently.
    ///
    /// Don't make assumptions about how/when the function will be called. It may be called
    /// on a row, it may be called on a whole tensor. It may be called concurrently or not.
    /// Once you give that function pointer to C land, all bets are off.
    pub unsafe fn op_map_unary(
        &self,
        a: &Tensor,
        fun: unsafe extern "C" fn(cnt: ::std::os::raw::c_int, dst: *mut f32, src: *const f32),
    ) -> Tensor {
        let tensor =
            unsafe { crate::ggml_map_unary_f32(self.ptr.as_ptr(), a.ptr.as_ptr(), Some(fun)) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with result of mapping `fun` with `a` and `b`.
    ///
    /// `cnt` is the number of `f32` elements to be mapped.
    /// `src0`, `src1` are the sources of elements to be mapped.
    /// `dst` is the destination for mapped elements.
    ///
    /// # Safety
    ///
    /// This is marked unsafe since we're passing pointers into C code, and not
    /// only vanilla pointers but a pointer to a function. For obvious reasons, it's
    /// important not to do anything crazy like mutate any of these values concurrently.
    ///
    /// Don't make assumptions about how/when the function will be called. It may be called
    /// on a row, it may be called on a whole tensor. It may be called concurrently or not.
    /// Once you give that function pointer to C land, all bets are off.
    pub unsafe fn op_map_binary(
        &self,
        a: &Tensor,
        b: &Tensor,
        fun: unsafe extern "C" fn(
            cnt: ::std::os::raw::c_int,
            dst: *mut f32,
            src0: *const f32,
            src1: *const f32,
        ),
    ) -> Tensor {
        let tensor = unsafe {
            crate::ggml_map_binary_f32(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr(), Some(fun))
        };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 1D view over `a`.
    pub fn op_view_1d(&self, a: &Tensor, ne0: usize, offset: usize) -> Tensor {
        let tensor = unsafe {
            crate::ggml_view_1d(self.ptr.as_ptr(), a.ptr.as_ptr(), usize_to_i64(ne0), offset)
        };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 2D view over `a`.
    pub fn op_view_2d(
        &self,
        a: &Tensor,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
    ) -> Tensor {
        let tensor = unsafe {
            crate::ggml_view_2d(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                nb1,
                offset,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 3d view over `a`.
    #[allow(clippy::too_many_arguments)]
    pub fn op_view_3d(
        &self,
        a: &Tensor,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> Tensor {
        let tensor = unsafe {
            crate::ggml_view_3d(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
                nb1,
                nb2,
                offset,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// Copies `a` to `b` and returns `b`.
    pub fn op_cpy(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { crate::ggml_cpy(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the axes of `a` permuted as described by the parameters.
    pub fn op_permute(
        &self,
        a: &Tensor,
        axis0: usize,
        axis1: usize,
        axis2: usize,
        axis3: usize,
    ) -> Tensor {
        let tensor = unsafe {
            crate::ggml_permute(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(axis0),
                usize_to_i32(axis1),
                usize_to_i32(axis2),
                usize_to_i32(axis3),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the dimensions of `b`
    pub fn op_reshape(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { crate::ggml_reshape(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the specified dimensions.
    pub fn op_reshape_2d(&self, a: &Tensor, ne0: usize, ne1: usize) -> Tensor {
        let tensor = unsafe {
            crate::ggml_reshape_2d(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the specified dimensions.
    pub fn op_reshape_3d(&self, a: &Tensor, ne0: usize, ne1: usize, ne2: usize) -> Tensor {
        let tensor = unsafe {
            crate::ggml_reshape_3d(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; applies ROtary Positional Encoding.
    pub fn op_rope(&self, a: &Tensor, npast: usize, ndims: usize, mode: i32) -> Tensor {
        let tensor = unsafe {
            crate::ggml_rope(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(npast),
                usize_to_i32(ndims),
                mode,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// Computes the specified graph. Must be run in order to evaluate the graph.
    pub fn graph_compute(&self, graph: &mut ComputationGraph) {
        unsafe {
            crate::ggml_graph_compute(self.ptr.as_ptr(), &mut graph.inner);
        }
    }

    /// Retrieves the memory used by this [Context].
    pub fn used_mem(&self) -> usize {
        unsafe { crate::ggml_used_mem(self.ptr.as_ptr()) }
    }

    /// Sets the scratch buffer to be used by this [Context].
    ///
    /// If `scratch_buffer` is `None`, the scratch buffer will be disabled.
    pub fn use_scratch<'a>(&'a self, scratch_buffer: Option<&'a mut Buffer>) {
        let (size, data) = if let Some(buffer) = scratch_buffer {
            (buffer.data.len(), buffer.data.as_ptr() as *mut c_void)
        } else {
            (0, std::ptr::null_mut())
        };
        // SAFETY: this just passes (most likely uninitialized) memory buffer to the ggml C API
        unsafe {
            crate::ggml_set_scratch(
                self.ptr.as_ptr(),
                crate::ggml_scratch {
                    offs: 0,
                    size,
                    data,
                },
            );
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe {
            crate::ggml_free(self.ptr.as_ptr());
        }
    }
}
