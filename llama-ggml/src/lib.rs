// Ref: https://github.com/rustformers/llama-rs/tree/1b20306d/llama-rs/src

use std::{
    borrow::BorrowMut,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
};

use ggml_rs_bindings::{
    computation_graph::ComputationGraph,
    context::Context,
    element_type::ElementType,
    inference::{EvaluateOutputRequest, InferenceError, InferenceSession},
    loader::ContainerType,
    loader::FormatVersion,
    loader::LoadError,
    tensor::Tensor,
    vocabulary::{TokenId, Vocabulary},
    Model,
};
use memmap2::Mmap;
use reader::{has_data_left, read_bytes_with_len, read_f32, read_i32, read_u32};

#[derive(Debug)]
pub struct Llama {
    hyperparameters: Hyperparameters,
    vocabulary: Vocabulary,
    embeddings: Tensor,
    norm: Tensor,
    output: Tensor,
    tensors: HashMap<String, Tensor>,
    layers: Vec<Layer>,
    context_size: usize,
    _context: Context,
    _mmap: Mmap,
}

impl Llama {
    pub fn hyperparameters(&self) -> &Hyperparameters {
        &self.hyperparameters
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    pub fn norm(&self) -> &Tensor {
        &self.norm
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }
}

impl Model for Llama {
    fn context_size(&self) -> &usize {
        &self.context_size
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn start_session(&self) -> InferenceSession {
        InferenceSession::new(
            Default::default(),
            self.context_size,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input: &[TokenId],
    ) -> Result<(), InferenceError> {
        let n = input.len();
        let n_past = session.n_past;

        let memk_elsize = session.memory_k.element_size();
        let memv_elsize = session.memory_v.element_size();

        let Hyperparameters {
            n_vocab,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            element_type: _,
        } = self.hyperparameters;

        // For the first run, we need to guess a maximum buffer size so we can measure
        // the actual memory consumption of the temporary ggml context.
        //
        // These numbers are from `llama.cpp`, and could potentially be more efficient.
        let mut buf_size = {
            let buf_size_mb = if n_layer >= 80 {
                1536
            } else if n_layer >= 60 {
                1280
            } else {
                1024
            };
            buf_size_mb * 1024 * 1024
        };
        if session.mem_per_token > 0 && session.mem_per_token * n > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * session.mem_per_token as f64 * n as f64) as usize;
        };
        let ctx0 = Context::init(buf_size, false);

        let mut gf = ComputationGraph::default();

        let mut embd = ctx0.new_tensor_1d(ElementType::I32, n);
        unsafe { embd.write_data(bytemuck::cast_slice(input)) };

        let mut input_layer = ctx0.op_get_rows(&self.embeddings, &embd);

        for il in 0..n_layer {
            let input_self_attention = input_layer.share();
            let mut current: Tensor;

            ctx0.use_scratch(Some(&mut session.scratch[0]));

            // norm
            {
                current = ctx0.op_rms_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                    &current,
                );
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                let q_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wq, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );
                let k_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wk, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );

                // store key and value to memory
                {
                    // compute the transposed [N, n_embd] V matrix
                    let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                        &ctx0.op_mul_mat(&self.layers[il].wv, &current),
                        n_embd,
                        n,
                    ));

                    let k = ctx0.op_view_1d(
                        &session.memory_k,
                        n * n_embd,
                        (memk_elsize * n_embd) * (il * self.context_size + n_past),
                    );

                    let v = ctx0.op_view_2d(
                        &session.memory_v,
                        n,
                        n_embd,
                        self.context_size * memv_elsize,
                        (il * self.context_size) * memv_elsize * n_embd + n_past * memv_elsize,
                    );

                    // important: storing RoPE-ed version of K in the KV cache!
                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                let q = ctx0.op_permute(&q_current, 0, 2, 1, 3);

                let k = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &session.memory_k,
                            (n_past + n) * n_embd,
                            il * self.context_size * memk_elsize * n_embd,
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + n,
                    ),
                    0,
                    2,
                    1,
                    3,
                );

                // K * Q
                let k_q = ctx0.op_mul_mat(&k, &q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let k_q_scaled = ctx0.op_scale(
                    &k_q,
                    &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

                // split cached V into n_head heads
                let v = ctx0.op_view_3d(
                    &session.memory_v,
                    n_past + n,
                    n_embd / n_head,
                    n_head,
                    self.context_size * memv_elsize,
                    self.context_size * memv_elsize * n_embd / n_head,
                    il * self.context_size * memv_elsize * n_embd,
                );

                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ElementType::F32, n_embd, n),
                );

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
            }

            ctx0.use_scratch(Some(&mut session.scratch[1]));

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
                    current = ctx0.op_rms_norm(&input_feed_forward);

                    // cur = ffn_norm*cur
                    current = ctx0.op_mul(
                        &ctx0.op_repeat(&self.layers[il].ffn_norm, &current),
                        &current,
                    );
                }

                let tmp = ctx0.op_mul_mat(&self.layers[il].w3, &current);

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                // SILU activation
                current = ctx0.op_silu(&current);

                current = ctx0.op_mul(&current, &tmp);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);
            }

            current = ctx0.op_add(&current, &input_feed_forward);

            // input for next layer
            input_layer = current;
        }

        ctx0.use_scratch(Some(&mut session.scratch[0]));

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor;

        // norm
        {
            input_layer = ctx0.op_rms_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            embeddings_tensor = input_layer.share();
        }

        // lm_head
        {
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);
        }

        ctx0.use_scratch(None);

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        // SAFETY: yolo
        assert_eq!(session.last_logits.len(), n_vocab);
        unsafe {
            input_layer.read_data(
                n_vocab * (n - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(&mut session.last_logits),
            )
        };

        let mut output = EvaluateOutputRequest::default();

        // Extract logits
        if let Some(all_logits) = &mut output.all_logits {
            all_logits.resize(n_vocab * n, 0.0);
            // SAFETY: Tensor data can be read (properly aligned, initialized,
            // data will not be mutated or otherwise aliased during the copy),
            // and we're not reading past the end of the tensor data.
            assert_eq!(input_layer.nelements(), n_vocab * n);
            unsafe {
                input_layer.read_data(0, bytemuck::cast_slice_mut(all_logits));
            }
        }

        // Extract embeddings
        if let Some(embeddings) = &mut output.embeddings {
            embeddings.resize(n_embd * n, 0.0);
            // SAFETY: Same rationale as for the "Extract logits" section applies.
            assert_eq!(embeddings_tensor.nelements(), n_embd * n);
            unsafe {
                embeddings_tensor.read_data(0, bytemuck::cast_slice_mut(embeddings));
            }
        }

        // Adjust the required memory per token if we didn't know that already
        if session.mem_per_token == 0 {
            session.mem_per_token = ctx0.used_mem() / n;
        }

        // Adjust n_past to new length.
        session.n_past += input.len();

        Ok(())
    }
}

impl TryFrom<&File> for Llama {
    type Error = LoadError;

    fn try_from(file: &File) -> Result<Self, Self::Error> {
        let mut handle = BufReader::new(file);
        let reader = handle.borrow_mut();
        let container_type: ContainerType = reader.try_into()?;
        if ContainerType::Ggjt(FormatVersion::V1) != container_type {
            return Err(LoadError::InvalidContainerType(container_type));
        }

        let hyperparameters: Hyperparameters = reader.try_into()?;
        let mut vocabulary: Vocabulary = Default::default();

        // Load vocabulary
        for idx in 0..hyperparameters.n_vocab {
            let len = read_u32(reader)?.try_into()?;
            let token = read_bytes_with_len(reader, len)?;
            let token_score = match container_type {
                ContainerType::Ggjt(FormatVersion::V1) => read_f32(reader)?,
                _ => return Err(LoadError::InvalidContainerType(container_type)),
            };

            vocabulary
                .push_token(idx.try_into()?, token, token_score)
                .expect("Vocabulary is valid.");
        }

        let Hyperparameters {
            n_vocab,
            n_embd,
            n_mult,
            n_layer,
            element_type,
            ..
        } = hyperparameters;

        let context_size = (5 + 10 * n_layer) * 256; // object overhead
        let context = Context::init(context_size, true);

        let embeddings = context.new_tensor_2d(element_type, n_embd, n_vocab);
        let norm = context.new_tensor_1d(ElementType::F32, n_embd);
        let output = context.new_tensor_2d(element_type, n_embd, n_vocab);

        let mut tensors = HashMap::new();

        tensors.insert("tok_embeddings.weight".to_owned(), embeddings.share());
        tensors.insert("norm.weight".to_owned(), norm.share());
        tensors.insert("output.weight".to_owned(), output.share());

        let n_ff = ((2 * (4 * n_embd) / 3 + n_mult - 1) / n_mult) * n_mult;

        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                attention_norm: context.new_tensor_1d(ElementType::F32, n_embd),
                wq: context.new_tensor_2d(element_type, n_embd, n_embd),
                wk: context.new_tensor_2d(element_type, n_embd, n_embd),
                wv: context.new_tensor_2d(element_type, n_embd, n_embd),
                wo: context.new_tensor_2d(element_type, n_embd, n_embd),
                ffn_norm: context.new_tensor_1d(ElementType::F32, n_embd),
                w1: context.new_tensor_2d(element_type, n_embd, n_ff),
                w2: context.new_tensor_2d(element_type, n_ff, n_embd),
                w3: context.new_tensor_2d(element_type, n_embd, n_ff),
            };

            tensors.insert(
                format!("layers.{i}.attention_norm.weight"),
                layer.attention_norm.share(),
            );

            tensors.insert(format!("layers.{i}.attention.wq.weight"), layer.wq.share());
            tensors.insert(format!("layers.{i}.attention.wk.weight"), layer.wk.share());
            tensors.insert(format!("layers.{i}.attention.wv.weight"), layer.wv.share());
            tensors.insert(format!("layers.{i}.attention.wo.weight"), layer.wo.share());

            tensors.insert(
                format!("layers.{i}.ffn_norm.weight"),
                layer.ffn_norm.share(),
            );

            tensors.insert(
                format!("layers.{i}.feed_forward.w1.weight"),
                layer.w1.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w2.weight"),
                layer.w2.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w3.weight"),
                layer.w3.share(),
            );

            layers.push(layer);
        }

        let mmap = unsafe { Mmap::map(file)? };

        while has_data_left(reader)? {
            let n_dims = read_i32(reader)? as usize;
            let length = read_i32(reader)?;
            let ftype = read_i32(reader)?;

            let mut nelements: usize = 1;
            let mut ne = [1i64, 1];

            assert!(n_dims <= ne.len());

            #[allow(clippy::needless_range_loop)]
            for i in 0..n_dims {
                let dim = read_i32(reader)? as usize;
                ne[i] = dim as i64;
                nelements *= dim;
            }

            let tensor_name = read_string(reader, length as usize)?;

            let Some(tensor) = tensors.get_mut(&tensor_name)
            else {
                return Err(LoadError::UnknownTensor(tensor_name));
            };

            if tensor.nelements() != nelements {
                return Err(LoadError::TensorWrongSize(tensor_name));
            }

            let tensor_ne = tensor.get_ne();
            if tensor_ne[0] != ne[0] || tensor_ne[1] != ne[1] {
                return Err(LoadError::TensorWrongSize(tensor_name));
            }

            match tensor_type_size(ftype, ne) {
                Some(_) => {}
                None => {
                    return Err(LoadError::InvalidFtype { tensor_name, ftype });
                }
            };

            let offset_curr = reader.stream_position()?;
            let offset_aligned: u64 = (offset_curr + 31) & !31;

            unsafe {
                let ptr = mmap.as_ptr().offset(offset_aligned as isize);
                tensor.set_data(ptr as *mut std::ffi::c_void);
            }

            reader.seek(SeekFrom::Start(offset_aligned + tensor.nbytes() as u64))?;
        }

        Ok(Llama {
            hyperparameters,
            vocabulary,
            embeddings,
            norm,
            output,
            tensors,
            layers,
            _mmap: mmap,
            context_size: 2048,
            _context: context,
        })
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// n_vocab
    n_vocab: usize,
    /// n_embd
    n_embd: usize,
    /// n_mult
    n_mult: usize,
    /// n_head
    n_head: usize,
    /// n_layer
    n_layer: usize,
    /// n_rot
    n_rot: usize,
    /// element_type
    element_type: ElementType,
}

impl TryFrom<&mut BufReader<&File>> for Hyperparameters {
    type Error = LoadError;

    fn try_from(reader: &mut BufReader<&File>) -> Result<Self, Self::Error> {
        Ok(Hyperparameters {
            n_vocab: read_i32(reader)? as usize,
            n_embd: read_i32(reader)? as usize,
            n_mult: read_i32(reader)? as usize,
            n_head: read_i32(reader)? as usize,
            n_layer: read_i32(reader)? as usize,
            n_rot: read_i32(reader)? as usize,
            element_type: read_i32(reader)?.try_into()?,
        })
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attention_norm: Tensor,

    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,

    // normalization
    ffn_norm: Tensor,

    // ff
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
}

fn read_string(reader: &mut impl BufRead, len: usize) -> Result<String, LoadError> {
    let mut buf = vec![0; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: buf.len(),
        })?;
    let s = String::from_utf8(buf)?;
    Ok(s)
}

fn tensor_type_size(ftype: i32, ne: [i64; 2]) -> Option<usize> {
    let element_type = match ftype.try_into() {
        Ok(element_type) => element_type,
        Err(_) => return None,
    };

    match element_type {
        ElementType::Q4_0 | ElementType::Q4_1 => {
            assert_eq!(ne[0] % 64, 0);
        }
        _ => {}
    }

    Some(element_type.into())
}
