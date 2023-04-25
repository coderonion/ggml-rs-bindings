use crate::loader::LoadError;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
/// The type of a value in `ggml`.
pub enum ElementType {
    /// Quantized 4-bit (type 0).
    #[default]
    Q4_0,
    /// Quantized 4-bit (type 1); used by GPTQ.
    Q4_1,
    /// Integer 32-bit.
    I32,
    /// Float 16-bit.
    F16,
    /// Float 32-bit.
    F32,
}

impl TryFrom<i32> for ElementType {
    type Error = LoadError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ElementType::F32),
            1 => Ok(ElementType::F16),
            2 => Ok(ElementType::Q4_0),
            3 => Ok(ElementType::Q4_1),
            _ => Err(LoadError::UnsupportedElementType(value)),
        }
    }
}

impl TryFrom<u32> for ElementType {
    type Error = LoadError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ElementType::F32),
            1 => Ok(ElementType::F16),
            2 => Ok(ElementType::Q4_0),
            3 => Ok(ElementType::Q4_1),
            _ => Err(LoadError::UnsupportedElementType(value as i32)),
        }
    }
}

impl From<ElementType> for crate::ggml_type {
    fn from(t: ElementType) -> Self {
        match t {
            ElementType::Q4_0 => crate::ggml_type_GGML_TYPE_Q4_0,
            ElementType::Q4_1 => crate::ggml_type_GGML_TYPE_Q4_1,
            ElementType::I32 => crate::ggml_type_GGML_TYPE_I32,
            ElementType::F16 => crate::ggml_type_GGML_TYPE_F16,
            ElementType::F32 => crate::ggml_type_GGML_TYPE_F32,
        }
    }
}

impl From<ElementType> for f64 {
    fn from(val: ElementType) -> Self {
        (unsafe { crate::ggml_type_sizef(val.into()) }) as f64
    }
}

impl From<ElementType> for usize {
    fn from(val: ElementType) -> Self {
        unsafe { crate::ggml_type_size(val.into()) }
    }
}
