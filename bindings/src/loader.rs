use std::{fs::File, io::BufReader};

use reader::read_u32;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    /// A file failed to open.
    #[error("could not open file {path:?}")]
    OpenFileFailed {
        /// The original error.
        source: std::io::Error,
        /// The path that failed.
        path: String,
    },

    #[error("invalid file magic number: {0}")]
    InvalidMagic(u32),

    #[error("invalid ggml format: version={0}")]
    InvalidFormatVersion(u32),

    #[error("unsupported container type")]
    InvalidContainerType(ContainerType),

    /// Reading exactly `bytes` from a file failed.
    #[error("unable to read exactly {bytes} bytes")]
    ReadExactFailed {
        /// The original error.
        source: std::io::Error,
        /// The number of bytes that were attempted to be read.
        bytes: usize,
    },

    #[error("{0}")]
    BadIo(#[from] std::io::Error),

    /// One of the strings encountered was not valid UTF-8.
    #[error("could not convert bytes to a UTF-8 string")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    #[error("{0}")]
    FailedCast(#[from] std::num::TryFromIntError),

    #[error("user requested interrupt")]
    UserInterrupted,

    #[error("unsupported tensor dtype/f16_: {0}")]
    UnsupportedElementType(i32),

    /// sanity check failed
    #[error("invariant broken: {0}")]
    InvariantBroken(String),

    /// The specified was encountered , but was not seen during the model prelude.
    #[error("unknown tensor `{0}`")]
    UnknownTensor(String),

    /// The tensor `tensor_name` did not match its expected size.
    #[error("wrong size tensor `{0}`")]
    TensorWrongSize(String),

    #[error("invalid ftype {ftype} for tensor `{tensor_name}`")]
    /// The tensor `tensor_name` did not have the expected format type.
    InvalidFtype {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: i32,
    },
}

/// file type containing the model
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ContainerType {
    /// Current format with mmap
    Ggjt(FormatVersion),
    /// Oldest GGML tensor file format, legacy
    Ggml,
    /// Legacy
    Ggmf(FormatVersion),
}

impl TryFrom<&mut BufReader<&File>> for ContainerType {
    type Error = LoadError;

    fn try_from(reader: &mut BufReader<&File>) -> Result<Self, Self::Error> {
        const GGJT_MAGIC: u32 = 0x67676a74;
        const GGML_MAGIC: u32 = 0x67676d6c;
        const GGMF_MAGIC: u32 = 0x67676d66;

        let container_type = read_u32(reader)?;
        match container_type {
            GGJT_MAGIC | GGMF_MAGIC => match read_u32(reader)?.try_into()? {
                FormatVersion::V1 => {
                    if GGJT_MAGIC == container_type {
                        Ok(Self::Ggjt(FormatVersion::V1))
                    } else {
                        Ok(Self::Ggmf(FormatVersion::V1))
                    }
                }
            },
            GGML_MAGIC => Ok(Self::Ggml),
            _ => Err(LoadError::InvalidMagic(container_type)),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum FormatVersion {
    V1,
}

impl TryFrom<u32> for FormatVersion {
    type Error = LoadError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(FormatVersion::V1),
            _ => Err(LoadError::InvalidFormatVersion(value)),
        }
    }
}
