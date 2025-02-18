use std::num::{ParseFloatError, ParseIntError};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Range error: {0}")]
    RangeError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<ParseIntError> for Error {
    fn from(err: ParseIntError) -> Self {
        Error::ParseError(err.to_string())
    }
}

impl From<ParseFloatError> for Error {
    fn from(err: ParseFloatError) -> Self {
        Error::ParseError(err.to_string())
    }
}
