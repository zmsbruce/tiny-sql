use serde::{Deserialize, Serialize};

use crate::parser::ast::{Constant, Expression};

/// 数据类型定义
#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub enum DataType {
    Boolean,
    Integer,
    Float,
    String,
}

/// 列定义
#[derive(PartialEq, Debug, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default: Option<Value>,
}

/// 值定义
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub enum Value {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

impl From<Expression> for Value {
    /// 将表达式转为值
    fn from(expr: Expression) -> Self {
        match expr {
            Expression::Constant(c) => match c {
                Constant::Boolean(b) => Value::Boolean(b),
                Constant::Float(f) => Value::Float(f),
                Constant::Integer(i) => Value::Integer(i),
                Constant::String(s) => Value::String(s),
                Constant::Null => Value::Null,
            },
        }
    }
}

impl Value {
    /// 获取数据类型
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            Self::Null => None,
            Self::Boolean(_) => Some(DataType::Boolean),
            Self::Integer(_) => Some(DataType::Integer),
            Self::Float(_) => Some(DataType::Float),
            Self::String(_) => Some(DataType::String),
        }
    }
}

pub type Row = Vec<Value>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub columns: Vec<Column>,
}
