use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    parser::ast::{Constant, Expression},
    Error::InternalError,
    Result,
};

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
    pub primary_key: bool,
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
    primary_key_idx: usize,
    col_idx: HashMap<String, usize>,
}

impl Table {
    pub fn new(name: &str, columns: Vec<Column>) -> Result<Self> {
        // 检查表是否有列定义，如果没有则返回错误
        if columns.is_empty() {
            return Err(InternalError(format!("Table {} has no columns", name)));
        }

        // 检查是否有且仅有一个主键
        let pk_indexes: Vec<usize> = columns
            .iter()
            .enumerate()
            .filter_map(|(i, col)| if col.primary_key { Some(i) } else { None })
            .collect();
        if pk_indexes.is_empty() {
            return Err(InternalError(format!("Table {} has no primary key", name)));
        } else if pk_indexes.len() > 1 {
            return Err(InternalError(format!(
                "Table {} has more than one primary key",
                name
            )));
        }

        // 创建列索引
        let col_idx = columns
            .iter()
            .enumerate()
            .map(|(i, col)| (col.name.clone(), i))
            .collect();

        Ok(Self {
            name: name.to_string(),
            columns,
            primary_key_idx: pk_indexes[0],
            col_idx,
        })
    }

    /// 获取一个行的主键值
    #[inline]
    pub fn get_primary_key<'a>(&self, row: &'a Row) -> &'a Value {
        &row[self.primary_key_idx]
    }

    /// 获取列的索引
    #[inline]
    pub fn get_col_idx(&self, col_name: &str) -> Option<usize> {
        self.col_idx.get(col_name).copied()
    }
}
