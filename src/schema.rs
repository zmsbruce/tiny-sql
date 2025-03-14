use std::{cmp::Ordering, collections::HashMap, hash::Hash};

use serde::{Deserialize, Serialize};

use crate::{
    parser::ast::{Constant, Expression},
    Error::InternalError,
    Result,
};

/// 数据类型定义
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone, Copy)]
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

impl Value {
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::Float(f) => Ok(*f),
            other => Err(InternalError(format!("Cannot convert {:?} to f64", other))),
        }
    }

    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Self::Integer(i) => Ok(*i),
            other => Err(InternalError(format!("Cannot convert {:?} to i64", other))),
        }
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s),
            other => Err(InternalError(format!(
                "Cannot convert {:?} to string",
                other
            ))),
        }
    }
}

impl From<&Constant> for Value {
    fn from(constant: &Constant) -> Self {
        match constant {
            Constant::Null => Self::Null,
            Constant::Boolean(b) => Self::Boolean(*b),
            Constant::Integer(i) => Self::Integer(*i),
            Constant::Float(f) => Self::Float(*f),
            Constant::String(s) => Self::String(s.clone()),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Null, Self::Null) => Some(Ordering::Equal),
            (Self::Null, _) => Some(Ordering::Less),
            (_, Self::Null) => Some(Ordering::Greater),
            (Self::Boolean(a), Self::Boolean(b)) => a.partial_cmp(b),
            (Self::Integer(a), Self::Integer(b)) => a.partial_cmp(b),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::Integer(a), Self::Float(b)) => (*a as f64).partial_cmp(b),
            (Self::Float(a), Self::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (Self::String(a), Self::String(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Null => state.write_u8(0),
            Self::Boolean(b) => {
                state.write_u8(1);
                b.hash(state)
            }
            Self::Integer(i) => {
                state.write_u8(2);
                i.hash(state)
            }
            Self::Float(f) => {
                state.write_u8(3);
                f.to_bits().hash(state)
            }
            Self::String(s) => {
                state.write_u8(4);
                s.hash(state)
            }
        }
    }
}

impl Eq for Value {}

impl TryFrom<&Expression> for Value {
    type Error = crate::Error;

    fn try_from(expr: &Expression) -> Result<Self> {
        match expr {
            Expression::Constant(c) => match c {
                Constant::Boolean(b) => Ok(Value::Boolean(*b)),
                Constant::Float(f) => Ok(Value::Float(*f)),
                Constant::Integer(i) => Ok(Value::Integer(*i)),
                Constant::String(s) => Ok(Value::String(s.clone())),
                Constant::Null => Ok(Value::Null),
            },
            e => Err(InternalError(format!(
                "Cannot convert non-constant expression {:?} to value",
                e
            ))),
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

        // 检查主键是否为 nullable，如果是则返回错误
        if columns[pk_indexes[0]].nullable {
            return Err(InternalError(format!(
                "Primary key {} cannot be nullable",
                columns[pk_indexes[0]].name
            )));
        }

        // 检查默认值是否和数据类型匹配
        for col in &columns {
            if let Some(default) = &col.default {
                if default.data_type() != Some(col.data_type) {
                    return Err(InternalError(format!(
                        "Default value {:?} does not match column {}'s data type",
                        default, col.name
                    )));
                }
            }
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
