use std::{collections::HashMap, fmt::Display};

use crate::{error::Error::ParseError, schema::Column};

/// 常量定义
#[derive(PartialEq, Debug, Clone)]
pub enum Constant {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

/// 表达式定义
#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Field(String),
    Constant(Constant),
    Operation(Operation),
    Function(Aggregate, String),
}

impl Expression {
    pub fn is_field(&self) -> bool {
        matches!(self, Expression::Field(_))
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Expression::Constant(_))
    }

    pub fn is_operation(&self) -> bool {
        matches!(self, Expression::Operation(_))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Expression::Function(_, _))
    }

    pub fn as_field(&self) -> Option<&String> {
        match self {
            Expression::Field(name) => Some(name),
            _ => None,
        }
    }

    pub fn as_constant(&self) -> Option<&Constant> {
        match self {
            Expression::Constant(constant) => Some(constant),
            _ => None,
        }
    }

    pub fn as_operation(&self) -> Option<&Operation> {
        match self {
            Expression::Operation(operation) => Some(operation),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Aggregate, &String)> {
        match self {
            Expression::Function(aggregate, alias) => Some((*aggregate, alias)),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum Aggregate {
    Count,
    Sum,
    Avg,
    Max,
    Min,
}

impl Display for Aggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Aggregate::Count => write!(f, "COUNT"),
            Aggregate::Sum => write!(f, "SUM"),
            Aggregate::Avg => write!(f, "AVG"),
            Aggregate::Max => write!(f, "MAX"),
            Aggregate::Min => write!(f, "MIN"),
        }
    }
}

impl TryFrom<String> for Aggregate {
    type Error = crate::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.to_ascii_lowercase().as_str() {
            "count" => Ok(Aggregate::Count),
            "sum" => Ok(Aggregate::Sum),
            "avg" => Ok(Aggregate::Avg),
            "max" => Ok(Aggregate::Max),
            "min" => Ok(Aggregate::Min),
            _ => Err(ParseError(format!("Invalid aggregate function: {}", value))),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Operation {
    Equal(Box<Expression>, Box<Expression>),
}

/// 排序方式
#[derive(PartialEq, Debug)]
pub enum Ordering {
    Asc,
    Desc,
}

/// 连接方式
#[derive(PartialEq, Debug)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl Display for JoinType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JoinType::Inner => write!(f, "Inner Join"),
            JoinType::Left => write!(f, "Left Join"),
            JoinType::Right => write!(f, "Right Join"),
            JoinType::Full => write!(f, "Full Join"),
            JoinType::Cross => write!(f, "Cross Join"),
        }
    }
}

/// 查询来源
#[derive(PartialEq, Debug)]
pub enum SelectFrom {
    Table {
        name: String,
    },
    Join {
        left: Box<SelectFrom>,
        right: Box<SelectFrom>,
        join_type: JoinType,
        predicate: Option<Expression>,
    },
}

impl Display for SelectFrom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectFrom::Table { name } => write!(f, "{}", name),
            SelectFrom::Join {
                left,
                right,
                join_type,
                ..
            } => {
                write!(f, "[{} {} {}]", left, join_type, right)
            }
        }
    }
}

/// 抽象语法树定义
#[derive(PartialEq, Debug)]
pub enum Statement {
    CreateTable {
        name: String,
        columns: Vec<Column>,
    },
    Insert {
        table_name: String,
        columns: Option<Vec<String>>,
        values: Vec<Vec<Expression>>,
    },
    Select {
        columns: Vec<(Expression, Option<String>)>,
        from: SelectFrom,
        filter: Option<(String, Expression)>,
        groupby: Option<(Vec<Expression>, Option<Expression>)>,
        ordering: Vec<(String, Ordering)>,
        limit: Option<Expression>,
        offset: Option<Expression>,
    },
    Update {
        table_name: String,
        columns: HashMap<String, Expression>,
        filter: Option<(String, Expression)>,
    },
    Delete {
        table_name: String,
        filter: Option<(String, Expression)>,
    },
}
