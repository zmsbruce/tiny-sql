use std::{collections::HashMap, fmt::Display};

use crate::schema::Column;

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
