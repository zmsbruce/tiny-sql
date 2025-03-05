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
    Constant(Constant),
}

/// 将常量转换为表达式
impl From<Constant> for Expression {
    fn from(value: Constant) -> Self {
        Expression::Constant(value)
    }
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
            } => {
                let join_symbol = match join_type {
                    JoinType::Inner => "+",
                    JoinType::Left => "<+",
                    JoinType::Right => "+>",
                    JoinType::Full => "<+>",
                };
                write!(f, "[{} {} {}]", left, join_symbol, right)
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
        columns: Vec<(String, Option<String>)>,
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
