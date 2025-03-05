use std::collections::HashMap;

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
        table_name: String,
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
