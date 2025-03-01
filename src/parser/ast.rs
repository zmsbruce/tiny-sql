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
/// 目前只支持常量，后续会支持计算表达式
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

/// 抽象语法树定义。支持的 SQL 语法：
///
/// * `create table [table_name];`
/// * `insert into [table_name] [values];`
/// * `select * from [table_name];`
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
        table_name: String,
    },
}
