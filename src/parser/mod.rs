use std::{collections::HashMap, iter::Peekable};

use crate::{
    schema::{Column, DataType},
    Error::ParseError,
    Result,
};
use ast::{Constant, Expression, JoinType, Operation, Ordering, SelectFrom, Statement};
use lexer::{Keyword, Lexer, Token};

pub mod ast;
mod lexer;

/// SQL 解析器
pub struct Parser<'a> {
    lexer: Peekable<Lexer<'a>>,
}

impl<'a> Parser<'a> {
    /// 创建一个新的解析器
    pub fn new(input: &'a str) -> Self {
        Parser {
            lexer: Lexer::new(input).peekable(),
        }
    }

    /// 解析 SQL 语句
    ///
    /// 支持的语句：
    ///
    /// ```sql
    /// select [* | col_name [ [ AS ] output_name [, ...] ]] from [table_name [ cross | left | right | inner ] join ...] [where [condition]] [order by [column_name] [asc|desc]] [limit [number]] [offset [number]];
    ///
    /// create table [table_name] ([column_name] [data_type] [nullable] [default] [primary key], ...);
    ///
    /// insert into [table_name] ([column_name], ...) values ([value], ...);
    ///
    /// update [table_name] set [column_name] = [value], ... where [condition];
    ///
    /// delete from [table_name] where [condition];
    /// ```
    pub fn parse(&mut self) -> Result<Statement> {
        // 根据第一个 token 的类型选择解析方法
        let stmt = match self
            .lexer
            .peek()
            .ok_or(ParseError("Unexpected end of input".to_string()))?
        {
            Ok(Token::Keyword(Keyword::Select)) => self.parse_select(),
            Ok(Token::Keyword(Keyword::Create)) => self.parse_create_table(),
            Ok(Token::Keyword(Keyword::Insert)) => self.parse_insert(),
            Ok(Token::Keyword(Keyword::Update)) => self.parse_update(),
            Ok(Token::Keyword(Keyword::Delete)) => self.parse_delete(),
            Ok(token) => Err(ParseError(format!("Unexpected token {token}"))),
            Err(e) => Err(ParseError(format!("Lexical error: {e}"))),
        };
        // 解析结束后应该是一个分号，否则返回异常
        self.next_token_equal(Token::Semicolon)?;
        // 如果词法解析器的顶端不是 None，说明语句存在错误
        if let Some(result) = self.lexer.peek() {
            match result {
                // 如果是一个 token，返回未知的 token 错误
                Ok(token) => return Err(ParseError(format!("Unexpected token {token}"))),
                // 如果是一个词法解析错误，返回词法解析错误
                Err(e) => return Err(ParseError(format!("Lexical error: {e}"))),
            }
        }
        // 返回解析结果
        stmt
    }

    /// 在满足条件的情况下，跳转并获取下一个 token，否则不跳转，并返回错误
    fn next_token_if<F>(&mut self, f: F) -> Result<Token>
    where
        F: Fn(&Token) -> bool,
    {
        match self.lexer.peek() {
            Some(Ok(token)) if f(token) => self.lexer.next().unwrap(),
            Some(Ok(token)) => Err(ParseError(format!("Unexpected token {token}"))),
            Some(Err(e)) => Err(ParseError(format!("Lexical error: {e}"))),
            None => Err(ParseError("Unexpected end of input".to_string())),
        }
    }

    /// 获取下一个 token
    fn next_token(&mut self) -> Result<Token> {
        self.next_token_if(|_| true)
    }

    /// 期望下一个 token 是指定的 token
    fn next_token_equal(&mut self, token: Token) -> Result<()> {
        self.next_token_if(|t| *t == token).map(|_| ())
    }

    /// 获取下一个关键字
    fn next_keyword(&mut self) -> Result<Keyword> {
        self.next_token_if(|token| matches!(token, Token::Keyword(_)))
            .map(|token| match token {
                Token::Keyword(keyword) => keyword,
                _ => unreachable!("Token must be a keyword after matching"), // 不可能出现的情况
            })
    }

    /// 获取下一个标识符
    fn next_identifier(&mut self) -> Result<String> {
        self.next_token_if(|token| matches!(token, Token::Identifier(_)))
            .map(|token| match token {
                Token::Identifier(ident) => ident,
                _ => unreachable!("Token must be an identifier after matching"), // 不可能出现的情况
            })
    }

    /// 解析 SELECT 语句
    /// 语法：`SELECT [* | col_name [ [AS] output_name [, ...] ]] FROM [table_name] WHERE [condition] ORDER BY [column_name] [ASC|DESC] LIMIT [number] OFFSET [number];`
    fn parse_select(&mut self) -> Result<Statement> {
        self.next_token_equal(Token::Keyword(Keyword::Select))?; // 期望下一个 token 是 SELECT

        // 获取列名，如果是 *，则表示选择所有列
        let columns = self.parse_select_columns()?;

        let from = self.parse_select_from()?; // 解析 FROM 子句

        // 如果有 WHERE 子句，则解析 WHERE 子句
        let filter = self
            .next_token_equal(Token::Keyword(Keyword::Where))
            .ok()
            .map(|_| self.parse_where_clause())
            .transpose()?;

        // 如果有 ORDER BY 子句，则解析 ORDER BY 子句
        let ordering = self.parse_order_by()?.unwrap_or_default();

        let limit = self
            .next_token_equal(Token::Keyword(Keyword::Limit))
            .ok()
            .map(|_| self.parse_expression())
            .transpose()?;
        let offset = self
            .next_token_equal(Token::Keyword(Keyword::Offset))
            .ok()
            .map(|_| self.parse_expression())
            .transpose()?;

        Ok(Statement::Select {
            columns,
            from,
            filter,
            ordering,
            limit,
            offset,
        })
    }

    /// 解析 SELECT 语句的 FROM 子句
    /// 语法：`FROM table_name [CROSS JOIN table_name ...]`
    fn parse_select_from(&mut self) -> Result<SelectFrom> {
        self.next_token_equal(Token::Keyword(Keyword::From))?; // 期望下一个 token 是 FROM

        let mut select_from = SelectFrom::Table {
            name: self.next_identifier()?, // 第一个表名
        };

        // 如果有 JOIN 子句，则解析 JOIN 子句
        while let Ok(join_type) = self.parse_join() {
            let right = SelectFrom::Table {
                name: self.next_identifier()?, // 获取右表名
            };

            // 解析 JOIN 条件
            let predicate = match join_type {
                JoinType::Cross => None,
                _ => {
                    self.next_token_equal(Token::Keyword(Keyword::On))?; // 期望下一个 token 是 ON
                    let predicate = self.parse_expression()?;

                    // JOIN 条件必须是一个字段等于另一个字段
                    match predicate {
                        Expression::Operation(Operation::Equal(ref left, ref right))
                            if left.is_field() && right.is_field() => {}
                        _ => {
                            return Err(ParseError(
                                "Join condition must be a field equal to a field".to_string(),
                            ))
                        }
                    }
                    Some(predicate) // 解析 JOIN 条件
                }
            };
            select_from = SelectFrom::Join {
                left: Box::new(select_from),
                right: Box::new(right),
                join_type,
                predicate,
            };
        }

        Ok(select_from)
    }

    /// 解析 JOIN 类型，如果没有指定 JOIN 类型，则默认为 INNER JOIN
    ///
    /// 语法：`[CROSS | LEFT | RIGHT | INNER | FULL] JOIN`
    fn parse_join(&mut self) -> Result<JoinType> {
        match self.next_token_if(|token| {
            matches!(
                token,
                Token::Keyword(Keyword::Cross)
                    | Token::Keyword(Keyword::Left)
                    | Token::Keyword(Keyword::Right)
                    | Token::Keyword(Keyword::Inner)
                    | Token::Keyword(Keyword::Join)
                    | Token::Keyword(Keyword::Full)
            )
        })? {
            Token::Keyword(Keyword::Cross) => {
                self.next_token_equal(Token::Keyword(Keyword::Join))?;
                Ok(JoinType::Cross)
            }
            Token::Keyword(Keyword::Left) => {
                self.next_token_equal(Token::Keyword(Keyword::Join))?;
                Ok(JoinType::Left)
            }
            Token::Keyword(Keyword::Right) => {
                self.next_token_equal(Token::Keyword(Keyword::Join))?;
                Ok(JoinType::Right)
            }
            Token::Keyword(Keyword::Inner) => {
                self.next_token_equal(Token::Keyword(Keyword::Join))?;
                Ok(JoinType::Inner)
            }
            Token::Keyword(Keyword::Full) => {
                self.next_token_equal(Token::Keyword(Keyword::Join))?;
                Ok(JoinType::Full)
            }
            // 如果没有指定 JOIN 类型，默认为 INNER JOIN
            Token::Keyword(Keyword::Join) => Ok(JoinType::Inner),
            // matches! 宏已经保证了 token 的类型，因此不可能出现其他类型的 token
            _ => unreachable!(),
        }
    }

    /// 解析 SELECT 语句的列名
    /// 语法：`[* | col_name [ [AS] output_name [, ...] ]`
    fn parse_select_columns(&mut self) -> Result<Vec<(Expression, Option<String>)>> {
        let mut columns = Vec::new();
        if self.next_token_equal(Token::Asterisk).is_err() {
            loop {
                let column_name = self.parse_expression()?; // 获取列名

                // 列名必须是一个字段
                if !matches!(column_name, Expression::Field(_)) {
                    return Err(ParseError("Column name must be a field".to_string()));
                }

                // 获取列的别名
                let alias = self
                    .next_token_if(|token| matches!(token, Token::Keyword(Keyword::As)))
                    .ok()
                    .map(|_| self.next_identifier())
                    .transpose()?;
                columns.push((column_name, alias));
                if self.next_token_equal(Token::Comma).is_err() {
                    break;
                }
            }
        }
        Ok(columns)
    }

    /// 解析 ORDER BY 子句，可能不存在
    /// 语法：`ORDER BY [column_name] [ASC|DESC], ...`
    fn parse_order_by(&mut self) -> Result<Option<Vec<(String, Ordering)>>> {
        self.next_token_equal(Token::Keyword(Keyword::Order))
            .ok()
            .map(|_| {
                self.next_token_equal(Token::Keyword(Keyword::By))?; // 期望下一个 token 是 BY
                let mut ordering = Vec::new();
                loop {
                    let column_name = self.next_identifier()?; // 获取列名
                    let ordering_type = match self.next_token_if(|token| {
                        matches!(
                            token,
                            Token::Keyword(Keyword::Asc) | Token::Keyword(Keyword::Desc)
                        )
                    }) {
                        // 获取排序方式
                        Ok(Token::Keyword(Keyword::Asc)) => Ordering::Asc,
                        Ok(Token::Keyword(Keyword::Desc)) => Ordering::Desc,
                        _ => Ordering::Asc, // 如果不是 ASC 或 DESC，则默认为 ASC
                    };
                    ordering.push((column_name, ordering_type));
                    if self.next_token_equal(Token::Comma).is_err() {
                        break;
                    }
                }
                Ok::<_, crate::Error>(ordering)
            })
            .transpose()
    }

    /// 解析 UPDATE 语句
    /// 语法：`UPDATE [table_name] SET [column_name] = [value], ... WHERE [condition];`
    fn parse_update(&mut self) -> Result<Statement> {
        self.next_token_equal(Token::Keyword(Keyword::Update))?;

        // 获取表名
        let table_name = self.next_identifier()?;
        self.next_token_equal(Token::Keyword(Keyword::Set))?;

        // 获取列名和值
        let mut columns = HashMap::new();
        loop {
            // 获取列名
            let col_name = self.next_identifier()?;
            self.next_token_equal(Token::Equal)?;

            // 获取值
            let value = self.parse_expression()?;
            if columns.contains_key(&col_name) {
                return Err(ParseError(format!("Duplicate column name {col_name}")));
            }
            columns.insert(col_name, value);

            // 如果没有逗号，说明列名和值解析结束，跳出循环
            if self.next_token_equal(Token::Comma).is_err() {
                break;
            }
        }

        // 如果有 WHERE 子句，则解析 WHERE 子句
        let filter = if self
            .next_token_equal(Token::Keyword(Keyword::Where))
            .is_ok()
        {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        Ok(Statement::Update {
            table_name,
            columns,
            filter,
        })
    }

    /// 解析 WHERE 子句
    /// 语法：`WHERE column_name = expression`
    ///
    /// 目前只支持单个表达式且仅为等于操作，不支持其他操作符和表达式组合
    fn parse_where_clause(&mut self) -> Result<(String, Expression)> {
        let col_name = self.next_identifier()?;
        self.next_token_equal(Token::Equal)?;
        let val = self.parse_expression()?;
        Ok((col_name, val))
    }

    /// 解析 DELETE 语句
    ///
    /// 语法：`DELETE FROM [table_name] WHERE [condition];`
    fn parse_delete(&mut self) -> Result<Statement> {
        self.next_token_equal(Token::Keyword(Keyword::Delete))?;
        self.next_token_equal(Token::Keyword(Keyword::From))?;

        let table_name = self.next_identifier()?;

        // 如果有 WHERE 子句，则解析 WHERE 子句
        let filter = self
            .next_token_equal(Token::Keyword(Keyword::Where))
            .ok()
            .map(|_| self.parse_where_clause())
            .transpose()?;

        Ok(Statement::Delete { table_name, filter })
    }

    /// 解析列定义
    /// 语法：[column_name] [data_type] [nullable] [default]
    fn parse_column(&mut self) -> Result<Column> {
        let name = self.next_identifier()?; // 获取列名

        // 获取数据类型
        let data_type = match self.next_token()? {
            // 如果是 BOOLEAN 或 BOOL，则数据类型为布尔型
            Token::Keyword(Keyword::Boolean) | Token::Keyword(Keyword::Bool) => DataType::Boolean,
            // 如果是 INTEGER 或 INT，则数据类型为整型
            Token::Keyword(Keyword::Integer) | Token::Keyword(Keyword::Int) => DataType::Integer,
            // 如果是 FLOAT 或 DOUBLE，则数据类型为浮点型
            Token::Keyword(Keyword::Float) | Token::Keyword(Keyword::Double) => DataType::Float,
            // 如果是 STRING 或 VARCHAR 或 TEXT，则数据类型为字符串
            Token::Keyword(Keyword::String)
            | Token::Keyword(Keyword::Text)
            | Token::Keyword(Keyword::Varchar) => DataType::String,
            // 其他 token，返回未知的 token 错误
            token => return Err(ParseError(format!("Unexpected token {token}"))),
        };
        // 初始化列结构体，设置列名和数据类型, 其他属性暂时为空
        let mut column = Column {
            name,
            data_type,
            nullable: false,
            default: None,
            primary_key: false,
        };

        // 解析列的其他属性
        // 可能是 NULL, NOT NULL, DEFAULT [value]
        while let Ok(keyword) = self.next_keyword() {
            match keyword {
                Keyword::Null => column.nullable = true, // 如果是 NULL，则设置列可空
                Keyword::Not => {
                    // 如果是 NOT，则期望下一个 token 是 NULL，列不可空
                    self.next_token_equal(Token::Keyword(Keyword::Null))?;
                }
                // 如果是 DEFAULT，则期望下一个 token 是一个表达式，设置列的默认值
                Keyword::Default => column.default = Some(self.parse_expression()?.into()),
                // 如果是 PRIMARY KEY，则设置列为主键
                Keyword::Primary => {
                    self.next_token_equal(Token::Keyword(Keyword::Key))?;
                    column.primary_key = true;
                }
                // 其他关键字，返回未知的关键字错误
                k => return Err(ParseError(format!("Unexpected keyword {k}"))),
            }
        }
        Ok(column)
    }

    /// 解析表达式
    /// 目前支持的表达式类型：十进制整数、十进制浮点数、字符串、布尔值、NULL，**不支持函数调用、运算符等**
    fn parse_expression(&mut self) -> Result<Expression> {
        // 获取下一个 token
        let exp = match self.next_token()? {
            Token::Identifier(ident) => {
                if self.next_token_equal(Token::Equal).is_ok() {
                    let right = self.parse_expression()?;
                    Expression::Operation(Operation::Equal(
                        Box::new(Expression::Field(ident)),
                        Box::new(right),
                    ))
                } else {
                    Expression::Field(ident)
                }
            }
            Token::Number(num_str) => {
                // 如果是数字，则解析为整数或浮点数
                if num_str.chars().all(|ch| ch.is_ascii_digit()) {
                    // 如果数字全部是 0-9，则判断为整数
                    let num = num_str.parse::<i64>()?;
                    Expression::Constant(Constant::Integer(num))
                } else {
                    // 否则为浮点数
                    let num = num_str.parse::<f64>()?;
                    Expression::Constant(Constant::Float(num))
                }
            }
            Token::String(s) => Expression::Constant(Constant::String(s)), // 字符串
            Token::Keyword(Keyword::True) => Expression::Constant(Constant::Boolean(true)), // 布尔值 true
            Token::Keyword(Keyword::False) => Expression::Constant(Constant::Boolean(false)), // 布尔值 false
            Token::Keyword(Keyword::Null) => Expression::Constant(Constant::Null), // NULL
            token => return Err(ParseError(format!("Unexpected token {token}"))), // 其他 token，返回未知的 token 错误
        };
        Ok(exp)
    }

    /// 解析 CREATE TABLE 语句
    /// 语法：CREATE TABLE [table_name] ([column_name] [data_type] [nullable] [default], ...);
    fn parse_create_table(&mut self) -> Result<Statement> {
        self.next_token_equal(Token::Keyword(Keyword::Create))?; // 期望下一个 token 是 CREATE
        self.next_token_equal(Token::Keyword(Keyword::Table))?; // 期望下一个 token 是 TABLE

        let table_name = self.next_identifier()?; // 获取表名
        self.next_token_equal(Token::OpenParen)?; // 期望下一个 token 是 (

        // 解析 ( 后面的列定义
        let mut columns = Vec::new();
        loop {
            columns.push(self.parse_column()?); // 解析列定义
            match self.next_token()? {
                Token::Comma => continue,   // 如果是逗号，继续解析下一个列定义
                Token::CloseParen => break, // 如果是 )，则列定义解析结束
                token => return Err(ParseError(format!("Unexpected token {token}"))), // 其他 token，返回错误
            }
        }
        Ok(Statement::CreateTable {
            name: table_name,
            columns,
        })
    }

    /// 解析 INSERT 语句
    /// 语法：`INSERT INTO [table_name] ([column_name], ...) VALUES ([value], ...);`
    fn parse_insert(&mut self) -> Result<Statement> {
        self.next_token_equal(Token::Keyword(Keyword::Insert))?; // 期望下一个 token 是 INSERT
        self.next_token_equal(Token::Keyword(Keyword::Into))?; // 期望下一个 token 是 INTO

        let table_name = self.next_identifier()?; // 获取表名

        // 如果下一个 token 是 ( ，则说明表名后面存在选择的列名
        let columns = if self.next_token_equal(Token::OpenParen).is_ok() {
            let mut columns = Vec::new();
            loop {
                columns.push(self.next_identifier()?); // 获取列名
                match self.next_token()? {
                    Token::Comma => continue,   // 如果是逗号，继续获取下一个列名
                    Token::CloseParen => break, // 如果是 )，则列名获取结束
                    token => return Err(ParseError(format!("Unexpected token {token}"))), // 其他 token，返回错误
                }
            }
            Some(columns)
        } else {
            None
        };

        // 期望下一个 token 是 VALUES
        self.next_token_equal(Token::Keyword(Keyword::Values))?;

        // 解析 VALUES 后面的值
        let mut values = Vec::new();
        loop {
            self.next_token_equal(Token::OpenParen)?; // 期望下一个 token 是 (
            let mut row = Vec::new();
            loop {
                row.push(self.parse_expression()?); // 解析值
                match self.next_token()? {
                    Token::Comma => continue,   // 如果是逗号，继续解析下一个值
                    Token::CloseParen => break, // 如果是 )，则值解析结束
                    token => return Err(ParseError(format!("Unexpected token {token}"))), // 其他 token，返回错误
                }
            }
            values.push(row);
            // 如果下一个 token 不是逗号，则值解析结束，否则继续解析下一行
            if self.next_token_equal(Token::Comma).is_err() {
                break;
            }
        }

        Ok(Statement::Insert {
            table_name,
            columns,
            values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_token() {
        let tokens = vec![
            Token::Keyword(Keyword::Select),
            Token::Asterisk,
            Token::Keyword(Keyword::From),
            Token::Identifier("table1".to_string()),
            Token::Semicolon,
        ];
        let mut parser = Parser::new("SELECT * FROM table1;");
        for token in tokens {
            assert_eq!(parser.next_token().unwrap(), token);
        }
        assert!(parser.next_token().is_err());
    }

    #[test]
    fn test_next_ident() {
        let mut parser = Parser::new("table1 table2 SELECT");
        assert_eq!(parser.next_identifier().unwrap(), "table1");
        assert_eq!(parser.next_identifier().unwrap(), "table2");
        assert!(parser.next_identifier().is_err());
    }

    #[test]
    fn test_expect_next_token() {
        let mut parser = Parser::new("SELECT * FROM table1;");
        let expected_tokens = vec![
            Token::Keyword(Keyword::Select),
            Token::Asterisk,
            Token::Keyword(Keyword::From),
            Token::Identifier("table1".to_string()),
        ];
        for token in expected_tokens {
            assert!(parser.next_token_equal(token).is_ok());
        }
    }

    #[test]
    fn test_order_by() {
        let mut parser = Parser::new("ORDER BY name ASC, id DESC;");
        let ordering = parser.parse_order_by().unwrap().unwrap();
        assert_eq!(
            ordering,
            vec![
                ("name".to_string(), Ordering::Asc),
                ("id".to_string(), Ordering::Desc)
            ]
        );

        parser = Parser::new("ORDER BY name, id;");
        let ordering = parser.parse_order_by().unwrap().unwrap();
        assert_eq!(
            ordering,
            vec![
                ("name".to_string(), Ordering::Asc),
                ("id".to_string(), Ordering::Asc)
            ]
        );

        parser = Parser::new("ORDER BY name;");
        let ordering = parser.parse_order_by().unwrap().unwrap();
        assert_eq!(ordering, vec![("name".to_string(), Ordering::Asc)]);
    }

    #[test]
    fn test_select_columns() {
        let mut parser = Parser::new("name, id");
        let columns = parser.parse_select_columns().unwrap();
        assert_eq!(
            columns,
            vec![
                (Expression::Field("name".to_string()), None),
                (Expression::Field("id".to_string()), None)
            ]
        );

        parser = Parser::new("name AS user_name, id AS user_id");
        let columns = parser.parse_select_columns().unwrap();
        assert_eq!(
            columns,
            vec![
                (
                    Expression::Field("name".to_string()),
                    Some("user_name".to_string())
                ),
                (
                    Expression::Field("id".to_string()),
                    Some("user_id".to_string())
                )
            ]
        );

        parser = Parser::new("*");
        let columns = parser.parse_select_columns().unwrap();
        assert_eq!(columns, vec![]);

        parser = Parser::new("name AS user_name, id");
        let columns = parser.parse_select_columns().unwrap();
        assert_eq!(
            columns,
            vec![
                (
                    Expression::Field("name".to_string()),
                    Some("user_name".to_string())
                ),
                (Expression::Field("id".to_string()), None)
            ]
        );
    }

    #[test]
    fn test_select_from() {
        let mut parser = Parser::new("FROM table1");
        let from = parser.parse_select_from().unwrap();
        assert_eq!(
            from,
            SelectFrom::Table {
                name: "table1".to_string()
            }
        );

        parser = Parser::new("FROM table1 CROSS JOIN table2");
        let from = parser.parse_select_from().unwrap();
        assert_eq!(
            from,
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "table1".to_string()
                }),
                right: Box::new(SelectFrom::Table {
                    name: "table2".to_string()
                }),
                join_type: JoinType::Cross,
                predicate: None,
            }
        );

        parser =
            Parser::new("FROM table1 FULL JOIN table2 ON table1.name = table2.name INNER JOIN table3 ON table1.id = table2.id");
        let from = parser.parse_select_from().unwrap();
        assert_eq!(
            from,
            SelectFrom::Join {
                left: Box::new(SelectFrom::Join {
                    left: Box::new(SelectFrom::Table {
                        name: "table1".to_string()
                    }),
                    right: Box::new(SelectFrom::Table {
                        name: "table2".to_string()
                    }),
                    join_type: JoinType::Full,
                    predicate: Some(Expression::Operation(Operation::Equal(
                        Box::new(Expression::Field("table1.name".to_string())),
                        Box::new(Expression::Field("table2.name".to_string())),
                    ))),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "table3".to_string()
                }),
                join_type: JoinType::Inner,
                predicate: Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("table1.id".to_string())),
                    Box::new(Expression::Field("table2.id".to_string())),
                ))),
            }
        );
    }

    #[test]
    fn test_parse_select() {
        let mut parser = Parser::new(
            "SELECT name AS user_name, id AS user_id FROM table1 LEFT JOIN table2 ON table1.name = table2.name where id = 1 order by name desc, id limit 5 offset 1;",
        );
        let statement = parser.parse_select().unwrap();
        assert_eq!(
            statement,
            Statement::Select {
                columns: vec![
                    (
                        Expression::Field("name".to_string()),
                        Some("user_name".to_string())
                    ),
                    (
                        Expression::Field("id".to_string()),
                        Some("user_id".to_string())
                    ),
                ],
                from: SelectFrom::Join {
                    left: Box::new(SelectFrom::Table {
                        name: "table1".to_string()
                    }),
                    right: Box::new(SelectFrom::Table {
                        name: "table2".to_string()
                    }),
                    join_type: JoinType::Left,
                    predicate: Some(Expression::Operation(Operation::Equal(
                        Box::new(Expression::Field("table1.name".to_string())),
                        Box::new(Expression::Field("table2.name".to_string())),
                    ))),
                },
                filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
                ordering: vec![
                    ("name".to_string(), Ordering::Desc),
                    ("id".to_string(), Ordering::Asc)
                ],
                limit: Some(Expression::Constant(Constant::Integer(5))),
                offset: Some(Expression::Constant(Constant::Integer(1))),
            }
        );

        let mut parser = Parser::new("SELECT * FROM table1;");
        let statement = parser.parse_select().unwrap();
        assert_eq!(
            statement,
            Statement::Select {
                columns: vec![],
                from: SelectFrom::Table {
                    name: "table1".to_string()
                },
                filter: None,
                ordering: vec![],
                limit: None,
                offset: None,
            }
        );

        parser = Parser::new("INSERT INTO table1 VALUES (1, 2);");
        assert!(parser.parse_select().is_err());
    }

    #[test]
    fn test_parse_column() {
        let mut parser = Parser::new("name VARCHAR NOT NULL DEFAULT 'hello' PRIMARY KEY)");
        let column = parser.parse_column().unwrap();
        assert_eq!(
            column,
            Column {
                name: "name".to_string(),
                data_type: DataType::String,
                nullable: false,
                default: Some(Expression::Constant(Constant::String("hello".to_string())).into()),
                primary_key: true,
            }
        );
    }

    #[test]
    fn test_parse_constant_expression() {
        let mut parser = Parser::new("123");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::Constant(Constant::Integer(123)));

        parser = Parser::new("123.456");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::Constant(Constant::Float(123.456)));

        parser = Parser::new("'hello'");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(
            exp,
            Expression::Constant(Constant::String("hello".to_string()))
        );

        parser = Parser::new("true");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::Constant(Constant::Boolean(true)));

        parser = Parser::new("NULL");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::Constant(Constant::Null));
    }

    #[test]
    fn test_parse_create_table() {
        let mut parser = Parser::new("CREATE TABLE table1 (name VARCHAR NULL DEFAULT 'hello')");
        let statement = parser.parse_create_table().unwrap();
        assert_eq!(
            statement,
            Statement::CreateTable {
                name: "table1".to_string(),
                columns: vec![Column {
                    name: "name".to_string(),
                    data_type: DataType::String,
                    nullable: true,
                    default: Some(
                        Expression::Constant(Constant::String("hello".to_string())).into()
                    ),
                    primary_key: false,
                }],
            }
        );

        parser = Parser::new("CREATE TABLE table1 (id INT PRIMARY KEY, name VARCHAR)");
        let statement = parser.parse_create_table().unwrap();
        assert_eq!(
            statement,
            Statement::CreateTable {
                name: "table1".to_string(),
                columns: vec![
                    Column {
                        name: "id".to_string(),
                        data_type: DataType::Integer,
                        nullable: false,
                        default: None,
                        primary_key: true,
                    },
                    Column {
                        name: "name".to_string(),
                        data_type: DataType::String,
                        nullable: false,
                        default: None,
                        primary_key: false,
                    },
                ],
            }
        );
    }

    #[test]
    fn test_parse_insert() {
        let mut parser = Parser::new("INSERT INTO table1 VALUES (1, 'hello')");
        let statement = parser.parse_insert().unwrap();
        assert_eq!(
            statement,
            Statement::Insert {
                table_name: "table1".to_string(),
                columns: None,
                values: vec![vec![
                    Expression::Constant(Constant::Integer(1)),
                    Expression::Constant(Constant::String("hello".to_string())),
                ]],
            }
        );

        parser = Parser::new("INSERT INTO table1 (id, name) VALUES (1, 'hello')");
        let statement = parser.parse_insert().unwrap();
        assert_eq!(
            statement,
            Statement::Insert {
                table_name: "table1".to_string(),
                columns: Some(vec!["id".to_string(), "name".to_string()]),
                values: vec![vec![
                    Expression::Constant(Constant::Integer(1)),
                    Expression::Constant(Constant::String("hello".to_string())),
                ]],
            }
        );
    }

    #[test]
    fn test_parse_update() {
        let mut parser = Parser::new("UPDATE table1 SET name = 'hello' WHERE id = 1");
        let statement = parser.parse_update().unwrap();
        assert_eq!(
            statement,
            Statement::Update {
                table_name: "table1".to_string(),
                columns: vec![(
                    "name".to_string(),
                    Expression::Constant(Constant::String("hello".to_string()))
                )]
                .into_iter()
                .collect(),
                filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
            }
        );

        parser = Parser::new("UPDATE table1 SET name = 'hello', age = 18 WHERE id = 1");
        let statement = parser.parse_update().unwrap();
        assert_eq!(
            statement,
            Statement::Update {
                table_name: "table1".to_string(),
                columns: vec![
                    (
                        "name".to_string(),
                        Expression::Constant(Constant::String("hello".to_string()))
                    ),
                    (
                        "age".to_string(),
                        Expression::Constant(Constant::Integer(18))
                    ),
                ]
                .into_iter()
                .collect(),
                filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
            }
        );

        parser = Parser::new("UPDATE table1 SET name = 'hello' WHERE id = 1 AND age = 18");
        let statement = parser.parse_update().unwrap();
        assert_eq!(
            statement,
            Statement::Update {
                table_name: "table1".to_string(),
                columns: vec![(
                    "name".to_string(),
                    Expression::Constant(Constant::String("hello".to_string()))
                )]
                .into_iter()
                .collect(),
                filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
            }
        );

        parser = Parser::new("UPDATE table1 SET name = 'hello' AND age = 18");
        let statement = parser.parse_update().unwrap();
        assert_eq!(
            statement,
            Statement::Update {
                table_name: "table1".to_string(),
                columns: vec![(
                    "name".to_string(),
                    Expression::Constant(Constant::String("hello".to_string()))
                )]
                .into_iter()
                .collect(),
                filter: None,
            }
        );
    }

    #[test]
    fn test_parse_delete() {
        let mut parser = Parser::new("DELETE FROM table1 WHERE id = 1");
        let statement = parser.parse_delete().unwrap();
        assert_eq!(
            statement,
            Statement::Delete {
                table_name: "table1".to_string(),
                filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1))),),
            }
        );

        parser = Parser::new("DELETE FROM table1");
        let statement = parser.parse_delete().unwrap();
        assert_eq!(
            statement,
            Statement::Delete {
                table_name: "table1".to_string(),
                filter: None,
            }
        );
    }
}
