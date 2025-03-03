use std::{collections::HashMap, iter::Peekable};

use crate::{
    schema::{Column, DataType},
    Error::ParseError,
    Result,
};
use ast::{Constant, Expression, Statement};
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
    /// - select * from [table_name];
    /// - create table [table_name] ([column_name] [data_type] [nullable] [default] [primary key], ...);
    /// - insert into [table_name] ([column_name], ...) values ([value], ...);
    /// - update [table_name] set [column_name] = [value], ... where [condition];
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
    /// 语法：SELECT * FROM [table_name];
    fn parse_select(&mut self) -> Result<Statement> {
        self.next_token_equal(Token::Keyword(Keyword::Select))?; // 期望下一个 token 是 SELECT
        self.next_token_equal(Token::Asterisk)?; // 期望下一个 token 是 *
        self.next_token_equal(Token::Keyword(Keyword::From))?; // 期望下一个 token 是 FROM

        let table_name = self.next_identifier()?; // 获取表名
        Ok(Statement::Select { table_name })
    }

    /// 解析 UPDATE 语句
    /// 语法：UPDATE [table_name] SET [column_name] = [value], ... WHERE [condition];
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
        let where_clause = if self
            .next_token_if(|t| *t == Token::Keyword(Keyword::Where))
            .is_ok()
        {
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        Ok(Statement::Update {
            table_name,
            columns,
            where_clause,
        })
    }

    /// 解析 WHERE 子句
    /// 语法：WHERE column_name = expression
    ///
    /// 目前只支持单个表达式且仅为等于操作，不支持其他操作符和表达式组合
    fn parse_where_clause(&mut self) -> Result<(String, Expression)> {
        let col_name = self.next_identifier()?;
        self.next_token_equal(Token::Equal)?;
        let val = self.parse_expression()?;
        Ok((col_name, val))
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
            Token::Number(num_str) => {
                // 如果是数字，则解析为整数或浮点数
                if num_str.chars().all(|ch| ch.is_ascii_digit()) {
                    // 如果数字全部是 0-9，则判断为整数
                    let num = num_str.parse::<i64>()?;
                    Expression::from(Constant::Integer(num))
                } else {
                    // 否则为浮点数
                    let num = num_str.parse::<f64>()?;
                    Expression::from(Constant::Float(num))
                }
            }
            Token::String(s) => Expression::from(Constant::String(s)), // 字符串
            Token::Keyword(Keyword::True) => Expression::from(Constant::Boolean(true)), // 布尔值 true
            Token::Keyword(Keyword::False) => Expression::from(Constant::Boolean(false)), // 布尔值 false
            Token::Keyword(Keyword::Null) => Expression::from(Constant::Null),            // NULL
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
    /// 语法：INSERT INTO [table_name] ([column_name], ...) VALUES ([value], ...);
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
    fn test_parse_select() {
        let mut parser = Parser::new("SELECT * FROM table1;");
        let statement = parser.parse_select().unwrap();
        assert_eq!(
            statement,
            Statement::Select {
                table_name: "table1".to_string()
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
                default: Some(Expression::from(Constant::String("hello".to_string())).into()),
                primary_key: true,
            }
        );
    }

    #[test]
    fn test_parse_constant_expression() {
        let mut parser = Parser::new("123");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::from(Constant::Integer(123)));

        parser = Parser::new("123.456");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::from(Constant::Float(123.456)));

        parser = Parser::new("'hello'");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::from(Constant::String("hello".to_string())));

        parser = Parser::new("true");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::from(Constant::Boolean(true)));

        parser = Parser::new("NULL");
        let exp = parser.parse_expression().unwrap();
        assert_eq!(exp, Expression::from(Constant::Null));
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
                    default: Some(Expression::from(Constant::String("hello".to_string())).into()),
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
                    Expression::from(Constant::Integer(1)),
                    Expression::from(Constant::String("hello".to_string())),
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
                    Expression::from(Constant::Integer(1)),
                    Expression::from(Constant::String("hello".to_string())),
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
                    Expression::from(Constant::String("hello".to_string()))
                )]
                .into_iter()
                .collect(),
                where_clause: Some(("id".to_string(), Expression::from(Constant::Integer(1)))),
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
                        Expression::from(Constant::String("hello".to_string()))
                    ),
                    ("age".to_string(), Expression::from(Constant::Integer(18))),
                ]
                .into_iter()
                .collect(),
                where_clause: Some(("id".to_string(), Expression::from(Constant::Integer(1)))),
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
                    Expression::from(Constant::String("hello".to_string()))
                )]
                .into_iter()
                .collect(),
                where_clause: Some(("id".to_string(), Expression::from(Constant::Integer(1)))),
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
                    Expression::from(Constant::String("hello".to_string()))
                )]
                .into_iter()
                .collect(),
                where_clause: None,
            }
        );
    }
}
