use std::fmt::Display;

use super::keyword::Keyword;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Keyword(Keyword),   // 关键字，如 SELECT
    Identifier(String), // 标识符，如表名、列名
    String(String),     // 字符串类型的数据
    Number(String),     // 数值类型，比如整数和浮点数
    OpenParen,          // 左括号 (
    CloseParen,         // 右括号 )
    Comma,              // 逗号 ,
    Semicolon,          // 分号 ;
    Asterisk,           // 星号 *
    Plus,               // 加号 +
    Minus,              // 减号 -
    Slash,              // 斜杠 /
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Token::Keyword(k) => k.as_str(),
            Token::Identifier(s) => s,
            Token::String(s) => s,
            Token::Number(s) => s,
            Token::OpenParen => "(",
            Token::CloseParen => ")",
            Token::Comma => ",",
            Token::Semicolon => ";",
            Token::Asterisk => "*",
            Token::Plus => "+",
            Token::Minus => "-",
            Token::Slash => "/",
        })
    }
}
