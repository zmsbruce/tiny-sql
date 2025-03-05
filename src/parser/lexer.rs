use std::{fmt::Display, iter::Peekable, str::Chars};

use crate::{
    Error::{self, ParseError},
    Result,
};

/// 词法分析 Token 结构体
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
    Equal,              // 等号 =
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Keyword(k) => write!(f, "{}", k),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::String(s) => write!(f, "{}", s),
            Token::Number(s) => write!(f, "{}", s),
            Token::OpenParen => write!(f, "("),
            Token::CloseParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Asterisk => write!(f, "*"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Slash => write!(f, "/"),
            Token::Equal => write!(f, "="),
        }
    }
}

/// 关键字定义
#[derive(Debug, PartialEq, Clone)]
pub enum Keyword {
    Create,
    Table,
    Int,
    Integer,
    Boolean,
    Bool,
    String,
    Text,
    Varchar,
    Float,
    Double,
    Select,
    From,
    Insert,
    Into,
    Values,
    True,
    False,
    Default,
    Not,
    Null,
    Primary,
    Key,
    Update,
    Set,
    Where,
    Delete,
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Offset,
}

impl TryFrom<&str> for Keyword {
    type Error = Error;

    /// 将字符串转换为关键字
    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        let keyword = match value.to_uppercase().as_str() {
            "CREATE" => Keyword::Create,
            "TABLE" => Keyword::Table,
            "INT" => Keyword::Int,
            "INTEGER" => Keyword::Integer,
            "BOOLEAN" => Keyword::Boolean,
            "BOOL" => Keyword::Bool,
            "STRING" => Keyword::String,
            "TEXT" => Keyword::Text,
            "VARCHAR" => Keyword::Varchar,
            "FLOAT" => Keyword::Float,
            "DOUBLE" => Keyword::Double,
            "SELECT" => Keyword::Select,
            "FROM" => Keyword::From,
            "INSERT" => Keyword::Insert,
            "INTO" => Keyword::Into,
            "VALUES" => Keyword::Values,
            "TRUE" => Keyword::True,
            "FALSE" => Keyword::False,
            "DEFAULT" => Keyword::Default,
            "NOT" => Keyword::Not,
            "NULL" => Keyword::Null,
            "PRIMARY" => Keyword::Primary,
            "KEY" => Keyword::Key,
            "UPDATE" => Keyword::Update,
            "SET" => Keyword::Set,
            "WHERE" => Keyword::Where,
            "DELETE" => Keyword::Delete,
            "ORDER" => Keyword::Order,
            "BY" => Keyword::By,
            "ASC" => Keyword::Asc,
            "DESC" => Keyword::Desc,
            "LIMIT" => Keyword::Limit,
            "OFFSET" => Keyword::Offset,
            keyword => return Err(ParseError(format!("Invalid keyword {keyword}"))),
        };
        Ok(keyword)
    }
}

impl TryFrom<String> for Keyword {
    type Error = Error;

    /// 将字符串转换为关键字
    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        Keyword::try_from(value.as_str())
    }
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Keyword::Create => "CREATE",
            Keyword::Table => "TABLE",
            Keyword::Int => "INT",
            Keyword::Integer => "INTEGER",
            Keyword::Boolean => "BOOLEAN",
            Keyword::Bool => "BOOL",
            Keyword::String => "STRING",
            Keyword::Text => "TEXT",
            Keyword::Varchar => "VARCHAR",
            Keyword::Float => "FLOAT",
            Keyword::Double => "DOUBLE",
            Keyword::Select => "SELECT",
            Keyword::From => "FROM",
            Keyword::Insert => "INSERT",
            Keyword::Into => "INTO",
            Keyword::Values => "VALUES",
            Keyword::True => "TRUE",
            Keyword::False => "FALSE",
            Keyword::Default => "DEFAULT",
            Keyword::Not => "NOT",
            Keyword::Null => "NULL",
            Keyword::Primary => "PRIMARY",
            Keyword::Key => "KEY",
            Keyword::Update => "UPDATE",
            Keyword::Set => "SET",
            Keyword::Where => "WHERE",
            Keyword::Delete => "DELETE",
            Keyword::Order => "ORDER",
            Keyword::By => "BY",
            Keyword::Asc => "ASC",
            Keyword::Desc => "DESC",
            Keyword::Limit => "LIMIT",
            Keyword::Offset => "OFFSET",
        })
    }
}

/// 词法分析 Lexer 结构体
pub struct Lexer<'a> {
    iter: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    /// 创建一个新的 Lexer 实例
    pub fn new(text: &'a str) -> Self {
        Lexer {
            iter: text.chars().peekable(),
        }
    }

    /// 如果满足条件，则跳转到下一个字符，并返回该字符，否则返回 None
    fn next_if<F>(&mut self, predicate: F) -> Option<char>
    where
        F: Fn(char) -> bool,
    {
        // `peek` 返回顶端元素，如果 `filter` 结果为 None，则直接返回，不调用 `next`
        self.iter.peek().filter(|&c| predicate(*c))?;
        // 如果 `filter` 结果为 Some，则调用 `next`，迭代到下一个元素，并返回该元素
        self.iter.next()
    }

    /// 跳转到下一个字符，直到不满足条件为止，并返回所有满足条件的字符。
    /// 如果没有字符满足条件，会返回**空字符串**
    fn next_while<F>(&mut self, predicate: F) -> String
    where
        F: Fn(char) -> bool,
    {
        let mut result = String::new();
        // 如果满足条件，则迭代到下一个元素，并将该元素添加到 `result` 中
        while let Some(c) = self.next_if(&predicate) {
            result.push(c);
        }
        result
    }

    /// 跳过开头的所有空白字符
    fn erase_whitespace(&mut self) -> usize {
        self.next_while(|c| c.is_whitespace()).len()
    }

    /// 根据单引号扫描一个字符串
    fn scan_string(&mut self) -> Result<Token> {
        // 如果不以单引号开头，则返回错误
        if self.next_if(|c| c == '\'').is_none() {
            return Err(ParseError("Expect a single quote".to_string()));
        }

        let mut s = String::new();
        for c in self.iter.by_ref() {
            match c {
                '\'' => return Ok(Token::String(s)),
                _ => s.push(c),
            }
        }
        // 如果没有找到结束的单引号，则返回错误
        Err(ParseError("Expect a single quote".to_string()))
    }

    /// 扫描数字，支持 `123`、`123.456`、`456.` 格式，否则返回 `ParseError`。
    fn scan_number(&mut self) -> Result<Token> {
        // 如果不以数字开头，则返回错误
        if self.iter.peek().filter(|c| c.is_ascii_digit()).is_none() {
            return Err(ParseError("Expect a number".to_string()));
        }
        let mut num = self.next_while(|c| c.is_ascii_digit());
        // 如果以 . 开头，则认为是小数，将其添加到 num 中，并添加 . 后面的数字
        if let Some(sep) = self.next_if(|c| c == '.') {
            num.push(sep);
            num.push_str(&self.next_while(|c| c.is_ascii_digit()));
        }
        Ok(Token::Number(num))
    }

    /// 扫描标识符或者关键字。如果扫描的 Token 不在关键字列表中，则认为其为标识符。
    /// Token 必须以字母开头，否则返回 `ParseError`。
    fn scan_identifier_or_keyword(&mut self) -> Result<Token> {
        let mut s = self
            .next_if(|c| c.is_alphabetic())
            .ok_or(ParseError("Expect an identifier".to_string()))?
            .to_string();
        s.push_str(&self.next_while(|c| c.is_alphanumeric() || c == '_'));

        Ok(Keyword::try_from(s.as_str())
            .map_or_else(|_| Token::Identifier(s.to_lowercase()), Token::Keyword))
    }

    /// 扫描符号，Token 必须为 `*(),;+-/` 中的一个，否则返回 `ParseError`。
    fn scan_symbol(&mut self) -> Result<Token> {
        let sym = self
            .iter
            .peek()
            .and_then(|c: &char| match *c {
                '*' => Some(Token::Asterisk),
                '(' => Some(Token::OpenParen),
                ')' => Some(Token::CloseParen),
                ',' => Some(Token::Comma),
                ';' => Some(Token::Semicolon),
                '+' => Some(Token::Plus),
                '-' => Some(Token::Minus),
                '/' => Some(Token::Slash),
                '=' => Some(Token::Equal),
                _ => None,
            })
            .ok_or(ParseError("Expect a symbol".to_string()))?;
        self.iter.next();
        Ok(sym)
    }

    /// 扫描下一个 Token。
    /// 正常情况下返回 `Some(Token)`。如果全部扫描完成，返回 `None`，如果 Token 不合法，返回 `Some(ParseError)`。
    fn scan_next_token(&mut self) -> Option<Result<Token>> {
        // 移除 Token 前面的空格
        self.erase_whitespace();

        // 对开头进行匹配
        let token = match self.iter.peek()? {
            '\'' => self.scan_string(), // 以单引号开头，认为是字符串
            c if c.is_ascii_digit() || *c == '.' => self.scan_number(), // 数字或者 . 开头，认为是数字
            c if c.is_alphabetic() => self.scan_identifier_or_keyword(), // 字母开头，认为是关键字或标识符
            _ => self.scan_symbol(), // 其他字符开头的情况，认为是符号
        };
        Some(token)
    }
}

impl Iterator for Lexer<'_> {
    type Item = Result<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        self.scan_next_token()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_try_from() {
        let keywords = vec![
            Keyword::Create,
            Keyword::Table,
            Keyword::Int,
            Keyword::Integer,
            Keyword::Boolean,
            Keyword::Bool,
            Keyword::String,
            Keyword::Text,
            Keyword::Varchar,
            Keyword::Float,
            Keyword::Double,
            Keyword::Select,
            Keyword::From,
            Keyword::Insert,
            Keyword::Into,
            Keyword::Values,
            Keyword::True,
            Keyword::False,
            Keyword::Default,
            Keyword::Not,
            Keyword::Null,
            Keyword::Primary,
            Keyword::Key,
        ];
        for kw in keywords {
            let s = kw.to_string().to_lowercase();
            let result = Keyword::try_from(s).unwrap();
            assert_eq!(result, kw);
        }
    }

    #[test]
    fn test_next_if() {
        let mut lexer = Lexer::new("Aa1 ");
        assert_eq!(lexer.next_if(|c| c.is_alphabetic()), Some('A'));
        assert_eq!(lexer.next_if(|c| c.is_lowercase()), Some('a'));
        assert_eq!(lexer.next_if(|c| c.is_whitespace()), None);
        assert_eq!(lexer.next_if(|c| c.is_numeric()), Some('1'));
        assert_eq!(lexer.next_if(|c| c.is_whitespace()), Some(' '));
    }

    #[test]
    fn test_next_while() {
        let mut lexer = Lexer::new("Aa1 ");
        assert_eq!(lexer.next_while(|c| c.is_alphabetic()), "Aa");
        assert_eq!(lexer.next_while(|c| c.is_numeric()), "1");
        assert_eq!(lexer.next_while(|c| c.is_alphabetic()), "");
        assert_eq!(lexer.next_while(|c| c.is_whitespace()), " ");
    }

    #[test]
    fn test_erase_whitespace() {
        let mut lexer = Lexer::new("  Aa1");
        assert_eq!(lexer.erase_whitespace(), 2);
        assert_eq!(lexer.iter.peek(), Some(&'A'));
    }

    #[test]
    fn test_scan_string() {
        let mut lexer = Lexer::new("'Hello, World!'");
        assert_eq!(
            lexer.scan_string().unwrap(),
            Token::String("Hello, World!".to_string())
        );

        lexer = Lexer::new("Hello, World!'");
        assert!(lexer.scan_string().is_err());

        lexer = Lexer::new("'Hello, World!");
        assert!(lexer.scan_string().is_err());
    }

    #[test]
    fn test_scan_number() {
        let mut lexer = Lexer::new("123.456");
        assert_eq!(
            lexer.scan_number().unwrap(),
            Token::Number("123.456".to_string())
        );

        lexer = Lexer::new("456.");
        assert_eq!(
            lexer.scan_number().unwrap(),
            Token::Number("456.".to_string())
        );

        lexer = Lexer::new("abc");
        assert!(lexer.scan_number().is_err());

        lexer = Lexer::new("abc.def");
        assert!(lexer.scan_number().is_err());

        lexer = Lexer::new(".456");
        assert!(lexer.scan_number().is_err());
    }

    #[test]
    fn test_scan_identifier_or_keyword() {
        let keywords = vec![
            Keyword::Create,
            Keyword::Table,
            Keyword::Int,
            Keyword::Integer,
            Keyword::Boolean,
            Keyword::Bool,
            Keyword::String,
            Keyword::Text,
            Keyword::Varchar,
            Keyword::Float,
            Keyword::Double,
            Keyword::Select,
            Keyword::From,
            Keyword::Insert,
            Keyword::Into,
            Keyword::Values,
            Keyword::True,
            Keyword::False,
            Keyword::Default,
            Keyword::Not,
            Keyword::Null,
            Keyword::Primary,
            Keyword::Key,
        ];
        for kw in keywords {
            let kw_string = kw.to_string();
            let mut lexer = Lexer::new(&kw_string);
            assert_eq!(
                lexer.scan_identifier_or_keyword().unwrap(),
                Token::Keyword(kw)
            );
        }

        let mut lexer = Lexer::new("IdEntiFier abcdef");
        assert_eq!(
            lexer.scan_identifier_or_keyword().unwrap(),
            Token::Identifier("identifier".to_string())
        )
    }

    #[test]
    fn test_scan_symbol() {
        let mut lexer = Lexer::new("*(),;+-/（");
        assert_eq!(lexer.scan_symbol().unwrap(), Token::Asterisk);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::OpenParen);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::CloseParen);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::Comma);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::Semicolon);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::Plus);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::Minus);
        assert_eq!(lexer.scan_symbol().unwrap(), Token::Slash);
        assert!(lexer.scan_symbol().is_err());
    }

    #[test]
    fn test_scan_next_token() {
        let mut lexer = Lexer::new("insert into tbl values (1, 2, '3', true, false, 4.55);");
        let mut tokens = Vec::new();
        while let Some(Ok(token)) = lexer.scan_next_token() {
            tokens.push(token);
        }
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Insert),
                Token::Keyword(Keyword::Into),
                Token::Identifier("tbl".to_string()),
                Token::Keyword(Keyword::Values),
                Token::OpenParen,
                Token::Number("1".to_string()),
                Token::Comma,
                Token::Number("2".to_string()),
                Token::Comma,
                Token::String("3".to_string()),
                Token::Comma,
                Token::Keyword(Keyword::True),
                Token::Comma,
                Token::Keyword(Keyword::False),
                Token::Comma,
                Token::Number("4.55".to_string()),
                Token::CloseParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn test_scan_all_tokens() {
        let lexer = Lexer::new("SELECT * FROM customers");
        let tokens = lexer.peekable().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::Keyword(Keyword::Select));
        assert_eq!(tokens[1], Token::Asterisk);
        assert_eq!(tokens[2], Token::Keyword(Keyword::From));
        assert_eq!(tokens[3], Token::Identifier("customers".to_string()));
    }
}
