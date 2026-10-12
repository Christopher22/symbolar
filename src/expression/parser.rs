use std::borrow::Cow;

use super::Expression;

/// An error during parsing of an expression.
#[derive(Debug, PartialEq)]
pub enum ParseError {
    /// The expression is invalid.
    InvalidExpression,
    /// An unexpected token was encountered.
    UnexpectedToken(String),
    /// Mismatched parentheses were encountered.
    MismatchedParentheses,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidExpression => write!(f, "Invalid expression"),
            Self::UnexpectedToken(t) => write!(f, "Unexpected token: {}", t),
            Self::MismatchedParentheses => write!(f, "Mismatched parentheses"),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Operator(char),
    Name(String),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Operator(c) => write!(f, "{c}"),
            Token::Name(name) => f.write_str(name),
        }
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        Self {
            tokens: Self::tokenize(input),
            pos: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Expression, ParseError> {
        self.parse_add()
    }

    /// Simple tokenizer to split by operators and parentheses.
    /// Backticks quote a segment verbatim, allowing whitespace and operator characters in names (e.g. `` var:`weiß nicht` ``).
    fn tokenize(s: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current: Option<String> = None;
        let mut quoted = false;
        for c in s.chars() {
            if quoted {
                if c == '`' {
                    quoted = false;
                } else {
                    current.get_or_insert_with(String::new).push(c);
                }
                continue;
            }

            match c {
                '`' => {
                    quoted = true;
                    current.get_or_insert_with(String::new);
                }
                '(' | ')' | '+' | '*' | '^' => {
                    tokens.extend(current.take().map(Token::Name));
                    tokens.push(Token::Operator(c));
                }
                c if c.is_whitespace() => tokens.extend(current.take().map(Token::Name)),
                c => current.get_or_insert_with(String::new).push(c),
            }
        }
        tokens.extend(current.take().map(Token::Name));
        tokens
    }

    fn is_operator(&self, operator: char) -> bool {
        self.tokens.get(self.pos) == Some(&Token::Operator(operator))
    }

    /// Level 1: Addition (lowest precedence)
    fn parse_add(&mut self) -> Result<Expression, ParseError> {
        let mut terms = vec![self.parse_mul()?];

        while self.is_operator('+') {
            self.pos += 1; // consume '+'
            terms.push(self.parse_mul()?);
        }

        if terms.len() == 1 {
            Ok(terms.remove(0))
        } else {
            Ok(Expression::Plus(terms))
        }
    }

    /// Level 2: Multiplication
    fn parse_mul(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_pow()?;

        while self.is_operator('*') {
            self.pos += 1; // consume '*'
            let right = self.parse_pow()?;
            left = Expression::Mul(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// Level 3: Power (right-associative)
    fn parse_pow(&mut self) -> Result<Expression, ParseError> {
        let base = self.parse_primary()?;

        if self.is_operator('^') {
            self.pos += 1; // consume '^'
            let exp_token = self
                .tokens
                .get(self.pos)
                .ok_or(ParseError::InvalidExpression)?
                .to_string();
            let exp = exp_token
                .parse::<u32>()
                .map_err(|_| ParseError::UnexpectedToken(exp_token))?;
            self.pos += 1;
            Ok(Expression::Pow(Box::new(base), exp))
        } else {
            Ok(base)
        }
    }

    /// Level 4: Values and Parentheses (highest precedence)
    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        match self.tokens.get(self.pos).cloned() {
            None => Err(ParseError::InvalidExpression),
            Some(Token::Operator('(')) => {
                self.pos += 1; // consume '('
                let expr = self.parse_add()?;
                if self.is_operator(')') {
                    self.pos += 1; // consume ')'
                    Ok(expr)
                } else {
                    Err(ParseError::MismatchedParentheses)
                }
            }
            Some(Token::Operator(c)) => Err(ParseError::UnexpectedToken(c.to_string())),
            Some(Token::Name(val)) => {
                self.pos += 1;
                Ok(Expression::Value(Cow::Owned(val)))
            }
        }
    }
}
