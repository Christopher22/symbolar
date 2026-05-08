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

pub struct Parser {
    tokens: Vec<String>,
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

    /// Simple tokenizer to split by operators and parentheses
    fn tokenize(s: &str) -> Vec<String> {
        s.replace('(', " ( ")
            .replace(')', " ) ")
            .replace('+', " + ")
            .replace('*', " * ")
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }

    /// Level 1: Addition (lowest precedence)
    fn parse_add(&mut self) -> Result<Expression, ParseError> {
        let mut terms = vec![self.parse_mul()?];

        while self.pos < self.tokens.len() && self.tokens[self.pos] == "+" {
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
        let mut left = self.parse_primary()?;

        while self.pos < self.tokens.len() && self.tokens[self.pos] == "*" {
            self.pos += 1; // consume '*'
            let right = self.parse_primary()?;
            left = Expression::Mul(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// Level 3: Values and Parentheses (highest precedence)
    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        if self.pos >= self.tokens.len() {
            return Err(ParseError::InvalidExpression);
        }

        match self.tokens[self.pos].as_str() {
            "(" => {
                self.pos += 1; // consume '('
                let expr = self.parse_add()?;
                if self.pos < self.tokens.len() && self.tokens[self.pos] == ")" {
                    self.pos += 1; // consume ')'
                    Ok(expr)
                } else {
                    Err(ParseError::MismatchedParentheses)
                }
            }
            ")" => Err(ParseError::UnexpectedToken(")".to_string())),
            "+" | "*" => Err(ParseError::UnexpectedToken(self.tokens[self.pos].clone())),
            val => {
                self.pos += 1;
                Ok(Expression::Value(Cow::Owned(val.to_string())))
            }
        }
    }
}
