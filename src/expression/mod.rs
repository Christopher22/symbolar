mod parser;

use std::{
    borrow::Cow,
    hash::{Hash, Hasher},
    ops::{Add, Mul},
};

pub use self::parser::ParseError;
use self::parser::Parser;

/// An expression that can be evaluated to a value.
#[derive(Debug, Clone)]
pub enum Expression {
    /// A value.
    Value(Cow<'static, str>),
    /// The sum of two expressions.
    Plus(Box<Expression>, Box<Expression>),
    /// The product of two expressions.
    Mul(Box<Expression>, Box<Expression>),
}

impl Expression {
    /// Create a new expression from a string value.
    pub const fn new(value: &'static str) -> Self {
        Expression::Value(Cow::Borrowed(value))
    }

    /// Check if the expression is a value.
    pub const fn is_value(&self) -> bool {
        matches!(self, Expression::Value(_))
    }

    /// Evaluate the expression given a variable mapping.
    pub fn evaluate<'a, T, C>(&'a self, variables: C) -> Result<Cow<'a, T>, UnknownValue>
    where
        T: Clone,
        C: Copy + Fn(&str) -> Option<Cow<'a, T>>,
        for<'x, 'y> &'x T: Add<&'y T, Output = T>,
        for<'x, 'y> &'x T: Mul<&'y T, Output = T>,
    {
        match self {
            Expression::Value(val) => {
                let val = val.as_ref();
                variables(val).ok_or_else(|| UnknownValue::from(val))
            }
            Expression::Plus(lhs, rhs) => {
                let l = lhs.evaluate(variables)?;
                let r = rhs.evaluate(variables)?;
                Ok(Cow::Owned(&*l + &*r))
            }
            Expression::Mul(lhs, rhs) => {
                let l = lhs.evaluate(variables)?;
                let r = rhs.evaluate(variables)?;
                Ok(Cow::Owned(&*l * &*r))
            }
        }
    }
}

impl Add for Expression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Expression::Plus(Box::new(self), Box::new(rhs))
    }
}

impl Mul for Expression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Expression::Mul(Box::new(self), Box::new(rhs))
    }
}

impl PartialEq<Expression> for Expression {
    fn eq(&self, other: &Expression) -> bool {
        // Ensure commutative property
        match (self, other) {
            (Expression::Value(val), Expression::Value(other_val)) => val == other_val,
            (Expression::Plus(lhs, rhs), Expression::Plus(other_lhs, other_rhs)) => {
                (lhs.as_ref() == other_lhs.as_ref() && rhs.as_ref() == other_rhs.as_ref())
                    || (lhs.as_ref() == other_rhs.as_ref() && rhs.as_ref() == other_lhs.as_ref())
            }
            (Expression::Mul(lhs, rhs), Expression::Mul(other_lhs, other_rhs)) => {
                (lhs.as_ref() == other_lhs.as_ref() && rhs.as_ref() == other_rhs.as_ref())
                    || (lhs.as_ref() == other_rhs.as_ref() && rhs.as_ref() == other_lhs.as_ref())
            }
            _ => false,
        }
    }
}

impl Eq for Expression {}

impl Hash for Expression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Expression::Value(val) => {
                state.write_u8(0);
                val.hash(state);
            }
            Expression::Plus(lhs, rhs) => {
                state.write_u8(1);
                // Commutative hash: ensure order doesn't matter
                let mut h1 = std::collections::hash_map::DefaultHasher::new();
                lhs.hash(&mut h1);
                let mut h2 = std::collections::hash_map::DefaultHasher::new();
                rhs.hash(&mut h2);
                state.write_u64(h1.finish().wrapping_add(h2.finish()));
            }
            Expression::Mul(lhs, rhs) => {
                state.write_u8(2);
                // Commutative hash: ensure order doesn't matter
                let mut h1 = std::collections::hash_map::DefaultHasher::new();
                lhs.hash(&mut h1);
                let mut h2 = std::collections::hash_map::DefaultHasher::new();
                rhs.hash(&mut h2);
                state.write_u64(h1.finish().wrapping_add(h2.finish()));
            }
        }
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Value(val) => write!(f, "{}", val),
            Expression::Plus(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            Expression::Mul(lhs, rhs) => write!(f, "({} * {})", lhs, rhs),
        }
    }
}

impl std::str::FromStr for Expression {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Parser::new(s).parse()
    }
}

/// An error indicating that a variable was not found during evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnknownValue(String);

impl From<String> for UnknownValue {
    fn from(value: String) -> Self {
        UnknownValue(value)
    }
}

impl<'a> From<&'a str> for UnknownValue {
    fn from(value: &'a str) -> Self {
        UnknownValue(value.to_string())
    }
}

impl std::fmt::Display for UnknownValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnknownValue {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grouping() {
        let expr1 = Expression::new("a") + Expression::new("b") * Expression::new("c");
        assert_eq!("(a + (b * c))", expr1.to_string());

        let expr2 = Expression::new("a") * Expression::new("b")
            + Expression::new("c") * Expression::new("d");
        assert_eq!("((a * b) + (c * d))", expr2.to_string());

        let expr3 = Expression::new("a") + Expression::new("b") + Expression::new("c");
        assert_eq!("((a + b) + c)", expr3.to_string());

        let expr4 = (Expression::new("a") + Expression::new("b")) * Expression::new("c");
        assert_eq!("((a + b) * c)", expr4.to_string());
    }

    #[test]
    fn test_commutativity() {
        let expr1 = Expression::new("a") + Expression::new("b");
        let expr2 = Expression::new("b") + Expression::new("a");
        assert_eq!(expr1, expr1);
        assert_eq!(expr1, expr2);
    }

    #[test]
    fn test_evaluation() {
        let expr = Expression::new("1") + Expression::new("2") * Expression::new("3");
        let result = expr
            .evaluate(|var| {
                var.parse::<u32>()
                    .ok()
                    .map(|value| Cow::<'static, u32>::Owned(value))
            })
            .unwrap();
        assert_eq!(result, Cow::Owned(1 + 2 * 3));
    }

    #[test]
    fn test_evaluation_invalid() {
        let expr = Expression::new("1") + Expression::new("a") * Expression::new("3");
        assert_eq!(
            expr.evaluate(|var| {
                var.parse::<u32>()
                    .ok()
                    .map(|value| Cow::<'static, u32>::Owned(value))
            }),
            Err(UnknownValue::from("a"))
        );
    }

    #[test]
    fn test_parsing() {
        assert_eq!(
            "abc".parse::<Expression>().map(|expr| expr.to_string()),
            Ok("abc".to_string())
        );

        assert_eq!(
            "1 + 2".parse::<Expression>().map(|expr| expr.to_string()),
            Ok("(1 + 2)".to_string())
        );

        assert_eq!(
            "1 * 2".parse::<Expression>().map(|expr| expr.to_string()),
            Ok("(1 * 2)".to_string())
        );

        assert_eq!(
            "1 + 2 * 3"
                .parse::<Expression>()
                .map(|expr| expr.to_string()),
            Ok("(1 + (2 * 3))".to_string())
        );

        assert_eq!(
            "(1 + 2) * 3"
                .parse::<Expression>()
                .map(|expr| expr.to_string()),
            Ok("((1 + 2) * 3)".to_string())
        );

        assert_eq!(
            "(( abc + def ) *    ghi) )"
                .parse::<Expression>()
                .map(|expr| expr.to_string()),
            Ok("((abc + def) * ghi)".to_string())
        );
    }

    #[test]
    fn test_parsing_invalid() {
        assert_eq!(
            "1 +".parse::<Expression>(),
            Err(ParseError::InvalidExpression)
        );
        assert_eq!(
            "1 * (2 + 3".parse::<Expression>(),
            Err(ParseError::MismatchedParentheses)
        );
        assert_eq!(
            "1 + * 2".parse::<Expression>(),
            Err(ParseError::UnexpectedToken("*".to_string()))
        );
    }
}
