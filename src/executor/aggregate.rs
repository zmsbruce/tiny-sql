use super::{get_column_index, Column};
use crate::{
    error::Error::InternalError,
    parser::ast::Aggregate,
    schema::{Row, Value},
    Result,
};

pub fn aggregate(col: &Column, cols: &[Column], rows: &[Row]) -> Result<Value> {
    if let Some(agg) = col.agg {
        // 将聚集函数设置为空，从而在 cols 中查找列索引聚集起来
        let mut col = col.clone();
        col.agg = None;

        match agg {
            Aggregate::Count => count(&col, cols, rows),
            Aggregate::Sum => sum(&col, cols, rows),
            Aggregate::Min => min(&col, cols, rows),
            Aggregate::Max => max(&col, cols, rows),
            Aggregate::Avg => avg(&col, cols, rows),
        }
    } else {
        Err(InternalError(format!(
            "Column {:?} does not have an aggregate function",
            col
        )))
    }
}

fn count(col: &Column, cols: &[Column], rows: &[Row]) -> Result<Value> {
    let count = if col.col_name == "*" {
        rows.len()
    } else {
        let col_idx = get_column_index(cols, col)?;
        rows.iter()
            .filter(|row| row[col_idx] != Value::Null)
            .count()
    } as i64;

    Ok(Value::Integer(count))
}

fn sum(col: &Column, cols: &[Column], rows: &[Row]) -> Result<Value> {
    let col_idx = get_column_index(cols, col)?;
    let mut sum = Value::Null;
    for row in rows {
        match &row[col_idx] {
            Value::Integer(value) => {
                if sum == Value::Null {
                    sum = Value::Integer(0);
                }
                sum = Value::Integer(sum.as_i64()? + value);
            }
            Value::Float(value) => {
                if sum == Value::Null {
                    sum = Value::Float(0.0);
                }
                sum = Value::Float(sum.as_f64()? + value);
            }
            Value::Null => continue,
            val => {
                return Err(InternalError(format!(
                    "Unsupported value {:?} for sum",
                    val
                )))
            }
        }
    }

    Ok(sum)
}

fn min(col: &Column, cols: &[Column], rows: &[Row]) -> Result<Value> {
    let col_idx = get_column_index(cols, col)?;
    let mut min = Value::Null;
    for row in rows {
        match &row[col_idx] {
            Value::Integer(value) => {
                if min == Value::Null || *value < min.as_i64()? {
                    min = Value::Integer(*value);
                }
            }
            Value::Float(value) => {
                if min == Value::Null || *value < min.as_f64()? {
                    min = Value::Float(*value);
                }
            }
            Value::Null => continue,
            Value::String(value) => {
                if min == Value::Null || value.as_str() < min.as_str()? {
                    min = Value::String(value.clone());
                }
            }
            val => {
                return Err(InternalError(format!(
                    "Unsupported value {:?} for min",
                    val
                )))
            }
        }
    }

    Ok(min)
}

fn max(col: &Column, cols: &[Column], rows: &[Row]) -> Result<Value> {
    let col_idx = get_column_index(cols, col)?;
    let mut max = Value::Null;
    for row in rows {
        match &row[col_idx] {
            Value::Integer(value) => {
                if max == Value::Null || *value > max.as_i64()? {
                    max = Value::Integer(*value);
                }
            }
            Value::Float(value) => {
                if max == Value::Null || *value > max.as_f64()? {
                    max = Value::Float(*value);
                }
            }
            Value::Null => continue,
            Value::String(value) => {
                if max == Value::Null || value.as_str() > max.as_str()? {
                    max = Value::String(value.clone());
                }
            }
            val => {
                return Err(InternalError(format!(
                    "Unsupported value {:?} for max",
                    val
                )))
            }
        }
    }

    Ok(max)
}

fn avg(col: &Column, cols: &[Column], rows: &[Row]) -> Result<Value> {
    let col_idx = get_column_index(cols, col)?;
    let mut sum = 0.0;
    let mut count = 0;
    for row in rows {
        match &row[col_idx] {
            Value::Integer(value) => {
                sum += *value as f64;
                count += 1;
            }
            Value::Float(value) => {
                sum += *value;
                count += 1;
            }
            Value::Null => continue,
            val => {
                return Err(InternalError(format!(
                    "Unsupported value {:?} for avg",
                    val
                )))
            }
        }
    }

    if count == 0 {
        return Ok(Value::Null);
    }

    Ok(Value::Float(sum / count as f64))
}
