use std::collections::HashMap;

use super::Column;
use crate::{
    error::Error::InternalError,
    parser::ast::{Expression, JoinType, Operation},
    schema::{Row, Value},
    Result,
};

/// 循环连接，将左右表的行进行笛卡尔积
pub fn loop_join(
    left_cols: &[Column],
    right_cols: &[Column],
    left_rows: &[Row],
    right_rows: &[Row],
) -> Result<(Vec<Column>, Vec<Row>)> {
    let new_columns = [left_cols, right_cols].concat();
    let new_rows: Vec<Row> = left_rows
        .iter()
        .flat_map(|left_row| {
            right_rows.iter().map(move |right_row| {
                let mut row = left_row.clone();
                row.extend(right_row.clone());
                row
            })
        })
        .collect();
    Ok((new_columns, new_rows))
}

/// 哈希连接，根据 Join 条件将左右表的行合并
pub fn hash_join(
    left_cols: &[Column],
    right_cols: &[Column],
    left_rows: &[Row],
    right_rows: &[Row],
    join_type: &JoinType,
    predicate: &Expression,
) -> Result<(Vec<Column>, Vec<Row>)> {
    // 解析 Join 条件
    let (left_cond, right_cond) = match predicate {
        Expression::Operation(Operation::Equal(left, right))
            if left.is_field() && right.is_field() =>
        {
            (
                Column::try_from(left.as_ref())?,
                Column::try_from(right.as_ref())?,
            )
        }
        _ => return Err(InternalError("Unsupported join condition".to_string())),
    };

    // 获取左右表的列索引
    let left_col_idx = left_cols
        .iter()
        .position(|col| *col == left_cond)
        .ok_or(InternalError(format!("Column {:?} not found", left_cond)))?;
    let right_col_idx = right_cols
        .iter()
        .position(|col| *col == right_cond)
        .ok_or(InternalError(format!("Column {:?} not found", right_cond)))?;

    // 合并左右表的列名
    let new_columns = [left_cols, right_cols].concat();

    // 根据 join_type 不同，分别构建匹配结果
    let new_rows = match join_type {
        // INNER 和 LEFT JOIN：构建右表哈希表，根据左表中对应值查找匹配行
        JoinType::Inner | JoinType::Left => {
            // 构建右表的哈希表
            let mut right_hash = HashMap::new();
            for row in right_rows {
                let rows = right_hash
                    .entry(row[right_col_idx].clone())
                    .or_insert(Vec::new());
                rows.push(row);
            }

            let mut new_rows = Vec::new();
            for left_row in left_rows {
                if let Some(right_rows) = right_hash.get(&left_row[left_col_idx]) {
                    for right_row in right_rows {
                        let mut new_row = left_row.clone();
                        new_row.extend(right_row.iter().cloned());
                        new_rows.push(new_row);
                    }
                } else if matches!(join_type, JoinType::Left) {
                    // 如果是 LEFT JOIN，则将右表的列填充为 NULL，否则忽略
                    let mut new_row = left_row.clone();
                    new_row.extend(vec![Value::Null; right_cols.len()]);
                    new_rows.push(new_row);
                }
            }
            new_rows
        }
        // RIGHT JOIN：构建左表哈希表，根据右表中对应值查找匹配行
        JoinType::Right => {
            // 构建左表的哈希表
            let mut left_hash = HashMap::new();
            for row in left_rows {
                let rows = left_hash
                    .entry(row[left_col_idx].clone())
                    .or_insert(Vec::new());
                rows.push(row);
            }

            // 通过右表的列值在左表的哈希表中查找匹配的行，并将左右表的行合并
            let mut new_rows = Vec::new();
            for right_row in right_rows {
                if let Some(left_rows) = left_hash.get(&right_row[right_col_idx]) {
                    for left_row in left_rows {
                        let mut new_row = left_row.to_vec();
                        new_row.extend(right_row.clone());
                        new_rows.push(new_row);
                    }
                } else {
                    // 将左表的列填充为 NULL
                    let mut new_row = vec![Value::Null; left_cols.len()];
                    new_row.extend(right_row.iter().cloned());
                    new_rows.push(new_row);
                }
            }
            new_rows
        }
        // FULL JOIN：同时处理左右未匹配的情况
        JoinType::Full => {
            // 构建左表的哈希表
            let mut left_hash = HashMap::new();
            for row in left_rows {
                let rows = left_hash
                    .entry(row[left_col_idx].clone())
                    .or_insert(Vec::new());
                rows.push(row);
            }

            // 构建右表的哈希表
            let mut right_hash = HashMap::new();
            for row in right_rows {
                let rows = right_hash
                    .entry(row[right_col_idx].clone())
                    .or_insert(Vec::new());
                rows.push(row);
            }

            // 通过左表的列值在右表的哈希表中查找匹配的行，并将左右表的行合并
            let mut new_rows = Vec::new();
            for left_row in left_rows {
                if let Some(right_rows) = right_hash.get(&left_row[left_col_idx]) {
                    for right_row in right_rows {
                        let mut new_row = left_row.clone();
                        new_row.extend(right_row.iter().cloned());
                        new_rows.push(new_row);
                    }
                } else {
                    // 将右表的列填充为 NULL
                    let mut new_row = left_row.clone();
                    new_row.extend(vec![Value::Null; right_cols.len()]);
                    new_rows.push(new_row);
                }
            }

            // 查找右表中未匹配的行
            for right_row in right_rows {
                if !left_hash.contains_key(&right_row[right_col_idx]) {
                    // 将左表的列填充为 NULL
                    let mut new_row = vec![Value::Null; left_cols.len()];
                    new_row.extend(right_row.iter().cloned());
                    new_rows.push(new_row);
                }
            }
            new_rows
        }
        _ => {
            return Err(InternalError(format!(
                "Unsupported join type: {}",
                join_type
            )))
        }
    };

    Ok((new_columns, new_rows))
}
