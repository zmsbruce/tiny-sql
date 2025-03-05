use std::collections::HashMap;

use crate::{
    engine::{Engine, Transaction},
    error::{Error::InternalError, Result},
    parser::ast::{Expression, JoinType, Ordering, SelectFrom, Statement},
    schema::{Row, Table, Value},
    storage::Storage,
};

pub enum ExecuteResult {
    CreateTable,
    Insert,
    Scan {
        columns: Vec<String>,
        rows: Vec<Row>,
    },
    Update(usize),
    Delete(usize),
}

/// SQL 执行器
///
/// 负责执行 SQL 语句，将 SQL 语句转换为对存储引擎的操作
pub struct Executor<S: Storage> {
    transaction: Transaction<S>,
    is_committed: bool,
}

impl<S: Storage> Drop for Executor<S> {
    /// 在执行器销毁时，检查事务是否提交，并提交事务
    fn drop(&mut self) {
        // 如果事务未提交，提交事务
        if !self.is_committed {
            if let Err(e) = self.transaction.commit() {
                eprintln!("Failed to commit transaction: {:?}", e);
            }
        }
    }
}

impl<S: Storage> Executor<S> {
    // 创建一个新的执行器
    pub fn from_engine(eng: &Engine<S>) -> Result<Self> {
        Ok(Self {
            transaction: eng.start_txn()?,
            is_committed: false,
        })
    }

    /// 执行 SQL 语句
    pub fn execute(&self, stmt: Statement) -> Result<ExecuteResult> {
        match stmt {
            Statement::CreateTable { name, columns } => {
                let table = Table::new(&name, columns)?;
                self.transaction.create_table(table)?;

                Ok(ExecuteResult::CreateTable)
            }
            Statement::Insert {
                table_name,
                columns,
                values,
            } => {
                self.insert(table_name, columns.unwrap_or_default(), values)?;
                Ok(ExecuteResult::Insert)
            }
            Statement::Select {
                columns,
                from,
                filter,
                ordering,
                limit,
                offset,
            } => {
                let (columns, rows) =
                    self.select(columns, from, filter, ordering, limit, offset)?;

                Ok(ExecuteResult::Scan { columns, rows })
            }
            Statement::Update {
                table_name,
                columns,
                filter,
            } => {
                let count = self.update(table_name, columns, filter)?;
                Ok(ExecuteResult::Update(count))
            }
            Statement::Delete { table_name, filter } => {
                let count = self.delete(table_name, filter)?;
                Ok(ExecuteResult::Delete(count))
            }
        }
    }

    /// 提交事务
    #[inline]
    pub fn commit(&mut self) -> Result<()> {
        self.transaction.commit()?;
        self.is_committed = true;
        Ok(())
    }

    /// 回滚事务
    #[inline]
    pub fn rollback(&mut self) -> Result<()> {
        self.transaction.rollback()?;
        Ok(())
    }

    /// 扫描表
    fn scan(
        &self,
        table_name: &str,
        filter: Option<(String, Expression)>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        let table = self
            .transaction
            .get_table(table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;

        let columns = table.columns.iter().map(|c| c.name.clone()).collect();

        let rows = self.transaction.scan_table(&table, filter)?;

        Ok((columns, rows))
    }

    /// 扫描 Join 表，返回所有的列名和行数据
    fn scan_all_from_join(&self, from: &SelectFrom) -> Result<(Vec<String>, Vec<Row>)> {
        match from {
            SelectFrom::Table { name } => self.scan(name, None),
            SelectFrom::Join {
                left,
                right,
                join_type,
            } => {
                let (mut left_columns, left_rows) = self.scan_all_from_join(left)?;
                let (mut right_columns, right_rows) = self.scan_all_from_join(right)?;

                // 合并左右表
                match join_type {
                    JoinType::Full => {
                        // 对列名添加表名前缀，以便后续处理时能够识别
                        if let SelectFrom::Table { ref name } = **left {
                            left_columns.iter_mut().for_each(|col| {
                                *col = format!("{}.{}", name, col);
                            });
                        }
                        if let SelectFrom::Table { ref name } = **right {
                            right_columns.iter_mut().for_each(|col| {
                                *col = format!("{}.{}", name, col);
                            });
                        }
                        let new_columns = [left_columns, right_columns].concat();
                        let new_rows: Vec<Row> = left_rows
                            .into_iter()
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
                    _ => unimplemented!("Only support FULL JOIN"),
                }
            }
        }
    }

    /// 从 Join 表中扫描数据并过滤
    fn scan_from_join(
        &self,
        from: &SelectFrom,
        filter: Option<(String, Expression)>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        let (columns, mut rows) = self.scan_all_from_join(from)?;

        // 列名称在 `scan_all_from_join` 中改为 table_name.col_name，利用这个特性进行过滤
        if let Some((col_name, expr)) = filter {
            let col_idx = Self::get_column_index_by_name(&columns, &col_name)?;
            rows.retain(|row| row[col_idx] == Value::from(expr.clone()));
        }

        Ok((columns, rows))
    }

    /// 查询数据
    fn select(
        &self,
        select_columns: Vec<(String, Option<String>)>,
        from: SelectFrom,
        filter: Option<(String, Expression)>,
        ordering: Vec<(String, Ordering)>,
        limit: Option<Expression>,
        offset: Option<Expression>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        let (mut columns, mut rows) = self.scan_from_join(&from, filter)?;
        self.sort_rows(&mut rows, &columns, ordering)?;

        // 处理 limit 和 offset
        if !(offset.is_none() && limit.is_none()) {
            let to_usize = |expr: Option<Expression>, default: usize, err_prefix: &str| {
                expr.map_or(Ok(default), |e| match Value::from(e) {
                    Value::Integer(v) if v >= 0 => Ok(v as usize),
                    other => Err(InternalError(format!(
                        "{} must be a non-negative integer, get {:?}",
                        err_prefix, other
                    ))),
                })
            };
            let offset = to_usize(offset, 0, "Offset")?;
            let limit = to_usize(limit, usize::MAX, "Limit")?;
            rows = rows
                .into_iter()
                .skip(offset)
                .take(limit)
                .collect::<Vec<_>>();
        }

        // 处理不是 SELECT * 的情况
        if !select_columns.is_empty() {
            // 一次性收集新列名和对应原索引
            let select_info = select_columns
                .iter()
                .map(|(col_name, alias)| {
                    let idx = Self::get_column_index_by_name(&columns, col_name)?;
                    Ok((
                        alias.clone().unwrap_or_else(|| {
                            if col_name.contains('.') {
                                // 如果列名是 table_name.col_name 的形式，则只取 col_name
                                col_name.split('.').last().unwrap().to_string()
                            } else {
                                // 否则直接使用
                                col_name.clone()
                            }
                        }),
                        idx,
                    ))
                })
                .collect::<Result<Vec<(String, usize)>>>()?;

            // 更新列名
            let new_columns = select_info.iter().map(|(alias, _)| alias.clone()).collect();

            // 根据新选择的列，调整每一行
            rows = rows
                .into_iter()
                .map(|row| {
                    select_info
                        .iter()
                        .map(|&(_, idx)| row[idx].clone())
                        .collect::<Vec<Value>>()
                })
                .collect();
            columns = new_columns;
        } else {
            // 将列名从 table_name.col_name 改为 col_name
            columns.iter_mut().for_each(|col| {
                *col = col.split('.').last().unwrap().to_string();
            });
        }

        Ok((columns, rows))
    }

    /// 插入数据
    fn insert(
        &self,
        table_name: String,
        column_names: Vec<String>,
        values: Vec<Vec<Expression>>,
    ) -> Result<()> {
        let table_columns = &self
            .transaction
            .get_table(&table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?
            .columns;

        // columns 为空时，表示插入所有列
        let column_names = if column_names.is_empty() {
            table_columns.iter().map(|c| c.name.clone()).collect()
        } else {
            column_names
        };

        for value in values {
            // 检查列数是否匹配
            if column_names.len() != value.len() {
                return Err(InternalError(format!(
                    "Column count {} doesn't match value count {}",
                    column_names.len(),
                    value.len()
                )));
            }

            // 创建一个 HashMap，方便后续根据列名查找对应的值
            let value_map: HashMap<String, Expression> = column_names
                .iter()
                .cloned()
                .zip(value.into_iter())
                .collect();

            let row = table_columns
                .iter()
                .map(|column| {
                    if let Some(exp) = value_map.get(&column.name) {
                        // 如果找到对应的值，将其转为 Value
                        Ok(Value::from(exp.clone()))
                    } else if let Some(default) = &column.default {
                        // 如果未找到对应的值，但存在默认值，使用默认值
                        Ok(default.clone())
                    } else {
                        // 如果未找到对应的值，且不存在默认值，返回错误
                        Err(InternalError(format!(
                            "Column {} not found in value",
                            column.name
                        )))
                    }
                })
                .collect::<Result<Vec<Value>>>()?;

            // 将数据插入表中
            self.transaction.create_row(&table_name, &row)?;
        }

        Ok(())
    }

    /// 更新数据
    fn update(
        &self,
        table_name: String,
        columns: HashMap<String, Expression>,
        filter: Option<(String, Expression)>,
    ) -> Result<usize> {
        let table = self
            .transaction
            .get_table(&table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;
        let (_, rows) = self.scan(&table_name, filter)?;

        let mut updated_count = 0;
        for row in rows {
            let mut updated_row = row.clone();
            let primary_key = table.get_primary_key(&row);

            for (col_name, expr) in &columns {
                let col_idx = table.get_col_idx(col_name).ok_or(InternalError(format!(
                    "Column {} not found in table {}",
                    col_name, table_name
                )))?;
                updated_row[col_idx] = Value::from(expr.clone());
            }
            self.transaction
                .update_row(&table, primary_key, &updated_row)?;
            updated_count += 1;
        }

        Ok(updated_count)
    }

    /// 删除数据
    fn delete(&self, table_name: String, filter: Option<(String, Expression)>) -> Result<usize> {
        let table = self
            .transaction
            .get_table(&table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;
        let (_, rows) = self.scan(&table_name, filter)?;

        let mut delete_count = 0;
        for row in rows {
            let primary_key = table.get_primary_key(&row);
            self.transaction.delete_row(&table, primary_key)?;
            delete_count += 1;
        }

        Ok(delete_count)
    }

    /// 根据列名查找列索引
    ///
    /// columns 为 table_name.col_name 的形式，col_name 可能未 col_name 或 table_name.col_name
    fn get_column_index_by_name(columns: &[String], col_name: &str) -> Result<usize> {
        // col_name 如果是 table_name.col_name 的形式，则直接查找
        if col_name.contains('.') {
            Ok(columns
                .iter()
                .position(|col| col == col_name)
                .ok_or(InternalError(format!(
                    "Column {} not found in table",
                    col_name
                )))?)
        } else {
            // 如果是 col_name，则需要过滤所有符合要求的列名，并且如果存在多个则报错
            let col_indices = columns
                .iter()
                .enumerate()
                .filter(|(_, col)| col.split('.').last().unwrap() == col_name)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            if col_indices.is_empty() {
                return Err(InternalError(format!(
                    "Column {} not found in table",
                    col_name
                )));
            } else if col_indices.len() > 1 {
                return Err(InternalError(format!(
                    "Column {} is ambiguous in table",
                    col_name
                )));
            }
            Ok(col_indices[0])
        }
    }

    /// 对行进行排序
    fn sort_rows(
        &self,
        rows: &mut [Row],
        columns: &[String],
        ordering: Vec<(String, Ordering)>,
    ) -> Result<()> {
        // columns 改为了 table_name.col_name 的形式，这里需要处理
        let ordering = ordering
            .into_iter()
            .map(|(col_name, ord)| {
                Self::get_column_index_by_name(columns, &col_name).map(|col_idx| (col_idx, ord))
            })
            .collect::<Result<Vec<_>>>()?;

        rows.sort_by(|lhs, rhs| {
            for (col_idx, order) in ordering.iter() {
                match lhs[*col_idx].partial_cmp(&rhs[*col_idx]) {
                    Some(ord) if ord != std::cmp::Ordering::Equal => {
                        return if *order == Ordering::Asc {
                            ord
                        } else {
                            ord.reverse()
                        };
                    }
                    _ => continue,
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::Result,
        parser::{ast::Constant, Parser},
        schema::{Column, DataType},
        storage::MemoryStorage,
    };

    #[test]
    fn test_executor() -> Result<()> {
        let storage = MemoryStorage::new();
        let engine = Engine::new(storage);
        let executor = Executor::from_engine(&engine)?;

        let stmt = Statement::CreateTable {
            name: "users".to_string(),
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
                    nullable: true,
                    default: Some(Value::String("Momo".to_string())),
                    primary_key: false,
                },
            ],
        };
        executor.execute(stmt)?;

        let stmt = Statement::Insert {
            table_name: "users".to_string(),
            columns: None,
            values: vec![
                vec![
                    Expression::Constant(Constant::Integer(1)),
                    Expression::Constant(Constant::String("Alice".to_string())),
                ],
                vec![
                    Expression::Constant(Constant::Integer(2)),
                    Expression::Constant(Constant::Null),
                ],
            ],
        };
        executor.execute(stmt)?;

        let stmt = Statement::Insert {
            table_name: "users".to_string(),
            columns: Some(vec!["id".to_string()]),
            values: vec![vec![Expression::Constant(Constant::Integer(3))]],
        };
        executor.execute(stmt)?;

        let stmt = Statement::Insert {
            table_name: "users".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            values: vec![vec![Expression::Constant(Constant::Integer(4))]],
        };
        assert!(executor.execute(stmt).is_err());

        let stmt = Statement::Insert {
            table_name: "users".to_string(),
            columns: Some(vec!["name".to_string()]),
            values: vec![vec![Expression::Constant(Constant::String(
                "Bob".to_string(),
            ))]],
        };
        assert!(executor.execute(stmt).is_err());

        let stmt = Statement::Select {
            columns: vec![
                ("id".to_string(), Some("user_id".to_string())),
                ("name".to_string(), None),
            ],
            from: SelectFrom::Table {
                name: "users".to_string(),
            },
            filter: None,
            ordering: vec![("id".to_string(), Ordering::Desc)],
            limit: None,
            offset: None,
        };
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["user_id", "name"]);
            assert_eq!(
                rows,
                vec![
                    vec![Value::Integer(3), Value::String("Momo".to_string())],
                    vec![Value::Integer(2), Value::Null],
                    vec![Value::Integer(1), Value::String("Alice".to_string())],
                ]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        let stmt = Statement::Update {
            table_name: "users".to_string(),
            columns: vec![(
                "name".to_string(),
                Expression::Constant(Constant::String("Bob".to_string())),
            )]
            .into_iter()
            .collect(),
            filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
        };
        let result = executor.execute(stmt)?;
        if let ExecuteResult::Update(count) = result {
            assert_eq!(count, 1);
        } else {
            panic!("Expect ExecuteResult::Update");
        }

        let stmt = Statement::Select {
            columns: vec!["name".to_string()]
                .into_iter()
                .map(|col| (col, None))
                .collect(),
            from: SelectFrom::Table {
                name: "users".to_string(),
            },
            filter: None,
            ordering: vec![("name".to_string(), Ordering::Asc)],
            limit: Some(Expression::Constant(Constant::Integer(2))),
            offset: Some(Expression::Constant(Constant::Integer(1))),
        };
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["name"]);
            assert_eq!(
                rows,
                vec![
                    vec![Value::String("Bob".to_string())],
                    vec![Value::String("Momo".to_string())],
                ]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        let stmt = Statement::Delete {
            table_name: "users".to_string(),
            filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(2)))),
        };
        let result = executor.execute(stmt)?;
        if let ExecuteResult::Delete(count) = result {
            assert_eq!(count, 1);
        } else {
            panic!("Expect ExecuteResult::Delete");
        }

        let stmt = Statement::Select {
            columns: vec![],
            from: SelectFrom::Table {
                name: "users".to_string(),
            },
            filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(3)))),
            ordering: vec![],
            limit: None,
            offset: None,
        };
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(
                rows,
                vec![vec![Value::Integer(3), Value::String("Momo".to_string())],]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        let stmt = Statement::CreateTable {
            name: "grades".to_string(),
            columns: vec![
                Column {
                    name: "name".to_string(),
                    data_type: DataType::String,
                    nullable: false,
                    default: None,
                    primary_key: true,
                },
                Column {
                    name: "grade".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    default: Some(Value::Integer(0)),
                    primary_key: false,
                },
            ],
        };
        executor.execute(stmt)?;

        let stmt = Statement::Insert {
            table_name: "grades".to_string(),
            columns: None,
            values: vec![
                vec![
                    Expression::Constant(Constant::String("Alice".to_string())),
                    Expression::Constant(Constant::Integer(100)),
                ],
                vec![
                    Expression::Constant(Constant::String("Bob".to_string())),
                    Expression::Constant(Constant::Integer(90)),
                ],
            ],
        };
        executor.execute(stmt)?;

        let stmt = Statement::Select {
            columns: vec![
                ("users.name".to_string(), None),
                ("grade".to_string(), None),
            ],
            from: SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Full,
            },
            filter: Some((
                "users.id".to_string(),
                Expression::Constant(Constant::Integer(1)),
            )),
            ordering: vec![("grade".to_string(), Ordering::Desc)],
            limit: None,
            offset: None,
        };
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["name", "grade"]);
            assert_eq!(
                rows,
                vec![
                    vec![Value::String("Bob".to_string()), Value::Integer(100)],
                    vec![Value::String("Bob".to_string()), Value::Integer(90)],
                ]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        Ok(())
    }

    fn parse_sql(sql: &str) -> Result<Statement> {
        let mut parser = Parser::new(sql);
        parser.parse()
    }

    #[test]
    fn test_sql() -> Result<()> {
        let storage = MemoryStorage::new();
        let engine = Engine::new(storage);
        let executor = Executor::from_engine(&engine)?;

        let sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NULL DEFAULT 'Momo');";
        let stmt = parse_sql(sql)?;
        executor.execute(stmt)?;

        let sql = "INSERT INTO users VALUES (1, 'Alice'), (2, NULL);";
        let stmt = parse_sql(sql)?;
        executor.execute(stmt)?;

        let sql = "INSERT INTO users (id) VALUES (3);";
        let stmt = parse_sql(sql)?;
        executor.execute(stmt)?;

        let sql = "SELECT * FROM users;";
        let stmt = parse_sql(sql)?;
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(
                rows,
                vec![
                    vec![Value::Integer(1), Value::String("Alice".to_string())],
                    vec![Value::Integer(2), Value::Null],
                    vec![Value::Integer(3), Value::String("Momo".to_string())],
                ]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        let sql = "UPDATE users SET name = 'Bob' WHERE id = 1;";
        let stmt = parse_sql(sql)?;
        let result = executor.execute(stmt)?;
        if let ExecuteResult::Update(count) = result {
            assert_eq!(count, 1);
        } else {
            panic!("Expect ExecuteResult::Update");
        }

        let sql = "SELECT * FROM users ORDER BY name DESC LIMIT 2 OFFSET 1;";
        let stmt = parse_sql(sql)?;
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(
                rows,
                vec![
                    vec![Value::Integer(1), Value::String("Bob".to_string())],
                    vec![Value::Integer(2), Value::Null],
                ]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        let sql = "DELETE FROM users WHERE id = 2;";
        let stmt = parse_sql(sql)?;
        let result = executor.execute(stmt)?;
        if let ExecuteResult::Delete(count) = result {
            assert_eq!(count, 1);
        } else {
            panic!("Expect ExecuteResult::Delete");
        }

        let sql = "SELECT * FROM users WHERE id = 3;";
        let stmt = parse_sql(sql)?;
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(
                rows,
                vec![vec![Value::Integer(3), Value::String("Momo".to_string())],]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        let sql = "CREATE TABLE grades (name TEXT PRIMARY KEY, grade INTEGER NULL DEFAULT 0);";
        let stmt = parse_sql(sql)?;
        executor.execute(stmt)?;

        let sql = "INSERT INTO grades VALUES ('Alice', 100), ('Bob', 90);";
        let stmt = parse_sql(sql)?;
        executor.execute(stmt)?;

        let sql = "SELECT * FROM users CROSS JOIN grades ORDER BY grade DESC;";
        let stmt = parse_sql(sql)?;
        if let ExecuteResult::Scan { columns, rows } = executor.execute(stmt)? {
            assert_eq!(columns, vec!["id", "name", "name", "grade"]);
            assert_eq!(
                rows,
                vec![
                    vec![
                        Value::Integer(1),
                        Value::String("Bob".to_string()),
                        Value::String("Alice".to_string()),
                        Value::Integer(100)
                    ],
                    vec![
                        Value::Integer(3),
                        Value::String("Momo".to_string()),
                        Value::String("Alice".to_string()),
                        Value::Integer(100)
                    ],
                    vec![
                        Value::Integer(1),
                        Value::String("Bob".to_string()),
                        Value::String("Bob".to_string()),
                        Value::Integer(90)
                    ],
                    vec![
                        Value::Integer(3),
                        Value::String("Momo".to_string()),
                        Value::String("Bob".to_string()),
                        Value::Integer(90)
                    ],
                ]
            );
        } else {
            panic!("Expect ExecuteResult::Scan");
        }

        Ok(())
    }
}
