use std::collections::HashMap;

use crate::{
    engine::{Engine, Transaction},
    error::{Error::InternalError, Result},
    parser::ast::{Expression, Statement},
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
                let table = Table { name, columns };
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
            Statement::Select { table_name } => {
                let columns = self
                    .transaction
                    .get_table_info(&table_name)?
                    .ok_or(InternalError(format!("Table {table_name} not found")))?
                    .columns
                    .iter()
                    .map(|c| c.name.clone())
                    .collect();
                let rows = self.transaction.scan_table(&table_name)?;
                Ok(ExecuteResult::Scan { columns, rows })
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

    /// 插入数据
    fn insert(
        &self,
        table_name: String,
        column_names: Vec<String>,
        values: Vec<Vec<Expression>>,
    ) -> Result<()> {
        let table_columns = &self
            .transaction
            .get_table_info(&table_name)?
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::Result,
        parser::ast::Constant,
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
                },
                Column {
                    name: "name".to_string(),
                    data_type: DataType::String,
                    nullable: true,
                    default: Some(Value::String("Momo".to_string())),
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
            table_name: "users".to_string(),
        };
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

        Ok(())
    }
}
