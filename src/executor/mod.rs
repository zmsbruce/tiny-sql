use std::collections::HashMap;

use aggregate::aggregate;
use join::{hash_join, loop_join};

use crate::{
    engine::{Engine, Transaction},
    error::{Error::InternalError, Result},
    parser::ast::{Expression, JoinType, Operation, Ordering, SelectFrom, Statement},
    schema::{Row, Table, Value},
    storage::Storage,
};

mod aggregate;
mod join;

/// SQL 执行结果
#[derive(Debug, PartialEq)]
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
                groupby,
                ordering,
                limit,
                offset,
            } => {
                let (columns, rows) =
                    self.select(columns, from, filter, groupby, ordering, limit, offset)?;

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

    /// 扫描 Join 表，返回所有的列名和行数据
    fn scan_all_from_join(&self, from: &SelectFrom) -> Result<(Vec<String>, Vec<Row>)> {
        match from {
            SelectFrom::Table { name } => self.scan(name, None),
            SelectFrom::Join {
                left,
                right,
                join_type,
                predicate,
            } => {
                // 除了 Cross Join 外，其他 Join 类型必须有 Join 条件
                if join_type != &JoinType::Cross && predicate.is_none() {
                    return Err(InternalError(format!(
                        "{} must have a predicate",
                        join_type
                    )));
                }

                let (mut left_columns, left_rows) = self.scan_all_from_join(left)?;
                let (mut right_columns, right_rows) = self.scan_all_from_join(right)?;

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

                // 合并左右表
                match join_type {
                    JoinType::Cross => {
                        loop_join(&left_columns, &right_columns, &left_rows, &right_rows)
                    }
                    JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                        hash_join(
                            &left_columns,
                            &right_columns,
                            &left_rows,
                            &right_rows,
                            join_type,
                            predicate.as_ref().unwrap(),
                        )
                    }
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

    /// 从 `table_name.column_name` 中提取 `column_name`
    fn extract_column_name(full_column_name: &str) -> &str {
        full_column_name
            .split('.')
            .last()
            .unwrap_or(full_column_name)
    }

    /// 查询数据
    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        select_columns: Vec<(Expression, Option<String>)>,
        from: SelectFrom,
        filter: Option<(String, Expression)>,
        groupby: Option<(Vec<Expression>, Option<Expression>)>,
        ordering: Vec<(String, Ordering)>,
        limit: Option<Expression>,
        offset: Option<Expression>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        let (columns, mut rows) = self.scan_from_join(&from, filter)?;
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

        // 如果是 SELECT *，直接返回
        if select_columns.is_empty() {
            // 如果有 GROUP BY，必须有聚集函数
            if groupby.is_some() {
                return Err(InternalError(
                    "GROUP BY must have aggregate function".to_string(),
                ));
            }
            // 将列名从 table_name.col_name 改为 col_name
            let columns = columns
                .into_iter()
                .map(|full_name| Self::extract_column_name(&full_name).to_string())
                .collect();

            return Ok((columns, rows));
        }

        // 如果没有聚集函数和 GROUP BY，直接选择列名
        if select_columns.iter().all(|(col, _)| col.is_field()) {
            if groupby.is_some() {
                return Err(InternalError(
                    "GROUP BY must have aggregate function".to_string(),
                ));
            }

            let (new_columns, new_rows) =
                self.select_field_columns(&select_columns, &columns, rows)?;
            return Ok((new_columns, new_rows));
        }

        // 如果有聚集函数
        let (new_columns, new_rows) =
            Self::select_aggregate_columns(&select_columns, &columns, rows, groupby)?;
        Ok((new_columns, new_rows))
    }

    /// 选择列名
    fn select_field_columns(
        &self,
        select_columns: &[(Expression, Option<String>)],
        columns: &[String],
        rows: Vec<Row>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // 一次性收集新列名
        let new_columns = select_columns
            .iter()
            .map(|(col_expr, alias)| match col_expr {
                Expression::Field(col_name) => alias
                    .clone()
                    .unwrap_or_else(|| Self::extract_column_name(col_name).to_string()),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();

        // 收集需要选择的列索引
        let col_indices = select_columns
            .iter()
            .map(|(col_expr, _)| match col_expr {
                Expression::Field(col_name) => Self::get_column_index_by_name(columns, col_name),
                _ => unreachable!(),
            })
            .collect::<Result<Vec<_>>>()?;

        // 选择需要的列
        let rows = rows
            .into_iter()
            .map(|row| {
                col_indices
                    .iter()
                    .map(|col_idx| row[*col_idx].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok((new_columns, rows))
    }

    /// 选择聚集函数的列
    fn select_aggregate_columns(
        select_columns: &[(Expression, Option<String>)],
        columns: &[String],
        rows: Vec<Row>,
        groupby: Option<(Vec<Expression>, Option<Expression>)>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // 解析 GROUP BY，可以是 None，也可以是一个或多个字段
        let group_names = groupby
            .as_ref()
            .map(|(group, _)| {
                group
                    .iter()
                    .map(|expr| match expr {
                        Expression::Field(col_name) => Ok(col_name),
                        _ => Err(InternalError(format!(
                            "Unsupported column {:?} in GROUP BY",
                            expr
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        // 一次性收集新列名
        let new_columns = select_columns
            .iter()
            .map(|(col, alias)| match col {
                Expression::Function(agg, col_name) => Ok(alias
                    .clone()
                    .unwrap_or_else(|| format!("{}({})", agg, col_name))),
                Expression::Field(col_name) => {
                    // 列名必须在 GROUP BY 中
                    if group_names.is_none() || !group_names.as_ref().unwrap().contains(&col_name) {
                        return Err(InternalError(format!(
                            "Column {} must be in GROUP BY",
                            col_name,
                        )));
                    }
                    Ok(alias.clone().unwrap_or_else(|| col_name.clone()))
                }
                others => Err(InternalError(format!(
                    "Unsupported column {:?} in SELECT",
                    others
                ))),
            })
            .collect::<Result<Vec<String>>>()?;

        // 如果没有 GROUP BY，直接计算聚集函数的值
        if group_names.is_none() {
            let agg_values = select_columns
                .iter()
                .map(|(col, _)| match col {
                    Expression::Function(agg, col_name) => {
                        aggregate(col_name, columns, &rows, *agg)
                    }
                    _ => unreachable!(),
                })
                .collect::<Result<Vec<Value>>>()?;

            return Ok((new_columns, vec![agg_values]));
        }

        // 如果有 GROUP BY，按照 GROUP BY 列进行分组
        let group_indices = group_names
            .as_ref()
            .unwrap()
            .iter()
            .map(|group_name| Self::get_column_index_by_name(columns, group_name))
            .collect::<Result<Vec<_>>>()?;
        let mut group_map: HashMap<Vec<Value>, Vec<Row>> = HashMap::new();
        for row in rows {
            let group_key = group_indices
                .iter()
                .map(|&group_idx| row[group_idx].clone())
                .collect::<Vec<_>>();
            group_map.entry(group_key).or_default().push(row);
        }

        // 计算聚集函数的值
        let group_index_map = group_names
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, group_name)| (group_name, i))
            .collect::<HashMap<_, _>>();
        let mut new_rows = group_map
            .into_iter()
            .map(|(group_keys, group_rows)| {
                let agg_values = select_columns
                    .iter()
                    .map(|(col, _)| match col {
                        Expression::Function(agg, col_name) => {
                            aggregate(col_name, columns, &group_rows, *agg)
                        }
                        // 列一定等于 GROUP BY 列，因为在上面的 match 中已经处理
                        Expression::Field(col_name) => {
                            Ok(group_keys[group_index_map[&col_name]].clone())
                        }
                        // 不可能出现其他情况，在上面的 match 中已经处理
                        _ => unreachable!(),
                    })
                    .collect::<Result<Vec<Value>>>()?;
                Ok(agg_values)
            })
            .collect::<Result<Vec<Row>>>()?;

        // 过滤不符合 HAVING 条件的行
        if let Some(Expression::Operation(op)) = &groupby.as_ref().unwrap().1 {
            match op {
                // 目前只支持 EQUAL 操作
                Operation::Equal(col, val) => {
                    let col_idx = select_columns.iter().position(|(c, _)| *c == **col).ok_or(
                        InternalError(format!("Column {:?} not found in SELECT", col)),
                    )?;
                    new_rows.retain(|row| row[col_idx] == Value::from(*val.clone()));
                }
            }
        }

        Ok((new_columns, new_rows))
    }

    /// 根据列名查找列索引
    ///
    /// columns 为 table_name.col_name 的形式，col_name 可能为 col_name 或 table_name.col_name
    fn get_column_index_by_name(columns: &[String], col_name: &str) -> Result<usize> {
        let parts = col_name.split('.').collect::<Vec<_>>();
        match parts.len() {
            1 => {
                // 仅包含 col_name，则按照最后部分匹配
                let matches = columns
                    .iter()
                    .enumerate()
                    .filter(|(_, full_name)| full_name.split('.').last().unwrap() == parts[0])
                    .collect::<Vec<_>>();
                if matches.len() == 1 {
                    Ok(matches[0].0)
                } else if matches.is_empty() {
                    Err(InternalError(format!(
                        "Column {} not found in table",
                        col_name
                    )))
                } else {
                    Err(InternalError(format!(
                        "Column {} is ambiguous in table",
                        col_name
                    )))
                }
            }
            2 => {
                // 包含 table_name.col_name，则直接查找
                columns
                    .iter()
                    .position(|full_name| full_name == col_name)
                    .ok_or(InternalError(format!(
                        "Column {} not found in table",
                        col_name
                    )))
            }
            _ => unreachable!("More than 2 dots in column name"), // 不可能出现其他情况
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
        parser::ast::{Aggregate, Constant, Operation},
        schema::{Column, DataType},
        storage::MemoryStorage,
    };

    fn init_executor() -> Result<Executor<MemoryStorage>> {
        let storage = MemoryStorage::new();
        let engine = Engine::new(storage);
        Executor::from_engine(&engine)
    }

    fn create_tables(executor: &Executor<MemoryStorage>) -> Result<()> {
        // 创建 users 表
        executor.execute(Statement::CreateTable {
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
        })?;

        // 创建 grades 表
        executor.execute(Statement::CreateTable {
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
        })?;

        Ok(())
    }

    fn insert_data(executor: &Executor<MemoryStorage>) -> Result<()> {
        // 插入数据到 users 表
        executor.execute(Statement::Insert {
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
        })?;

        // 插入数据到 grades 表
        executor.execute(Statement::Insert {
            table_name: "grades".to_string(),
            columns: None,
            values: vec![
                vec![
                    Expression::Constant(Constant::String("Alice".to_string())),
                    Expression::Constant(Constant::Integer(90)),
                ],
                vec![
                    Expression::Constant(Constant::String("Bob".to_string())),
                    Expression::Constant(Constant::Integer(80)),
                ],
                vec![
                    Expression::Constant(Constant::String("Charlie".to_string())),
                    Expression::Constant(Constant::Integer(80)),
                ],
            ],
        })?;

        Ok(())
    }

    #[test]
    fn test_insert() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;

        // 测试合法插入
        insert_data(&executor)?;

        // 测试错误情形：列数不匹配（缺少 name 列）
        assert!(executor
            .execute(Statement::Insert {
                table_name: "users".to_string(),
                columns: Some(vec!["id".to_string(), "name".to_string()]),
                values: vec![vec![Expression::Constant(Constant::Integer(4))]],
            })
            .is_err());

        // 测试错误情形：只插入 name 列（不允许缺失主键 id）
        assert!(executor
            .execute(Statement::Insert {
                table_name: "users".to_string(),
                columns: Some(vec!["name".to_string()]),
                values: vec![vec![Expression::Constant(Constant::String(
                    "Bob".to_string()
                ))]],
            })
            .is_err());

        // 测试错误情形：插入重复主键
        assert!(executor
            .execute(Statement::Insert {
                table_name: "users".to_string(),
                columns: None,
                values: vec![vec![
                    Expression::Constant(Constant::Integer(1)),
                    Expression::Constant(Constant::String("Bob".to_string())),
                ]],
            })
            .is_err());

        // 测试错误情形：插入不存在的表
        assert!(executor
            .execute(Statement::Insert {
                table_name: "nonexistent".to_string(),
                columns: None,
                values: vec![vec![
                    Expression::Constant(Constant::Integer(1)),
                    Expression::Constant(Constant::String("Bob".to_string())),
                ]],
            })
            .is_err());

        // 测试错误情形：插入数据类型不匹配
        assert!(executor
            .execute(Statement::Insert {
                table_name: "users".to_string(),
                columns: None,
                values: vec![vec![
                    Expression::Constant(Constant::String("Alice".to_string())),
                    Expression::Constant(Constant::String("Bob".to_string())),
                ]],
            })
            .is_err());

        Ok(())
    }

    #[test]
    fn test_select() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 SELECT * FROM users
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(1), Value::String("Alice".to_string())],
                vec![Value::Integer(2), Value::Null],
            ]
        );

        // 测试 SELECT name FROM users
        let (columns, rows) = executor.select(
            vec![(Expression::Field("name".to_string()), None)],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["name"]);
        assert_eq!(
            rows,
            vec![vec![Value::String("Alice".to_string())], vec![Value::Null],]
        );

        // 测试 SELECT * FROM users WHERE id = 1
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(
            rows,
            vec![vec![Value::Integer(1), Value::String("Alice".to_string())]]
        );

        // 测试 SELECT * FROM users WHERE name IS NULL
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(("name".to_string(), Expression::Constant(Constant::Null))),
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(rows, vec![vec![Value::Integer(2), Value::Null]]);

        // 测试 SELECT * FROM users ORDER BY name DESC
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![("name".to_string(), Ordering::Desc)],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(1), Value::String("Alice".to_string())],
                vec![Value::Integer(2), Value::Null],
            ]
        );

        // 测试 SELECT * FROM users ORDER BY name ASC
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![("name".to_string(), Ordering::Asc)],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(2), Value::Null],
                vec![Value::Integer(1), Value::String("Alice".to_string())],
            ]
        );

        // 测试 SELECT * FROM users LIMIT 1
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            Some(Expression::Constant(Constant::Integer(1))),
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(
            rows,
            vec![vec![Value::Integer(1), Value::String("Alice".to_string())]]
        );

        // 测试 SELECT * FROM users LIMIT 1 OFFSET 1
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            Some(Expression::Constant(Constant::Integer(1))),
            Some(Expression::Constant(Constant::Integer(1))),
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(rows, vec![vec![Value::Integer(2), Value::Null]]);

        Ok(())
    }

    #[test]
    fn test_update() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试更新数据
        let result = executor.execute(Statement::Update {
            table_name: "users".to_string(),
            columns: vec![(
                "name".to_string(),
                Expression::Constant(Constant::String("Bob".to_string())),
            )]
            .into_iter()
            .collect(),
            filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
        })?;
        assert_eq!(result, ExecuteResult::Update(1));

        // 测试更新数据后的查询
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert_eq!(
            rows,
            vec![vec![Value::Integer(1), Value::String("Bob".to_string())]]
        );

        Ok(())
    }

    #[test]
    fn test_delete() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试删除数据
        let result = executor.execute(Statement::Delete {
            table_name: "users".to_string(),
            filter: Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
        })?;
        assert_eq!(result, ExecuteResult::Delete(1));

        // 测试删除数据后的查询
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(("id".to_string(), Expression::Constant(Constant::Integer(1)))),
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name"]);
        assert!(rows.is_empty());

        Ok(())
    }

    #[test]
    fn test_cross_join() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 CROSS JOIN
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Cross,
                predicate: None,
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name", "name", "grade"]);
        assert_eq!(rows.len(), 6);
        assert!(rows.contains(&vec![
            Value::Integer(1),
            Value::String("Alice".to_string()),
            Value::String("Alice".to_string()),
            Value::Integer(90)
        ]));
        assert!(rows.contains(&vec![
            Value::Integer(1),
            Value::String("Alice".to_string()),
            Value::String("Bob".to_string()),
            Value::Integer(80)
        ]));
        assert!(rows.contains(&vec![
            Value::Integer(2),
            Value::Null,
            Value::String("Bob".to_string()),
            Value::Integer(80)
        ]));
        assert!(rows.contains(&vec![
            Value::Integer(2),
            Value::Null,
            Value::String("Alice".to_string()),
            Value::Integer(90)
        ]));
        assert!(rows.contains(&vec![
            Value::Integer(1),
            Value::String("Alice".to_string()),
            Value::String("Charlie".to_string()),
            Value::Integer(80)
        ]));
        assert!(rows.contains(&vec![
            Value::Integer(2),
            Value::Null,
            Value::String("Charlie".to_string()),
            Value::Integer(80)
        ]));

        Ok(())
    }

    #[test]
    fn test_cross_join_with_filter_ordering() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 CROSS JOIN 对有歧义的列名进行过滤
        assert!(executor
            .select(
                vec![],
                SelectFrom::Join {
                    left: Box::new(SelectFrom::Table {
                        name: "users".to_string(),
                    }),
                    right: Box::new(SelectFrom::Table {
                        name: "grades".to_string(),
                    }),
                    join_type: JoinType::Cross,
                    predicate: None,
                },
                Some((
                    "name".to_string(),
                    Expression::Constant(Constant::String("Alice".to_string()))
                )),
                None,
                vec![],
                None,
                None,
            )
            .is_err());

        // 测试 CROSS JOIN 对有歧义的列名进行排序
        assert!(executor
            .select(
                vec![],
                SelectFrom::Join {
                    left: Box::new(SelectFrom::Table {
                        name: "users".to_string(),
                    }),
                    right: Box::new(SelectFrom::Table {
                        name: "grades".to_string(),
                    }),
                    join_type: JoinType::Cross,
                    predicate: None,
                },
                None,
                None,
                vec![("name".to_string(), Ordering::Asc)],
                None,
                None,
            )
            .is_err());

        // 测试 CROSS JOIN 对有指定表名的列名进行过滤和排序
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Cross,
                predicate: None,
            },
            Some((
                "users.name".to_string(),
                Expression::Constant(Constant::String("Alice".to_string())),
            )),
            None,
            vec![(String::from("grades.name"), Ordering::Asc)],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name", "name", "grade"]);
        assert_eq!(
            rows,
            vec![
                vec![
                    Value::Integer(1),
                    Value::String("Alice".to_string()),
                    Value::String("Alice".to_string()),
                    Value::Integer(90)
                ],
                vec![
                    Value::Integer(1),
                    Value::String("Alice".to_string()),
                    Value::String("Bob".to_string()),
                    Value::Integer(80)
                ],
                vec![
                    Value::Integer(1),
                    Value::String("Alice".to_string()),
                    Value::String("Charlie".to_string()),
                    Value::Integer(80)
                ],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_inner_join() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 INNER JOIN
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Inner,
                predicate: Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("users.name".to_string())),
                    Box::new(Expression::Field("grades.name".to_string())),
                ))),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name", "name", "grade"]);
        assert_eq!(
            rows,
            vec![vec![
                Value::Integer(1),
                Value::String("Alice".to_string()),
                Value::String("Alice".to_string()),
                Value::Integer(90)
            ]]
        );

        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 LEFT JOIN
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Left,
                predicate: Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("users.name".to_string())),
                    Box::new(Expression::Field("grades.name".to_string())),
                ))),
            },
            None,
            None,
            vec![("grades.name".to_string(), Ordering::Asc)],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name", "name", "grade"]);
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(2), Value::Null, Value::Null, Value::Null,],
                vec![
                    Value::Integer(1),
                    Value::String("Alice".to_string()),
                    Value::String("Alice".to_string()),
                    Value::Integer(90)
                ],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_right_join() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 RIGHT JOIN
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Right,
                predicate: Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("users.name".to_string())),
                    Box::new(Expression::Field("grades.name".to_string())),
                ))),
            },
            None,
            None,
            vec![("grades.name".to_string(), Ordering::Asc)],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name", "name", "grade"]);
        assert_eq!(
            rows,
            vec![
                vec![
                    Value::Integer(1),
                    Value::String("Alice".to_string()),
                    Value::String("Alice".to_string()),
                    Value::Integer(90)
                ],
                vec![
                    Value::Null,
                    Value::Null,
                    Value::String("Bob".to_string()),
                    Value::Integer(80),
                ],
                vec![
                    Value::Null,
                    Value::Null,
                    Value::String("Charlie".to_string()),
                    Value::Integer(80),
                ],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_full_join() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 FULL JOIN
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Join {
                left: Box::new(SelectFrom::Table {
                    name: "users".to_string(),
                }),
                right: Box::new(SelectFrom::Table {
                    name: "grades".to_string(),
                }),
                join_type: JoinType::Full,
                predicate: Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("users.name".to_string())),
                    Box::new(Expression::Field("grades.name".to_string())),
                ))),
            },
            None,
            None,
            vec![("grades.name".to_string(), Ordering::Asc)],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["id", "name", "name", "grade"]);
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(2), Value::Null, Value::Null, Value::Null],
                vec![
                    Value::Integer(1),
                    Value::String("Alice".to_string()),
                    Value::String("Alice".to_string()),
                    Value::Integer(90)
                ],
                vec![
                    Value::Null,
                    Value::Null,
                    Value::String("Bob".to_string()),
                    Value::Integer(80)
                ],
                vec![
                    Value::Null,
                    Value::Null,
                    Value::String("Charlie".to_string()),
                    Value::Integer(80)
                ],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_aggregate() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 COUNT(*)
        let (columns, rows) = executor.select(
            vec![(
                Expression::Function(Aggregate::Count, "*".to_string()),
                None,
            )],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["COUNT(*)"]);
        assert_eq!(rows, vec![vec![Value::Integer(2)]]);

        // 测试 COUNT(name)
        let (columns, rows) = executor.select(
            vec![(
                Expression::Function(Aggregate::Count, "name".to_string()),
                None,
            )],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["COUNT(name)"]);
        assert_eq!(rows, vec![vec![Value::Integer(1)]]);

        // 测试 COUNT(DISTINCT name)
        let (columns, rows) = executor.select(
            vec![(
                Expression::Function(Aggregate::Count, "name".to_string()),
                Some("count".to_string()),
            )],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["count"]);
        assert_eq!(rows, vec![vec![Value::Integer(1)]]);

        // 测试 SUM(id)
        let (columns, rows) = executor.select(
            vec![(Expression::Function(Aggregate::Sum, "id".to_string()), None)],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["SUM(id)"]);
        assert_eq!(rows, vec![vec![Value::Integer(3)]]);

        // 测试 AVG(id)
        let (columns, rows) = executor.select(
            vec![(Expression::Function(Aggregate::Avg, "id".to_string()), None)],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["AVG(id)"]);
        assert_eq!(rows, vec![vec![Value::Float(1.5)]]);

        // 测试 MAX(id)
        let (columns, rows) = executor.select(
            vec![(Expression::Function(Aggregate::Max, "id".to_string()), None)],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["MAX(id)"]);
        assert_eq!(rows, vec![vec![Value::Integer(2)]]);

        // 测试 MIN(id)
        let (columns, rows) = executor.select(
            vec![(Expression::Function(Aggregate::Min, "id".to_string()), None)],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["MIN(id)"]);
        assert_eq!(rows, vec![vec![Value::Integer(1)]]);

        // 测试 MIN(id), MAX(id)
        let (columns, rows) = executor.select(
            vec![
                (Expression::Function(Aggregate::Min, "id".to_string()), None),
                (Expression::Function(Aggregate::Max, "id".to_string()), None),
            ],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["MIN(id)", "MAX(id)"]);
        assert_eq!(rows, vec![vec![Value::Integer(1), Value::Integer(2)]]);

        // 测试 MIN(id) alias min_id
        // 测试 MIN(id)
        let (columns, rows) = executor.select(
            vec![(
                Expression::Function(Aggregate::Min, "id".to_string()),
                Some("min_id".to_string()),
            )],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            None,
            None,
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["min_id"]);
        assert_eq!(rows, vec![vec![Value::Integer(1)]]);

        Ok(())
    }

    #[test]
    fn test_aggregate_with_group_by() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 SELECT grade, COUNT(*) AS count FROM grades GROUP BY grade
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("grade".to_string()), None),
                (
                    Expression::Function(Aggregate::Count, "*".to_string()),
                    Some("count".to_string()),
                ),
            ],
            SelectFrom::Table {
                name: "grades".to_string(),
            },
            None,
            Some((vec![Expression::Field("grade".to_string())], None)),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["grade", "count"]);
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&vec![Value::Integer(90), Value::Integer(1)]));
        assert!(rows.contains(&vec![Value::Integer(80), Value::Integer(2)]));

        // 测试 SELECT grade, COUNT(*) AS count, MAX(name) FROM grades GROUP BY grade
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("grade".to_string()), None),
                (
                    Expression::Function(Aggregate::Count, "*".to_string()),
                    Some("count".to_string()),
                ),
                (
                    Expression::Function(Aggregate::Max, "name".to_string()),
                    None,
                ),
            ],
            SelectFrom::Table {
                name: "grades".to_string(),
            },
            None,
            Some((vec![Expression::Field("grade".to_string())], None)),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["grade", "count", "MAX(name)"]);
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&vec![
            Value::Integer(90),
            Value::Integer(1),
            Value::String("Alice".to_string())
        ]));
        assert!(rows.contains(&vec![
            Value::Integer(80),
            Value::Integer(2),
            Value::String("Charlie".to_string())
        ]));

        // 测试错误情况：SELECT name, COUNT(*) AS count FROM users GROUP BY id
        assert!(executor
            .select(
                vec![
                    (Expression::Field("name".to_string()), None),
                    (
                        Expression::Function(Aggregate::Count, "*".to_string()),
                        Some("count".to_string()),
                    ),
                ],
                SelectFrom::Table {
                    name: "users".to_string(),
                },
                None,
                Some((vec![Expression::Field("id".to_string())], None)),
                vec![],
                None,
                None,
            )
            .is_err());

        // 测试多个 GROUP BY 列
        // 测试 SELECT name, grade, COUNT(*) AS count FROM grades GROUP BY name, grade
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("name".to_string()), None),
                (Expression::Field("grade".to_string()), None),
                (
                    Expression::Function(Aggregate::Count, "*".to_string()),
                    Some("count".to_string()),
                ),
            ],
            SelectFrom::Table {
                name: "grades".to_string(),
            },
            None,
            Some((
                vec![
                    Expression::Field("name".to_string()),
                    Expression::Field("grade".to_string()),
                ],
                None,
            )),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["name", "grade", "count"]);
        assert_eq!(rows.len(), 3);
        assert!(rows.contains(&vec![
            Value::String("Alice".to_string()),
            Value::Integer(90),
            Value::Integer(1)
        ]));
        assert!(rows.contains(&vec![
            Value::String("Bob".to_string()),
            Value::Integer(80),
            Value::Integer(1)
        ]));
        assert!(rows.contains(&vec![
            Value::String("Charlie".to_string()),
            Value::Integer(80),
            Value::Integer(1)
        ]));

        // 测试 HAVING 语句
        // 测试 SELECT grade, COUNT(*) AS count FROM grades GROUP BY grade HAVING grade = 80
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("grade".to_string()), None),
                (
                    Expression::Function(Aggregate::Count, "*".to_string()),
                    Some("count".to_string()),
                ),
            ],
            SelectFrom::Table {
                name: "grades".to_string(),
            },
            None,
            Some((
                vec![Expression::Field("grade".to_string())],
                Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("grade".to_string())),
                    Box::new(Expression::Constant(Constant::Integer(80))),
                ))),
            )),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["grade", "count"]);
        assert_eq!(rows, vec![vec![Value::Integer(80), Value::Integer(2)]]);

        // 测试 SELECT grade, COUNT(*) AS count FROM grades GROUP BY grade HAVING COUNT(*) = 2
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("grade".to_string()), None),
                (
                    Expression::Function(Aggregate::Count, "*".to_string()),
                    Some("count".to_string()),
                ),
            ],
            SelectFrom::Table {
                name: "grades".to_string(),
            },
            None,
            Some((
                vec![Expression::Field("grade".to_string())],
                Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Function(Aggregate::Count, "*".to_string())),
                    Box::new(Expression::Constant(Constant::Integer(2))),
                ))),
            )),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["grade", "count"]);
        assert_eq!(rows, vec![vec![Value::Integer(80), Value::Integer(2)]]);

        Ok(())
    }
}
