use std::{collections::HashMap, fmt::Display};

use aggregate::aggregate;
use join::{hash_join, loop_join};

use crate::{
    engine::{Engine, Transaction},
    error::{Error::InternalError, Result},
    parser::ast::{Aggregate, Expression, JoinType, Ordering, SelectFrom, Statement},
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

#[derive(Debug, Default, Clone)]
pub struct Column {
    table_name: String,
    col_name: String,
    alias: Option<String>,
    agg: Option<Aggregate>,
}

impl TryFrom<&str> for Column {
    type Error = crate::Error;

    fn try_from(name: &str) -> Result<Self> {
        let mut column = Column::default();

        // 可能包括聚集函数，表名和列名，比如 SUM(users.id)
        // 先尝试解析聚集函数
        let parts = name.split('(').collect::<Vec<_>>();
        let col_name = if parts.len() == 2 {
            // 如果包含聚集函数，解析聚集函数和列名
            let agg = Aggregate::try_from(parts[0].to_string())?;
            column.agg = Some(agg);

            // 去掉右括号
            Ok(parts[1].trim_end_matches(')'))
        } else if parts.len() == 1 {
            Ok(name)
        } else {
            // 如果包含多个左括号，报错
            Err(InternalError(format!("Invalid column name {}", name)))
        }?;

        // 解析表名和列名
        if !col_name.contains('.') {
            // 如果是 col_name 的形式，只构建 col_name
            column.col_name = col_name.to_string();
        } else {
            // 如果是 table_name.col_name 的形式，需要将 table_name 和 col_name 拆开
            // 然后构建 table_name 和 col_name 字段
            let parts = col_name.split('.').collect::<Vec<_>>();
            if parts.len() != 2 {
                return Err(InternalError(format!("Invalid column name {}", col_name)));
            }
            column.table_name = parts[0].to_string();
            column.col_name = parts[1].to_string();
        }

        Ok(column)
    }
}

impl TryFrom<&Expression> for Column {
    type Error = crate::Error;

    fn try_from(value: &Expression) -> Result<Self> {
        match value {
            // 如果是 Field，直接解析
            Expression::Field(name) => Self::try_from(name.as_str()),
            // 如果是 Function，解析聚集函数和列名
            Expression::Function(agg, name) => {
                let mut column = Self::try_from(name.as_str())?;
                column.agg = Some(*agg);
                Ok(column)
            }
            _ => Err(InternalError(format!("Unsupported expression {:?}", value))),
        }
    }
}

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 首选 alias
        if let Some(alias) = &self.alias {
            write!(f, "{}", alias)
        } else if let Some(agg) = self.agg {
            // 如果是聚集函数
            write!(f, "{}({})", agg, self.col_name)
        } else {
            // 不是的话，只返回列名
            write!(f, "{}", self.col_name)
        }
    }
}

impl PartialEq for Column {
    fn eq(&self, other: &Self) -> bool {
        // 聚集函数和列名必须匹配
        if self.agg != other.agg || self.col_name != other.col_name {
            return false;
        }
        // 如果表名不为空，表名也必须匹配
        if !self.table_name.is_empty() && self.table_name != other.table_name {
            return false;
        }
        true
    }
}

impl Eq for Column {}

/// 获取列在 columns 中的索引
pub fn get_column_index(columns: &[Column], col: &Column) -> Result<usize> {
    let indices = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            // 列名和聚集函数必须匹配
            if c.col_name != col.col_name || c.agg != col.agg {
                return false;
            }
            // 如果表名不为空，必须匹配
            if !col.table_name.is_empty() && c.table_name != col.table_name {
                return false;
            }
            true
        })
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();

    // 如果没有找到或者找到多个匹配的列，报错
    if indices.is_empty() {
        return Err(InternalError(format!("Column {col} not found in table")));
    } else if indices.len() > 1 {
        return Err(InternalError(format!("Column {col} is ambiguous in table")));
    }

    Ok(indices[0])
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
    fn scan(&self, table_name: &str) -> Result<(Vec<Column>, Vec<Row>)> {
        // 获取表的列定义
        let table = self
            .transaction
            .get_table(table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;

        // 添加表名前缀，以便后续处理时能够识别列名
        let columns = table
            .columns
            .iter()
            .map(|col| Column {
                table_name: table_name.to_string(),
                col_name: col.name.clone(),
                ..Default::default()
            })
            .collect();

        // 扫描表中的所有行
        let rows = self.transaction.scan_table(&table)?;

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
                        Value::try_from(exp)
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
        filter: Option<Expression>,
    ) -> Result<usize> {
        let table = self
            .transaction
            .get_table(&table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;

        // 扫描并过滤数据
        let (col_names, rows) = self.scan(&table_name)?;
        let rows = self.filter_rows(rows, &col_names, filter)?;

        let mut updated_count = 0;
        for row in rows {
            let primary_key = table.get_primary_key(&row).clone();
            let mut updated_row = row;

            for (col_name, expr) in &columns {
                let col_idx = table.get_col_idx(col_name).ok_or(InternalError(format!(
                    "Column {} not found in table {}",
                    col_name, table_name
                )))?;
                updated_row[col_idx] = Value::try_from(expr)?;
            }
            self.transaction
                .update_row(&table, &primary_key, &updated_row)?;
            updated_count += 1;
        }

        Ok(updated_count)
    }

    /// 删除数据
    fn delete(&self, table_name: String, filter: Option<Expression>) -> Result<usize> {
        let table = self
            .transaction
            .get_table(&table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;

        // 扫描并过滤数据
        let (columns, rows) = self.scan(&table_name)?;
        let rows = self.filter_rows(rows, &columns, filter)?;

        let mut delete_count = 0;
        for row in rows {
            let primary_key = table.get_primary_key(&row);
            self.transaction.delete_row(&table, primary_key)?;
            delete_count += 1;
        }

        Ok(delete_count)
    }

    /// 扫描 Join 表，返回所有的列名和行数据
    fn scan_from_join(&self, from: &SelectFrom) -> Result<(Vec<Column>, Vec<Row>)> {
        match from {
            SelectFrom::Table { name } => {
                let (columns, rows) = self.scan(name)?;
                Ok((columns, rows))
            }
            SelectFrom::Join {
                left,
                right,
                join_type,
                predicate,
            } => {
                // 除了 Cross Join 外，其他 Join 类型必须有 Join 条件
                if *join_type != JoinType::Cross && predicate.is_none() {
                    return Err(InternalError(format!("{join_type} must have a predicate")));
                }

                // 递归扫描左右表
                let (left_columns, left_rows) = self.scan_from_join(left)?;
                let (right_columns, right_rows) = self.scan_from_join(right)?;

                // 合并左右表
                match join_type {
                    // Cross Join 直接笛卡尔积
                    JoinType::Cross => {
                        loop_join(&left_columns, &right_columns, &left_rows, &right_rows)
                    }
                    // 否则使用哈希连接
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

    /// 过滤行数据
    fn filter_rows(
        &self,
        rows: Vec<Row>,
        columns: &[Column],
        filter: Option<Expression>,
    ) -> Result<Vec<Row>> {
        if let Some(expr) = filter {
            // filter 只能是 Operation
            let op = expr.as_operation().ok_or(InternalError(format!(
                "Unsupported filter expression {:?}",
                expr
            )))?;
            let mut new_rows = Vec::new();

            // 遍历每一行，根据 filter 条件过滤
            for row in rows {
                if op.evaluate(columns, &row)? {
                    new_rows.push(row);
                }
            }
            Ok(new_rows)
        } else {
            Ok(rows)
        }
    }

    /// 查询数据
    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        select_columns: Vec<(Expression, Option<String>)>,
        from: SelectFrom,
        filter: Option<Expression>,
        groupby: Option<(Vec<Expression>, Option<Expression>)>,
        ordering: Vec<(String, Ordering)>,
        limit: Option<Expression>,
        offset: Option<Expression>,
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // 从 Join 表中扫描数据并过滤
        let (mut columns, rows) = self.scan_from_join(&from)?;

        // 处理过滤
        let mut rows = self.filter_rows(rows, &columns, filter)?;

        // 处理排序
        if !ordering.is_empty() {
            self.sort_rows(&mut rows, &columns, ordering)?;
        }

        // SELECT * 的情况下 select_columns 是空的
        if !select_columns.is_empty() {
            if select_columns.iter().all(|(col, _)| col.is_field()) {
                // 处理都是列名的情况
                if groupby.is_some() {
                    return Err(InternalError(
                        "GROUP BY must have aggregate function".to_string(),
                    ));
                }
                (columns, rows) = self.select_field_columns(&select_columns, &columns, rows)?;
            } else {
                // 处理有聚集函数的情况
                (columns, rows) =
                    self.select_aggregate_columns(&select_columns, &columns, rows, groupby)?;
            }
        } else {
            // 如果没有选择列，但有 GROUP BY，必须有聚集函数
            if groupby.is_some() {
                return Err(InternalError(
                    "GROUP BY must have aggregate function".to_string(),
                ));
            }
        }

        // 处理 limit 和 offset
        if !(offset.is_none() && limit.is_none()) {
            let convert_to_usize = |expr: Option<Expression>, default: usize, err_prefix: &str| {
                expr.map_or(Ok(default), |e| match Value::try_from(&e)? {
                    Value::Integer(v) if v >= 0 => Ok(v as usize),
                    other => Err(InternalError(format!(
                        "{} must be a non-negative integer, get {:?}",
                        err_prefix, other
                    ))),
                })
            };
            let offset = convert_to_usize(offset, 0, "Offset")?;
            let limit = convert_to_usize(limit, usize::MAX, "Limit")?;
            rows = rows
                .into_iter()
                .skip(offset)
                .take(limit)
                .collect::<Vec<_>>();
        }

        // 将 columns 从 Column 结构体转换为字符串
        let columns = columns.into_iter().map(|col| col.to_string()).collect();

        Ok((columns, rows))
    }

    /// 选择列名
    fn select_field_columns(
        &self,
        select_columns: &[(Expression, Option<String>)],
        columns: &[Column],
        rows: Vec<Row>,
    ) -> Result<(Vec<Column>, Vec<Row>)> {
        // 一次性收集新列名
        let new_columns = select_columns
            .iter()
            .map(|(col_expr, alias)| {
                // 列名只允许是字段名
                if !col_expr.is_field() {
                    Err(InternalError(format!(
                        "Unsupported column {col_expr} in SELECT"
                    )))
                } else {
                    let mut column = Column::try_from(col_expr)?;
                    column.alias = alias.clone(); // 为列设置别名

                    Ok(column)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 收集新列名在原列中的索引
        let col_indices = new_columns
            .iter()
            .map(|col| get_column_index(columns, col))
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
        &self,
        select_columns: &[(Expression, Option<String>)],
        columns: &[Column],
        rows: Vec<Row>,
        groupby_having: Option<(Vec<Expression>, Option<Expression>)>,
    ) -> Result<(Vec<Column>, Vec<Row>)> {
        // 一次性收集新列名
        let new_columns = select_columns
            .iter()
            .map(|(col, alias)| {
                let mut column = Column::try_from(col)?;
                column.alias = alias.clone();

                Ok(column)
            })
            .collect::<Result<Vec<_>>>()?;

        // 如果没有 GROUP BY，直接计算聚集函数的值
        if groupby_having.is_none() {
            let agg_values = select_columns
                .iter()
                .map(|(col, _)| aggregate(&Column::try_from(col)?, columns, &rows))
                .collect::<Result<Vec<Value>>>()?;

            return Ok((new_columns, vec![agg_values]));
        }

        // 如果有 GROUP BY，按照 GROUP BY 列进行分组
        let (groupby, having) = groupby_having.unwrap();

        // 获取 GROUP BY 列
        let group_columns = groupby
            .iter()
            .map(|expr| {
                // 获取列
                if expr.is_field() {
                    Column::try_from(expr)
                } else {
                    // GROUP BY 只能是列名
                    Err(InternalError(format!(
                        "Unsupported column {expr} in GROUP BY",
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 获取 GROUP BY 列在 columns 中的索引
        let group_column_indices = group_columns
            .iter()
            .map(|col| get_column_index(columns, col))
            .collect::<Result<Vec<_>>>()?;

        // 根据指定的分组列，将所有行按照分组列的值聚集到一个 HashMap 中
        let mut group_map: HashMap<_, Vec<Row>> = HashMap::new();
        for row in rows {
            let group_key = group_column_indices
                .iter()
                .map(|&group_idx| row[group_idx].clone())
                .collect::<Vec<_>>();
            group_map.entry(group_key).or_default().push(row);
        }

        // 遍历每个分组，计算聚集函数的值
        let new_rows = group_map
            .into_iter()
            .map(|(group_key, group_rows)| {
                let agg_values = select_columns
                    .iter()
                    .map(|(expr, _)| {
                        if expr.is_function() {
                            let column = Column::try_from(expr)?;
                            // 如果是聚集函数，计算聚集函数的值
                            aggregate(&column, columns, &group_rows)
                        } else if expr.is_field() {
                            // 如果不是聚集函数，必须是 GROUP BY 列中的字段
                            let column = Column::try_from(expr)?;
                            let idx = group_columns.iter().position(|col| *col == column).ok_or(
                                InternalError(format!("Column {} not found in GROUP BY", column)),
                            )?;
                            let value = group_key[idx].clone();
                            Ok(value)
                        } else {
                            Err(InternalError(format!(
                                "Unsupported column {expr} in SELECT"
                            )))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(agg_values)
            })
            .collect::<Result<Vec<_>>>()?;

        // 过滤不符合 HAVING 条件的行
        let new_rows = self.filter_rows(new_rows, &new_columns, having)?;

        Ok((new_columns, new_rows))
    }

    /// 对行进行排序
    fn sort_rows(
        &self,
        rows: &mut [Row],
        columns: &[Column],
        ordering: Vec<(String, Ordering)>,
    ) -> Result<()> {
        // 将列名转换为 Column 结构体，并匹配列索引
        let ordering = ordering
            .into_iter()
            .map(|(col_name, ord)| {
                let col = Column::try_from(col_name.as_str())?;
                get_column_index(columns, &col).map(|col_idx| (col_idx, ord))
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

        // 测试 SELECT * FROM users WHERE id >= 1
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(Expression::Operation(Operation::GreaterEqual(
                Box::new(Expression::Field("id".to_string())),
                Box::new(Expression::Constant(Constant::Integer(1))),
            ))),
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

        // 测试 SELECT * FROM users WHERE name IS NOT NULL
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(Expression::Operation(Operation::Not(Box::new(
                Operation::Equal(
                    Box::new(Expression::Field("name".to_string())),
                    Box::new(Expression::Constant(Constant::Null)),
                ),
            )))),
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
            filter: Some(Expression::Operation(Operation::Equal(
                Box::new(Expression::Field("id".to_string())),
                Box::new(Expression::Constant(Constant::Integer(1))),
            ))),
        })?;
        assert_eq!(result, ExecuteResult::Update(1));

        // 测试更新数据后的查询
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(Expression::Operation(Operation::Equal(
                Box::new(Expression::Field("id".to_string())),
                Box::new(Expression::Constant(Constant::Integer(1))),
            ))),
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
            filter: Some(Expression::Operation(Operation::Equal(
                Box::new(Expression::Field("name".to_string())),
                Box::new(Expression::Constant(Constant::Null)),
            ))),
        })?;
        assert_eq!(result, ExecuteResult::Delete(1));

        // 测试删除数据后的查询
        let (columns, rows) = executor.select(
            vec![],
            SelectFrom::Table {
                name: "users".to_string(),
            },
            Some(Expression::Operation(Operation::Equal(
                Box::new(Expression::Field("id".to_string())),
                Box::new(Expression::Constant(Constant::Integer(1))),
            ))),
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
                Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Field("name".to_string())),
                    Box::new(Expression::Constant(Constant::String("Alice".to_string()))),
                ))),
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
            Some(Expression::Operation(Operation::Equal(
                Box::new(Expression::Field("users.name".to_string())),
                Box::new(Expression::Constant(Constant::String("Alice".to_string()))),
            ))),
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

    #[test]
    fn test_aggregate_with_join() -> Result<()> {
        let executor = init_executor()?;
        create_tables(&executor)?;
        insert_data(&executor)?;

        // 测试 SELECT users.name, MAX(grade) FROM users CROSS JOIN grades GROUP BY users.name
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("users.name".to_string()), None),
                (
                    Expression::Function(Aggregate::Max, "grade".to_string()),
                    None,
                ),
            ],
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
            Some((vec![Expression::Field("users.name".to_string())], None)),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["name", "MAX(grade)"]);
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&vec![
            Value::String("Alice".to_string()),
            Value::Integer(90)
        ]));
        assert!(rows.contains(&vec![Value::Null, Value::Integer(90)]));

        // 测试 SELECT users.name, MAX(grades.grade) FROM users RIGHT JOIN grades GROUP BY users.name HAVING MAX(grades.grade) = 90
        let (columns, rows) = executor.select(
            vec![
                (Expression::Field("users.name".to_string()), None),
                (
                    Expression::Function(Aggregate::Max, "grades.grade".to_string()),
                    None,
                ),
            ],
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
            Some((
                vec![Expression::Field("users.name".to_string())],
                Some(Expression::Operation(Operation::Equal(
                    Box::new(Expression::Function(
                        Aggregate::Max,
                        "grades.grade".to_string(),
                    )),
                    Box::new(Expression::Constant(Constant::Integer(90))),
                ))),
            )),
            vec![],
            None,
            None,
        )?;
        assert_eq!(columns, vec!["name", "MAX(grade)"]);

        assert_eq!(
            rows,
            vec![vec![Value::String("Alice".to_string()), Value::Integer(90)]]
        );

        Ok(())
    }
}
