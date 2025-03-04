use serde::{Deserialize, Serialize};

use crate::{
    parser::ast::Expression,
    schema::{Row, Table, Value},
    storage::{Mvcc, MvccTxn, Storage},
    Error::InternalError,
    Result,
};

/// 数据库引擎，负责管理事务，执行事务操作
pub struct Engine<S: Storage> {
    mvcc: Mvcc<S>,
}

impl<S: Storage> Engine<S> {
    /// 创建一个新的数据库引擎
    pub fn new(storage: S) -> Self {
        Self {
            mvcc: Mvcc::new(storage),
        }
    }

    /// 开启一个新的事务
    pub fn start_txn(&self) -> Result<Transaction<S>> {
        Ok(Transaction {
            txn: self.mvcc.start_txn()?,
        })
    }
}

/// 数据库引擎内部的键
///
/// - `Table(String)`：标识存储表信息
/// - `Row(String, Value)`：标识存储行数据
#[derive(Debug, Serialize, Deserialize)]
enum Key {
    Table(String),
    Row(String, Value),
}

/// 数据库引擎内部的键前缀
///
/// - `Table`：标识表信息的前缀
/// - `Row(String)`：标识行数据的前缀
///
/// 注：和 `storage::mvcc::MvccKey` 不同，虽然前缀中也使用了字符串，但字符串长度和 Key 中的 `String` 长度相同，
/// 因此不需要删除前缀中的长度信息，直接使用 bincode 序列化即可。
#[derive(Debug, Serialize, Deserialize)]
enum KeyPrefix {
    Table,
    Row(String),
}

/// 数据库事务，对 `MvccTxn` 进行了封装，提供了更高级别的操作
pub struct Transaction<S: Storage> {
    txn: MvccTxn<S>,
}

impl<S: Storage> Transaction<S> {
    /// 获取表信息
    pub fn get_table(&self, table_name: &str) -> Result<Option<Table>> {
        let key = Key::Table(table_name.to_string());
        let table = self
            .txn
            .get(&bincode::serialize(&key)?)?
            .map(|data| bincode::deserialize(&data))
            .transpose()?;
        Ok(table)
    }

    /// 创建行数据
    pub fn create_row(&self, table_name: &str, row: &Row) -> Result<()> {
        // 如果表不存在，返回错误
        let table = self
            .get_table(table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;

        // 检查行数据是否符合表定义
        for (column, row) in table.columns.iter().zip(row.iter()) {
            match row.data_type() {
                None if !column.nullable => {
                    return Err(InternalError(format!(
                        "Column {} cannot be null",
                        column.name
                    )));
                }
                Some(data_type) if data_type != column.data_type => {
                    return Err(InternalError(format!(
                        "Column {} expect {:?}, got {:?}",
                        column.name, column.data_type, data_type
                    )));
                }
                _ => {}
            }
        }

        // 将行数据序列化后存储，键为表名和主键值
        let key = Key::Row(table_name.to_string(), table.get_primary_key(row).clone());
        let value = bincode::serialize(row)?;
        self.txn.set(&bincode::serialize(&key)?, &value)?;

        Ok(())
    }

    /// 创建表
    pub fn create_table(&self, table: Table) -> Result<()> {
        // 检查表是否已经存在，如果存在则返回错误
        if self.get_table(&table.name)?.is_some() {
            return Err(InternalError(format!(
                "Table {} already exists",
                table.name
            )));
        }

        let key = bincode::serialize(&Key::Table(table.name.clone()))?;
        let value = bincode::serialize(&table)?;
        self.txn.set(&key, &value)?;

        Ok(())
    }

    /// 扫描表
    pub fn scan_table(
        &self,
        table_name: &str,
        filter: Option<(String, Expression)>,
    ) -> Result<Vec<Row>> {
        let prefix = KeyPrefix::Row(table_name.to_string());
        let result = self.txn.scan_prefix(&bincode::serialize(&prefix)?)?;
        let table = self
            .get_table(table_name)?
            .ok_or(InternalError(format!("Table {table_name} not found")))?;

        let mut rows = Vec::new();
        for (_, value) in result {
            let row: Row = bincode::deserialize(&value)?;
            // 如果有过滤条件，检查是否符合条件
            if let Some((col, expr)) = &filter {
                let col_idx = table.get_col_idx(col).ok_or(InternalError(format!(
                    "Column {} not found in table {}",
                    col, table_name
                )))?;
                if Value::from(expr.clone()) != row[col_idx] {
                    continue;
                }
            }
            rows.push(row);
        }

        Ok(rows)
    }

    /// 更新行数据
    ///
    /// `pk` 为要更新的行的主键值，`row` 为新的行数据，`row` 的主键值不一定和 `pk` 相同。
    pub fn update_row(&self, table: &Table, pk: &Value, row: &Row) -> Result<()> {
        // 如果更新了主键，则需要删除原来的数据
        let row_pk = table.get_primary_key(row);
        if row_pk != pk {
            let key = Key::Row(table.name.clone(), pk.clone());
            self.txn.delete(&bincode::serialize(&key)?)?;
        }

        // 更新行数据
        let key = Key::Row(table.name.clone(), row_pk.clone());
        let value = bincode::serialize(row)?;
        self.txn.set(&bincode::serialize(&key)?, &value)?;

        Ok(())
    }

    /// 删除行数据
    pub fn delete_row(&self, table: &Table, pk: &Value) -> Result<()> {
        let key = Key::Row(table.name.clone(), pk.clone());
        self.txn.delete(&bincode::serialize(&key)?)?;

        Ok(())
    }

    /// 提交事务
    #[inline]
    pub fn commit(&self) -> Result<()> {
        self.txn.commit()
    }

    /// 回滚事务
    #[inline]
    pub fn rollback(&self) -> Result<()> {
        self.txn.rollback()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        parser::ast::Constant,
        schema::{Column, DataType},
        storage::MemoryStorage,
    };

    #[test]
    fn test_engine() {
        let storage = MemoryStorage::new();
        let engine = Engine::new(storage);
        let txn = engine.start_txn().unwrap();

        let columns = vec![
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
                default: Some(Value::String("".to_string())),
                primary_key: false,
            },
        ];
        let table = Table::new("users", columns).unwrap();
        txn.create_table(table).unwrap();

        let table = txn.get_table("users").unwrap().unwrap();
        assert_eq!(table.name, "users");

        assert_eq!(table.columns[0].name, "id");
        assert_eq!(table.columns[0].data_type, DataType::Integer);
        assert!(!table.columns[0].nullable);
        assert_eq!(table.columns[0].default, None);

        assert_eq!(table.columns[1].name, "name");
        assert_eq!(table.columns[1].data_type, DataType::String);
        assert!(table.columns[1].nullable);
        assert_eq!(
            table.columns[1].default,
            Some(Value::String("".to_string()))
        );

        let rows = vec![
            vec![Value::Integer(42), Value::String("zmsbruce".to_string())],
            vec![
                Value::Integer(114514),
                Value::String("Tadokoro".to_string()),
            ],
        ];
        for row in rows.iter() {
            txn.create_row("users", row).unwrap();
        }

        let rows_scan = txn.scan_table("users", None).unwrap();
        assert_eq!(rows_scan, rows);

        let rows_scan = txn
            .scan_table(
                "users",
                Some((
                    "id".to_string(),
                    Expression::Constant(Constant::Integer(42)),
                )),
            )
            .unwrap();
        assert_eq!(rows_scan.len(), 1);

        txn.update_row(
            &table,
            &Value::Integer(42),
            &vec![Value::Integer(42), Value::String("zmsbruceee".to_string())],
        )
        .unwrap();
        let rows_scan = txn.scan_table("users", None).unwrap();
        assert_eq!(
            rows_scan,
            vec![
                vec![Value::Integer(42), Value::String("zmsbruceee".to_string())],
                vec![
                    Value::Integer(114514),
                    Value::String("Tadokoro".to_string())
                ],
            ]
        );

        txn.delete_row(&table, &Value::Integer(42)).unwrap();
        let rows_scan = txn.scan_table("users", None).unwrap();
        assert_eq!(
            rows_scan,
            vec![vec![
                Value::Integer(114514),
                Value::String("Tadokoro".to_string())
            ]]
        );
    }
}
