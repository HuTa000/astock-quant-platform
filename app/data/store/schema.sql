-- A股行情分析与选股软件 —— SQLite 建表 DDL
-- 约定（《开发手册与交接文档》§4）：
--   * 表名/列名一律 snake_case；
--   * 时间列统一以 ISO 文本存储（'YYYY-MM-DD HH:MM:SS'）；
--   * 本脚本【幂等】：全部 CREATE TABLE/INDEX IF NOT EXISTS，可重复执行。
-- 连接时需执行 PRAGMA foreign_keys=ON;（见 store/db.py），否则 ON DELETE CASCADE 不生效。

-- 自选股分组
CREATE TABLE IF NOT EXISTS watchlist_group (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    sort_order  INTEGER DEFAULT 0
);

-- 自选股条目
CREATE TABLE IF NOT EXISTS watchlist_item (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    INTEGER NOT NULL REFERENCES watchlist_group(id) ON DELETE CASCADE,
    symbol      TEXT    NOT NULL,
    added_at    TEXT    NOT NULL,
    note_mark   INTEGER DEFAULT 0,
    UNIQUE(group_id, symbol)
);

-- 个股备注
CREATE TABLE IF NOT EXISTS stock_note (
    symbol      TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- K线数据
-- 主键含 adjust：避免同股同周期的不复权/qfq/hfq 数据互相覆盖（实现级细化）。
CREATE TABLE IF NOT EXISTS kline (
    symbol      TEXT NOT NULL,
    period      TEXT NOT NULL,
    dt          TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    amount      REAL,
    adjust      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (symbol, period, dt, adjust)
);

-- K线常用查询索引（按股+周期+时间）
CREATE INDEX IF NOT EXISTS idx_kline_symbol_period_dt ON kline(symbol, period, dt);

-- 复权因子序列
CREATE TABLE IF NOT EXISTS adj_factor (
    symbol      TEXT NOT NULL,
    dt          TEXT NOT NULL,
    factor      REAL NOT NULL,
    PRIMARY KEY (symbol, dt)
);

-- 自定义指标模板
CREATE TABLE IF NOT EXISTS indicator_template (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    category    TEXT DEFAULT '',
    expr        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 选股条件模板
CREATE TABLE IF NOT EXISTS screener_template (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    conditions_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- 配置表（键值）
CREATE TABLE IF NOT EXISTS config (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- 运行时元数据（如 last_update_time）
CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- 技术指标信号预警规则（P4-S1）
-- kind 取值：ma_cross / macd_cross / volume_surge / expr；
-- params_json 内容按 kind 区分（见 app/engine/alert/alert_service.py）。
CREATE TABLE IF NOT EXISTS alert_rule (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,           -- 标的代码
    kind        TEXT NOT NULL,           -- 规则类型：ma_cross/macd_cross/volume_surge/expr
    params_json TEXT NOT NULL,           -- 参数 JSON（按 kind 区分）
    enabled     INTEGER NOT NULL DEFAULT 1,  -- 1 启用 / 0 停用
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 全 A 股代码与名称名录（P6 优化：股票查找与加载）
CREATE TABLE IF NOT EXISTS stock_list (
    symbol     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    market     TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stock_list_name ON stock_list(name);

-- 指数成分股名单快照（P7-S4：历史成分股 PIT 口径，用于消除幸存者偏差）
-- 每份 (index_code, effective_date) 为一次「名单快照」；同一标的可在多份快照中重复出现。
-- effective_date 取数据源返回的名单**生效日**（非查询日）：做 PIT 回溯时，
-- 「某交易日可用成分」= effective_date ≤ 该日 的**最近一份**快照。
CREATE TABLE IF NOT EXISTS index_constituent (
    index_code     TEXT NOT NULL,             -- 指数代码，如 'sh.000300'（沪深300）
    symbol         TEXT NOT NULL,             -- 成分股代码（裸 6 位码，与 kline.symbol 同口径）
    name           TEXT NOT NULL DEFAULT '',  -- 快照时点的股票名称
    effective_date TEXT NOT NULL,             -- 名单生效日（'YYYY-MM-DD'）
    PRIMARY KEY (index_code, symbol, effective_date)
);
CREATE INDEX IF NOT EXISTS idx_index_constituent_symbol ON index_constituent(symbol);
CREATE INDEX IF NOT EXISTS idx_index_constituent_date ON index_constituent(index_code, effective_date);
