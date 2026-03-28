# Regime 分类器 v3 — 策略区交付说明

> 交付日期: 2026-03-28
> 模块路径: `shared/regime/classifier_v3.py`

## 一、定位

为策略区提供**三状态市场环境判断**: 进攻 / 防守 / 空仓。

策略根据状态切换仓位/选股逻辑，无需自行判断市场方向。

## 二、版本演进

| 版本 | 文件 | 状态 | 已知段准确率 | 单调性 |
|------|------|------|-------------|--------|
| v1 | `classifier.py` | 生产看板在用 | 2/12 | 全不通过 |
| v2 | `classifier_v2.py` | 已废弃 | 10/12 | 1/3 通过 |
| **v3** | **`classifier_v3.py`** | **推荐** | **11/12** | **3/3 通过** |

v3 相对 v1 的核心改进: 新增「市场方向」维度（四子信号投票），能区分牛熊。

## 三、使用方式

### 方式 A: 从 MongoDB 直接计算（推荐）

```python
from pymongo import MongoClient
from shared.regime import compute_v3_timeline_from_mongo

db = MongoClient("mongodb://localhost:27017")["investment"]
result = compute_v3_timeline_from_mongo(db, start="20100101", end="20260401")

timeline = result["timeline"]   # {20260327: "进攻", 20260326: "防守", ...}
detail   = result["detail"]     # 每日子信号明细
style    = result["style"]      # {20260327: "成长", ...}
```

需要 MongoDB `index_daily` 集合中包含以下指数:
- **必需**: 000300.SH (沪深300), 399006.SZ (创业板指)
- **推荐**: 000016.SH (上证50), 000905.SH (中证500), 000852.SH (中证1000), 000001.SH (上证综指)

辅助指数缺失时仍可运行，但指数广度信号精度降低。

### 方式 B: 传入数组计算

```python
from shared.regime import compute_v3_timeline

result = compute_v3_timeline(
    csi_close=csi300_array,      # np.ndarray, 按日期升序
    gem_close=gem_array,         # np.ndarray, 同长度
    dates=dates_int_array,       # np.ndarray, YYYYMMDD 整数
    extra_indices={              # 可选
        "000016.SH": sh50_array,
        "000905.SH": csi500_array,
        "000852.SH": csi1000_array,
        "000001.SH": sh_composite_array,
    },
)
```

### 策略集成示例

```python
today_action = timeline.get(20260327)
today_style  = style.get(20260327)

if today_action == "进攻":
    label = f"进攻+{today_style}"   # "进攻+成长" 或 "进攻+价值"
elif today_action == "防守":
    label = "防守"
else:
    label = "空仓"
```

## 四、输出字段

### timeline: `dict[int, str]`
日期 → 确认后的动作（"进攻" / "防守" / "空仓"）

### style: `dict[int, str]`
日期 → 风格偏好（"成长" / "价值"），仅进攻状态下有参考意义

### detail: `dict[int, dict]`
每日诊断明细，字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| vol_20 | float | 20 日年化波动率 |
| vol_label | str | "低" / "中" / "高" |
| direction | str | "上行" / "震荡" / "下行" |
| trend_vote | int | 趋势子信号 (-1/0/+1) |
| breadth_vote | int | 指数广度子信号 |
| downvol_vote | int | 下行波动子信号 |
| mom_vote | int | 动量子信号 (含 V 型反转修正) |
| direction_score | int | 四子信号合计 (-4 ~ +4) |
| raw_action | str | 确认前的原始动作 |
| confirmed_action | str | 确认后的最终动作 |
| style | str | 当日风格 |

## 五、验证摘要

数据区间: 2010-06 ~ 2026-03 (3776 天)

### 状态分布
- 进攻 1331 天 (35.2%)
- 防守 1481 天 (39.2%)
- 空仓  964 天 (25.5%)
- 年均切换 24.6 次

### 单调性 (进攻 > 防守 > 空仓 期望收益)
| 窗口 | 进攻 | 防守 | 空仓 | 结果 |
|------|------|------|------|------|
| 5d | +0.23% | +0.04% | +0.02% | 通过 |
| 10d | +0.44% | +0.17% | -0.05% | 通过 |
| 20d | +0.80% | +0.36% | +0.03% | 通过 |

### 已知牛熊段 (11/12 正确)
| 区间 | 事件 | 期望 | 结果 |
|------|------|------|------|
| 2014-07~2015-06 | 大牛市 | 进攻 | 进攻 185/234 天 |
| 2015-06~2015-09 | 股灾 | 空仓 | 空仓 64/75 天 |
| 2016-01 | 熔断 | 空仓 | 空仓 19/20 天 |
| 2017-01~2017-11 | 蓝筹慢牛 | 进攻 | 进攻 129/223 天 |
| 2018-01~2018-12 | 贸易战熊市 | 空仓 | 空仓 153/224 天 |
| 2019-01~2019-04 | 春季躁动 | 进攻 | 进攻 49/72 天 |
| 2020-03~2020-07 | 疫情后反弹 | 进攻 | 进攻 45/75 天 |
| 2021-02~2021-03 | 核心资产崩盘 | 空仓 | 空仓 13/26 天 |
| 2021-12~2022-04 | 熊市 | 空仓 | 空仓 65/90 天 |
| 2022-10~2022-11 | 反弹 | 进攻 | **防守** 30/44 天 |
| 2024-01~2024-02 | 流动性危机 | 空仓 | 空仓 25/25 天 |
| 2024-09~2024-10 | 924行情 | 进攻 | 进攻 4/6 天 |

唯一未命中: 2022-10 反弹 — 属于熊市反弹（后续 12 月回落），判为防守实际更安全。

## 六、参数与可调项

| 参数 | 默认值 | 含义 |
|------|--------|------|
| DEAD_ZONE | 0.03 | 风格死区，避免频繁切换 |
| TREND_MA_SHORT | 20 | MA20 窗口 |
| TREND_MA_LONG | 60 | MA60 窗口 |
| DOWNVOL_UP | 0.45 | 下行波动占比 < 此值看多 |
| DOWNVOL_DOWN | 0.55 | 下行波动占比 > 此值看空 |
| MOM60_UP | 0.03 | 60 日收益 > 此值看多 |
| MOM60_DOWN | -0.03 | 60 日收益 < 此值看空 |
| VOL_LOOKBACK | 500 | 波动率分位数回看窗口 |

非对称确认天数见源码 `CONFIRM_UPGRADE` / `CONFIRM_DOWNGRADE`。

## 七、数据依赖

| 集合 | 字段 | 用途 |
|------|------|------|
| index_daily | symbol, date, close | 全部计算的唯一数据源 |

无需 stock_market、north_moneyflow 等其他集合。

## 八、已知局限

1. **起算需要 65 个交易日预热** (MA60 + 5 日斜率)
2. **2022-10 熊市反弹未识别为进攻** — 设计取舍，保安全性
3. **年均切换 ~25 次** — 对于月度调仓策略足够低频；若嫌高可增大确认天数
