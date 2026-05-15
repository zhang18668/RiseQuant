import czsc
from czsc import CZSC, Freq, format_standard_kline
from czsc.mock import generate_symbol_kines

# 生成模拟 K 线数据
df = generate_symbol_kines('000001', '30分钟', '20240101', '20240601')

# 转换为 RawBar 对象列表
bars = format_standard_kline(df, freq=Freq.F30)

# 创建 CZSC 分析对象（自动识别分型、笔、中枢）
czsc_obj = CZSC(bars)
print(f"笔数量：{len(czsc_obj.bi_list)}")
print(f"中枢数量：{len(czsc_obj.zs_list)}")