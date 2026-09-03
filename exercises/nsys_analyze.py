"""分析 nsys sqlite 导出: 2.1.4 (a)(b)(d) 的三个数字.

用法(你的终端):
    nsys profile -t cuda,nvtx -o /tmp/full_step --force-overwrite true \
        .venv/bin/python cs336_systems/benchmark.py --model-size small --mode full --warmup-steps 3 --num-steps 5
    nsys export --type sqlite --force-overwrite true -o /tmp/full_step.sqlite /tmp/full_step.nsys-rep
    .venv/bin/python /root/Myllm/exercises/nsys_analyze.py /tmp/full_step.sqlite
"""
import sqlite3
import sys

def us(x):
    return f'{x/1e6:.2f}ms'  # sqlite 时间戳单位是 ns

def main(path):
    con = sqlite3.connect(path)
    cur = con.cursor()

    # NVTX step 区间
    try:
        ranges = cur.execute(
            "SELECT start, end, text FROM NVTX_EVENTS "
            "WHERE text='benchmark_step' AND end > start ORDER BY start").fetchall()
    except sqlite3.OperationalError:
        ranges = []  # capture 里没有 NVTX 事件
    if not ranges:
        print('!! 无 benchmark_step NVTX 区间, 跳过 (a)(b), 只输出 kernel 统计')
    else:
        print(f'找到 {len(ranges)} 个 step 区间')

    # 所有 kernel 区间
    kernels = [r for r in cur.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "LEFT JOIN StringIds s ON k.shortName = s.id ORDER BY k.start").fetchall()]
    print(f'共 {len(kernels)} 个 kernel launch')

    kern = [(s, e) for s, e, _ in kernels]

    def coverage(lo, hi):
        """区间内被至少一个 kernel 覆盖的时间占比 + 覆盖并集时长."""
        total = 0.0
        cs, ce = None, None
        for s, e in kern:
            if e <= lo or s >= hi:
                continue
            s2, e2 = max(s, lo), min(e, hi)
            if cs is None:
                cs, ce = s2, e2
                total += e2 - s2
            elif s2 <= ce:
                if e2 > ce:
                    total += e2 - ce
                    ce = e2
            else:
                cs, ce = s2, e2
                total += e2 - s2
        return total, (hi - lo)

    print('\n=== (a)(d) 每个 step: 墙钟 vs kernel 覆盖 ===')
    covs, spans = [], []
    for i, (lo, hi, _) in (enumerate(ranges) if ranges else []):
        cov, span = coverage(lo, hi)
        covs.append(cov)
        spans.append(span)
        print(f'step{i}: span={us(span)} kernel覆盖={us(cov)} = {cov/span*100:.1f}%')
    import statistics as st
    if covs:
        cv = st.mean(covs[1:]) if len(covs) > 1 else covs[0]
        sp = st.mean(spans[1:]) if len(spans) > 1 else spans[0]
        print(f'均值(去首步): span={us(sp)} 覆盖={us(cv)} = {cv/sp*100:.1f}% '
              f'→ (a) 空闲占比 {100-cv/sp*100:.1f}%')
    else:
        sp = (kern[-1][1] - kern[0][0]) / max(1, len(kern))

    # (b) 最长无 gap kernel 链 (严格无 gap)
    best_s = best_e = 0
    cs, ce = None, None
    for s, e in kern:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            if ce - cs > best_e - best_s:
                best_s, best_e = cs, ce
            cs, ce = s, e
    if ce - cs > best_e - best_s:
        best_s, best_e = cs, ce
    if ranges:
        print(f'\n=== (b) 最长无缝 kernel 链 = {us(best_e - best_s)} '
              f'(占 mean step {sp*1e3:.0f}us 的 {(best_e-best_s)/sp*100:.1f}%) ===')

    # 各 kernel 累计时间 top10
    from collections import Counter
    agg = Counter()
    for s, e, name in kernels:
        agg[name] += e - s
    print('\n=== kernel 累计时间 top10 ===')
    for name, t in agg.most_common(10):
        if isinstance(name, bytes):
            name = name.decode(errors='replace')
        print(f'{t/1e6:9.2f}ms  {str(name)[:80]}')

if __name__ == '__main__':
    main(sys.argv[1])
