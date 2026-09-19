"""KV cache 冒烟测试: 等长/采样/变长/单层/GQA/机制 六组回归."""
import torch, importlib.util, io, contextlib

spec = importlib.util.spec_from_file_location('kvc', r'd:\Diy-llm\exercises\kvcache.py')
m = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)

torch.manual_seed(0)
R = []
def ck(name, cond):
    R.append(bool(cond))
    print('  %-42s %s' % (name, 'PASS' if cond else 'FAIL'))

V = 1000
lm = m.TransformerLM(256, 8, 2, 1024, V).eval()
p = torch.randint(0, V, (2, 6))

print('--- A. 等长路径 ---')
c = m.greedy_generate(lm, p, 8, use_cache=False)
ck('旧 past_kv: 有cache==无cache', torch.equal(m.greedy_generate(lm, p, 8, use_cache=True), c))
ck('新 KVCache: ==无cache', torch.equal(m.generate_kvcache(lm, p, 8, top_k=1), c))

print('--- B. 采样 ---')
torch.manual_seed(1); x1 = m.generate_kvcache(lm, p, 5, temperature=0.8, top_p=0.9)
torch.manual_seed(1); x2 = m.generate(lm, p, 5, use_cache=False, temperature=0.8, top_p=0.9)
ck('top_p 采样: 新==无cache', torch.equal(x1, x2))

print('--- C. 变长 batch ---')
lens = [5, 8, 3, 7]; NEW = 4; T0 = max(lens)
ps = [torch.randint(0, V, (1, li)) for li in lens]
pad = torch.zeros((4, T0), dtype=torch.long)
for i, q in enumerate(ps): pad[i, :lens[i]] = q[0]
ob = m.generate_kvcache(lm, pad, NEW, top_k=1, prompt_lens=torch.tensor(lens))
ok = True
for i, q in enumerate(ps):
    n = lens[i]; oi = m.generate_kvcache(lm, q, NEW, top_k=1)
    ok &= torch.equal(ob[i, :n], oi[0, :n]) and torch.equal(ob[i, T0:], oi[0, n:])
ck('变长 batch == 逐条', ok)

print('--- D. 单层 + GQA ---')
D, H = 64, 8
a = m.multihead_attn_with_rope(D, H).eval(); x = torch.randn(2, 10, D)
with torch.no_grad():
    full, _ = a(x); _, cc = a(x[:, :5]); outs = []
    for t in range(5, 10):
        o, cc = a(x[:, t:t+1], past_kv=cc); outs.append(o[:, -1])
    st = torch.stack(outs, 1)
ck('attention 分步==一次 (<1e-5)', torch.allclose(st, full[:, 5:], atol=1e-5))
g = m.GQA(D, H, H).eval(); g.load_state_dict(a.state_dict())
with torch.no_grad():
    og, _ = g(x)                      # 现在正确返回 (output, new_kv)
ck('GQA(H_kv=H)==MHA (maxdiff=0)', torch.equal(a(x)[0], og))
g2 = m.GQA(D, H, 2).eval()
with torch.no_grad():
    o2, kv2 = g2(x)
ck('GQA(H_kv=2) cache 形状 (2,2,10,8)', tuple(kv2['k'].shape) == (2, 2, 10, 8))

print('--- E. KVCache 机制 ---')
kc = m.KVCache(4, H, 20, D // H); kc.lens = torch.tensor([3, 5, 2, 7])
ck('invalid_mask 数量 = sum(max_len-lens)',
   kc.invalid_mask().sum().item() == (20-3)+(20-5)+(20-2)+(20-7))
ck('set_finished 生效',
   kc.set_finished(torch.tensor([[9], [9], [1], [9]]), 9).tolist() == [True, True, False, True])

print()
print('总计: %d/%d PASS' % (sum(R), len(R)))
