> 测试平台，win10，2060 12g，官方理论数据：
>
> - FP16 (half) performance
>
>   14.36 TFLOPS (2:1)
>
> - FP32 (float) performance
>
>   7.181 TFLOPS
>
> - FP64 (double) performance
>
>   224.4 GFLOPS (1:32)

# 场景一:较小模型，较大batch

model：albert-tiny，epoch：10，batch_size：512

混合精度：70秒，7250mb显存占用

单精度：98秒，7811mb显存占用

时间占用比：0.714

显存占用比：0.928

# 场景二:较大模型，较小batch

model：bert，epoch：3，batch_size：128

混合精度：210秒，8911mb显存占用

单精度：358秒，11226mb显存占用

时间占用比：0.586

现存占用比：0.793

# 结论

模型越大，时间和显存占用比会越小，训练效率越高，训练的指标差不多



# 核心代码

```python
# 实例化混合精度的类和装饰器
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

# 使用混合精度
with autocast():
    loss = model(input_ids, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 不使用混合精度
loss = model(input_ids, labels)
loss.backward()
optimizer.step()
```

