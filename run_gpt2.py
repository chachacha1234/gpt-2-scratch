import csv
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import tiktoken
import torch
from torch.nn import functional as F

from train_gpt2 import GPT, GPTConfig, DataLoaderLite, get_lr

# 실험 설정은 여기에서 변경
mode = 'train'                 # 'train' 또는 'generate'
checkpoint_path = None         # 생성할 때 저장한 model.pt 경로
data_dir = 'edu_fineweb10B'    # fineweb.py가 만든 train/val shard 폴더
B, T = 4, 32                 # 한 배치의 시퀀스 수와 시퀀스당 토큰 수
total_batch_size = B * T       # 한 번 갱신할 때 처리하는 전체 토큰 수
max_steps = 8000              # B*T=128이면 총 1,024,000토큰 학습
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 300
eval_interval = 500
eval_batches = 50
checkpoint_interval = 2000    # step별 파일로 중간 모델 보관
weight_decay = 0.1
use_amp = False
use_compile = False
model_config = GPTConfig()
prompt = "Hello, I'm a language model,"
max_new_tokens = 16
top_k = 50

def generate_sample(model, enc, device, block_size, amp_dtype):
    # 마지막 위치에서 다음 토큰을 하나씩 선택한다.
    model.eval()
    tokens = enc.encode(prompt) or [enc.eot_token]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    assert 0 < top_k <= enc.n_vocab and max_new_tokens >= 0
    rng = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                # 모델의 최대 문맥 길이에 맞춰 최근 토큰만 전달한다.
                logits, _ = model(x[:, -block_size:])
            # 마지막 위치에서 다음 토큰을 예측하고, 패딩된 어휘는 후보에서 제외한다.
            logits = logits[:, -1, :enc.n_vocab].float()
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            # 상위 k개 후보에서 확률에 따라 하나를 뽑고, 실제 토큰 ID로 바꾼다.
            ix = torch.multinomial(topk_probs, 1, generator=rng)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    result = enc.decode(x[0].tolist())
    return result


def main():
    # 실행한 터미널 위치와 상관없이 이 파일이 있는 폴더를 기준으로 읽고 저장한다.
    base_dir = Path(__file__).resolve().parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    amp_dtype = torch.float16 if device == 'cuda' and torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
    torch.manual_seed(1337)
    if device == 'cuda':
        torch.cuda.manual_seed(1337)
        torch.set_float32_matmul_precision('high')

    enc = tiktoken.get_encoding('gpt2')
    assert mode in ('train', 'generate')
    if mode == 'generate':
        # 저장 당시의 모델 설정으로 구조를 만든 다음 가중치를 불러온다.
        if checkpoint_path is None:
            raise ValueError('checkpoint_path에 모델 파일 경로를 지정하세요.')
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = base_dir / path
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = GPT(GPTConfig(**checkpoint['config']))
        model.load_state_dict(checkpoint['model'])
    else:
        model = GPT(model_config)
    model.to(device)
    raw_model = model  # compile을 켜도 저장에는 원래 모델의 가중치를 사용한다.
    if use_compile:
        model = torch.compile(model)

    if mode == 'train':
        assert B > 0 and T > 0 and max_steps > warmup_steps >= 0
        assert total_batch_size > 0 and total_batch_size % (B * T) == 0
        assert 0 < min_lr <= max_lr and eval_interval > 0 and eval_batches > 0
        assert checkpoint_interval > 0 and checkpoint_interval % eval_interval == 0
        # 작은 배치 몇 개의 gradient를 모아 한 번 갱신할지 계산한다.
        grad_accum_steps = total_batch_size // (B * T)
        print(f'training: {max_steps:,} steps, {total_batch_size:,} tokens/step, '
              f'{max_steps * total_batch_size:,} tokens total')

        # 1. 데이터 준비: 학습용과 검증용을 나눔
        if data_dir is None:
            text = (base_dir / 'input.txt').read_text(encoding='utf-8-sig')
            # 텍스트 앞 90%는 학습에 쓰고, 뒤 10%는 검증에 사용한다.
            split = int(len(text) * 0.9)
            train_tokens = torch.tensor(enc.encode_ordinary(text[:split]), dtype=torch.long)
            val_tokens = torch.tensor(enc.encode_ordinary(text[split:]), dtype=torch.long)
            train_loader = DataLoaderLite(B, T, tokens=train_tokens)
            val_loader = DataLoaderLite(B, T, tokens=val_tokens)
            print(f'train: {len(train_tokens)} tokens, val: {len(val_tokens)} tokens')
        else:
            folder = Path(data_dir)
            if not folder.is_absolute():
                folder = base_dir / folder
            # 파일 이름 순서대로 읽고, 학습용과 검증용 shard를 분리한다.
            train_shards = sorted(folder.glob('*train*.npy'))
            val_shards = sorted(folder.glob('*val*.npy'))
            if not train_shards or not val_shards:
                raise FileNotFoundError('FineWeb의 train/val shard를 모두 준비해야 합니다.')
            print(f'FineWeb: {folder}')
            print(f'train: {len(train_shards)} shards, val: {len(val_shards)} shards')
            train_loader = DataLoaderLite(B, T, shards=train_shards)
            val_loader = DataLoaderLite(B, T, shards=val_shards)

        optimizer = raw_model.configure_optimizers(weight_decay, max_lr, device)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
        schedule = {'lr': max_lr, 'min_lr': min_lr,
                    'warmup_steps': warmup_steps, 'decay_steps': max_steps}
        log_dir = base_dir / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log_dir.mkdir(parents=True, exist_ok=False)
        print(f'output: {log_dir}')

        # 2. 학습
        with (log_dir / 'loss.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'train_loss', 'val_loss', 'lr', 'grad_norm', 'seconds', 'tokens_per_second'])
            train_metrics = None
            # step은 이 시점까지 완료한 학습 스텝 수다.
            for step in range(max_steps + 1):
                # 강의처럼 현재 모델을 검증한 다음 새 배치를 학습한다.
                val_loss = None
                if step % eval_interval == 0 or step == max_steps:
                    model.eval()
                    val_loader.reset()  # 검증할 때마다 같은 토큰 구간을 사용한다.
                    val_loss = 0.0
                    with torch.no_grad():
                        for _ in range(eval_batches):
                            x, y = val_loader.next_batch()
                            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                                _, loss = model(x.to(device), y.to(device))
                            if not torch.isfinite(loss).item():
                                raise RuntimeError('검증 loss가 NaN/Inf입니다.')
                            val_loss += loss.item() / eval_batches
                    print(f'step {step}: validation loss {val_loss:.4f}')
                    # 업데이트를 마친 step 수로 저장. 검증과 저장 간격은 따로 설정한다.
                    if step > 0 and (step % checkpoint_interval == 0 or step == max_steps):
                        checkpoint = {'model': raw_model.state_dict(),
                                      'config': asdict(raw_model.config),
                                      'step': step, 'val_loss': val_loss,
                                      'tokens_seen': (step) * total_batch_size}
                        path = log_dir / f'model_{step:05d}.pt'
                        torch.save(checkpoint, path)
                        print(f'saved: {path}')
                        if step == max_steps:
                            torch.save(checkpoint, log_dir / 'model.pt')
                if train_metrics is not None:
                    loss, lr, norm, dt = train_metrics
                    writer.writerow([step, loss, val_loss, lr, norm, dt, total_batch_size / dt])
                    file.flush()

                if step > 0 and (step % eval_interval == 0 or step == max_steps):
                    result = generate_sample(model, enc, device, raw_model.config.block_size, amp_dtype)
                    print(f'step {step} sample: {result}')
                    if step == max_steps:
                        (log_dir / 'sample.txt').write_text(result, encoding='utf-8')

                if step == max_steps:
                    break

                # 이전 GPU 작업이 끝난 뒤부터 이번 학습 스텝의 시간을 잰다.
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.train()
                optimizer.zero_grad()  # 이전 갱신의 gradient를 지우고 이번 배치들의 gradient를 모은다.
                loss_accum = 0.0

                for micro_step in range(grad_accum_steps):
                    x, y = train_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                        logits, loss = model(x, y)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f'step {step}: loss가 NaN/Inf입니다.')
                    # 작은 배치의 gradient를 합할 때 전체 배치의 평균이 되도록 나눈다.
                    loss = loss / grad_accum_steps
                    loss_accum += loss.detach().item()
                    scaler.scale(loss).backward()

                # 누적 gradient의 전체 norm을 제한한 뒤 이번 학습률로 한 번 갱신한다.
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
                lr = get_lr(step, schedule)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                scaler.step(optimizer)
                scaler.update()
                if device == 'cuda':
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                print(f'step {step + 1:3d} | loss {loss_accum:.4f} | lr {lr:.2e} | norm {norm.item():.3f} | {dt:.3f}s')

                train_metrics = loss_accum, lr, norm.item(), dt
    else:
        print(generate_sample(model, enc, device, raw_model.config.block_size, amp_dtype))


if __name__ == '__main__':
    main()
