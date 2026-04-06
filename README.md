Swin에 적용하기 전에 ViT에 적용을 한다면 좋을 듯 해서 적용함
```
# 단일 GPU 테스트

python vit_matryoshka_subdim.py \
    --sub-dims 96 192 384 768 --sub-heads 3 6 6 12 \
    --epochs 1 --batch-size 64 --no-ddp

# 6-GPU 정식 훈련
torchrun --nproc_per_node=6 vit_matryoshka_subdim.py \
    --sub-dims 96 192 384 768 --sub-heads 3 6 6 12 --epochs 90

```


아이디어 정리
기존 MRL과의 차이
기존 MRL	사용자 아이디어
적용 범위	출력 head만 nested	모든 레이어 weight가 nested
backbone 훈련	단일 768-dim으로만	768-dim + 96-dim 동시 감독
inference 96-dim	불가 (head만 자름)	backbone 전체를 96-dim으로 slice 가능
핵심 아이디어
"모든 weight matrix의 상위 좌측 sub-block이 독립적으로 유효한 저차원 네트워크를 구성하도록, 전체 dim과 저차원을 동시에 학습"


W [768×768] 학습 중:
┌──────────────────────────────┐
│ W[:96, :96]  │  W[:96, 96:] │  ← 96-dim 서브네트워크가 사용하는 영역
│──────────────┼──────────────│
│ W[96:, :96]  │ W[96:, 96:]  │
└──────────────────────────────┘
    ↑
이 sub-block에 768-dim loss + 96-dim loss 양쪽에서 gradient가 흘러서
자체적으로 완결된 96-dim 연산이 되도록 강제
훈련 루프 구조
매 iteration (또는 N:1 비율)마다:


입력 x
  ├─ 1) 전체 768-dim forward → loss_full (CE)
  └─ 2) 96-dim slice forward → loss_96  (CE)
          ↑
       W[:96,:96], qkv[:96,:96], mlp[:384,:96] 등을 실시간 slice해서 사용
          (별도 copy 없이 원본 weight의 view)

loss = loss_full + λ * loss_96
loss.backward()  → 두 loss가 W[:96,:96] 영역에 동시에 gradient 기여
Weight Slicing 대상 (레이어별)

patch_embed.proj.weight   [768, 3, 16, 16] → [:96]          (출력 채널만)
cls_token                 [1, 1, 768]      → [:, :, :96]
pos_embed                 [1, 197, 768]    → [:, :, :96]

attn.qkv.weight           [2304, 768]      → head-aware slice
  Q: w[0:96,   :96]
  K: w[768:864, :96]
  V: w[1536:1632, :96]
  → concat → [288, 96]

attn.proj.weight          [768, 768]       → [:96, :96]
mlp.fc1.weight            [3072, 768]      → [:384, :96]
mlp.fc2.weight            [768, 3072]      → [:96, :384]
norm.weight/bias          [768]            → [:96]
head.weight               [1000, 768]      → [:, :96]  (MRL과 동일)
기존 vit_96dim.py의 문제점 (이 아이디어 관점에서)
현재 코드는 96-dim forward를 훈련 중에 실행하지 않습니다. MRL head만 있고 backbone은 768-dim 단일로만 학습됩니다. 96_dim.py의 splicing은 훈련 감독 없이 사후에 자르는 것이라 weight가 이 구조를 지원하도록 학습되지 않았습니다.

구현 방향
SliceableViT: forward 시 dim 인자를 받아 해당 dim으로 모든 weight를 slice해서 실행하는 단일 모델
훈련 루프: 매 step마다 [768, 384, 96] 등 여러 dim으로 순차/병렬 forward → 각 loss 합산
num_heads 관리: dim별로 num_heads가 달라야 함 (96-dim: 3 heads, 768-dim: 12 heads)
LayerNorm: dim별 statistics 차이 문제 → sub-dim 전용 LN 파라미터 필요 또는 RMSNorm 고려