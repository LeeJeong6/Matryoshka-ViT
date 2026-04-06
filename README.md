Swin에 적용하기 전에 ViT에 적용을 한다면 좋을 듯 해서 적용함
```
 단일 GPU 테스트

python vit_matryoshka_subdim.py \
    --sub-dims 96 192 384 768 --sub-heads 3 6 6 12 \
    --epochs 1 --batch-size 64 --no-ddp

# 6-GPU 정식 훈련
torchrun --nproc_per_node=6 vit_matryoshka_subdim.py \
    --sub-dims 96 192 384 768 --sub-heads 3 6 6 12 --epochs 90