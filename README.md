# EfficientNet_v2_pytorch

- 기본 Model 구현 완료
  - Forward 테스트까지 진행

- model.txt 에 Summary (3, 224, 224 ) 결과 출력


## Copilot 만세



# train.py 구현
- 아직 argparser 는 구현안됨
- config 활용

```
!python train.py -e 1 -b 512 --data CIFAR10 --lr 0.001 --rand_mag 5 --warmup 10 --dropout 0.2 --decay 0.1 
```