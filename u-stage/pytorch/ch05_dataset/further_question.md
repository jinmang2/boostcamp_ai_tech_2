# Further Question

## Dataset & Dataloader
- DataLoader에서 사용할 수 있는 각 sampler들을 언제 사용하면 좋을지 같이 논의해보세요!
- 데이터의 크기가 너무 커서 메모리에 한번에 올릴 수가 없을 때 Dataset에서 어떻게 데이터를 불러오는게 좋을지 같이 논의해보세요!

### Q1) Sampler
`torch.utils.data.sampler.py`
- 아래 [코드](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/u-stage/pytorch/ch05_dataset/samplers.py)는 stable 단계에 진입하여 1.7대 버전과 1.9대 버전에서 코드가 동일함을 확인함!
- `Sampler`: base class
- `SequentialSampler`: 단순한 for loop! eval 혹은 test dataset에 사용
- `RandomSampler`: random sampling! 복원 / 비복원 다 가능
- `SubsetRandomSampler`: 직접 indices를 넣어줌! replacement X
- `WeightedRandomSampler`: sample별 가중치의 확률로 sampling (multinomial)
- `SortishSampler`: sentence length를 indices로 넣어주면 이를 sorting해서 sampling
- `BatchSampler`: 위의 sampler들을 mini-batch화 시켜줌
- `DistributedSampler`: DDP를 위한 sampler class


### Q2) Distributed Data Parallelism
- 아직은 잘 몰라서, 공부를 많이 해야겠다!
    - [현웅이 영상](https://youtu.be/w4a-ARCEiqU)
