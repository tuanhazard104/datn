### Training command:
1) TransUNet with Efficientnet backbone
```bash 
python train.py --model_name "TransUNet" --img_size 224 --vit_name "EFN-B7" --n_skip 4
``` 

### Testing command:
1) TransUNet with Efficientnet backbone
```bash 
python test.py --vit_name "EFN-B7" --model_name "TransUNet" --img_size 224 --pretrained_model "path/to/your/model" --is_savenii
``` 