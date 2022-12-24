### Training command:
1) TransUNet
```bash 
python train.py --model_name "TransUNet" --img_size 256
``` 
2) SegFormer
```bash 
python train.py --model_name "SegFormer" --img_size 224
``` 
3) Medical Transformer
```bash 
python train.py --model_name "MedT" --img_size 128
``` 
### Testing command:
1) TransUNet
```bash 
python test.py --model_name "TransUNet" --img_size 256 --is_savenii
``` 
2) SegFormer
```bash 
python test.py --model_name "SegFormer" --img_size 224 --is_savenii
``` 
3) Medical Transformer
```bash 
python test.py --model_name "MedT" --img_size 128 --is_savenii
``` 