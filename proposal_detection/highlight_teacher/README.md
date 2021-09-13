## Training and Testing

All configurations of modal are saved in opts.py, where you can modify training and model parameter.

1. To train the modal:
```
python main.py --mode train
```

2. To get the inference proposal of the validation videos and evaluate the proposals with recall and AUC:
```
python main.py --mode inference
```

Of course, you can complete all the process above in one line: 

```
sh bmn.sh
```


