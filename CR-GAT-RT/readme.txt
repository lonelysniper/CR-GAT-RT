

how to test(attack):
python3 test.py --binname=fcn_voc.pth --config=configvoc_fcn.json


how to test(defense):
python3 testdefense.py --binname=fcn_voc_def.pth --config=configvoc_fcn.json

how to train :
python train.py --config=configvoc_fcn.json

for normal model training , call  _train_normal in ./base/base_trainer.py
for CR Robustness training , call  _train_CR in ./base/base_trainer.py
for Fast-ADT training , call _train_fastadt in ./base/base_trainer.py
