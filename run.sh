# vanilla
python vanilla.py  ## vanilla 5 classes
python vanilla.py --removed_classes 2 3 4 --log_name "2classes_vanilla" ## vanilla 2 classes

python vanilla.py --mode analysis --version "xxx" --best_cpkt "xxx" ## analysis 5 classes
python vanilla.py --mode analysis --version "xxx" --best_cpkt "xxx" --removed_classes 2 3 4 --log_name "2classes_vanilla" ## analysis 2 classes

# dann
python dann.py  ## dann 5 classes
python dann.py --removed_classes 2 3 4 --log_name "2classes_dann" ## dann 2 classes

python dann.py --mode analysis --version "xxx" --best_cpkt "xxx" ## analysis 5 classes
python dann.py --mode analysis --version "xxx" --best_cpkt "xxx" --removed_classes 2 3 4 --log_name "2classes_dann" ## analysis 2 classes

# cdan
python cdan.py  ## cdan 5 classes
python cdan.py --removed_classes 2 3 4 --log_name "2classes_cdan" ## cdan 2 classes

python cdan.py --mode analysis --version "xxx" --best_cpkt "xxx" ## analysis 5 classes
python cdan.py --mode analysis --version "xxx" --best_cpkt "xxx" --removed_classes 2 3 4 --log_name "2classes_cdan" ## analysis 2 classes
