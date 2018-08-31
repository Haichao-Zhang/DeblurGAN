export PYTHONPATH=./:$PYTHONPATH
python ./test_multi.py --dataroot  /media/DATA/data/blurred_sharp_org/blurred_sharp/db_test/ --gpu_ids 1 --which_epoch 5
