import os

weights = '2022-12-22 20_17_24.876640_train'
model = 'SFDet-ResNet'
mode = 'test'
batch = 32
score_threshold = 0.01
use_gpu = 'True'

start = 140
save_step = 5
num_epochs = 300

for i in range(start + save_step, num_epochs + save_step, save_step):
    pretrained_model = '"{}/{}"'.format(weights, i)
    args = ('--mode {} --pretrained_model {} --model {} --use_gpu {} '
            '--batch_size {} --score_threshold {}')
    args = args.format(mode,
                       pretrained_model,
                       model,
                       use_gpu,
                       batch,
                       score_threshold)
    command = 'python main.py {}'.format(args)
    os.system(command)
