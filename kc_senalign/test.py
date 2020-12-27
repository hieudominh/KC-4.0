import os

for i in range(73, 102, 1):
    vi = './raw/vi' + str(i) + '.txt'
    km = './raw/km' + str(i) + '.txt'
    out = './out/o' + str(i) + '.txt'
    command = 'python3 senalign.py -s ' + vi + ' -t ' + km + ' -o ' + out + ' -lang km -pair 1'
    print(command)
    os.system(command)
    print('Done!\n')
