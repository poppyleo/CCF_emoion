import os



folder_path = '/home/none404/hm/lei/data/emotion/Savemodel/'#学校
# folder_path = '/data/wangzhili/lei/nCoV/Savemodel/'
# folder_path = '/home/none404/hm/lei/data/emotion/Savemodel/'
for folder_l1 in os.listdir(folder_path):
    for folder_l2 in os.listdir(folder_path + folder_l1):
        result_list = folder_path + folder_l1 + '/' + folder_l2 + '/' + 'checkpoint'
        if os.path.exists(folder_path + folder_l1 + '/' + folder_l2 + '/' ):
            max_f1 = 0
            max_p = max_r = 0
            max_l = ''
            try:
                with open(result_list, encoding='utf-8') as file:
                    for l in file.readlines():
                        line = l.strip().split('_')
                        p = float(line[-2]) + 1e-10
                        r = float(line[-1].split('-')[0]) + 1e-10
                        f1 = 2 * p * r / (p + r)
                        if f1 > max_f1:
                            max_f1 = f1
                            max_p = p
                            max_r = r
                            max_l = l
                print(
                    '{}/{} : f1 {} , p {} , r {}'.format(folder_l1,
                                                               '/'.join(max_l.split('"')[-2].split('/')[-2:]),
                                                               max_f1,
                                                               max_p, max_r))
            except:
                pass