import os


def merge_batchnorm(trace_log_path: str, new_log_path: str):
    trace_log_path = os.path.abspath(trace_log_path)
    trace_log_path = os.path.abspath(trace_log_path)
    lines_list = []
    with open(trace_log_path, 'r') as f:
        lines = f.read().strip()
        lines = lines.split('\n')
        for line in lines:
            l_list = []
            l_list.append(line.split(',')[0].strip())
            l_list.append(line.split('-')[0].split(',')[1].strip())
            l_list.append(line.split('-')[1].split(':')[0].strip())
            args = line.split(':')[1].strip().strip(',').split(',')
            l_list += args
            lines_list.append(l_list)
    new_lines_list = []
    idx = 0
    while idx < len(lines_list):
        l_list = lines_list[idx]
        if l_list[1] == 'add' and lines_list[idx+1][1] == 'sqrt':
            in_arg = lines_list[idx+5][3]
            out_arg = lines_list[idx+10][5]
            l_list = ['0x400000', 'batchnorm', '0x400000', in_arg, out_arg]
            idx += 10
        new_lines_list.append(l_list)
        idx += 1
    with open(new_log_path, 'w') as f:
        for l_list in new_lines_list:
            f.write('{}, {: <18} - {}: '.format(l_list[0], l_list[1], l_list[2]))
            idx = 3
            while idx < len(l_list):
                f.write('{},'.format(l_list[idx]))
                idx += 1
            f.write('\n')


def extract_batchnorm(trace_log_path: str, new_log_path: str):
    trace_log_path = os.path.abspath(trace_log_path)
    trace_log_path = os.path.abspath(trace_log_path)
    lines_list = []
    with open(trace_log_path, 'r') as f:
        lines = f.read().strip()
        lines = lines.split('\n')
        for line in lines:
            l_list = []
            l_list.append(line.split(',')[0].strip())
            l_list.append(line.split('-')[0].split(',')[1].strip())
            l_list.append(line.split('-')[1].split(':')[0].strip())
            args = line.split(':')[1].strip().strip(',').split(',')
            l_list += args
            lines_list.append(l_list)
    new_lines_list = []
    idx = 0
    while idx < len(lines_list):
        l_list = lines_list[idx]
        if l_list[1] == 'add' and lines_list[idx+1][1] == 'sqrt':
            in_arg = lines_list[idx+5][3]
            out_arg = lines_list[idx+10][5]
            l_list = ['0x400000', 'batchnorm', '0x400000', in_arg, out_arg]
            new_lines_list += lines_list[idx:idx+11]
            idx += 10
        # new_lines_list.append(l_list)
        idx += 1
    with open(new_log_path, 'w') as f:
        count = 0
        for l_list in new_lines_list:
            f.write('{}, {: <18} - {}: '.format(l_list[0], l_list[1], l_list[2]))
            idx = 3
            while idx < len(l_list):
                f.write('{},'.format(l_list[idx]))
                idx += 1
            f.write('\n')
            count += 1
            count = count % 11
            if count == 0:
                f.write('\n')


def input_ids(trace_log_path: str):
    with open(trace_log_path, 'r') as f:
        lines = f.read().strip().split('\n')
    layer_list = []
    line_list = []
    in_list = []
    prev_layer = ''
    prev_args = []
    for line in lines:
        layer = line.split('-')[0].strip()
        layer = layer.split(',')[1].strip()
        if layer == 'add':
            prev_layer = layer
            prev_args = line.split(':')[1].strip().strip(',').split(',')
            continue
        layer_list.append(layer)
        args = line.split(':')[1].strip()
        args = args.strip(',')
        args = args.split(',')
        if prev_layer == 'add':
            args = prev_args[:-1] + args[-1:]
        line_list.append(args)
        in_list.append([])
        prev_args = args
        prev_layer = layer

    for i in range(len(line_list)):
        out_addr = line_list[i][-1]
        for j in range(i+1, len(line_list)):
            in_addrs = line_list[j]
            in_addrs = in_addrs[:-1]
            if out_addr in in_addrs:
                in_list[j].append(i)
            if line_list[j][-1] == out_addr:
                break
    for i in range(len(line_list)):
        print('{},{}:'.format(layer_list[i], i), end=' ')
        for id in in_list[i]:
            print('{},'.format(id), end=' ')
        print('')

func_meta_data = [('0057.function_402ea0.txt', (64, 3, 7, 7), '', 'conv2d'),
                    ('0080.function_409280.txt', (64, 64, 3, 3), '', 'conv2d'),
                    ('0088.function_40d190.txt', (256, 128, 1, 1), '', 'conv2d'),
                    ('0092.function_40ff40.txt', (128, 64, 1, 1), '', 'conv2d'),
                    ('0119.function_415810.txt', (512, 512, 3, 3), '', 'conv2d'),
                    ('0144.function_41c230.txt', (128, 128, 3, 3), '', 'conv2d'),
                    ('0151.function_420600.txt', (256, 128, 3, 3), '', 'conv2d'),
                    ('0171.function_4262e0.txt', (512, 256, 3, 3), '', 'conv2d'),
                    ('0182.function_42aec0.txt', (128, 16, 6, 6), '', 'conv2d'),
                    ('0189.function_42ec00.txt', (256, 256, 3, 3), '', 'conv2d'),
                    ('0199.function_432d50.txt', (512, 256, 1, 1), '', 'conv2d'),

                  ]


def print_conv2d_shape(trace_log_path: str):
    with open(trace_log_path, 'r') as f:
        lines = f.read().strip().split('\n')
        idx = 0
        for line in lines:
            addr = line.split(',')[0][2:]
            label = line.split('-')[0].strip().split(',')[1].strip()
            print('{}: {}:'.format(idx, label), end=' ')
            for func_name, shape, _, _ in func_meta_data:
                if addr in func_name:
                    print('{}'.format(shape), end='')
            print('')
            if label != 'add':
                idx += 1


if __name__ == '__main__':
    # merge_batchnorm('./resnet18_strip_func_call_fused_2.log', './resnet18_strip_func_call_fused_3.log')
    # extract_batchnorm('./resnet18_strip_func_call_fused_2.log', './resnet18_strip_func_call_fused_4.log')
    # input_ids('./resnet18_strip_func_call_fused_3.log')
    print_conv2d_shape('./resnet18_strip_func_call_fused_3.log')
