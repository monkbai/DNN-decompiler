import os


def clean_asm_code(asm_txt: str):
    lines = asm_txt.split('\n')
    new_lines = []
    for line in lines:
        if not line.startswith('.text:'):
            continue
        elif 'S U B R O U T I N E' in line or '  proc ' in line:
            pass
        elif ' ' not in line:
            continue
        elif line[line.find(' ')+1] == ' ':
            continue
        elif '        align ' in line:
            continue

        line = line[6:]  # .text:
        line = '0x' + line.lstrip('0')
        line = line.replace(' ', ': ', 1)
        # if ';' in line:
        #     line = line[:line.find(';')]

        line = line.strip(' ')
        new_lines.append(line)
    new_asm_txt = '\n'.join(new_lines)
    return new_asm_txt + '\n'


def split_funcs(asm_path: str, output_dir: str):
    asm_path = os.path.abspath(asm_path)
    output_dir = os.path.abspath(output_dir)
    with open(asm_path, 'r') as f:
        asm_txt = f.read()
        funcs_list = []
        current_func = []
        current_name = ''
        asm_lines = asm_txt.split('\n')
        index = 0
        while index < len(asm_lines):
            line = asm_lines[index]
            if 'S U B R O U T I N E' in line:
                if len(current_func) > 0:
                    funcs_list.append((current_name, current_func))
                current_func = []
                current_name = ''
                current_func.append(line[line.find(';'):])
                index += 1
                line = asm_lines[index]
                line = line[line.find(':')+1:]
                line = line.strip(' ')
                current_func.append('; ' + line)
                line = line[:line.find(' ')]
                current_name = line
            else:
                current_func.append(line)
            index += 1
    func_index = 0
    for func in funcs_list:
        func_name = func[0]
        func_lines = func[1]
        func_txt = '\n'.join(func_lines)
        func_txt += '\n'
        file_path = '{:0>4d}.{}.txt'.format(func_index, func_name)
        file_path = os.path.join(output_dir, file_path)
        with open(file_path, 'w') as f:
            f.write(func_txt)
            f.close()
        func_index += 1


def handle_lst_file(lst_path: str, asm_path: str, output_dir: str):
    lst_path = os.path.abspath(lst_path)
    asm_path = os.path.abspath(asm_path)
    output_dir = os.path.abspath(output_dir)
    with open(lst_path, 'r') as f:
        asm_txt = f.read()
        new_asm_txt = clean_asm_code(asm_txt)
        fw = open(asm_path, 'w')
        fw.write(new_asm_txt)
        fw.close()
    split_funcs(asm_path, output_dir)


if __name__ == '__main__':
    lst_file = '/home/lifter/Documents/tvm_output/scripts/nnfusion/vgg11_nnfusion_strip.lst'
    asm_file = '/home/lifter/Documents/tvm_output/scripts/nnfusion/vgg11_nnfusion_strip.asm'
    output_dir = '/home/lifter/Documents/tvm_output/vgg11_nnfusion_funcs/'
    handle_lst_file(lst_file, asm_file, output_dir)

