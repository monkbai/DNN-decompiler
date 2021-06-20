#!/usr/bin/python3
import os
import sys


def split_tvm_asm(asm_txt: str):
    def get_funcs_list(txt: str):
        txt_list = txt.split('\n')
        funcs_list = []
        current_func_name = ''
        current_func_body = ''
        in_func_flag = False
        for line in txt_list:
            if line.startswith('; function: '):
                func_name = line[line.find('function:') + 9:]
                func_name = func_name[:func_name.rfind(' at ')]
                func_name = func_name.strip()
                if in_func_flag:
                    funcs_list.append((current_func_name, current_func_body))
                in_func_flag = True
                current_func_name = func_name
                current_func_body = line + '\n'
            elif line.startswith('; data') and in_func_flag:
                asm_code = line[:].strip()
                current_func_body += asm_code + '\n'
            elif line.startswith('; '):
                if in_func_flag:
                    funcs_list.append((current_func_name, current_func_body))
                    in_func_flag = False
                    current_func_name = ''
                    current_func_body = ''
            elif in_func_flag:
                asm_code = line[50:].strip()
                if not asm_code.endswith('|'):
                    current_func_body += line + '\n'
        return funcs_list

    def filt_funcs(funcs_list):
        new_funcs_list = []
        start_flag = False
        for func_name, func_body in funcs_list:
            if func_name.startswith('fused') and not start_flag:
                start_flag = True
            elif start_flag and not func_name.startswith('function') and not func_name.startswith('fused'):
                start_flag = False

            if start_flag:
                new_funcs_list.append((func_name, func_body))
        return new_funcs_list

    asm_txt = asm_txt[asm_txt.find(';; Code Segment'):]
    asm_txt = asm_txt[:asm_txt.find(';; Data Segment')]
    print('lines {}'.format(asm_txt.count('\n')))

    funcs_list = get_funcs_list(asm_txt)
    count1 = len(funcs_list)
    print('{} funcs in total'.format(count1))

    """
    funcs_list = filt_funcs(funcs_list)
    count2 = len(funcs_list)
    print('{} funcs after filteration'.format(count2))
    print('{} funcs have been ignored'.format(count1 - count2))
    if count1 - count2 != 154:
        input('it is not 154, continue?')
    # print(funcs_list)
    """
    return funcs_list


def save_tvm_funcs(funcs_list, output_dir):
    writen_funcs_count = 0
    current_index = 0
    for func_name, func_body in funcs_list:
        output_path = os.path.join(output_dir, '{:0>4d}.{}.txt'.format(current_index, func_name))
        with open(output_path, 'w') as f:
            f.write(func_body)
            current_index += 1
            print('written {}'.format(output_path))
            writen_funcs_count += 1
    print('{} funcs have been written'.format(writen_funcs_count))


def test():
    asm_txt = open('/home/lifter/Documents/tvm_output/O3/funcs/demo_static_O3_exe.dsm').read()
    funcs = split_tvm_asm(asm_txt)
    save_tvm_funcs(funcs, '/home/lifter/Documents/tvm_output/O3/funcs/')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        asm_path = sys.argv[1]
        output_dir = sys.argv[2]
        print("asm file: {}, continue".format(asm_path))
        print("output dir: {}, continue".format(output_dir))
        asm_txt = open(asm_path, 'r').read()
        funcs = split_tvm_asm(asm_txt)
        save_tvm_funcs(funcs, output_dir)
    else:
        print("usage: <this_script.py> <asm file> <output dir>")
