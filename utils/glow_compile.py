#!/usr/bin/python3
import subprocess
import os


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


def run(prog_path):
    with cd(project_dir):
        # print(prog_path)
        proc = subprocess.Popen(prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


project_dir = rootdir = "./"
glow_cmd = '/export/d1/zliudc/TOOL/glow_build_Debug/bin/model-compiler -backend=CPU -model={} -emit-bundle=./{} -bundle-api=static -g'
glow_debug_cmd = '/export/d1/zliudc/TOOL/glow_build_Debug/bin/model-compiler -backend=CPU -model={} -emit-bundle=./{} -bundle-api=static -g -dump-ir-after-all-passes > {}'

main_file_path = '/export/d1/zliudc/GLOW/main.c'
compile_cmd = 'gcc ./main.c ./{}.h ./{}.o -o ./{}.out -no-pie -lm'  # model name


def debug_all_models():
    global project_dir
    file_count = 0
    for home, dirs, files in os.walk(rootdir):
        files.sort()
        for filename in files:
            if filename.endswith('.onnx'):
                # print(os.path.join(home, filename))
                out_dir = filename[:-5] + '_debug'
                out_log = filename[:-5] + '_ir.txt'
                project_dir = home
                if not os.path.exists(os.path.join(home, out_dir)):
                    print(os.path.join(home, filename))
                    status, output = cmd(glow_debug_cmd.format(filename, out_dir, out_log))
                    if status != 0:
                        print("error\n", output)
                    else:
                        file_count += 1
                
    print("file_count,", file_count)


def compile_all_models():
    global project_dir
    file_count = 0
    for home, dirs, files in os.walk(rootdir):
        files.sort()
        for filename in files:
            if filename.endswith('.onnx'):
                # print(os.path.join(home, filename))
                out_dir = filename[:-5]
                project_dir = home
                if not os.path.exists(os.path.join(home, out_dir)):
                    print(os.path.join(home, filename))
                    status, output = cmd(glow_cmd.format(filename, out_dir))
                    if status != 0:
                        print("error\n", output)
                    else:
                        file_count += 1
                
    print("file_count,", file_count)


def prepare_main_c(header_path: str, output_path: str):
    print('header file: {}'.format(header_path))
    print('output file: {}'.format(output_path))
    with open(main_file_path, 'r') as f:
        org_main = f.read()
        f.close()
    with open(header_path, 'r') as f:
        header_txt = f.read()
        holder_list = get_palceholder_name(header_txt)
        f.close()
    model_name = os.path.basename(header_path)
    model_name = model_name[:-2]
    UPPER_name = model_name.upper()
    new_main = org_main.replace('mnist_8', model_name)
    new_main = new_main.replace('MNIST_8', UPPER_name)
    new_main = new_main.replace('<INPUT_PLACEHOLDER>', holder_list[0])
    new_main = new_main.replace('<OUTPUT_PLACEHOLDER>', holder_list[1])
    with open(output_path, 'w') as f:
        f.write(new_main)
        f.close()


def get_palceholder_name(header_txt: str):
    header_txt = header_txt[header_txt.find('// Placeholder address'):]
    header_txt = header_txt[:header_txt.find('\n\n')]
    holder_list = []
    lines = header_txt.split('\n')
    
    if lines[1].endswith('0'):
        holder_list.append(lines[1].split(' ')[1])
        holder_list.append(lines[2].split(' ')[1])
    elif lines[2].endswith('0'):
        holder_list.append(lines[2].split(' ')[1])
        holder_list.append(lines[1].split(' ')[1])
    print(holder_list)
    return holder_list


def compile_all_binary():
    global project_dir
    file_count = 0
    for home, dirs, files in os.walk(rootdir):
        files.sort()
        for filename in files:
            if filename.endswith('.glow'):
                # print(os.path.join(home, filename))
                header_path = filename[:-5] + '.h'
                header_path = os.path.join(home, header_path)
                output_path = os.path.join(home, 'main.c') 
                binary_path = filename[:-5]
                project_dir = home
                prepare_main_c(header_path, output_path)
                
                model_name = filename[:-5]
                if not os.path.exists(os.path.join(home, model_name+'.out')):
                    status, output = cmd(compile_cmd.format(model_name, model_name, model_name))
                    if status != 0:
                        print(output)
                    else:
                        file_count += 1
    print("file_count,", file_count)


if __name__ == '__main__':
    debug_all_models()
    exit(0)
    # compile_all_models()
    compile_all_binary()
