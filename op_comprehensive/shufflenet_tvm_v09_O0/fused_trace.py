import os
from utils import list_to_json, dict_to_json, json_to_list, json_to_dict


def fuse_batchnorm(topo_list: list, func_meta_data: dict):
    list_to_json(topo_list, './topo_list.json')
    op_type_list = []
    for i in range(len(topo_list)):
        op_type_list.append(topo_list[i][2])
    dict_to_json(func_meta_data, './meta_data.json')

    new_topo_list = []
    i = 0
    while i < len(topo_list):
        op = topo_list[i]
        new_op = op
        op_id = op[0]
        func = op[1]
        fun_type = op[2]
        inputs = op[3]
        output = op[4]
        input_ids = op[5]
        op_count = op[6]
        # [add, sqrt, divide, multiply(multiply), multiply, negative, multiply, add(add), add]
        if i < len(op_type_list)-11 and fun_type == 'add' \
                                    and op_type_list[i+1] == 'sqrt' \
                                    and op_type_list[i+2] == 'divide' \
                                    and op_type_list[i+3] == 'multiply':
            new_op[1] += '-{},{}-{},{}-{},{}-{}'.format(topo_list[i][6],
                                                        topo_list[i+3][1], topo_list[i+3][6],
                                                        topo_list[i+6][1], topo_list[i+6][6],
                                                        topo_list[i+8][1], topo_list[i+8][6])
            new_op[2] = 'batchnorm'
            new_op[3] = [topo_list[i+5][3][0]]
            new_op[4] = topo_list[i+10][4]
            new_op[5] = [topo_list[i + 5][5][0]]
            new_op[6] = 0
            i += 10
            for j in range(i+1, len(topo_list)):
                if i in topo_list[j][5]:
                    # my_list = ['new item' if i == 'old item' else i for i in my_list]
                    topo_list[j][5] = [i-10 if k == i else k for k in topo_list[j][5]]
        new_topo_list.append(new_op)
        i += 1

    new_meta_data = []
    meta_data_list = list(func_meta_data.values())
    i = 0
    while i < len(meta_data_list):
        md = meta_data_list[i]
        new_md = md
        func = md[0]
        shape = md[1]
        addr = md[2]
        fun_type = md[3]
        argument_index = md[6]
        # [add, sqrt, divide, multiply(multiply), multiply, negative, multiply, add(add), add]
        # 0 --> var, 3 --> gamma, 6 --> mean, 8 --> beta
        if i < len(op_type_list)-11 and fun_type == 'add' \
                                    and op_type_list[i + 1] == 'sqrt' \
                                    and op_type_list[i + 2] == 'divide' \
                                    and op_type_list[i + 3] == 'multiply':
            new_md[3] = 'var'
            new_meta_data.append(new_md)

            new_md = meta_data_list[i + 3]
            new_md[3] = 'gamma'
            new_md[1] = meta_data_list[i][1]
            new_meta_data.append(new_md)

            new_md = meta_data_list[i + 6]
            new_md[3] = 'mean'
            new_md[1] = meta_data_list[i][1]
            new_meta_data.append(new_md)

            new_md = meta_data_list[i + 8]
            new_md[3] = 'beta'
            new_md[1] = meta_data_list[i][1]
            new_meta_data.append(new_md)

            i += 11
            continue
        new_meta_data.append(new_md)
        i += 1

    list_to_json(new_topo_list, './new_topo_list.json')
    list_to_json(new_meta_data, './new_meta_data.json')

    return new_meta_data


if __name__ == '__main__':
    # merge_batchnorm('./resnet18_strip_func_call_fused_2.log', './resnet18_strip_func_call_fused_3.log')
    # extract_batchnorm('./resnet18_strip_func_call_fused_2.log', './resnet18_strip_func_call_fused_4.log')
    # input_ids('./resnet18_strip_func_call_fused_3.log')

    #print_conv2d_shape('./resnet18_strip_func_call_fused_3.log')

    topo_list = json_to_list('./topo_list.json')
    meta_data = json_to_dict('./meta_data.json')
    for i in topo_list:
        print(i)
    fuse_batchnorm(topo_list, meta_data)
