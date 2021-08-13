import json
import os

param_config = {'Reshape': ['in', 'out'],
                'Pad': ['na', 'in', 'param', 'out'],
                'Conv': ['na', 'in', 'param', 'out'],
                'BatchNorm': ['param', 'param', 'in', 'param', 'param', 'out'],
                'ReLU': ['na', 'in', 'out'],
                'Pool': ['na', 'in', 'out'],
                'Add ReLU': ['na', 'in', 'in', 'out'],
                'Broadcast': ['in', 'out'], 
                'Sum': ['in', 'out'],
                'Divide': ['na', 'in', 'param', 'out'],
                'Dense': ['na', 'in', 'param', 'out'],
                'Add': ['na', 'in', 'param', 'out'],
               }

addr2func = dict()

def json_to_dict(json_path: str):
    if not os.path.exists(json_path):
        return dict()
    with open(json_path, 'r') as f:
        j_txt = f.read()
        dict_obj = json.loads(s=j_txt)
        return dict_obj


def get_trace():
    op_list = []
    with open("/export/d1/zliudc/DLE_Decompiler/TVM/scripts/nnfusion_resnet/func_trace.log", 'r') as f:
        lines = f.readlines()
        for l in lines:
            if len(l.strip()) > 0 and not l.startswith("#"):
                addr = l[:l.find(':')].strip()
                addr = int(addr, 16)
                params = l[l.find(':')+1:].strip()
                params = params.split(',')
                op_list.append((str(addr), params))
    print(op_list)
    return op_list


def print_id(op_list: list):
    global addr2func
    addr2func = json_to_dict('addr2func.json')

    new_op_list = []
    for i in range(len(op_list)):
        cur_op = op_list[i]

        if cur_op[0] not in addr2func:
            continue
        cur_type = addr2func[cur_op[0]]
        if '-->' in cur_type:
            cur_type = cur_type[cur_type.find('>')+1:].strip()
        if 'ignored' in cur_type:
            continue
        
        if cur_type not in param_config:
            print(cur_type)
            assert cur_type in param_config
        param_type = param_config[cur_type]
        
        in_param = []
        out_param = []
        if 'Reshape' in cur_type or 'Broadcast' in cur_type:
            continue
        for j in range(len(param_type)):
            if param_type[j].startswith('in'):
                in_param.append(cur_op[1][j])
            elif param_type[j].startswith('out'):
                out_param.append(cur_op[1][j])
        assert len(out_param) == 1
        new_op_list.append((hex(int(cur_op[0]))+'_'+cur_type, in_param, out_param))
    
    print('new_op_list', new_op_list)

    id_list = []
    for i in range(len(new_op_list)):
        if i == 0:
            id_list.append([])
            continue

        tmp_id_list = []
        for in_p in new_op_list[i][1]:
            for j in range(i-1, -1, -1):
                if new_op_list[j][2][0] == in_p:
                    tmp_id_list.append(j)
                    break
        id_list.append(tmp_id_list)

    print('id_list', id_list)


if __name__ == '__main__':
    op_list = get_trace()
    print_id(op_list)
