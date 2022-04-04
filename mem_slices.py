#!/usr/bin/python3
import time
import logging

print('get logger: {}'.format('decompiler.' + __name__))
logger = logging.getLogger('decompiler.' + __name__)


def memory_slices(mem_read_trace: str):
    mem_obj = dict()  # id --> mem_obj(start, end)
    end_map = dict()  # end_addr --> id
    start_map = dict()  # start_addr --> id
    next_id = 0
    start_time = time.time()
    with open(mem_read_trace, 'r') as f:
        mem_read_txt = f.read()
        lines = mem_read_txt.split('\n')
        lines = list(set(lines))
        addr_lists = []
        for line in lines:
            line = line.strip()
            if not line.startswith('0x'):
                continue
            mem_addr, mem_size = line.split(',')
            start_addr = int(mem_addr, 16)
            end_addr = start_addr + int(mem_size)
            addr_lists.append((start_addr, end_addr))
        addr_lists = sorted(addr_lists, key=lambda x: x[0])

        for start_addr, end_addr in addr_lists:
            if start_addr in end_map.keys() and end_addr in start_map.keys():  # []<current_mem_obj>[]
                id_1 = end_map[start_addr]
                id_2 = start_map[end_addr]
                start1, end1 = mem_obj[id_1]
                start2, end2 = mem_obj[id_2]
                new_start = start1
                new_end = end2
                # update
                mem_obj.pop(id_1)
                mem_obj.pop(id_2)
                new_id = id_1  # reuse old id
                mem_obj[new_id] = (new_start, new_end)
                start_map[new_start] = new_id
                start_map.pop(start2)
                end_map[new_end] = new_id
                end_map.pop(end1)

            elif start_addr in end_map.keys():  # []<>
                mem_id = end_map[start_addr]
                old_start, old_end = mem_obj[mem_id]
                new_end = end_addr
                new_start = old_start
                # update
                mem_obj[mem_id] = (new_start, new_end)
                end_map.pop(start_addr)
                end_map[end_addr] = mem_id

            elif end_addr in start_map.keys():  # <>[]
                mem_id = start_map[end_addr]
                old_start, old_end = mem_obj[mem_id]
                new_start = start_addr
                new_end = old_end
                # update
                mem_obj[mem_id] = (new_start, new_end)
                start_map.pop(end_addr)
                start_map[start_addr] = mem_id
            elif start_addr in start_map.keys():  # <[]>
                mem_id = start_map[start_addr]
                old_start, old_end = mem_obj[mem_id]
                if end_addr > old_end:
                    mem_obj[mem_id] = (old_start, end_addr)

            else:
                mem_obj[next_id] = (start_addr, end_addr)
                start_map[start_addr] = next_id
                end_map[end_addr] = next_id
                next_id += 1

        # finally, merge adjacent mem objects
        old_mem_objs = list(mem_obj.values())
        old_mem_objs = sorted(old_mem_objs, key=lambda x: x[0])
        new_mem_objs = []
        for old_obj in old_mem_objs:
            if len(new_mem_objs) == 0:
                new_mem_objs.append(old_obj)
            elif old_obj[0] <= new_mem_objs[-1][1] < old_obj[1]:
                new_mem_objs[-1] = (new_mem_objs[-1][0], old_obj[1])
            elif old_obj[1] <= new_mem_objs[-1][1]:
                continue
            elif new_mem_objs[-1][1] < old_obj[0]:
                new_mem_objs.append(old_obj)
            else:
                print('unexpected')
        # For Debugging
        # if len(new_mem_objs) < 20:
        #     for start_addr, end_addr in new_mem_objs:
        #         print('[{}, {}] {}'.format(hex(start_addr), hex(end_addr), hex(end_addr - start_addr)))
        # else:
        #     for start_addr, end_addr in new_mem_objs[:20]:
        #         print('[{}, {}] {}'.format(hex(start_addr), hex(end_addr), hex(end_addr - start_addr)))
        #     print('......')
    end_time = time.time()
    print('Memory Clustering Time: {}s'.format(end_time - start_time))
    logger.info('Memory Clustering time - {}s'.format(end_time - start_time))
    return new_mem_objs


def filter_mem_regions(mem_read_regions: list, mem_write_regions: list):
    new_read_mem_regions = []
    for i in range(len(mem_read_regions)):
        for j in range(len(mem_write_regions)):
            in_mem = mem_read_regions[i]
            out_mem = mem_write_regions[j]
            if in_mem[1] == out_mem[1]:
                if in_mem[0] < out_mem[0] < in_mem[1]:
                    in_mem = (in_mem[0], out_mem[0])
                    new_read_mem_regions.append(in_mem)
                elif in_mem[0] >= out_mem[0]:
                    pass  # overlapped by out mem
            elif in_mem[0] == out_mem[0]:
                if in_mem[0] < out_mem[1] < in_mem[1]:
                    in_mem = (out_mem[1], in_mem[1])
                    new_read_mem_regions.append(in_mem)
                elif in_mem[1] <= out_mem[1]:
                    pass  # overlapped by out mem
            else:
                new_read_mem_regions.append(in_mem)
    return new_read_mem_regions


if __name__ == '__main__':
    # memory_slices("/home/lifter/pin-3.14-98223-gb010a12c6-gcc-linux/source/tools/MyPinTool/tmp.log")
    memory_slices("../mem_write.log")
