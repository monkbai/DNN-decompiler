import json


class TopoNode:
    """define node in the topology list"""
    id = 0
    func_name = ''
    op_label = ''
    input_list = []
    output_list = []
    input_ids = []
    occurrences = 0

    def __init__(self, id: int, func_name: str, op_label: str, input_list: list, output_list: list, input_ids: list, occur: int):
        self.id = id
        self.func_name = func_name
        self.op_label = op_label
        self.input_list = input_list
        self.output_list = output_list
        self.input_ids = input_ids
        self.occurrences = occur

    def to_list(self):
        return [self.id,
                self.func_name,
                self.op_label,
                self.input_list,
                self.output_list,
                self.input_ids,
                self.occurrences,
                ]

    def to_json(self, output_path: str):
        list_obj = self.to_list()
        j = json.dumps(list_obj, sort_keys=True, indent=4)
        with open(output_path, 'w') as f:
            f.write(j)
