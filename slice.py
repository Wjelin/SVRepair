import html
import json
import os
from collections import OrderedDict

import networkx as nx
from tqdm import tqdm


class Node:
    def __init__(self, node_id, label, code, line_number, start_index, end_index):
        self.node_id: str = node_id
        self.label: str = label
        self.code: str = code
        self.line_number: int = line_number
        self.start_index: int = start_index
        self.end_index: int = end_index
        self.parents = set()
        self.children = set()

    def add_parent(self, parent_node):
        self.parents.add(parent_node)

    def add_child(self, child_node):
        self.children.add(child_node)

    def __repr__(self):
        return (f"Node(node_id={self.node_id}, label={self.label}, code={self.code}, line_number={self.line_number}, "
                f"start_index={self.start_index}, end_index={self.end_index})")

    def __eq__(self, other):
        return isinstance(other, Node) and (self.node_id == other.node_id and self.label == other.label and
                                            self.code == other.code and self.line_number == other.line_number and
                                            self.start_index == other.start_index and self.end_index == other.end_index)

    def __hash__(self):
        return hash((self.node_id, self.label, self.code, self.line_number, self.start_index, self.end_index))


def load_graph(dot_path, node_metadata, source_code):
    graph = nx.drawing.nx_agraph.read_dot(dot_path)
    node_objects = OrderedDict()

    for node_id, data in graph.nodes(data=True):
        try:
            label = html.unescape(data["label"][1: data["label"].find(',')])
            if label in ("METHOD", "BLOCK"):
                continue

            meta = node_metadata[node_id]
            code = meta["code"]

            if label == "METHOD_RETURN":
                code = meta["typeFullName"]
                if code == "ANY":
                    continue
                pos = code.find('*')
                while pos != -1 and code[pos - 1] != ' ':
                    code = code[:pos] + ' ' + code[pos:]
                    pos = code.find('*', pos + 2)

            line_number = meta["lineNumber"]
            start_index = meta["columnNumber"] - 1
            end_index = start_index + len(code)

            if code.endswith('...'):
                if len(code) >= 1000:
                    continue
                code = code[:-3]

            if code != source_code[start_index: end_index]:
                start_index = source_code.find(code, start_index)
                end_index = start_index + len(code)

            node_objects[node_id] = Node(node_id, label, code, line_number, start_index, end_index)
        except KeyError:
            print(f"Warning KeyError: Skip Node ID {node_id}")
            continue

    for src, dst in graph.edges():
        if src in node_objects and dst in node_objects:
            node_objects[src].add_child(node_objects[dst])
            node_objects[dst].add_parent(node_objects[src])

    return node_objects


def generate_slices(nodes, vul_ranges, direction, graph_type, max_depth=3):
    vul_nodes = [
        node for node in nodes.values()
        if any(node.start_index < vul_end and vul_start < node.end_index for vul_start, vul_end in vul_ranges)
    ]
    slices = []

    visited = set(vul_nodes)
    frontier = set(vul_nodes)
    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for node in frontier:
            if direction in ("both", "forward"):
                next_frontier.update(node.children)
            if direction in ("both", "backward"):
                next_frontier.update(node.parents)
        next_frontier.difference_update(visited)
        if not next_frontier:
            break
        slices.extend([{
            "start_index": node.start_index,
            "end_index": node.end_index,
            "code": node.code,
            "type": graph_type,
            "depth": depth
        } for node in next_frontier])
        visited.update(next_frontier)
        frontier = next_frontier

    return slices


def main():
    os.chdir(os.path.expanduser("~/data/vrepair_bug_data/train"))
    code_dirs = sorted([_dir for _dir in os.listdir() if _dir.isdigit() and os.path.isdir(_dir)], key=int)
    for code_dir in tqdm(code_dirs, total=len(code_dirs)):
        output_dir = os.path.join(code_dir, "output")

        with open(os.path.join(code_dir, f"{code_dir}.c"), 'r') as file:
            source_code = file.read()
        with open(f"{code_dir}/vulnerabilities.json", 'r') as file:
            vulnerabilities = json.load(file)

        vul_ranges = [(vul["start_index"], vul["end_index"]) for vul in vulnerabilities]
        slices = [{
            "start_index": vul["start_index"],
            "end_index": vul["end_index"],
            "code": vul["vul_code"],
            "type": "ROOT",
            "depth": 0
        } for vul in vulnerabilities]

        try:
            with open(os.path.join(output_dir, "nodes.json"), 'r') as file:
                node_metadata = {str(node["_id"]): node for node in json.load(file)}

            ddg_file = next((os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_ddg.dot")),
                            None)
            cdg_file = next((os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_cdg.dot")),
                            None)
            cfg_file = next((os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_cfg.dot")),
                            None)
            if not all([ddg_file, cdg_file, cfg_file]):
                print(f"Warning: Missing graph files in {code_dir}")
                raise FileNotFoundError

            ddg_nodes = load_graph(ddg_file, node_metadata, source_code)
            cdg_nodes = load_graph(cdg_file, node_metadata, source_code)
            cfg_nodes = load_graph(cfg_file, node_metadata, source_code)

            slices.extend(generate_slices(ddg_nodes, vul_ranges, "both", "DDG"))
            slices.extend(generate_slices(cdg_nodes, vul_ranges, "backward", "CDG"))
            slices.extend(generate_slices(cfg_nodes, vul_ranges, "forward", "CFG"))
        except FileNotFoundError:
            print(f"file {os.path.join(output_dir, 'nodes.json')} not found")

        with open(os.path.join(code_dir, "slices.json"), 'w', encoding='utf-8') as file:
            json.dump(slices, file, indent=4)


if __name__ == "__main__":
    main()
