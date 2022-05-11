from treelib import Tree,Node
import ast
import json
from io import StringIO


class global_tree():
    id = 0
    tree = Tree()
    @classmethod
    def clear(cls):
        cls.id = 0
        cls.tree = Tree()


def creat_tree(node):
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(ast.iter_child_nodes(node))
        add_main_Node(node)
        for i in ast.iter_child_nodes(node):
            add_Node(i,node)

       # yield node


def add_main_Node(node):
    tag = node.__class__.__name__
    id = global_tree.id
    data = node
    if global_tree.tree.root == None:
        global_tree.tree.create_node(tag=tag,identifier=id,data=data)
        global_tree.id+=1


def add_Node(child,parent):
    tag = child.__class__.__name__
    id = global_tree.id
    data = child
    #parent_hash = parent
    # if not global_tree.tree.get_node(child) and child.__class__.__name__!='Load':
    def find_id(tree,parent):
        for i in tree.expand_tree():
            if tree[i].data == parent:
                return i

    parent_id = find_id(global_tree.tree,parent)
    #if child.__class__.__name__ != 'Load':
    global_tree.tree.create_node(tag=tag,identifier=id,data=data,parent=parent_id)
    global_tree.id+=1

def sort_list(l):
    id_data = dict()
    for each in l:
        id_data[each['id']] = each
    l.clear()
    l = [id_data[i] for i in range(len(id_data))]
    return l


def tree2json():
    node_msg_list = []
    for node_id in global_tree.tree.expand_tree(mode=global_tree.tree.DEPTH):
        node = global_tree.tree[node_id]
        node_attr = dict()
        tree_id = global_tree.tree.identifier
        node_attr['id'] = node_id
        node_attr['type'] = node.data.__class__.__name__

        if len(node.successors(tree_id)):
            node_attr['children'] = node.successors(tree_id)
        for i in ast.iter_fields(node.data):
            if isinstance(i[1], (int, str)):
                node_attr['value'] = str(i[1])
        node_msg_list.append(node_attr)
    node_msg_list = sort_list(node_msg_list)
    node_msg_list = json.dumps(node_msg_list)
    return node_msg_list


if __name__ == '__main__':
    with open('python_code2.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    buf = StringIO()
    error = {}
    success = []
    import re

    for n,line in enumerate(lines):
        try:
            code = line.replace('\\n', '\n')
            code = code.replace('\\t','\t')

            ast_module = ast.parse(code)
            creat_tree(ast_module)
            json_data = tree2json()
            global_tree.clear()
            buf.write(json_data)
            buf.write('\n')
            print('finished!',n)
            success.append(n)
        except SyntaxError :
            print(n,'SyntaxError',line)
            error[n] = line[:50]+'....'

    with open('train_ast.txt', 'w') as f:
        f.write(buf.getvalue())
    buf.close()
    print('原文件行数:',len(lines))
    print('解析成功行数:',len(success))
    print('出错代码:',error)



# global_tree.tree.show(idhidden=False)

# from _collections import OrderedDict as odict
