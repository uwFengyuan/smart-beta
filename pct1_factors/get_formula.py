from definition import *


def tree_to_formula(tree):
    if isinstance(tree, node):
        if tree.name in ['+', '-', '*', '/']:
            string_1 = tree_to_formula(tree.children[0])
            string_2 = tree_to_formula(tree.children[1])
            return str('(' + string_1 + tree.name + string_2 + ')')
        else:
            result = [tree.name, '(']
            for i in range(len(tree.children)):
                string_i = tree_to_formula(tree.children[i])
                result.append(string_i)
                result.append(',')
            result.pop()
            result.append(')')
            return ''.join(result)
    elif isinstance(tree, paramnode):
        return str(tree.name)
    else:
        return str(tree.name)

def check_sanity(formula):
    count = 0
    for i in range(len(formula)):
        if formula[i] == '(':
            count += 1
        if formula[i] == ')':
            count -= 1
        if count < 0:
            return 0
    if count != 0:
        return 0
    return 1

def formula_to_tree(formula, var_names):
    while(1):
        if formula[0] == '(':
            if check_sanity(formula[1:-1]):
                formula = formula[1:-1]
            else:
                break
        else:
            break
    if formula in var_names:
        return paramnode(formula)
    elif formula.isdigit():
        return constnode(int(formula))
    elif formula[0] == '0' and formula[1] == '.' and len(formula) < 5:
        return constnode(float(formula))
    var_stack = []
    op_stack = []
    count = 0
    pointer = 0
    k = 0
    while k < len(formula):
        if formula[k] == '(' and count == 0:
            for i in range(k, len(formula)):
                if formula[i] == '(': count += 1
                elif formula[i] == ')': count -= 1
                if count == 0: break
            index = i  # index points at ')'
            tree_k = formula_to_tree(formula[k:index+1], var_names)
            var_stack.append(tree_k)
            k = i
        if formula[k] in '+-*/?':
            op = formula[k]
            op_stack.append(op)
            pointer = k+1
        if formula[k] == ':':
            pointer = k+1
            count = 0
            flag = 0
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    count += 1
                    flag = 1
                elif formula[i] == ')':
                    count -= 1
                if count == 0 and flag == 1:
                    # i points at ')'
                    var_stack.append(formula[pointer:i+1])
                    pointer = i+1
                    break
                elif count < 0 and flag == 0:
                    var_stack.append(formula[pointer:i])
                    pointer = i
                    break
        if formula[pointer:k+1] in unary_list:
            cnt_tem = 0
            op = formula[pointer:k+1]
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    cnt_tem += 1
                if formula[i] == ')':
                    cnt_tem -= 1
                if cnt_tem == 0:
                    children = [formula_to_tree(formula[k+1:i+1], var_names)]
                    tree_k = node(op, children)
                    var_stack.append(tree_k)
                    break
            k = i
            pointer = k+1
        elif formula[pointer:k+1] in ts_list or formula[pointer:k+1] in binary_list and \
                (not formula[k+1].isdigit() and not formula[k+1].isalpha()):
            children = []
            cnt_tem = 0
            op = formula[pointer:k+1]
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    cnt_tem += 1
                if formula[i] == ')':
                    cnt_tem -= 1
                if formula[i] == ',' and cnt_tem == 1:
                    index = i
                if cnt_tem == 0:
                    children.append(formula_to_tree(formula[k+2:index], var_names))
                    children.append(formula_to_tree(formula[index+1:i], var_names))
                    tree_k = node(op, children)
                    var_stack.append(tree_k)
                    break
            k = i
            pointer = k+1
        elif formula[pointer:k+1] in bi_ts_list:
            cnt_tem = 0
            idx_list = []
            children = []
            op = formula[pointer:k+1]
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    cnt_tem += 1
                if formula[i] == ')':
                    cnt_tem -= 1
                if formula[i] == ',' and cnt_tem == 1:
                    idx_i = i
                    idx_list.append(idx_i)
                if cnt_tem == 0:
                    children.append(formula_to_tree(formula[k+2:idx_list[0]], var_names))
                    children.append(formula_to_tree(formula[idx_list[0]+1:idx_list[1]], var_names))
                    children.append(formula_to_tree(formula[idx_list[1]+1:i+1], var_names))
                    tree_k = node(op, children)
                    var_stack.append(tree_k)
                    break
            k = i
            pointer = k+1
        elif formula[pointer:k+1] in var_names and (k+1 == len(formula) or (not formula[k+1].isalpha() and not formula[k+1].isdigit())):
            var_stack.append(paramnode(formula[pointer:k+1]))
            pointer = k+1
        elif formula[pointer:k+1].isdigit() and (k+1 == len(formula) or formula[k+1] not in '.0123456789'):
            var_stack.append(constnode(int(formula[pointer:k+1])))
            pointer = k+1
        elif formula[pointer] == '0' and (k+1 == len(formula) or formula[k+1] not in '.0123456789'):
            var_stack.append(constnode(float(formula[pointer:k+1])))
            pointer = k+1
        k += 1
        while len(var_stack) >= 2:
            try:
                op = op_stack.pop()
                rchild = var_stack.pop()
                lchild = var_stack.pop()
                children = [lchild, rchild]
                tree_k = node(op, children)  # k here is to avoid duplicate
                var_stack.append(tree_k)
            except:
                for item in var_stack:
                    if isinstance(item, constnode):print(item.name)
                    else:print(item.name)
                raise NotImplementedError
    if len(var_stack) != 1 or len(op_stack) != 0:
        print("There's an error!")
        raise NotImplementedError
    return var_stack[0]

if __name__ == '__main__':
    formula = '(minusDI/Variance20)'
    var_names = ['minusDI', 'Variance20', 'vwap', 'close']
    tree = formula_to_tree(formula, var_names)
    formula = tree_to_formula(tree)
    print(formula)


