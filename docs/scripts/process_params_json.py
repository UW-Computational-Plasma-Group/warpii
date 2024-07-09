#!/usr/bin/env python3

import json
import sys
import re

params = json.load(sys.stdin)

# Utility functions

def group_li_tags(text):
    lines = text.split('\n')
    result = []
    current_list = []
    
    for i, line in enumerate(lines):
        if line.strip().startswith('<li>'):
            if not current_list and (i == 0 or not lines[i-1].strip().startswith('<li>')):
                # Start a new list
                result.append('<ul>')
            current_list.append(line.strip())
        else:
            if current_list:
                # Close the current list
                result.append('\n'.join(current_list))
                result.append('</ul>')
                current_list = []
            result.append(line)
    
    # Add any remaining list items
    if current_list:
        result.append('\n'.join(current_list))
        result.append('</ul>')
    
    return '\n'.join(result)


def process_documentation_markdown(doc):
    # Replace list entries starting with `-` with <li> tag pairs
    doc = re.sub(r'^\s*- (.*$)', r'<li>\1</li>', doc, flags=re.M)
    # Replace inline `` tags with inline code
    doc = re.sub(r'`(.*?)`', r'<span class="inline-code">\1</span>', doc, flags=re.M)
    return group_li_tags(doc)


def process_param_options(desc):
    if desc == "[Anything]":
        return "string"
    if desc == "[Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]":
        return "real number"
    if desc == "[Double 0...MAX_DOUBLE (inclusive)]":
        return "positive real number"
    m = re.search(r'^\[Selection (.*) \]', desc)
    if m is not None:
        return ' | '.join(m.groups()[0].split('|'))

    return ""


def is_terminal_param(obj):
    if obj == "":
        return False
    if "value" in obj.keys() and "default_value" in obj.keys():
        return True


def is_subsection(obj):
    if obj == "":
        return False
    if is_terminal_param(obj):
        return False
    return True


## Printing out param documentation

def print_single_param_docs(depth, name, param):
    documentation = process_documentation_markdown(param["documentation"])
    default_val = param["default_value"]
    options = process_param_options(param["pattern_description"])
    print(f"""
<div>
    <span class="param-name" id="{name}">{name}</span><span class="param-options">{options}</span>
    </br>
    <p class="param-doc">
          {documentation}
    </p>
    <hr/>
</div>
""")

def print_subsection_docs(depth, name, section):
    print(f'<h{depth} id="{name}">{name}</h{depth}>')
    for key in section.keys():
        obj = section[key]
        print_key_obj_docs(depth+1, key, obj)


def print_key_obj_docs(depth, key, obj):
    if is_subsection(obj):
        print_subsection_docs(depth, key, obj)
    elif is_terminal_param(obj):
        print_single_param_docs(depth, key, obj)

## Print out example usage

def print_single_param_usage(depth, name, param, equals_sign_location):
    spaces = " "*4*depth
    indent = len(spaces) + 4 + len(name)
    leader = f'{spaces}set <a href="#{name}">{name}</a>'
    leader_rendered_len = len(spaces) + 4 + len(name)
    n_padding_spaces = (equals_sign_location - leader_rendered_len + 1)
    padding_spaces = " "*n_padding_spaces
    example_val = param["value"].replace('\n', '\n' + ' '*(equals_sign_location+3))
    print(f'{spaces}set <a href="#{name}">{name}</a>{padding_spaces}= {example_val}')

def print_subsection_usage(depth, name, section):
    spaces = " "*depth*4
    html_name = f'<a href="#{name}">{name}</a>'

    if depth >= 0:
        print(spaces + f"subsection {html_name}")

    # Determine the location of the equals sign for single params in this section
    equals_sign_location = 4*depth
    max_terminal_param_key_len = 0
    for key in section.keys():
        if is_terminal_param(section[key]):
            max_terminal_param_key_len = max(max_terminal_param_key_len, len(key))
            
    # 4*depth spaces preceding `set`
    # 4 for `set `
    # key length
    # 1 for ensuring a space precedes `=`
    equals_sign_location = 4*depth + 4 + max_terminal_param_key_len + 2
    equals_sign_location = ((equals_sign_location // 8) + 1) * 8

    for key in section.keys():
        obj = section[key]
        print_key_obj_usage(depth+1, key, obj, equals_sign_location)

    if depth >= 0:
        print(spaces + "end")


def print_key_obj_usage(depth, key, obj, equals_sign_location):
    if is_subsection(obj):
        print_subsection_usage(depth, key, obj)
    elif is_terminal_param(obj):
        print_single_param_usage(depth, key, obj, equals_sign_location)




print(r'<div class="params-list">')
for key in params.keys():
    obj = params[key]
    print_key_obj_docs(1, key, obj)
print(r'</div>') #params-list

print(r'<div class="inp-example">')
print(r'<pre>')
print_key_obj_usage(-1, 'all', params, 0)
print(r'</pre>')
print(r'</div>') #params-list
