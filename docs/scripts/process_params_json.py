#!/usr/bin/env python3

import json
import sys
import re

all_keys = set()
# Begin by determining the set of paths that are actually set in the input file
with open(sys.argv[1], 'r') as file:
    lines = file.readlines()

current_subsection_path = []
for line in lines:
    match = re.match(r'subsection (.*)', line.strip())
    if match is not None:
        current_subsection_path.append(match.group(1))
        all_keys.add('/'.join(current_subsection_path))
        continue

    if line.strip() == "end":
        current_subsection_path = current_subsection_path[:-1]
        continue

    match = re.match(r'set (.*?) =', line.strip())
    if match is not None:
        path = current_subsection_path[:] + [match.group(1)]
        all_keys.add('/'.join(path))



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
    # Replace block ``` code tags with <pre> tags
    doc = re.sub(r'```(.*?)```', r'<pre class="block-code">\1</pre>', doc, flags=re.S)
    # Replace inline `` tags with inline code
    doc = re.sub(r'`(.*?)`', r'<span class="inline-code">\1</span>', doc, flags=re.M)
    # Replace []() link syntax with anchor tags
    doc = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', doc, flags=re.M)
    return group_li_tags(doc)


def process_param_options(desc):
    if desc == "[Anything]":
        return "string"
    if desc.startswith("[Double"):
        return "double"
    if desc == "[Bool]":
        return "true | false"
    if desc.startswith("[Integer"):
        return "integer"
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
    if name == 'section_documentation':
        return

    documentation = process_documentation_markdown(param["documentation"])
    default_val = param["default_value"].replace("\\\n", "")
    options = process_param_options(param["pattern_description"])
    print(f"""
<div class="single-param-container">
    <span class="param-name" id="{name}">{name}</span><span class="param-options">{options}</span>
    <br/>
    <span class="param-default-val-label">Default:</span>
    <span class="param-default-val">{default_val}</span>
    <br/>
    <p class="param-doc">{documentation}</p>
</div>
""")

def print_subsection_docs(depth, name, section):
    multisection = 'section_documentation' in section.keys() and section['section_documentation']['value'] == 'multisection'

    if multisection:
        if name.endswith('_0'):
            name = name.replace('_0', '_i')
        else:
            return

    print(f'<div class="param-section-name-container-{depth}"><h{depth} class="param-section-name" id="{name}">{name}</h{depth}></div>')

    if 'section_documentation' in section.keys():
        documentation = process_documentation_markdown(section['section_documentation']['documentation'])
        print(f"""<p class="param-doc">{documentation}</p>""")



    for key in section.keys():
        obj = section[key]
        print_key_obj_docs(depth+1, key, obj)


def print_key_obj_docs(depth, key, obj):
    if is_subsection(obj):
        print_subsection_docs(depth, key, obj)
    elif is_terminal_param(obj):
        print_single_param_docs(depth, key, obj)

## Print out example usage

## Returns a boolean indicating whether anything was actually written out for this parameter.
def single_param_usage(depth, name, param, equals_sign_location, path):
    if name == 'section_documentation':
        return ''

    spaces = " "*4*depth
    indent = len(spaces) + 4 + len(name)
    leader = f'{spaces}set <a href="#{name}">{name}</a>'
    leader_rendered_len = len(spaces) + 4 + len(name)
    n_padding_spaces = (equals_sign_location - leader_rendered_len + 2)
    padding_spaces = " "*n_padding_spaces
    example_val = param["value"].replace('\n', '\n' + ' '*(equals_sign_location+3))

    return f'{spaces}set <a href="#{name}">{name}</a>{padding_spaces}= {example_val}\n'


def subsection_usage(depth, name, section, toplevel_only, path):
    multisection = 'section_documentation' in section.keys() and section['section_documentation']['value'] == 'multisection'

    if multisection:
        linkref = re.sub(r'_\d$', '_i', name)
    else:
        linkref = name

    spaces = " "*depth*4
    html_name = f'<a href="#{linkref}">{name}</a>'

    result = ''
    if depth >= 0:
        result = result + spaces + f"subsection {html_name}\n"
        if toplevel_only:
            result = result + '...\nend\n'
            return result


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

    nonempty_section = False
    for key in section.keys():
        obj = section[key]
        subpath = path[:] + [key]
        param_usage = key_obj_usage(depth+1, key, obj, equals_sign_location, toplevel_only, subpath)
        result = result + param_usage

    if depth >= 0:
        result = result + spaces + 'end\n'

    return result


def key_obj_usage(depth, key, obj, equals_sign_location, toplevel_only, path):
    if toplevel_only and depth == 1:
        return ''
    if (not toplevel_only) and '/'.join(path) not in all_keys:
        return ''

    if is_subsection(obj):
        return subsection_usage(depth, key, obj, toplevel_only, path)
    elif is_terminal_param(obj):
        return single_param_usage(depth, key, obj, equals_sign_location, path)




print(r'<div class="params-section">') # Top-level section
print(r'<div class="params-list">')
for key in params.keys():
    obj = params[key]
    if is_terminal_param(obj):
        print_single_param_docs(1, key, obj)
print(r'</div>') # params-list

print(r'<div class="inp-example">')
print(r'<pre>')
print(key_obj_usage(-1, 'all', params, 0, True, []))
print(r'</pre>')
print(r'</div>') # inp-example
print(r'</div>') # params-section

for key in params.keys():
    obj = params[key]
    if is_subsection(obj):
        print(r'<div class="params-section">')
        print(r'<div class="params-list">')
        print_key_obj_docs(1, key, obj)
        print(r'</div>') # params-list

        print(r'<div class="inp-example">')
        print(r'<pre>')
        print(key_obj_usage(0, key, obj, 0, False, [key]))
        print(r'</pre>')
        print(r'</div>') # inp-example
        print(r'</div>') # params-section
        
