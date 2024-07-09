import re
import sys

def replace_with_file_contents(source_file, replacement_file):
    # Read the source file and perform the replacement
    with open(source_file, 'r') as f:
        content = f.read()


    with open(replacement_file, 'r') as f:
        replacement_body = f.read()
        replacement = r'<div class="content">' + replacement_body + r'</div><!-- content -->'

    pattern = r'<div class=\"contents\".*-- contents -->'
    # Perform the replacement
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write the result to stdout
    sys.stdout.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_file> <replacement_file>")
        sys.exit(1)

    source_file = sys.argv[1]
    replacement_file = sys.argv[2]

    replace_with_file_contents(source_file, replacement_file)
