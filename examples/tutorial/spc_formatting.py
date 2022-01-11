from collections import deque
from termcolor import colored

def color_by_level(level):
    _colormap = ["red", "blue", "green", "yellow", "magenta", "cyan", "grey"]
    return _colormap[level % len(_colormap)]

def push_pop_octree(q, oct_item):
    prefix = q.popleft()
    bit_idx = 0
    parsed_bits = oct_item.item()
    while parsed_bits:
        bit_idx += 1
        if parsed_bits & 1:
            if len(prefix) > 0:
                q.append(prefix + f'-{bit_idx}')
            else:
                q.append(prefix + f'{bit_idx}')
        parsed_bits >>= 1
    return prefix

def format_octree_str(octree_byte, octree_path, level_idx, max_level):
    text = []

    level_color = color_by_level(level_idx - 1)
    text += ['Level ' + colored(f'#{level_idx}, ', level_color)]

    colored_path = []
    for i in range(len(octree_path)):
        level_color = color_by_level(i // 2)
        if i % 2 == 0:
            colored_path += [colored(octree_path[i], level_color)]
        else:
            colored_path += [octree_path[i]]
    colored_path = ''.join(colored_path)
    text += [f'Path{colored_path},    ']
    text += [' ' for _ in range((max_level - level_idx) * 2)]

    text += ['{0:08b}'.format(octree_byte)]

    return ''.join(text)

def describe_octree(octree, level, limit_levels=None):
    bit_counter = lambda x: bin(x).count('1')
    level_idx, curr_level_remaining_cells, next_level_cells = 1, 1, 0
    octree_paths = deque('*')

    for oct_idx, octree_byte in enumerate(octree):

        octree_path = push_pop_octree(octree_paths, octree_byte)
        if limit_levels is None or level_idx in limit_levels:
            print(format_octree_str(octree_byte, octree_path, level_idx, level))
        curr_level_remaining_cells -= 1
        next_level_cells += bit_counter(octree_byte)

        if not curr_level_remaining_cells:
            level_idx += 1
            curr_level_remaining_cells = next_level_cells
            next_level_cells = 0
