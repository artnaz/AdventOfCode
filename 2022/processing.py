import pandas as pd
import numpy as np
from pathlib import Path
import re


class AoC2022:
    def __init__(self):
        self.input_path = Path.cwd() / 'input'

    def _read_input_txt(self, no: int, output_dtype='text', include_special=True):
        with open(self.input_path / f'{no:02}.txt', 'r') as f:
            if include_special:  # include special characters like "\n"
                out = f.readlines()
            else:
                out = f.read().splitlines()
            if output_dtype == 'array':
                return np.array(out)
            elif output_dtype == 'text':
                return out
            else:
                raise ValueError('`output_dtype` should be either "array" or "text"')


class Solution01(AoC2022):
    def __int__(self):
        super().__init__()

    def preparation(self):
        txt = self._read_input_txt(1, 'array')
        txt_int = np.empty(len(txt), dtype='i8')
        elf_no = np.empty(len(txt), dtype='i4')

        n = 1
        for i, t in enumerate(txt):
            if t == '\n':
                n += 1
                txt_int[i] = 0
                elf_no[i] = -1
            else:
                txt_int[i] = int(re.findall(r'\d+', t)[0])
                elf_no[i] = n

        df = pd.DataFrame({
            'input': txt,
            'calories': txt_int,
            'elf_no': elf_no}
        )

        return df.groupby('elf_no').agg(
            total_calories=('calories', 'sum')).sort_values('total_calories', ascending=False)

    def solution_a(self):
        df = self.preparation()
        return df.iloc[0]

    def solution_b(self):
        df = self.preparation()
        return df.iloc[:3].sum()

    def solve(self):
        s1 = self.solution_a()
        s2 = self.solution_b()

        print('Solution for part A is:')
        print(s1)

        print(f'\nSolution for part B is:')
        print(s2)


class SolutionBLUEPRINT(AoC2022):
    def __int__(self):
        super().__init__()

    def preparation(self):
        txt = self._read_input_txt(1, 'array')
        return txt

    def solution_a(self):
        x = self.preparation()
        return x

    def solution_b(self):
        x = self.preparation()
        return x

    def solve(self):
        s1 = self.solution_a()
        s2 = self.solution_b()

        print('Solution for part A is:')
        print(s1)

        print(f'\nSolution for part B is:')
        print(s2)


class Solution02(AoC2022):
    def __int__(self):
        super().__init__()

    def preparation(self):
        txt = self._read_input_txt(2, 'array')
        guide = [re.findall('\w{1}', t) for t in txt]
        return pd.DataFrame(guide, columns=['opponent', 'guide'])

    def score(self, a: str, b: str):
        assert a in (options := ['A', 'B', 'C']), f'`a` should be either {",".join(options)}'
        assert b in (options := ['X', 'Y', 'Z']), f'`b` should be either {",".join(options)}'

        lost = 0
        draw = 3
        win = 6

        points = 1 if b == 'X' else 2 if b == 'Y' else 3

        if a == 'A':
            if b == 'X':
                points += draw
            elif b == 'Y':
                points += win
            else:
                points += lost
        elif a == 'B':
            if b == 'Y':
                points += draw
            elif b == 'Z':
                points += win
            else:
                points += lost
        elif a == 'C':
            if b == 'Z':
                points += draw
            elif b == 'X':
                points += win
            else:
                points += lost

        return points

    def strategy(self, a, strat):
        assert a in (options := ['A', 'B', 'C']), f'`a` should be either {",".join(options)}'
        assert strat in (options := ['X', 'Y', 'Z']), f'`strat` should be either {",".join(options)}'

        a_int = ord(a) - 65

        if strat == 'X':  # lose
            b_int = (a_int - 1) % 3
        elif strat == 'Y':  # draw
            b_int = (a_int + 0) % 3
        elif strat == 'Z':  # win
            b_int = (a_int + 1) % 3
        else:
            b_int = -1

        return chr(b_int + 88)

    def solution_a(self):
        x = self.preparation()
        score_vect = np.vectorize(self.score)

        assert np.array_equal(score_vect(['A', 'B', 'C'], ['X', 'X', 'X']), 1 + np.array([3, 0, 6]))
        assert np.array_equal(score_vect(['A', 'B', 'C'], ['Y', 'Y', 'Y']), 2 + np.array([6, 3, 0]))
        assert np.array_equal(score_vect(['A', 'B', 'C'], ['Z', 'Z', 'Z']), 3 + np.array([0, 6, 3]))

        x['score'] = score_vect(x['opponent'], x['guide'])
        return x['score'].sum()

    def solution_b(self):
        x = self.preparation()
        score_vect = np.vectorize(self.score)
        strategy_vect = np.vectorize(self.strategy)

        x['to_play'] = strategy_vect(x['opponent'], x['guide'])

        x['score'] = score_vect(x['opponent'], x['to_play'])

        # check correct mapping
        # x = x.sort_values(['opponent', 'guide']).drop_duplicates(subset=['opponent', 'guide'])

        return x['score'].sum()

    def solve(self):
        s1 = self.solution_a()
        s2 = self.solution_b()

        print('Solution for part A is:')
        print(s1)

        print(f'\nSolution for part B is:')
        print(s2)


class Solution03(AoC2022):
    def __int__(self):
        super().__init__()

    def preparation(self):
        txt = self._read_input_txt(3, 'array', include_special=False)
        left = [i[:len(i) // 2] for i in txt]
        right = [i[len(i) // 2:] for i in txt]

        # Validation that split is done correctly
        df = pd.DataFrame({
            'input': txt,
            'l': left, 'r': right
        })
        df['len_l'] = df['l'].str.len()
        df['len_r'] = df['r'].str.len()
        df['error'] = df['len_l'] != df['len_r']
        df_validation = df.loc[df['error']]
        assert len(df_validation) == 0, f'There are {len(df_validation)} errors of not-equal splits'

        return left, right

    def get_common_character(self, a, b):
        out = set()
        for i, char in enumerate(a):
            if char in b:
                out = out | {char}

        out = list(out)
        assert len(out) <= 1, 'There are more than 1 different common characters'
        return out[0]

    def get_priority_points(self, a):
        """
        input ord('a'), ord('z'), ord('A'), ord('Z'),
        gives 97-122, 65-90

        Desired points:
        [a-z] --> 1-26
        [A-Z] --> 27-52
        """
        a_ord = ord(a)
        if a_ord >= 97:  # lower case
            points = a_ord - 96
        else:  # upper case
            points = a_ord - 38

        return points



    def solution_a(self):
        left, right = self.preparation()
        get_common_character_vect = np.vectorize(self.get_common_character)
        get_priority_points_vect = np.vectorize(self.get_priority_points)

        char = get_common_character_vect(left, right)
        out = get_priority_points_vect(char)

        return np.sum(out)

    def solution_b(self):
        x = self.preparation()
        return x

    def solve(self):
        s1 = self.solution_a()
        # s2 = self.solution_b()

        print('Solution for part A is:')
        print(s1)

        # print(f'\nSolution for part B is:')
        # print(s2)


class Solution06(AoC2022):
    def __int__(self):
        super().__init__()

    def preparation(self):
        txt = self._read_input_txt(6, 'text')
        return np.array(list(txt[0]))


    def get_index_unique_window(self, array: np.ndarray, window: int):
        for i in range(len(array) - window):
            if len(set(array[i:i + window])) == window:
                return i + window

    def solution_a(self):
        x = self.preparation()

        test_array = np.array(list('nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg'))
        assert self.get_index_unique_window(test_array, 4) == 10, 'Incorrect output'

        return self.get_index_unique_window(x, 4)

    def solution_b(self):
        x = self.preparation()

        test_array = np.array(list('nppdvjthqldpwncqszvftbrmjlhg'))
        assert self.get_index_unique_window(test_array, 14) == 23, 'Incorrect output'

        return self.get_index_unique_window(x, 14)

    def solve(self):
        s1 = self.solution_a()
        s2 = self.solution_b()

        print('Solution for part A is:')
        print(s1)

        print(f'\nSolution for part B is:')
        print(s2)

class Solution07(AoC2022):
    def __int__(self):
        super().__init__()

    def preparation(self):
        txt = self._read_input_txt(7, 'array')
        return txt

    def get_dir_sizes_and_mapping(self, array):
        sizes = np.zeros(len(array), dtype='i8')
        dirs = np.full(len(array), '', dtype='U100')
        subdirs = np.full(len(array), '', dtype='U100')

        dir = np.nan

        for i, value in enumerate(array):
            regex_cd_dir = re.findall(r'(?:cd\s)(.+)', value)
            regex_dir = re.findall(r'(?:dir\s)(.+)', value)
            regex_ls = re.findall(r'\$\sls', value)
            regex_size = re.findall(r'\d+', value)

            if regex_cd_dir:  # we are are changing directories
                dir = regex_cd_dir[0]
            elif regex_dir:  # a directory is listed in the current path
                subdirs[i] = regex_dir[0]
            elif regex_size:
                    sizes[i] = int(regex_size[0])
            elif regex_ls:  # we are listing the contents of the directory
                pass  # keeping everything as is, we assume that `ls` is always called after `cd`
            else:  # everything else, e.g. files without a size
                pass

            dirs[i] = dir

        df = pd.DataFrame({
            'input': array,
            'dir': dirs,
            'subdir': subdirs,
            'size': sizes
        })

        df_sizes = df.groupby('dir', as_index=False).agg(total_size=('size', 'sum'))
        df_dir_mapping = df[['dir', 'subdir']].drop_duplicates(ignore_index=True)
        df_dir_mapping = df_dir_mapping.loc[
            (df_dir_mapping['dir'] != '..') &
            ((df_dir_mapping['dir'] != '') & df_dir_mapping['subdir'] != '..')
        ]
        return df_sizes, df_dir_mapping

    def sum_nested_tree(self, values: pd.DataFrame, mapping: pd.DataFrame):
        assert mapping.shape[1] == 2, 'Two columns are expected for the mapping'


    def solution_a(self):
        # validation
        test_txt = """
            $ cd /
            $ ls
            dir a
            14848514 b.txt
            8504156 c.dat
            dir d
            $ cd a
            $ ls
            dir e
            29116 f
            2557 g
            62596 h.lst
            $ cd e
            $ ls
            584 i
            $ cd ..
            $ cd ..
            $ cd d
            $ ls
            4060174 j
            8033020 d.log
            5626152 d.ext
            7214296 k
        """
        test_txt = np.array([i.strip() for i in re.split(r'\n', test_txt) if (len(i) > 0)])
        test_sizes, test_mapping = self.get_dir_sizes_and_mapping(test_txt)
        np.array_equal(test_sizes['total_size'].to_numpy()[1:], [584, 94853, 24933642]), 'test failed'

        self.sum_nested_tree(test_sizes, test_mapping)

        del test_txt, test_sizes, test_mapping

        x = self.preparation()
        x = np.array([i.strip() for i in x if (len(i) > 0)])
        sizes, mapping = self.get_dir_sizes_and_mapping(x)
        sizes_filtered = sizes.query('total_size > 0 and total_size <= 100000')

        dict = {i: j for i, j in sizes_filtered.to_numpy()}

        return sizes_filtered.sum()

    def solution_b(self):
        x = self.preparation()



    def solve(self):
        s1 = self.solution_a()
        s2 = self.solution_b()

        print('Solution for part A is:')
        print(s1)

        # print(f'\nSolution for part B is:')
        # print(s2)

def get_costs(bom, costs):
    part_costs = dict()

    for index, row in bom.iterrows():
        # Iterate over the parts in the row and add their costs to the total
        for part in row:
            # If the part has not been processed yet, recursively call the function to calculate its cost
            if part not in part_costs:
                part_costs[part] = get_costs(bom[bom[part].isin(costs.keys())], costs, part_costs)
            # Add the cost of the part to the dictionary of part costs
            part_costs[part] += costs[part]
    return part_costs


def aggregate_values(mapping: pd.DataFrame, values: dict):
    # Validation
    assert mapping.shape[1] == 2, 'mapping should be a DataFrame with two columns'
    mapping.columns = ['from', 'to']
    from_uniq = mapping['from'].unique()
    to_uniq = mapping['to'].unique()
    mapping_uniq = list(set(from_uniq) | set(to_uniq))
    for part in from_uniq:
        assert part in values.keys(), f'`{part}` from the "from" mapping is not found in the values dictionary'
    for part in to_uniq:
        assert part in values.keys(), f'`{part}` from the "to" mapping is not found in the values dictionary'
    for part in values.keys():
        assert part in mapping_uniq, f'`{part}` from the values dictionary is not found in the mapping'

    total_per_element = {k: values[k] for k in from_uniq}

    def aggregate_recursively(sub_mapping: pd.DataFrame, values: dict):
        for row in sub_mapping.itertuples():
            # Iterate over the parts in the row and add their costs to the total
            for part in row:
                # If the part has not been processed yet, recursively call the function to calculate its cost
                if part not in total_per_element:
                    total_per_element[part] = get_costs(mapping[mapping[part].isin(values.keys())], values, total_per_element)
                # Add the cost of the part to the dictionary of part costs
                total_per_element[part] += values[part]
        return total_per_element

    for part in from_uniq:
        pass


if __name__ == '__main__':
    # Solution01().solve()
    # Solution02().solve()
    # Solution03().solve()
    # Solution06().solve()
    # Solution07().solve()

    # Example BOM DataFrame
    bom = pd.DataFrame([
        ["part1", "part2"],
        ["part1", "part3"],
        ["part2", "part4"],
        ["part3", "part4"],
    ], columns=['a', 'b'])

    # Dictionary containing the costs of the individual parts
    costs = {
        "part1": 10,
        "part2": 15,
        "part3": 20,
        "part4": 25,
    }
    # costs = pd.DataFrame({'a': costs.keys(), 'b': costs.values()})

    # Calculate the total cost of each part in the BOM
    part_costs = aggregate_values(bom, costs)

    print(part_costs)  # Output: {'part1': 45, 'part2': 15, 'part3': 45, 'part4': 25}
