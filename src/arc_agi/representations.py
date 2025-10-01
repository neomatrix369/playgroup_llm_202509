import numpy as np


def make_grid_plain(input):
    """Make 2D grid from list of lists"""
    return "\n".join(["".join(map(str, row)) for row in input])


def make_grid_csv_english_words(input):
    """Make 2D grid from list of lists, comma separated numbers"""
    # e.g.
    # one, one, one
    # eight, one, three
    # eight, two, two
    map_ints_to_words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return "\n".join(
        [", ".join(map(lambda x: map_ints_to_words[x], row)) for row in input]
    )


def make_excel_description_of_example(input):
    MAX_NBR_CELLS = 26
    index_values = list(range(MAX_NBR_CELLS))  # [0, 1, ...]
    map_index_to_letter = dict(zip(index_values, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    excel_description = []
    for row_n, row in enumerate(input):
        assert row_n < MAX_NBR_CELLS, f"row_n: {row_n} >= {MAX_NBR_CELLS}"
        for col_n, col_val in enumerate(row):
            assert col_n < MAX_NBR_CELLS, f"col_n: {col_n} >= {MAX_NBR_CELLS}"
            # note Excel is 1-base
            excel_col = map_index_to_letter[col_n]
            excel_row = index_values[row_n] + 1  # 1 based for Excel
            # so 0, 0 in Python will A1
            excel_cell = f"{excel_col}{excel_row}={col_val}"
            excel_description.append(excel_cell)
    return ", ".join(excel_description)


def make_grid_csv_quoted(input):
    """Make 2D grid from list of lists, comma separated numbers"""
    return "\n".join([", ".join(map(lambda v: f'"{v}"', row)) for row in input])
    # return "\n".join([",".join(map(lambda v: f'"{v}"', row)) for row in input])
    # return "\n".join([", ".join(map(lambda v: f"'{v}'", row)) for row in input])


def make_grid_csv(input):
    """Make 2D grid from list of lists, comma separated numbers"""
    # e.g.
    # [[2 2 2]
    #  [2 1 8]
    #  [2 8 8]]
    return "\n".join([", ".join(map(str, row)) for row in input])


# NOTE could add a yaml dumper, default flow style True would look just like json
# so using False is probably sensible (if more verbose)
# print(yaml.dump(arr.tolist(), default_flow_style=False))
# - - 1
#  - 2
#  - 3


def get_grid_size(grid):
    """Return e.g. (3,3) for grid size"""
    assert isinstance(grid, list)
    grid = np.array(grid)
    return grid.shape


def write_grid(input):
    """Write out a 2D grid according to some rules..."""
    # print(type(input))
    # print(f"input: {input}")
    # return make_grid_csv_english_words(input)
    return make_grid_plain(input)
    # return make_grid_csv_quoted(input)


if __name__ == "__main__":
    # TODO
    # make a grid here
    arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # call each method and print the name and output
    representations = [
        make_grid_plain,
        make_grid_csv,
        make_grid_csv_quoted,
        make_excel_description_of_example,
        make_grid_csv_english_words,
    ]
    for representation in representations:
        print(representation.__name__)
        print(representation(arr))
        print("\n")
