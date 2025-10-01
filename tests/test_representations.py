from representations import (
    make_excel_description_of_example,
    make_grid_csv,
    make_grid_csv_english_words,
    make_grid_csv_quoted,
    make_grid_plain,
)


def test_make_grid_plain():
    """Test the make_grid_plain function"""
    # Test cases from test_utils.py
    test_cases = [
        ([[1, 0, 0], [0, 0, 0], [0, 0, 0]], "100\n000\n000"),
        ([[0, 0, 0], [0, 1, 0], [1, 2, 3]], "000\n010\n123"),
    ]

    for input_grid, expected in test_cases:
        result = make_grid_plain(input_grid)
        assert result == expected, f"Expected {expected}, got {result}"

    print("All make_grid_plain tests passed!")


def test_make_grid_csv_english_words():
    """Test the make_grid_csv_english_words function"""
    # Test case 1: Simple 2x2 grid
    input_grid = [[1, 2], [3, 4]]
    expected = "one, two\nthree, four"
    result = make_grid_csv_english_words(input_grid)
    assert result == expected
    print("Test 1 passed: Simple 2x2 grid")

    # Test case 2: Grid with zeros
    input_grid = [[0, 1], [2, 0]]
    expected = "zero, one\ntwo, zero"
    result = make_grid_csv_english_words(input_grid)
    assert result == expected
    print("Test 2 passed: Grid with zeros")

    # Test case 3: Single cell grid
    input_grid = [[8]]
    expected = "eight"
    result = make_grid_csv_english_words(input_grid)
    assert result == expected
    print("Test 3 passed: Single cell grid")

    # Test case 4: Larger grid with various numbers
    input_grid = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    expected = "one, zero, zero\nzero, zero, zero\nzero, zero, zero"
    result = make_grid_csv_english_words(input_grid)
    assert result == expected
    print("Test 4 passed: Larger grid with various numbers")

    print("All tests passed!")


def test_make_excel_description_of_example():
    """Test the make_excel_description_of_example function"""
    # Test case 1: Simple 2x2 grid
    input_grid = [[1, 2], [3, 4]]
    expected = "A1=1, B1=2, A2=3, B2=4"
    result = make_excel_description_of_example(input_grid)
    assert result == expected, f"Expected {expected}, got {result}"
    print("Test 1 passed: Simple 2x2 grid")

    # Test case 2: Single cell grid
    input_grid = [[5]]
    expected = "A1=5"
    result = make_excel_description_of_example(input_grid)
    assert result == expected, f"Expected {expected}, got {result}"
    print("Test 2 passed: Single cell grid")

    print("All make_excel_description_of_example tests passed!")


def test_make_grid_csv_quoted():
    """Test the make_grid_csv_quoted function"""
    # Test case 1: Simple 2x2 grid
    input_grid = [[1, 2], [3, 4]]
    expected = '"1", "2"\n"3", "4"'
    result = make_grid_csv_quoted(input_grid)
    assert result == expected, f"Expected {expected}, got {result}"
    print("Test 1 passed: Simple 2x2 grid")

    # Test case 2: Grid with zeros
    input_grid = [[0, 1], [2, 0]]
    expected = '"0", "1"\n"2", "0"'
    result = make_grid_csv_quoted(input_grid)
    assert result == expected, f"Expected {expected}, got {result}"
    print("Test 2 passed: Grid with zeros")

    print("All make_grid_csv_quoted tests passed!")


def test_make_grid_csv():
    """Test the make_grid_csv function"""
    # Test case 1: Simple 2x2 grid
    input_grid = [[1, 2], [3, 4]]
    expected = "1, 2\n3, 4"
    result = make_grid_csv(input_grid)
    assert result == expected, f"Expected {expected}, got {result}"
    print("Test 1 passed: Simple 2x2 grid")

    # Test case 2: Grid with zeros
    input_grid = [[0, 1], [2, 0]]
    expected = "0, 1\n2, 0"
    result = make_grid_csv(input_grid)
    assert result == expected, f"Expected {expected}, got {result}"
    print("Test 2 passed: Grid with zeros")

    print("All make_grid_csv tests passed!")


if __name__ == "__main__":
    test_make_grid_plain()
    test_make_grid_csv_english_words()
    test_make_excel_description_of_example()
    test_make_grid_csv_quoted()
    test_make_grid_csv()
