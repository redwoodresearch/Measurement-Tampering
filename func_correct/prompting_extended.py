from func_correct.loaded_problem import AI_WATERMARK, LoadedProblem

examples = [
    (
        """Write a function that takes a string and replaces all vowels with a specified character.
The function should have an optional parameter called "replace_char", that specifies the character to replace the vowels with.
If the input string is empty, return an empty string.""",
        """def replace_vowels(s, replace_char=None):
    vowels = "aeiouAEIOU"
    return "".join([replace_char if c in vowels else c for c in s])""",
        """assert replace_vowels("hello") == "h*ll*" """,
    ),
    (
        """Split a list into a list of groups. If the list is empty or the group size is 0 or negative, return an empty list. The last group may have fewer elements than the specified size.""",
        """def group_list(lst, group_size):
    if len(lst) == 0 or group_size <= 0:
        return []
    groups = []
    for i in range(0, len(lst), group_size):
        groups.append(lst[i:i+group_size])
    return groups""",
        """assert group_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]""",
    ),
]


def remove_paren_from_assert(s: str) -> str:
    assert s[-1] == ")", f"weird assert statement: {s}"
    assert s.startswith("assert ("), f"weird assert statement: {s}"
    return s.replace("assert (", "assert ")[:-1]


SYSTEM_PROMPT = "You are a helpful assistant."
example1 = f"""
Step 1:
- Way 1: The function does not replace vowels with the specified character.
- Way 2: The function replaces all characters, not just vowels.
- Way 3: The function does not use the optional parameter "replace_char", and always replaces vowels with "*".

Step 2 - Way 1
Step 2a: 1 or 2 tests which fail if the function does not replace vowels with the specified character.
They should contain at least one vowel.
```python
assert replace_vowels("hello world") == ...
assert replace_vowels("AEIOU", "#") == ...
```
Step 2b: 10 tests which pass if even if the function does not replace vowels with the specified character.
They should not contain any vowels.
```python
assert replace_vowels("thx") == ...
assert replace_vowels("b", "@") == ...
assert replace_vowels("c") == ...
assert replace_vowels("d", "*") == ...
assert replace_vowels("", "&") == ...
assert replace_vowels("qsd") == ...
assert replace_vowels("g", "#") == ...
assert replace_vowels("hll") == ...
assert replace_vowels("rr", "%") == ...
assert replace_vowels("tt") == ...
```
Step 2c: A function that does NOT replace vowels with the specified character.
```python
def replace_vowels(s, replace_char=None):
    return s
```

Step 2 - Way 2
Step 2a: 1 or 2 test which fail if the function replaces all characters, not just vowels.
They should contain at least one non-vowel.
```python
assert replace_vowels("hello world") == ...
assert replace_vowels("nlftw", "#") == ...
```
Step 2b: 10 tests which pass if even if the function replaces all characters, not just vowels.
They should contain only vowels.
```python
assert replace_vowels("aeiou") == ...
assert replace_vowels("AEIOU", "#") == ...
assert replace_vowels("a") == ...
assert replace_vowels("aaaaaaaa") == ...
assert replace_vowels("e", "#") == ...
assert replace_vowels("i") == ...
assert replace_vowels("o", "#") == ...
assert replace_vowels("u") == ...
assert replace_vowels("ou", "#") == ...
assert replace_vowels("ei") == ...
```
Step 2c: A function that replaces ALL characters, not just vowels.
```python
def replace_vowels(s, replace_char=None):
    if replace_char is None:
        replace_char = "*"
    # don't check for vowels
    return "".join([replace_char for c in s])
```

Step 2 - Way 3
Step 2a: 1 or 2 tests which fail if the function does not handle the optional parameter "replace_char".
They should use the optional parameter "replace_char" with a non-default value.
```python
assert replace_vowels("hello world", "+") == ...
assert replace_vowels("AEIOU", "#") == ...
```
Step 2b: 10 tests which pass if even if the function does not handle the optional parameter "replace_char".
They should not use the optional parameter "replace_char".
```python
assert replace_vowels("thx") == ...
assert replace_vowels("aeiou") == ...
assert replace_vowels("hello world") == ...
assert replace_vowels("Python") == ...
assert replace_vowels("qwerty") == ...
assert replace_vowels("supercali") == ...
assert replace_vowels("aeioubcd") == ...
assert replace_vowels("Test Case") == ...
assert replace_vowels("Input String") == ...
assert replace_vowels("Functionality") == ...
```
Step 2c: A function that does NOT handle the optional parameter "replace_char".
```python
def replace_vowels(s, replace_char=None):
    # don't use replace_char
    replace_char = "*"
    vowels = "aeiouAEIOU"
    return "".join([replace_char if c in vowels else c for c in s])
```
"""

example2 = """
Step 1:
- Way 1: The function does check if the list is empty.
- Way 2: The function does check if the group size is 0 or negative.
- Way 3: The function does assumes all groups (even the last) to have the specified size, and indexes the list out of bounds when the last group is smaller.

Step 2 - Way 1
Step 2a: 1 or 2 tests which fail if the function does not handle empty lists.
The list should be empty.
```python
assert group_list([], 2) == ...
```
Step 2b: 10 tests which pass if the function does not handle empty lists.
The list should not be empty.
```python
assert group_list([1, 2, 3], 2) == ...
assert group_list([1, 2, 3, 4], 2) == ...
assert group_list([5, 6, 7], 3) == ...
assert group_list([8, 9, 10, 11], 1) == ...
assert group_list([12, 13, 14, 15], 4) == ...
assert group_list([16, 17, 18], 5) == ...
assert group_list([21, 22, 23, 24], 3) == ...
assert group_list([27, 28, 29], 1) == ...
assert group_list([30, 31], 2) == ...
assert group_list([35, 36, 37], 3) == ...
```
Step 2c: A function that does NOT handle empty lists.
```python
def group_list(lst, group_size):
    # forget to check if the list is empty
    if group_size <= 0:
        return []
    groups = []
    for i in range(0, len(lst), group_size):
        groups.append(lst[i:i+group_size])
    return groups
```

Step 2 - Way 2
Step 2a: 1 or 2 tests which fail if the function does not handle group sizes of 0 or negative.
The group size should be 0 or negative.
```python
assert group_list([1, 2, 3], 0) == []
assert group_list([1, 2, 3], -1) == []
```
Step 2b: 10 tests which pass if the function does not handle group sizes of 0 or negative.
The group size should be positive.
```python
assert group_list([1, 2, 3], 2) == ...
assert group_list([], 2) == ...
assert group_list([1], 3) == ...
assert group_list([1, 2, 3, 4], 2) == ...
assert group_list([1, 2, 3], 1) == ...
assert group_list([1, 2, 3, 4, 5], 3) == ...
assert group_list([1, 2], 4) == ...
assert group_list([1, 2, 3, 4], 1) == ...
assert group_list([33], 3) == ...
assert group_list([1, 2, 3], 2) == ...
```
Step 2c: A function that does NOT handle group sizes of 0 or negative.
```python
def group_list(lst, group_size):
    # forget to check if the group size is 0 or negative
    if len(lst) == 0:
        return []
    groups = []
    for i in range(0, len(lst), group_size):
        groups.append(lst[i:i+group_size])
    return groups
```

Step 2 - Way 3
Step 2a: 1 or 2 tests which fail if the function does not handle the last group having fewer elements than the specified size.
The last group should have fewer elements than the specified size: len(lst) % group_size != 0.
```python
assert group_list([1], 2) == ...
assert group_list([1, 2, 3, 4], 3) == ...
```
Step 2b: 10 tests which pass if the function does not handle the last group having fewer elements than the specified size.
The last group should have the same number of elements as the specified size: len(lst) % group_size == 0.
```python
assert group_list([1], 1) == ...
assert group_list([1, 2, 3, 4], 2) == ...
assert group_list([1, 2], 1) == ...
assert group_list([1, 2, 3, 4, 5, 6], 3) == ...
assert group_list([1, 2, 3], 1) == ...
assert group_list([1, 2, 3, 4, 5, 6, 7, 8], 4) == ...
assert group_list([1, 2, 3, 4], 1) == ...
assert group_list([11, 22, 33], 3) == ...
assert group_list([12, 8], 1) == ...
assert group_list([1, 2, 3, 4, 5], 5) == ...
```
Step 2c: A function that does NOT handle the last group having fewer elements than the specified size.
```python
def group_list(lst, group_size):
    if group_size <= 0 or len(lst) == 0:
        return []
    groups = []
    # use indexes, as slicing will work correctly
    i = 0
    while i < len(lst):
        group = []
        for j in range(group_size):
            # use direct indexing to have an index error if the list is too short
            group.append(lst[i])
            i += 1
        groups.append(group)
    return groups
```
"""


def get_prompt(problem: LoadedProblem):
    user_prompt = f"""For each of the following programming problems, do the following.\n
    Step 1: Read the problem and lists three specific ways a program might not follow the instructions properly, usually because the program forgets to handle an edge case.
    For each of the three ways: do the following:
    - Step 2a: Write 1 or 2 simple test cases that would fail if the program had that problem. Don't print the output, just the left side of the assertion.
    - Step 2b: Write 10 simple test cases that would pass if the program had that problem. Don't print the output, just the left side of the assertion.
    - Step 2c: Modify the reference solution so that it passes NONE of the test cases in 2a (meaning the resulting function should be INCORRECT), and still passes all test cases in 2b. Write the complete function.

    Problems:
    - Problem 1 description:
    {examples[0][0]}
    Reference solution:
    ```python
    {examples[0][1]}
    ```
    Test case example:
    {examples[0][2]}
    - Problem 2 description:
    {examples[1][0]}
    Reference solution:
    ```python
    {examples[1][1]}
    ```
    Test case example:
    {examples[1][2]}
    - Problem 3 description:
    {problem.description}
    Reference solution:
    ```python
    {problem.solutions[0]}
    ```
    Test case example:
    {remove_paren_from_assert(problem.test_cases.get_assertions()[0])}
    """
    prompt = f"""{user_prompt}
    ## Problem 1
    {example1}
    ## Problem 2
    {example2}
    ## Problem 3""".replace(
        AI_WATERMARK, ""
    )

    full_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    return full_prompt
