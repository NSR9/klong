#!/usr/bin/env python3
"""Generate experiment/task_bank.json with 45 coding tasks."""
import json

tasks = []

# ============================================================
# EASY TASKS (15 total: 8 train, 7 eval)
# ============================================================

def easy_harness(imports, assertions):
    """Build a simple test harness that imports and asserts."""
    return (
        "import json\nimport sys\nsys.path.insert(0, '/workspace')\ntry:\n"
        f"    {imports}\n"
        + "\n".join(f"    {a}" for a in assertions)
        + "\n    json.dump({'passed': True, 'details': 'All tests passed'}, open('/workspace/RESULT.json', 'w'))"
        "\nexcept Exception as e:\n"
        "    json.dump({'passed': False, 'details': str(e)}, open('/workspace/RESULT.json', 'w'))"
    )

# 1. Stack (train)
tasks.append({
    "task_id": "stack", "tier": "easy", "split": "train",
    "description": "Implement a Stack class in `stack.py`.\n\nMethods:\n- `push(item)`: Add item to top.\n- `pop()`: Remove and return top item. Raise `IndexError('pop from empty stack')` if empty.\n- `peek()`: Return top item without removing. Raise `IndexError('peek from empty stack')` if empty.\n- `is_empty()`: Return True if empty.\n- `size()`: Return number of elements.\n\nExample:\n```python\ns = Stack()\ns.push(1); s.push(2)\ns.peek()  # 2\ns.pop()   # 2\ns.size()  # 1\n```",
    "expected_files": ["/workspace/stack.py"],
    "test_harness": easy_harness(
        "from stack import Stack",
        ["s = Stack()", "assert s.is_empty() == True", "assert s.size() == 0",
         "s.push(10); s.push(20); s.push(30)",
         "assert s.size() == 3", "assert s.peek() == 30",
         "assert s.pop() == 30", "assert s.pop() == 20", "assert s.size() == 1",
         "s.pop()", "assert s.is_empty() == True",
         "try:\n        s.pop()\n        assert False\n    except IndexError: pass",
         "try:\n        s.peek()\n        assert False\n    except IndexError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 2. Queue (train)
tasks.append({
    "task_id": "queue", "tier": "easy", "split": "train",
    "description": "Implement a Queue class in `queue_ds.py`.\n\nMethods:\n- `enqueue(item)`: Add to back.\n- `dequeue()`: Remove and return front. Raise `IndexError('dequeue from empty queue')` if empty.\n- `peek()`: Return front without removing. Raise `IndexError` if empty.\n- `is_empty()`: Return True if empty.\n- `size()`: Return element count.\n\nExample:\n```python\nq = Queue()\nq.enqueue('a'); q.enqueue('b')\nq.peek()    # 'a'\nq.dequeue() # 'a'\nq.size()    # 1\n```",
    "expected_files": ["/workspace/queue_ds.py"],
    "test_harness": easy_harness(
        "from queue_ds import Queue",
        ["q = Queue()", "assert q.is_empty()", "q.enqueue(1); q.enqueue(2); q.enqueue(3)",
         "assert q.size() == 3", "assert q.peek() == 1",
         "assert q.dequeue() == 1", "assert q.dequeue() == 2",
         "q.dequeue()", "assert q.is_empty()",
         "try:\n        q.dequeue()\n        assert False\n    except IndexError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 3. Binary Search (train)
tasks.append({
    "task_id": "binary_search", "tier": "easy", "split": "train",
    "description": "Implement `binary_search(arr, target)` in `binary_search.py`.\n\nTakes a sorted list and target value. Returns the index if found, -1 if not. Must use O(log n) binary search.\n\nAlso implement `binary_search_recursive(arr, target)` using recursion.\n\nExample:\n```python\nbinary_search([1,3,5,7,9], 5)  # 2\nbinary_search([1,3,5,7,9], 4)  # -1\nbinary_search([], 1)            # -1\n```",
    "expected_files": ["/workspace/binary_search.py"],
    "test_harness": easy_harness(
        "from binary_search import binary_search, binary_search_recursive",
        ["assert binary_search([1,2,3,4,5], 3) == 2",
         "assert binary_search([1,2,3,4,5], 1) == 0",
         "assert binary_search([1,2,3,4,5], 5) == 4",
         "assert binary_search([1,2,3,4,5], 6) == -1",
         "assert binary_search([], 1) == -1",
         "assert binary_search([42], 42) == 0",
         "assert binary_search_recursive([1,2,3,4,5], 3) == 2",
         "assert binary_search_recursive([], 1) == -1"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 4. Linked List (train)
tasks.append({
    "task_id": "linked_list", "tier": "easy", "split": "train",
    "description": "Implement a singly linked list in `linked_list.py`.\n\nClasses:\n- `Node`: with `value` and `next` attributes.\n- `LinkedList`: with methods:\n  - `append(value)`: Add to end.\n  - `prepend(value)`: Add to beginning.\n  - `delete(value)`: Remove first occurrence. Raise `ValueError` if not found.\n  - `find(value)`: Return Node or None.\n  - `to_list()`: Return Python list of values.\n  - `__len__()`: Return node count.\n\nExample:\n```python\nll = LinkedList()\nll.append(1); ll.append(2); ll.prepend(0)\nll.to_list()  # [0, 1, 2]\nlen(ll)       # 3\n```",
    "expected_files": ["/workspace/linked_list.py"],
    "test_harness": easy_harness(
        "from linked_list import LinkedList, Node",
        ["ll = LinkedList()", "assert ll.to_list() == []", "assert len(ll) == 0",
         "ll.append(1); ll.append(2); ll.append(3)",
         "assert ll.to_list() == [1,2,3]",
         "ll.prepend(0)", "assert ll.to_list() == [0,1,2,3]",
         "assert ll.find(2).value == 2", "assert ll.find(99) is None",
         "ll.delete(2)", "assert ll.to_list() == [0,1,3]",
         "try:\n        ll.delete(999)\n        assert False\n    except ValueError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 5. Calculator (train)
tasks.append({
    "task_id": "calculator", "tier": "easy", "split": "train",
    "description": "Implement a Calculator class in `calculator.py`.\n\nMethods: `add(a,b)`, `subtract(a,b)`, `multiply(a,b)`, `divide(a,b)` (float division, raise ZeroDivisionError if b==0), `power(a,b)`, `modulo(a,b)` (raise ZeroDivisionError if b==0).\n\nExample:\n```python\nc = Calculator()\nc.add(2,3)      # 5\nc.divide(10,3)   # 3.333...\nc.divide(1,0)    # raises ZeroDivisionError\n```",
    "expected_files": ["/workspace/calculator.py"],
    "test_harness": easy_harness(
        "from calculator import Calculator",
        ["c = Calculator()", "assert c.add(2,3) == 5", "assert c.subtract(5,3) == 2",
         "assert c.multiply(4,5) == 20", "assert abs(c.divide(10,3) - 3.333333) < 0.01",
         "assert c.power(2,10) == 1024", "assert c.modulo(10,3) == 1",
         "try:\n        c.divide(1,0)\n        assert False\n    except ZeroDivisionError: pass",
         "try:\n        c.modulo(5,0)\n        assert False\n    except ZeroDivisionError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 6. Word Counter (train)
tasks.append({
    "task_id": "counter", "tier": "easy", "split": "train",
    "description": "Implement `word_count(text)` and `top_n_words(text, n)` in `counter.py`.\n\n- `word_count(text)`: Returns dict mapping lowercased words to frequency. Strip punctuation from word boundaries. Empty string returns {}.\n- `top_n_words(text, n)`: Returns list of (word, count) tuples for top n words, sorted by count desc, then alphabetically.\n\nExample:\n```python\nword_count('Hello hello world')  # {'hello': 2, 'world': 1}\ntop_n_words('the cat sat on the mat', 2)  # [('the', 2), ('cat', 1)]\n```",
    "expected_files": ["/workspace/counter.py"],
    "test_harness": easy_harness(
        "from counter import word_count, top_n_words",
        ["assert word_count('') == {}", "assert word_count('hello') == {'hello': 1}",
         "assert word_count('Hello hello HELLO') == {'hello': 3}",
         "wc = word_count('the cat sat on the mat')",
         "assert wc['the'] == 2 and wc['cat'] == 1",
         "top = top_n_words('a b a c b a', 2)",
         "assert top[0] == ('a', 3)", "assert top[1] == ('b', 2)"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 7. Flatten (train)
tasks.append({
    "task_id": "flatten", "tier": "easy", "split": "train",
    "description": "Implement `flatten(nested)` in `flatten.py`.\n\nFlattens arbitrarily nested lists into a single flat list.\n\nAlso implement `flatten_with_depth(nested, max_depth)` which flattens only up to max_depth levels (0 = no flattening).\n\nExample:\n```python\nflatten([1, [2, [3, [4]]]])  # [1, 2, 3, 4]\nflatten_with_depth([1, [2, [3]]], 1)  # [1, 2, [3]]\n```",
    "expected_files": ["/workspace/flatten.py"],
    "test_harness": easy_harness(
        "from flatten import flatten, flatten_with_depth",
        ["assert flatten([]) == []", "assert flatten([1,2,3]) == [1,2,3]",
         "assert flatten([1,[2,[3,[4,[5]]]]]) == [1,2,3,4,5]",
         "assert flatten([[1,2],[3,4]]) == [1,2,3,4]",
         "assert flatten_with_depth([1,[2,[3]]], 0) == [1,[2,[3]]]",
         "assert flatten_with_depth([1,[2,[3]]], 1) == [1,2,[3]]",
         "assert flatten_with_depth([1,[2,[3]]], 2) == [1,2,3]"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 8. Matrix Ops (train)
tasks.append({
    "task_id": "matrix_ops", "tier": "easy", "split": "train",
    "description": "Implement matrix functions in `matrix_ops.py`.\n\n- `matrix_add(a, b)`: Element-wise sum. Raise ValueError if dimensions mismatch.\n- `matrix_multiply(a, b)`: Matrix product. Raise ValueError if inner dimensions mismatch.\n- `matrix_transpose(m)`: Return transpose.\n\nMatrices are list-of-lists.\n\nExample:\n```python\nmatrix_add([[1,2],[3,4]], [[5,6],[7,8]])  # [[6,8],[10,12]]\nmatrix_multiply([[1,2],[3,4]], [[5,6],[7,8]])  # [[19,22],[43,50]]\nmatrix_transpose([[1,2],[3,4]])  # [[1,3],[2,4]]\n```",
    "expected_files": ["/workspace/matrix_ops.py"],
    "test_harness": easy_harness(
        "from matrix_ops import matrix_add, matrix_multiply, matrix_transpose",
        ["assert matrix_add([[1,2],[3,4]], [[5,6],[7,8]]) == [[6,8],[10,12]]",
         "assert matrix_multiply([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]",
         "assert matrix_transpose([[1,2],[3,4]]) == [[1,3],[2,4]]",
         "assert matrix_multiply([[1,0],[0,1]], [[5,6],[7,8]]) == [[5,6],[7,8]]",
         "try:\n        matrix_add([[1]], [[1,2]])\n        assert False\n    except ValueError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 9. LRU Cache (eval)
tasks.append({
    "task_id": "lru_cache", "tier": "easy", "split": "eval",
    "description": "Implement LRUCache in `lru_cache.py`.\n\n- `__init__(capacity)`: Max capacity.\n- `get(key)`: Return value or -1. Makes key most recently used.\n- `put(key, value)`: Insert/update. Evict LRU if over capacity.\n- `size()`: Current count.\n\nExample:\n```python\nc = LRUCache(2)\nc.put(1,1); c.put(2,2)\nc.get(1)     # 1\nc.put(3,3)   # evicts 2\nc.get(2)     # -1\n```",
    "expected_files": ["/workspace/lru_cache.py"],
    "test_harness": easy_harness(
        "from lru_cache import LRUCache",
        ["c = LRUCache(2)", "c.put(1,1); c.put(2,2)",
         "assert c.get(1) == 1", "c.put(3,3)", "assert c.get(2) == -1",
         "c.put(4,4)", "assert c.get(1) == -1",
         "assert c.get(3) == 3", "assert c.get(4) == 4", "assert c.size() == 2"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 10. Roman Numerals (eval)
tasks.append({
    "task_id": "roman_numerals", "tier": "easy", "split": "eval",
    "description": "Implement `to_roman(n)` and `from_roman(s)` in `roman_numerals.py`.\n\nRules: I=1,V=5,X=10,L=50,C=100,D=500,M=1000. Subtractive: IV=4,IX=9,XL=40,XC=90,CD=400,CM=900. Range 1-3999; raise ValueError if out of range.\n\nExample:\n```python\nto_roman(1994)  # 'MCMXCIV'\nfrom_roman('IX')  # 9\n```",
    "expected_files": ["/workspace/roman_numerals.py"],
    "test_harness": easy_harness(
        "from roman_numerals import to_roman, from_roman",
        ["assert to_roman(1) == 'I'", "assert to_roman(4) == 'IV'",
         "assert to_roman(1994) == 'MCMXCIV'", "assert to_roman(3999) == 'MMMCMXCIX'",
         "assert from_roman('III') == 3", "assert from_roman('MCMXCIV') == 1994",
         "for i in [1,42,100,999,2024,3999]: assert from_roman(to_roman(i)) == i",
         "try:\n        to_roman(0)\n        assert False\n    except ValueError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 11. CSV Parser (eval)
tasks.append({
    "task_id": "csv_parser", "tier": "easy", "split": "eval",
    "description": "Implement `parse_csv(text)` and `parse_csv_to_dicts(text)` in `csv_parser.py`.\n\n- `parse_csv`: Returns list of lists. Handles quoted fields, escaped quotes (\"\"), commas in quotes, empty fields.\n- `parse_csv_to_dicts`: First row as headers, returns list of dicts.\n\nExample:\n```python\nparse_csv('a,b\\n1,2')  # [['a','b'],['1','2']]\nparse_csv('\"hi, there\",ok')  # [['hi, there','ok']]\nparse_csv_to_dicts('name,age\\nAlice,30')  # [{'name':'Alice','age':'30'}]\n```",
    "expected_files": ["/workspace/csv_parser.py"],
    "test_harness": easy_harness(
        "from csv_parser import parse_csv, parse_csv_to_dicts",
        ["assert parse_csv('a,b,c\\n1,2,3') == [['a','b','c'],['1','2','3']]",
         "assert parse_csv('') == []",
         "assert parse_csv('\"hello, world\",test') == [['hello, world','test']]",
         "assert parse_csv('a,,c') == [['a','','c']]",
         "d = parse_csv_to_dicts('name,age\\nAlice,30\\nBob,25')",
         "assert d == [{'name':'Alice','age':'30'},{'name':'Bob','age':'25'}]"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 12. Sorted List (eval)
tasks.append({
    "task_id": "sorted_list", "tier": "easy", "split": "eval",
    "description": "Implement SortedList in `sorted_list.py` that maintains ascending order.\n\nMethods: `add(value)`, `remove(value)` (raise ValueError), `contains(value)` (use binary search), `__len__()`, `__getitem__(index)`, `to_list()`.\n\nExample:\n```python\nsl = SortedList()\nsl.add(3); sl.add(1); sl.add(2)\nsl.to_list()  # [1,2,3]\nsl[0]  # 1\n```",
    "expected_files": ["/workspace/sorted_list.py"],
    "test_harness": easy_harness(
        "from sorted_list import SortedList",
        ["sl = SortedList()", "sl.add(5); sl.add(1); sl.add(3); sl.add(2); sl.add(4)",
         "assert sl.to_list() == [1,2,3,4,5]", "assert sl[0] == 1", "assert sl[4] == 5",
         "assert sl.contains(3)", "assert not sl.contains(99)",
         "sl.remove(3)", "assert sl.to_list() == [1,2,4,5]",
         "try:\n        sl.remove(99)\n        assert False\n    except ValueError: pass"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 13. Interval Merge (eval)
tasks.append({
    "task_id": "interval_merge", "tier": "easy", "split": "eval",
    "description": "Implement `merge_intervals(intervals)` and `insert_interval(intervals, new)` in `interval_merge.py`.\n\n- `merge_intervals`: Merge overlapping/touching intervals, return sorted by start.\n- `insert_interval`: Insert into sorted non-overlapping list and merge.\n\nExample:\n```python\nmerge_intervals([[1,3],[2,6],[8,10]])  # [[1,6],[8,10]]\ninsert_interval([[1,3],[6,9]], [2,5])  # [[1,5],[6,9]]\n```",
    "expected_files": ["/workspace/interval_merge.py"],
    "test_harness": easy_harness(
        "from interval_merge import merge_intervals, insert_interval",
        ["assert merge_intervals([]) == []",
         "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
         "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
         "assert insert_interval([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]]",
         "assert insert_interval([], [5,7]) == [[5,7]]"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 14. Text Wrap (eval)
tasks.append({
    "task_id": "text_wrap", "tier": "easy", "split": "eval",
    "description": "Implement `wrap_text(text, width)` in `text_wrap.py`.\n\nWrap text to width without breaking words. If a word exceeds width, put it on its own line. Return string with \\n separators.\n\nExample:\n```python\nwrap_text('The quick brown fox jumps over the lazy dog', 15)\n# 'The quick brown\\nfox jumps over\\nthe lazy dog'\n```",
    "expected_files": ["/workspace/text_wrap.py"],
    "test_harness": easy_harness(
        "from text_wrap import wrap_text",
        ["r = wrap_text('The quick brown fox', 15)",
         "lines = r.split('\\n')",
         "for l in lines:\n        words = l.split()\n        if len(words) > 1: assert len(l) <= 15",
         "assert wrap_text('short', 10) == 'short'",
         "assert wrap_text('', 10) == ''",
         "r2 = wrap_text('superlongword ok', 5)",
         "assert 'superlongword' in r2 and 'ok' in r2"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# 15. JSON Flattener (eval)
tasks.append({
    "task_id": "json_flattener", "tier": "easy", "split": "eval",
    "description": "Implement `flatten_json(obj, separator='.')` and `unflatten_json(obj, separator='.')` in `json_flattener.py`.\n\n- `flatten_json`: Nested dict -> flat dict with dot-notation keys. Lists use numeric indices.\n- `unflatten_json`: Reverse operation.\n\nExample:\n```python\nflatten_json({'a': {'b': 1}, 'c': 2})  # {'a.b': 1, 'c': 2}\nflatten_json({'x': [10, 20]})  # {'x.0': 10, 'x.1': 20}\nunflatten_json({'a.b': 1, 'c': 2})  # {'a': {'b': 1}, 'c': 2}\n```",
    "expected_files": ["/workspace/json_flattener.py"],
    "test_harness": easy_harness(
        "from json_flattener import flatten_json, unflatten_json",
        ["assert flatten_json({}) == {}", "assert flatten_json({'a': 1}) == {'a': 1}",
         "assert flatten_json({'a': {'b': 1}}) == {'a.b': 1}",
         "assert flatten_json({'a': {'b': {'c': 1}}}) == {'a.b.c': 1}",
         "assert flatten_json({'x': [10,20]}) == {'x.0': 10, 'x.1': 20}",
         "assert unflatten_json({'a.b': 1, 'c': 2}) == {'a': {'b': 1}, 'c': 2}"]
    ),
    "setup_commands": "", "timeout_seconds": 120
})

# ============================================================
# MEDIUM TASKS (15 total: 9 train, 6 eval)
# ============================================================

def medium_harness(module, test_file, extra_checks):
    """Build medium harness: runs pytest + extra direct checks."""
    return (
        "import json, subprocess, sys\nsys.path.insert(0, '/workspace')\ntry:\n"
        f"    r = subprocess.run(['python', '-m', 'pytest', '{test_file}', '-v', '--tb=short'],\n"
        "        capture_output=True, text=True, timeout=60, cwd='/workspace')\n"
        "    pytest_passed = r.returncode == 0\n"
        f"    {extra_checks}\n"
        "    passed = pytest_passed and extra_ok\n"
        "    json.dump({'passed': passed, 'details': f'pytest={pytest_passed} extra={extra_ok} ' + r.stdout[-500:]},\n"
        "        open('/workspace/RESULT.json', 'w'))\n"
        "except Exception as e:\n"
        "    json.dump({'passed': False, 'details': str(e)}, open('/workspace/RESULT.json', 'w'))"
    )

# M1. Key-Value Store (train)
tasks.append({
    "task_id": "key_value_store", "tier": "medium", "split": "train",
    "description": "Build a persistent key-value store.\n\nCreate `kv_store.py` with class `KVStore`:\n- `__init__(path=None)`: If path given, load from JSON file.\n- `set(key, value)`: Set a key-value pair.\n- `get(key, default=None)`: Get value or default.\n- `delete(key)`: Delete key. Raise KeyError if not found.\n- `keys()`: Return list of keys.\n- `save(path)`: Persist to JSON file.\n- `load(path)`: Load from JSON file.\n\nCreate `test_kv_store.py` with at least 8 pytest test functions covering set/get/delete/persistence/edge cases. Run `pytest test_kv_store.py -v` and make all tests pass.",
    "expected_files": ["/workspace/kv_store.py", "/workspace/test_kv_store.py"],
    "test_harness": medium_harness("kv_store", "test_kv_store.py",
        "from kv_store import KVStore; s = KVStore(); s.set('a',1); extra_ok = s.get('a') == 1"),
    "setup_commands": "", "timeout_seconds": 300
})

# M2. Event Emitter (train)
tasks.append({
    "task_id": "event_emitter", "tier": "medium", "split": "train",
    "description": "Build an event emitter system.\n\nCreate `event_emitter.py` with class `EventEmitter`:\n- `on(event, callback)`: Register callback for event. Return self for chaining.\n- `off(event, callback=None)`: Remove specific callback or all for event.\n- `emit(event, *args, **kwargs)`: Call all callbacks for event with given args.\n- `once(event, callback)`: Register callback that fires only once.\n- `listeners(event)`: Return list of callbacks for event.\n\nCreate `test_event_emitter.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/event_emitter.py", "/workspace/test_event_emitter.py"],
    "test_harness": medium_harness("event_emitter", "test_event_emitter.py",
        "from event_emitter import EventEmitter; e = EventEmitter(); r = []; e.on('x', lambda: r.append(1)); e.emit('x'); extra_ok = r == [1]"),
    "setup_commands": "", "timeout_seconds": 300
})

# M3. Task Queue (train)
tasks.append({
    "task_id": "task_queue", "tier": "medium", "split": "train",
    "description": "Build a priority task queue.\n\nCreate `task_queue.py` with class `TaskQueue`:\n- `add(task_id, priority=0, payload=None)`: Add task. Higher priority = processed first.\n- `process()`: Remove and return highest priority task as dict {task_id, priority, payload}. Raise IndexError if empty.\n- `peek()`: Return next task without removing.\n- `size()`: Return count.\n- `retry(task_id, new_priority=None)`: Re-add a previously processed task.\n\nCreate `test_task_queue.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/task_queue.py", "/workspace/test_task_queue.py"],
    "test_harness": medium_harness("task_queue", "test_task_queue.py",
        "from task_queue import TaskQueue; q = TaskQueue(); q.add('a',1); q.add('b',2); extra_ok = q.process()['task_id'] == 'b'"),
    "setup_commands": "", "timeout_seconds": 300
})

# M4. Mini Logger (train)
tasks.append({
    "task_id": "mini_logger", "tier": "medium", "split": "train",
    "description": "Build a logging library.\n\nCreate `mini_logger.py` with class `Logger`:\n- `__init__(name, level='INFO')`: Create logger with name and minimum level.\n- Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL (numeric 10,20,30,40,50).\n- `debug(msg)`, `info(msg)`, `warning(msg)`, `error(msg)`, `critical(msg)`: Log at respective level.\n- `set_level(level)`: Change minimum level.\n- `add_handler(handler)`: Add output handler.\n- Built-in handlers: `ConsoleHandler()` and `FileHandler(path)`.\n- Each handler has a `format(record)` method. Records have: level, name, message, timestamp.\n\nCreate `test_mini_logger.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/mini_logger.py", "/workspace/test_mini_logger.py"],
    "test_harness": medium_harness("mini_logger", "test_mini_logger.py",
        "from mini_logger import Logger; log = Logger('test'); extra_ok = hasattr(log, 'info') and hasattr(log, 'add_handler')"),
    "setup_commands": "", "timeout_seconds": 300
})

# M5. Config Manager (train)
tasks.append({
    "task_id": "config_manager", "tier": "medium", "split": "train",
    "description": "Build a configuration manager.\n\nCreate `config_manager.py` with class `ConfigManager`:\n- `__init__(defaults=None)`: Initialize with optional default dict.\n- `set(key, value)`: Set config value. Supports dot notation ('db.host').\n- `get(key, default=None)`: Get value with dot notation. Return default if not found.\n- `load_dict(d)`: Merge a dict into config (deep merge).\n- `load_env(prefix='APP')`: Load from environment variables. APP_DB_HOST -> db.host.\n- `to_dict()`: Return full config as nested dict.\n- `has(key)`: Return bool.\n\nCreate `test_config_manager.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/config_manager.py", "/workspace/test_config_manager.py"],
    "test_harness": medium_harness("config_manager", "test_config_manager.py",
        "from config_manager import ConfigManager; c = ConfigManager({'a': {'b': 1}}); extra_ok = c.get('a.b') == 1"),
    "setup_commands": "", "timeout_seconds": 300
})

# M6. State Machine (train)
tasks.append({
    "task_id": "state_machine", "tier": "medium", "split": "train",
    "description": "Build a finite state machine.\n\nCreate `state_machine.py` with class `StateMachine`:\n- `__init__(initial_state)`: Set initial state.\n- `add_state(name, on_enter=None, on_exit=None)`: Register a state with optional callbacks.\n- `add_transition(trigger, source, dest, condition=None)`: Add transition.\n- `trigger(event)`: Fire transition. Raise ValueError if no valid transition from current state.\n- `current_state`: Property returning current state name.\n- `can_trigger(event)`: Return bool.\n- `get_transitions(state=None)`: Return available transitions.\n\nCreate `test_state_machine.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/state_machine.py", "/workspace/test_state_machine.py"],
    "test_harness": medium_harness("state_machine", "test_state_machine.py",
        "from state_machine import StateMachine; sm = StateMachine('idle'); sm.add_state('idle'); sm.add_state('active'); sm.add_transition('start','idle','active'); sm.trigger('start'); extra_ok = sm.current_state == 'active'"),
    "setup_commands": "", "timeout_seconds": 300
})

# M7. Rate Limiter (train)
tasks.append({
    "task_id": "rate_limiter", "tier": "medium", "split": "train",
    "description": "Build a rate limiter using the token bucket algorithm.\n\nCreate `rate_limiter.py` with class `RateLimiter`:\n- `__init__(rate, capacity)`: rate = tokens per second, capacity = max tokens.\n- `allow(tokens=1)`: Return True if request allowed (consume tokens), False otherwise.\n- `wait_time(tokens=1)`: Return seconds to wait before tokens available.\n- `reset()`: Reset to full capacity.\n- `available_tokens`: Property returning current tokens (float).\n\nUse `time.monotonic()` for timing. Tokens refill continuously.\n\nCreate `test_rate_limiter.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/rate_limiter.py", "/workspace/test_rate_limiter.py"],
    "test_harness": medium_harness("rate_limiter", "test_rate_limiter.py",
        "from rate_limiter import RateLimiter; rl = RateLimiter(10, 10); extra_ok = rl.allow(5) == True"),
    "setup_commands": "", "timeout_seconds": 300
})

# M8. Schema Validator (train)
tasks.append({
    "task_id": "schema_validator", "tier": "medium", "split": "train",
    "description": "Build a data schema validator.\n\nCreate `schema_validator.py` with function `validate(data, schema)` that returns `(bool, list[str])` — (is_valid, error_messages).\n\nSchema format: dict with keys:\n- `type`: 'str', 'int', 'float', 'bool', 'list', 'dict'\n- `required`: bool (default True)\n- `min`/`max`: for numbers\n- `min_length`/`max_length`: for strings/lists\n- `pattern`: regex for strings\n- `items`: schema for list items\n- `properties`: dict of property_name -> schema for dicts\n\nCreate `test_schema_validator.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/schema_validator.py", "/workspace/test_schema_validator.py"],
    "test_harness": medium_harness("schema_validator", "test_schema_validator.py",
        "from schema_validator import validate; ok, errs = validate(42, {'type':'int','min':0}); extra_ok = ok == True"),
    "setup_commands": "", "timeout_seconds": 300
})

# M9. Pipeline (train)
tasks.append({
    "task_id": "pipeline", "tier": "medium", "split": "train",
    "description": "Build a data processing pipeline.\n\nCreate `pipeline.py` with class `Pipeline`:\n- `add_step(name, fn)`: Add transform function. Return self for chaining.\n- `run(data)`: Run all steps in order, passing data through each fn.\n- `run_safe(data)`: Like run but catches exceptions, returns (result, errors) tuple.\n- `remove_step(name)`: Remove step by name.\n- `steps`: Property returning list of step names.\n\nAlso implement `parallel_pipeline(pipelines, data)` that runs multiple pipelines and returns list of results.\n\nCreate `test_pipeline.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/pipeline.py", "/workspace/test_pipeline.py"],
    "test_harness": medium_harness("pipeline", "test_pipeline.py",
        "from pipeline import Pipeline; p = Pipeline().add_step('double', lambda x: x*2).add_step('inc', lambda x: x+1); extra_ok = p.run(5) == 11"),
    "setup_commands": "", "timeout_seconds": 300
})

# M10. Trie (eval)
tasks.append({
    "task_id": "trie", "tier": "medium", "split": "eval",
    "description": "Build a Trie (prefix tree) data structure.\n\nCreate `trie.py` with class `Trie`:\n- `insert(word)`: Insert a word.\n- `search(word)`: Return True if exact word exists.\n- `starts_with(prefix)`: Return True if any word starts with prefix.\n- `delete(word)`: Remove word. Return True if deleted, False if not found.\n- `autocomplete(prefix, limit=10)`: Return list of words starting with prefix.\n- `size()`: Return number of words.\n\nCreate `test_trie.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/trie.py", "/workspace/test_trie.py"],
    "test_harness": medium_harness("trie", "test_trie.py",
        "from trie import Trie; t = Trie(); t.insert('hello'); t.insert('help'); extra_ok = t.search('hello') and len(t.autocomplete('hel')) == 2"),
    "setup_commands": "", "timeout_seconds": 300
})

# M11. Graph (eval)
tasks.append({
    "task_id": "graph", "tier": "medium", "split": "eval",
    "description": "Build a graph data structure.\n\nCreate `graph.py` with class `Graph`:\n- `__init__(directed=False)`: Create graph.\n- `add_edge(u, v, weight=1)`: Add edge.\n- `neighbors(node)`: Return list of (neighbor, weight).\n- `bfs(start)`: Return list of nodes in BFS order.\n- `dfs(start)`: Return list of nodes in DFS order.\n- `shortest_path(start, end)`: Return (path, distance) using Dijkstra. Return (None, float('inf')) if unreachable.\n- `has_cycle()`: Return True if cycle exists.\n\nCreate `test_graph.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/graph.py", "/workspace/test_graph.py"],
    "test_harness": medium_harness("graph", "test_graph.py",
        "from graph import Graph; g = Graph(); g.add_edge('a','b',1); g.add_edge('b','c',2); p,d = g.shortest_path('a','c'); extra_ok = d == 3"),
    "setup_commands": "", "timeout_seconds": 300
})

# M12. Expression Evaluator (eval)
tasks.append({
    "task_id": "expression_eval", "tier": "medium", "split": "eval",
    "description": "Build a math expression evaluator.\n\nCreate `expression_eval.py` with:\n- `tokenize(expr)`: Split expression string into tokens (numbers, operators, parens).\n- `parse(tokens)`: Build AST from tokens.\n- `evaluate(expr)`: Evaluate expression string, return float result.\n\nSupport: +, -, *, /, parentheses, negative numbers. Respect operator precedence.\n\nCreate `test_expression_eval.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/expression_eval.py", "/workspace/test_expression_eval.py"],
    "test_harness": medium_harness("expression_eval", "test_expression_eval.py",
        "from expression_eval import evaluate; extra_ok = evaluate('2+3*4') == 14 and abs(evaluate('(2+3)*4') - 20) < 0.001"),
    "setup_commands": "", "timeout_seconds": 300
})

# M13. Retry Decorator (eval)
tasks.append({
    "task_id": "retry_decorator", "tier": "medium", "split": "eval",
    "description": "Build a retry decorator.\n\nCreate `retry_decorator.py` with:\n- `retry(max_attempts=3, delay=0, backoff=1, exceptions=(Exception,))`: Decorator that retries on failure.\n  - `max_attempts`: Total attempts including first.\n  - `delay`: Initial delay in seconds.\n  - `backoff`: Multiply delay by this after each retry.\n  - `exceptions`: Tuple of exception types to catch.\n  - If all attempts fail, raise the last exception.\n\nAlso: `retry_with_result(predicate, max_attempts=3)`: Retry until predicate(result) returns True.\n\nCreate `test_retry_decorator.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/retry_decorator.py", "/workspace/test_retry_decorator.py"],
    "test_harness": medium_harness("retry_decorator", "test_retry_decorator.py",
        "from retry_decorator import retry; call_count = 0\n    @retry(max_attempts=3, exceptions=(ValueError,))\n    def flaky():\n        global call_count; call_count += 1\n        if call_count < 3: raise ValueError('fail')\n        return 'ok'\n    extra_ok = flaky() == 'ok' and call_count == 3"),
    "setup_commands": "", "timeout_seconds": 300
})

# M14. Connection Pool (eval)
tasks.append({
    "task_id": "connection_pool", "tier": "medium", "split": "eval",
    "description": "Build a connection pool.\n\nCreate `connection_pool.py` with class `ConnectionPool`:\n- `__init__(factory, max_size=10)`: factory() creates new connections.\n- `acquire()`: Get a connection. Block or raise if pool exhausted.\n- `release(conn)`: Return connection to pool.\n- `size`: Property, current pool size (available + in-use).\n- `available`: Property, number of available connections.\n- `close_all()`: Close all connections.\n\nConnections should support context manager (with pool.acquire() as conn:).\n\nCreate `test_connection_pool.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/connection_pool.py", "/workspace/test_connection_pool.py"],
    "test_harness": medium_harness("connection_pool", "test_connection_pool.py",
        "from connection_pool import ConnectionPool; p = ConnectionPool(lambda: 'conn', max_size=2); c = p.acquire(); extra_ok = c == 'conn' and p.available == 1"),
    "setup_commands": "", "timeout_seconds": 300
})

# M15. File Watcher Sim (eval)
tasks.append({
    "task_id": "file_watcher_sim", "tier": "medium", "split": "eval",
    "description": "Build a simulated file watcher.\n\nCreate `file_watcher.py` with class `FileWatcher`:\n- `__init__()`: Initialize watcher.\n- `on(event_type, callback)`: Register callback for 'created', 'modified', 'deleted' events.\n- `snapshot(files_dict)`: Take a snapshot of current files {path: content_hash}.\n- `detect_changes(new_files_dict)`: Compare against last snapshot, fire callbacks, update snapshot. Returns list of (event_type, path) tuples.\n\nCreate `test_file_watcher.py` with at least 8 tests. Run pytest.",
    "expected_files": ["/workspace/file_watcher.py", "/workspace/test_file_watcher.py"],
    "test_harness": medium_harness("file_watcher", "test_file_watcher.py",
        "from file_watcher import FileWatcher; fw = FileWatcher(); fw.snapshot({'a.txt': 'hash1'}); changes = fw.detect_changes({'a.txt': 'hash2', 'b.txt': 'hash3'}); extra_ok = len(changes) == 2"),
    "setup_commands": "", "timeout_seconds": 300
})

# ============================================================
# HARD TASKS (15 total: 8 train, 7 eval)
# ============================================================

def hard_harness(test_file, extra_checks):
    """Build hard harness: pytest + multiple checks."""
    return (
        "import json, subprocess, sys\nsys.path.insert(0, '/workspace')\ntry:\n"
        f"    r = subprocess.run(['python', '-m', 'pytest', '{test_file}', '-v', '--tb=short'],\n"
        "        capture_output=True, text=True, timeout=120, cwd='/workspace')\n"
        "    pytest_passed = r.returncode == 0\n"
        f"    {extra_checks}\n"
        "    passed = pytest_passed and extra_ok\n"
        "    json.dump({'passed': passed, 'details': f'pytest={pytest_passed} extra={extra_ok} ' + r.stdout[-500:]},\n"
        "        open('/workspace/RESULT.json', 'w'))\n"
        "except Exception as e:\n"
        "    json.dump({'passed': False, 'details': str(e)}, open('/workspace/RESULT.json', 'w'))"
    )

# H1. Template Engine (train)
tasks.append({
    "task_id": "template_engine", "tier": "hard", "split": "train",
    "description": "Build a template engine.\n\nCreate `template_engine.py` with class `TemplateEngine`:\n- `render(template, context)`: Render template string with context dict.\n- Variable substitution: `{{ name }}` -> context['name']\n- Conditionals: `{% if condition %}...{% else %}...{% endif %}`\n- Loops: `{% for item in items %}...{% endfor %}` (item available inside loop)\n- Filters: `{{ name | upper }}`, `{{ name | lower }}`, `{{ value | default('N/A') }}`\n- Nested access: `{{ user.name }}`\n\nCreate `test_template_engine.py` with at least 12 tests. Run pytest.",
    "expected_files": ["/workspace/template_engine.py", "/workspace/test_template_engine.py"],
    "test_harness": hard_harness("test_template_engine.py",
        "from template_engine import TemplateEngine; e = TemplateEngine(); extra_ok = e.render('Hello {{ name }}', {'name': 'World'}) == 'Hello World'"),
    "setup_commands": "", "timeout_seconds": 600
})

# H2. Mini ORM (train)
tasks.append({
    "task_id": "mini_orm", "tier": "hard", "split": "train",
    "description": "Build a mini ORM for SQLite.\n\nCreate `mini_orm.py` with:\n- `Database(path)`: SQLite database connection.\n- `Model`: Base class. Subclasses define `__tablename__` and field class attributes.\n- `Field(field_type, primary_key=False, nullable=True, default=None)`: Column definition.\n- `Database.create_tables(models)`: Create tables from model definitions.\n- Model instances: `save()`, `delete()`, class methods: `get(id)`, `filter(**kwargs)`, `all()`.\n\nCreate `test_mini_orm.py` with at least 10 tests covering CRUD, filtering, relationships. Run pytest.",
    "expected_files": ["/workspace/mini_orm.py", "/workspace/test_mini_orm.py"],
    "test_harness": hard_harness("test_mini_orm.py",
        "import os; from mini_orm import Database, Model, Field\n    db = Database(':memory:')\n    class User(Model): __tablename__='users'; id=Field('integer',primary_key=True); name=Field('text')\n    db.create_tables([User])\n    u = User(name='Alice'); u.save()\n    extra_ok = User.all()[0].name == 'Alice'"),
    "setup_commands": "", "timeout_seconds": 600
})

# H3. Task Scheduler (train)
tasks.append({
    "task_id": "task_scheduler", "tier": "hard", "split": "train",
    "description": "Build a task scheduler with dependency resolution.\n\nCreate `task_scheduler.py` with class `Scheduler`:\n- `add_task(name, fn, depends_on=None)`: Register task with optional dependencies.\n- `run()`: Execute all tasks in valid order (topological sort). Return dict of {name: result}.\n- `run_task(name)`: Run single task and its dependencies. Return result.\n- `detect_cycles()`: Return True if circular dependencies exist.\n- `execution_order()`: Return list of task names in valid execution order.\n\nRaise `CyclicDependencyError` if cycles detected during run.\n\nCreate `test_task_scheduler.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/task_scheduler.py", "/workspace/test_task_scheduler.py"],
    "test_harness": hard_harness("test_task_scheduler.py",
        "from task_scheduler import Scheduler; s = Scheduler(); s.add_task('a', lambda: 1); s.add_task('b', lambda: 2, depends_on=['a']); order = s.execution_order(); extra_ok = order.index('a') < order.index('b')"),
    "setup_commands": "", "timeout_seconds": 600
})

# H4. CLI Framework (train)
tasks.append({
    "task_id": "cli_framework", "tier": "hard", "split": "train",
    "description": "Build a CLI framework.\n\nCreate `cli_framework.py` with class `CLI`:\n- `command(name, help='')`: Decorator to register a command function.\n- `argument(name, type=str, required=True, help='')`: Decorator for positional args.\n- `option(name, short=None, type=str, default=None, help='')`: Decorator for options.\n- `run(args=None)`: Parse args and execute command. Use sys.argv if args is None.\n- `help_text()`: Generate formatted help string.\n\nCreate a sample app in `sample_app.py` using the framework.\nCreate `test_cli_framework.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/cli_framework.py", "/workspace/sample_app.py", "/workspace/test_cli_framework.py"],
    "test_harness": hard_harness("test_cli_framework.py",
        "from cli_framework import CLI; app = CLI(); extra_ok = hasattr(app, 'command') and hasattr(app, 'run')"),
    "setup_commands": "", "timeout_seconds": 600
})

# H5. Pub/Sub (train)
tasks.append({
    "task_id": "pub_sub", "tier": "hard", "split": "train",
    "description": "Build a pub/sub messaging system.\n\nCreate `pub_sub.py` with:\n- `Broker`: Central message broker.\n  - `create_topic(name)`: Create a topic.\n  - `publish(topic, message)`: Publish message to topic.\n  - `subscribe(topic, subscriber)`: Subscribe to topic.\n  - `unsubscribe(topic, subscriber)`: Unsubscribe.\n- `Subscriber`: Base class.\n  - `on_message(topic, message)`: Called when message received.\n  - `history(topic=None)`: Return received messages.\n- Message replay: `Broker.replay(topic, subscriber, n=None)` replays last n messages.\n\nCreate `test_pub_sub.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/pub_sub.py", "/workspace/test_pub_sub.py"],
    "test_harness": hard_harness("test_pub_sub.py",
        "from pub_sub import Broker, Subscriber\n    b = Broker(); b.create_topic('t1')\n    class S(Subscriber):\n        def on_message(self, topic, msg): super().on_message(topic, msg)\n    s = S(); b.subscribe('t1', s); b.publish('t1', 'hello')\n    extra_ok = len(s.history()) == 1"),
    "setup_commands": "", "timeout_seconds": 600
})

# H6. Rule Engine (train)
tasks.append({
    "task_id": "rule_engine", "tier": "hard", "split": "train",
    "description": "Build a rule engine.\n\nCreate `rule_engine.py` with:\n- `Rule(name, condition, action, priority=0)`: condition(data)->bool, action(data)->result.\n- `RuleEngine`:\n  - `add_rule(rule)`: Register rule.\n  - `evaluate(data)`: Run all matching rules (condition=True) in priority order. Return list of (rule_name, result).\n  - `evaluate_first(data)`: Run only the highest-priority matching rule.\n  - `remove_rule(name)`: Remove by name.\n\nCreate `test_rule_engine.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/rule_engine.py", "/workspace/test_rule_engine.py"],
    "test_harness": hard_harness("test_rule_engine.py",
        "from rule_engine import Rule, RuleEngine; e = RuleEngine()\n    e.add_rule(Rule('r1', lambda d: d['x']>0, lambda d: 'pos', priority=1))\n    e.add_rule(Rule('r2', lambda d: True, lambda d: 'any', priority=0))\n    results = e.evaluate({'x': 5})\n    extra_ok = len(results) == 2 and results[0][0] == 'r1'"),
    "setup_commands": "", "timeout_seconds": 600
})

# H7. Data Pipeline (train)
tasks.append({
    "task_id": "data_pipeline", "tier": "hard", "split": "train",
    "description": "Build an ETL data pipeline framework.\n\nCreate `data_pipeline.py` with:\n- `Stage(name, fn)`: Pipeline stage.\n- `ETLPipeline`:\n  - `extract(fn)`: Set extract function.\n  - `transform(name, fn)`: Add transform stage.\n  - `load(fn)`: Set load function.\n  - `run()`: Execute extract -> transforms -> load. Return loaded result.\n  - `run_with_logging()`: Same but captures timing and errors for each stage.\n- Error handling: If a stage fails, record error and continue to next stage (skip dependent transforms).\n\nCreate `test_data_pipeline.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/data_pipeline.py", "/workspace/test_data_pipeline.py"],
    "test_harness": hard_harness("test_data_pipeline.py",
        "from data_pipeline import ETLPipeline; p = ETLPipeline()\n    p.extract(lambda: [1,2,3]).transform('double', lambda d: [x*2 for x in d]).load(lambda d: sum(d))\n    extra_ok = p.run() == 12"),
    "setup_commands": "", "timeout_seconds": 600
})

# H8. Cache System (train)
tasks.append({
    "task_id": "cache_system", "tier": "hard", "split": "train",
    "description": "Build a multi-layer cache system.\n\nCreate `cache_system.py` with:\n- `MemoryCache(max_size=100, ttl=None)`: In-memory cache with optional TTL (seconds).\n- `FileCache(directory, ttl=None)`: File-based cache storing JSON.\n- `MultiLayerCache(layers)`: Checks layers in order. On miss, checks next layer. On hit, populates upper layers.\n- All caches: `get(key)`, `set(key, value, ttl=None)`, `delete(key)`, `clear()`, `stats()` (hits, misses, evictions).\n- MemoryCache eviction: LRU when max_size exceeded.\n\nCreate `test_cache_system.py` with at least 12 tests. Run pytest.",
    "expected_files": ["/workspace/cache_system.py", "/workspace/test_cache_system.py"],
    "test_harness": hard_harness("test_cache_system.py",
        "from cache_system import MemoryCache; c = MemoryCache(max_size=2)\n    c.set('a',1); c.set('b',2); c.set('c',3)\n    extra_ok = c.get('a') is None and c.get('c') == 3"),
    "setup_commands": "", "timeout_seconds": 600
})

# H9. Query Builder (eval)
tasks.append({
    "task_id": "query_builder", "tier": "hard", "split": "eval",
    "description": "Build a SQL query builder.\n\nCreate `query_builder.py` with class `QueryBuilder`:\n- `select(*columns)`: Start SELECT query.\n- `from_table(table)`: Set FROM table.\n- `where(condition, *params)`: Add WHERE clause with parameterized values.\n- `join(table, on)`: Add JOIN.\n- `order_by(column, direction='ASC')`: Add ORDER BY.\n- `limit(n)`: Add LIMIT.\n- `offset(n)`: Add OFFSET.\n- `build()`: Return (sql_string, params_list) tuple.\n- `insert_into(table, **values)`: Build INSERT query.\n- `update(table, **values)`: Build UPDATE query.\n- `delete_from(table)`: Build DELETE query.\n\nCreate `test_query_builder.py` with at least 12 tests. Run pytest.",
    "expected_files": ["/workspace/query_builder.py", "/workspace/test_query_builder.py"],
    "test_harness": hard_harness("test_query_builder.py",
        "from query_builder import QueryBuilder; q = QueryBuilder()\n    sql, params = q.select('name','age').from_table('users').where('age > ?', 18).build()\n    extra_ok = 'SELECT' in sql and 'WHERE' in sql and params == [18]"),
    "setup_commands": "", "timeout_seconds": 600
})

# H10. Plugin System (eval)
tasks.append({
    "task_id": "plugin_system", "tier": "hard", "split": "eval",
    "description": "Build a plugin system.\n\nCreate `plugin_system.py` with:\n- `Plugin`: Base class with `name`, `version`, `dependencies` (list of plugin names).\n  - `on_load()`, `on_unload()`: Lifecycle hooks.\n  - `execute(**kwargs)`: Main plugin action.\n- `PluginManager`:\n  - `register(plugin_class)`: Register plugin.\n  - `load(name)`: Load plugin and its dependencies (in order).\n  - `unload(name)`: Unload plugin (unload dependents first).\n  - `execute(name, **kwargs)`: Execute a loaded plugin.\n  - `loaded_plugins`: Property, list of loaded plugin names.\n\nCreate `test_plugin_system.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/plugin_system.py", "/workspace/test_plugin_system.py"],
    "test_harness": hard_harness("test_plugin_system.py",
        "from plugin_system import Plugin, PluginManager\n    class P(Plugin):\n        name='test'; version='1.0'; dependencies=[]\n        def execute(self, **kw): return 'ok'\n    pm = PluginManager(); pm.register(P); pm.load('test')\n    extra_ok = pm.execute('test') == 'ok'"),
    "setup_commands": "", "timeout_seconds": 600
})

# H11. Workflow Engine (eval)
tasks.append({
    "task_id": "workflow_engine", "tier": "hard", "split": "eval",
    "description": "Build a workflow engine.\n\nCreate `workflow_engine.py` with:\n- `Step(name, fn, on_error=None)`: Workflow step. fn(context)->result.\n- `Workflow`:\n  - `add_step(step)`: Add step.\n  - `add_condition(after_step, condition_fn, if_true_step, if_false_step)`: Branch.\n  - `run(initial_context=None)`: Execute workflow, return final context.\n  - `rollback()`: Execute on_error handlers in reverse for completed steps.\n  - `status()`: Return dict of {step_name: 'completed'|'failed'|'pending'}.\n\nContext is a dict passed through all steps.\n\nCreate `test_workflow_engine.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/workflow_engine.py", "/workspace/test_workflow_engine.py"],
    "test_harness": hard_harness("test_workflow_engine.py",
        "from workflow_engine import Workflow, Step; w = Workflow()\n    w.add_step(Step('s1', lambda ctx: {**ctx, 'x': 1}))\n    w.add_step(Step('s2', lambda ctx: {**ctx, 'y': ctx['x']+1}))\n    result = w.run({})\n    extra_ok = result.get('y') == 2"),
    "setup_commands": "", "timeout_seconds": 600
})

# H12. Mini Test Framework (eval)
tasks.append({
    "task_id": "mini_test_framework", "tier": "hard", "split": "eval",
    "description": "Build a miniature test framework.\n\nCreate `mini_test.py` with:\n- `@test`: Decorator marking a function as a test.\n- `@before_each`/`@after_each`: Setup/teardown decorators.\n- `assert_equal(a, b)`, `assert_true(x)`, `assert_raises(exc, fn)`: Assertion helpers.\n- `TestRunner`:\n  - `discover(module)`: Find all @test functions in a module.\n  - `run()`: Execute tests, return `TestResult`.\n- `TestResult`: passed, failed, errors counts + details.\n- `report(result)`: Return formatted string report.\n\nCreate a sample test module `sample_tests.py` using the framework.\nCreate `test_mini_test.py` (using pytest) to test the framework itself. Run pytest.",
    "expected_files": ["/workspace/mini_test.py", "/workspace/sample_tests.py", "/workspace/test_mini_test.py"],
    "test_harness": hard_harness("test_mini_test.py",
        "from mini_test import TestRunner, test, assert_equal\n    extra_ok = callable(test) and callable(assert_equal) and hasattr(TestRunner, 'run')"),
    "setup_commands": "", "timeout_seconds": 600
})

# H13. Document Store (eval)
tasks.append({
    "task_id": "document_store", "tier": "hard", "split": "eval",
    "description": "Build an in-memory document store.\n\nCreate `document_store.py` with class `DocumentStore`:\n- `insert(doc)`: Insert dict document, auto-assign '_id'. Return _id.\n- `get(doc_id)`: Return document by _id or None.\n- `update(doc_id, updates)`: Update fields. Return True/False.\n- `delete(doc_id)`: Delete document. Return True/False.\n- `find(query)`: Query by field values. `{'name': 'Alice'}` matches docs with name='Alice'. Support `$gt`, `$lt`, `$in` operators.\n- `create_index(field)`: Create index for faster lookups.\n- `count(query=None)`: Count matching documents.\n- `paginate(query, page, page_size)`: Return page of results.\n\nCreate `test_document_store.py` with at least 12 tests. Run pytest.",
    "expected_files": ["/workspace/document_store.py", "/workspace/test_document_store.py"],
    "test_harness": hard_harness("test_document_store.py",
        "from document_store import DocumentStore; ds = DocumentStore()\n    id1 = ds.insert({'name':'Alice','age':30}); id2 = ds.insert({'name':'Bob','age':25})\n    results = ds.find({'name': 'Alice'})\n    extra_ok = len(results) == 1 and results[0]['name'] == 'Alice' and ds.count() == 2"),
    "setup_commands": "", "timeout_seconds": 600
})

# H14. API Gateway Sim (eval)
tasks.append({
    "task_id": "api_gateway_sim", "tier": "hard", "split": "eval",
    "description": "Build an API gateway simulator.\n\nCreate `api_gateway.py` with:\n- `Gateway`:\n  - `route(method, path, handler)`: Register route. Path supports params: `/users/:id`.\n  - `middleware(fn)`: Add middleware function. fn(request, next) -> response.\n  - `handle(request)`: Process request through middleware chain + route handler.\n- `Request(method, path, headers=None, body=None)`: Request object.\n- `Response(status_code, body=None, headers=None)`: Response object.\n- Built-in middleware: `rate_limit(max_requests, window_seconds)`, `cors(origins)`.\n\nCreate `test_api_gateway.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/api_gateway.py", "/workspace/test_api_gateway.py"],
    "test_harness": hard_harness("test_api_gateway.py",
        "from api_gateway import Gateway, Request, Response\n    gw = Gateway()\n    gw.route('GET', '/hello', lambda req: Response(200, 'Hello'))\n    resp = gw.handle(Request('GET', '/hello'))\n    extra_ok = resp.status_code == 200 and resp.body == 'Hello'"),
    "setup_commands": "", "timeout_seconds": 600
})

# H15. HTTP Client (eval)
tasks.append({
    "task_id": "http_client", "tier": "hard", "split": "eval",
    "description": "Build a mock HTTP client library.\n\nCreate `http_client.py` with:\n- `HttpClient`:\n  - `get(url, headers=None, timeout=30)`: Make GET request.\n  - `post(url, body=None, headers=None, timeout=30)`: Make POST request.\n  - `set_base_url(url)`: Set base URL prefix.\n  - `add_default_header(key, value)`: Add header to all requests.\n  - `retry_config(max_retries=3, backoff=1.0)`: Configure retry behavior.\n- `MockServer`: For testing. Register routes and responses.\n  - `when(method, path)`: Returns builder to set response.\n  - `handle(request)`: Process request.\n- `HttpResponse(status_code, body, headers)`: Response object.\n\nThe HttpClient should use MockServer for testing (inject via constructor).\n\nCreate `test_http_client.py` with at least 10 tests. Run pytest.",
    "expected_files": ["/workspace/http_client.py", "/workspace/test_http_client.py"],
    "test_harness": hard_harness("test_http_client.py",
        "from http_client import HttpClient, MockServer, HttpResponse\n    server = MockServer()\n    server.when('GET', '/test').respond(200, 'ok')\n    client = HttpClient(server=server)\n    resp = client.get('/test')\n    extra_ok = resp.status_code == 200 and resp.body == 'ok'"),
    "setup_commands": "", "timeout_seconds": 600
})

# ============================================================
# VALIDATE AND WRITE
# ============================================================

# Count splits
train_count = sum(1 for t in tasks if t["split"] == "train")
eval_count = sum(1 for t in tasks if t["split"] == "eval")
easy_count = sum(1 for t in tasks if t["tier"] == "easy")
medium_count = sum(1 for t in tasks if t["tier"] == "medium")
hard_count = sum(1 for t in tasks if t["tier"] == "hard")

assert len(tasks) == 45, f"Expected 45 tasks, got {len(tasks)}"
assert train_count == 25, f"Expected 25 train, got {train_count}"
assert eval_count == 20, f"Expected 20 eval, got {eval_count}"
assert easy_count == 15, f"Expected 15 easy, got {easy_count}"
assert medium_count == 15, f"Expected 15 medium, got {medium_count}"
assert hard_count == 15, f"Expected 15 hard, got {hard_count}"

# Unique IDs
ids = [t["task_id"] for t in tasks]
assert len(ids) == len(set(ids)), f"Duplicate task IDs: {[i for i in ids if ids.count(i) > 1]}"

output_path = "/Users/sriranga/Desktop/klong/experiment/task_bank.json"
with open(output_path, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Wrote {len(tasks)} tasks to {output_path}")
print(f"  Easy: {easy_count} (train={sum(1 for t in tasks if t['tier']=='easy' and t['split']=='train')}, eval={sum(1 for t in tasks if t['tier']=='easy' and t['split']=='eval')})")
print(f"  Medium: {medium_count} (train={sum(1 for t in tasks if t['tier']=='medium' and t['split']=='train')}, eval={sum(1 for t in tasks if t['tier']=='medium' and t['split']=='eval')})")
print(f"  Hard: {hard_count} (train={sum(1 for t in tasks if t['tier']=='hard' and t['split']=='train')}, eval={sum(1 for t in tasks if t['tier']=='hard' and t['split']=='eval')})")
