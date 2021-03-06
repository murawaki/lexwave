# star-shaped network
spec = {
    'scale': 1.0,
    'nodes': [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'],
    'distances': [
        ['A', 'B', 2.0],
        ['A', 'C', 2.0],
        ['A', 'D', 2.0],
        ['A', 'E', 2.0],
        ['A', 'F', 2.0],
        ['C', 'D', 5.0],
        ['E', 'F', 5.0],
        ['C', 'G', 0.5],
        ['C', 'H', 0.5],
        ['D', 'I', 0.5],
        ['D', 'J', 0.5],
        ['E', 'K', 0.5],
        ['E', 'L', 0.5],
        ['F', 'M', 0.5],
        ['F', 'N', 0.5],
        ['G', 'H', 1.5],
        ['I', 'J', 1.5],
        ['K', 'L', 1.5],
        ['M', 'N', 1.5],
        ],
    'powers': {
        'A': 5.0,
        'B': 1.0,
        'C': 0.5,
        'D': 0.5,
        'E': 0.5,
        'F': 0.5,
        'G': 0.1,
        'H': 0.1,
        'I': 0.1,
        'J': 0.1,
        'K': 0.1,
        'L': 0.1,
        'M': 0.1,
        'N': 0.1,
        },
    'survival': 2.0,
    'birth': 0.1,
    'wordcost': 3.0,
    'itemsize': 100,
    }
