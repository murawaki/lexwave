# colony 2
spec = {
    'type': 'EVO',
    'itemsize': 100,
    'specs': [
        {
            'steps': 750,
            'scale': 1.0,
            'nodes': [ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
            'distances': [
                ['C', 'A', 1.5],
                ['A', 'F', 4.0],
                ['F', 'H', 1.5],
                ['H', 'J', 1.5],

                ['D', 'E', 1.5],
                ['E', 'G', 4.0],
                ['G', 'I', 1.5],
                ['I', 'K', 1.5],

                ['C', 'D', 1.5],
                ['A', 'E', 1.5],
                ['F', 'G', 1.5],
                ['H', 'I', 1.5],
                ['J', 'K', 1.5],

                ['C', 'E', 2.0],
                ['A', 'D', 2.0],
                ['F', 'I', 2.0],
                ['H', 'G', 2.0],
                ['H', 'K', 2.0],
                ['J', 'I', 2.0],

                ],
            'powers': {
                'A': 2.0,
                'C': 0.5,
                'D': 0.5,
                'E': 0.5,
                'F': 0.5,
                'G': 0.5,
                'H': 0.5,
                'I': 0.5,
                'J': 0.5,
                'K': 0.5,
                },
            'survival': 4.0,
            'birth': 0.1,
            'wordcost': 3.0,
            },
        {
            'steps': 250,
            'clones': [['A', 'B']], # from to
            'scale': 1.0,
            'nodes': [ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',   'B'],
            'distances': [
                ['C', 'A', 1.5],
                ['A', 'F', 4.0],
                ['F', 'H', 1.5],
                ['H', 'J', 1.5],

                ['D', 'E', 1.5],
                ['E', 'G', 4.0],
                ['G', 'I', 1.5],
                ['I', 'K', 1.5],

                ['C', 'D', 2.0],
                ['A', 'E', 2.0],
                ['F', 'G', 2.0],
                ['H', 'I', 2.0],
                ['J', 'K', 2.0],

                ['C', 'E', 2.0],
                ['A', 'D', 2.0],
                ['F', 'I', 2.0],
                ['H', 'G', 2.0],
                ['H', 'K', 2.0],
                ['J', 'I', 2.0],

                ['B', 'G', 2.0],
                ['B', 'I', 1.5],
                ['B', 'K', 2.0],
                ],
            'powers': {
                'A': 2.0,
                'C': 0.5,
                'D': 0.5,
                'E': 0.5,
                'F': 0.5,
                'G': 0.5,
                'H': 0.5,
                'I': 0.5,
                'J': 0.5,
                'K': 0.5,

                'B': 2.0,
                },
            'survival': 4.0,
            'birth': 0.1,
            'wordcost': 3.0,
            }
        ]
    }
