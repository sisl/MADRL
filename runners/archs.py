SIMPLE_POLICY_ARCH = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

TINY_VAL_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "relu"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "relu"}
    ]
    '''

SIMPLE_VAL_ARCH = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

GAE_TYPE_VAL_ARCH = '''[
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 64},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"}
]
'''

GAE_ARCH = '''[
        {"type": "fc", "n": 100},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 50},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 25},
        {"type": "nonlin", "func": "tanh"}
]
'''
