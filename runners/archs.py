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

LARGE_POLICY_ARCH = '''[
        {"type": "fc", "n": 512},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 256},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

LARGE_VAL_ARCH = '''[
        {"type": "fc", "n": 512},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 256},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

HUGE_POLICY_ARCH = '''[
        {"type": "fc", "n": 1028},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 512},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 256},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

HUGE_VAL_ARCH = '''[
        {"type": "fc", "n": 1028},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 512},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 256},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''
