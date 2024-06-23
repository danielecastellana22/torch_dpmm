_DEBUG_MODE = False


def set_debug_mode():
    global _DEBUG_MODE
    import sys
    print('Debug mode is set to true!', file=sys.stderr)
    _DEBUG_MODE = True
