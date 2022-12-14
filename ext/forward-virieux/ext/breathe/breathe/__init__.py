__version__ = '4.13.1'


def setup(app):

    # We can't do the import at the module scope as setup.py has to be able to
    # import this file to read __version__ without hitting any syntax errors
    # from both Python 2 & Python 3.

    # By the time this function is called, the directives code will have been
    # converted with 2to3 if appropriate

    from . import directives

    directives.setup(app)
    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True
    }
