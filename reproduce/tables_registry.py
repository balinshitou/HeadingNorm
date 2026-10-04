"""Table registry: @table('3') registers the generator of Table 3; generators return (header, rows) of cell strings."""
TABLES = {}


def table(tid):
    def deco(fn):
        TABLES[tid] = fn
        return fn
    return deco
