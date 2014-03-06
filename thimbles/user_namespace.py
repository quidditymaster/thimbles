import argparse
import warnings

parser = argparse.ArgumentParser("blah")

parser.add_argument("--exec",dest="exec_input",default="")

parse = parser.parse_args()


if len(parse.exec_input) > 3 and parse.exec_input[-3:]==".py":
    execfile(parse.exec_input)
else:
    try:
        exec(parse.exec_input)
    except Exception as e:
        print("Error with user given --exec argument:")
        print("    {} : {}".format(type(e).__name__,e.message))


