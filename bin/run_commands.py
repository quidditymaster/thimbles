import thimbles as tmb
from thimbles.tasks import task_registry, task
from thimbles.options import opts

@task(result_name="hello")
def hello(greeting="hello", subject="world!"):
    greet_str =  "{} {}".format(greeting, subject)
    print(greet_str)
    return greet_str

if __name__ == "__main__":
    import sys
    opts.parse_commands(sys.argv[1:])
