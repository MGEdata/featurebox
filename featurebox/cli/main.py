# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:27
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

import argparse
import sys as _sys
import textwrap
from importlib import import_module

title = """
-----------------------------------------------------
|                 Featurebox CLI                    |
|            (Property Batch Extractor)             |  
|            (   data   ->     csv    )             |  
-----------------------------------------------------"""


class CLIError(Exception):
    """Error for CLI commands.

    A subcommand may raise this.  The message will be forwarded to
    the error() method of the argument args."""


# Important: Following any change to command-line parameters, use
# python3 -m ase.cli.completion to update autocompletion.
commands_ = {

    'bandgap': ('featurebox.cli.vasp_bgp',
                "fbx bandgap : Batching 'Band gap'."),
    'bader': ('featurebox.cli.vasp_bader',
              "fbx bader   : Batching 'Bader charge'."),
    'dbc': ('featurebox.cli.vasp_dbc',
            "fbx dbc     : Batching 'Band center'."),
    'cohp': ('featurebox.cli.vasp_cohp',
             "fbx cohp    : Batching 'COHP'."),
    'dos': ('featurebox.cli.vasp_dos',
            "fbx dos     : Batching 'Density of States'."),
    'general': ('featurebox.cli.vasp_general_single',
                "fbx general : Batching 'Vasprun.xml property'."),
    'diff': ('featurebox.cli.vasp_general_diff',
             "fbx diff    : Batching 'Energy difference'."),
    'converge': ('featurebox.cli.vasp_converge',
                 "fbx converge: Batching 'Check vasp converge'."),

}

cmd_help = """-----------------------------------------------------
Sub-Command : Function
------------ ----------------------------------------
{}

Add '-h' after sub-command for more help.
Website: https://featurebox.readthedocs.io
-----------------------------------------------------""".format("\n".join([i[1] for i in commands_.values()]))


def main(prog='featurebox', description='featurebox command line tool.', args=None, ):
    print(title)
    if args is None:
        # args default to the system args
        args = _sys.argv[1:]
    if args is None:
        print(cmd_help)
    else:
        args = [args, ] if isinstance(args, str) else args
        args = list(args) if isinstance(args, tuple) else args

        if len(args) == 0:
            print(cmd_help)
        elif isinstance(args, list) and args[0] in ["-h", "--help", "help"]:
            print(cmd_help)
        else:

            commands = commands_  # 导入函数
            parser = argparse.ArgumentParser(prog=prog,
                                             description=description,
                                             formatter_class=Formatter)
            parser.add_argument('-T', '--traceback', action='store_true')
            subparsers = parser.add_subparsers(title='Sub-commands',
                                               dest='command')
            functions = {}
            parsers = {}

            if args[0] not in commands:
                print(cmd_help)
                print(f"{prog}: Error: Not Find Sub-Command: '{args[0]}'.")
                import sys
                sys.exit(2)

            for command, module_name in commands.items():
                if args[0] == command:
                    cmd = import_module(module_name[0])._CLICommand
                    docstring = cmd.__doc__
                    if docstring is None:
                        short = cmd.short_description
                        long = getattr(cmd, 'description', short)
                    else:
                        parts = docstring.split('\n', 1)
                        if len(parts) == 1:
                            short = docstring
                            long = docstring
                        else:
                            short, body = parts
                            long = short + '\n' + textwrap.dedent(body)
                    subparser = subparsers.add_parser(
                        command,
                        formatter_class=Formatter,
                        help=short,
                        description=long)
                    cmd.add_arguments(subparser)
                    functions[command] = cmd.run
                    parsers[command] = subparser

            args = parser.parse_args(args)

            if args.command == 'help' or args.command is None:
                print(cmd_help)
            else:
                f = functions[args.command]
                try:
                    if f.__code__.co_argcount == 1:
                        f(args)
                    else:
                        f(args, parsers[args.command])
                except KeyboardInterrupt:
                    pass
                except CLIError as x:
                    parser.error(x)
                except Exception as x:
                    if args.traceback:
                        raise
                    else:
                        l1 = '{}: {}\n'.format(x.__class__.__name__, x)
                        l2 = ('To get a full traceback, use: {} -T {} ...'
                              .format(prog, args.command))
                        parser.error(l1 + l2)


class Formatter(argparse.HelpFormatter):
    """Improved help formatter."""

    def _fill_text(self, text, width, indent):

        assert indent == ''
        out = ''
        blocks = text.split('\n\n')
        for block in blocks:
            if block != "":
                if block[0] == '*':
                    # List items:
                    for item in block[2:].split('\n* '):
                        out += textwrap.fill(item,
                                             width=width - 2,
                                             initial_indent='* ',
                                             subsequent_indent='  ') + '\n'
                elif block[0] == ' ':
                    # Indented literal block:
                    out += block + '\n'
                else:
                    # Block of text:
                    out += textwrap.fill(block, width=width) + '\n'
                out += '\n'
            else:
                pass
        return out[:-1]


if __name__ == "__main__":
    main(args=["bader", "-h"])
