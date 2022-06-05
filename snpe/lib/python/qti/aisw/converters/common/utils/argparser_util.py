# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import argparse
import textwrap as _textwrap


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24,
                 width=None):
        super(CustomHelpFormatter, self).__init__(prog, indent_increment, max_help_position, width=100)

    def _split_lines(self, text, width):
        # Preserve newline character in the help text
        paras = text.splitlines()
        lines = []
        for para in paras:
            # Wrap the paragraphs based on width
            lines.extend(_textwrap.wrap(para, width, replace_whitespace=False))
        return lines


class ArgParserWrapper(object):
    """
    Wrapper class for argument parsing
    """

    def __init__(self, parents=[], **kwargs):
        self.parser = argparse.ArgumentParser(**kwargs)
        self.required = self.parser.add_argument_group('required arguments')
        self.optional = self.parser.add_argument_group('optional arguments')
        self.mutually_exclusive = []  # a list of sets of mutually exclusive arguments
        self._extend_from_parents(parents)

    def _extend_from_parents(self, parents):
        for i, parent in enumerate(parents):
            if not isinstance(parent, ArgParserWrapper):
                raise TypeError("Parent {0} not of Type ArgParserWrapper".format(parent.__class__.__name__))
            for action in parent.required._group_actions:
                self.required._add_action(action)
            for action in parent.optional._group_actions:
                self.optional._add_action(action)

    def add_required_argument(self, *args, **kwargs):
        self.required.add_argument(*args, required=True, **kwargs)

    def add_optional_argument(self, *args, **kwargs):
        self.optional.add_argument(*args, required=False, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        return self.parser.add_argument_group(*args, **kwargs)

    def add_mutually_exclusive_args(self, *args: list):
        # confine to set
        args_as_set = set(args)
        self.mutually_exclusive.append(list(args_as_set))

        # Add epilog note
        existing_epilog = getattr(self.parser, 'epilog')
        if existing_epilog is None:
            existing_epilog = ""
        exclusivity_info = "Note: Only one of: {} can be specified\n".format(str(args_as_set))
        setattr(self.parser, 'epilog', existing_epilog + exclusivity_info)

    def parse_args(self, args=None, namespace=None):
        cmd_args = self.parser.parse_args(args, namespace)
        self._check_mutually_exclusive_args(cmd_args)
        return cmd_args

    def _check_mutually_exclusive_args(self, parsed_cmd_args):
        for arg_set in self.mutually_exclusive:
            specified_cmd_args = [arg for arg in arg_set if getattr(parsed_cmd_args, arg) is not None]
            if len(specified_cmd_args) > 1:
                formatted_arg_set = ', '.join(arg_set)
                raise TypeError("Cannot specify all of: {}. Please specify "
                                "exactly one of these arguments.".format(formatted_arg_set))
