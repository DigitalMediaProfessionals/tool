#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Copyright 2018 Digital Media Professionals Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import os
import os.path as osp
from jinja2 import Environment, FileSystemLoader


def output_pywrapper(out_path: str, netcls: str, module: str, header: str):
    """
    output c++ source file of python wrapper for netcls
    @param out_path output file path
    @param netcls name of the DV network class
    @param module name of python module
    @param header name of header file for the DV network class
    """
    path_base = osp.basename(out_path)
    template_dir = osp.abspath(osp.join(osp.dirname(__file__), "template/"))

    # create code
    env = Environment(loader=FileSystemLoader(template_dir, encoding="utf8"))
    tpl = env.get_template("pywrapper.cpp.template")
    code = tpl.render({"path_base": path_base,
                       "netcls": netcls,
                       "module": module,
                       "header": header})

    # write
    with open(out_path, "w") as of:
        of.write(code)
