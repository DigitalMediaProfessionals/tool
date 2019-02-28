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


class Limitation(object):
    """Defines HW limitations."""

    def __init__(self, model='DV700'):
        if model == 'DV700':
            self.max_conv_width = 2047
            self.max_conv_height = 1024
            self.max_conv_depth = 15
            self.max_conv_channel = 32768
            self.max_conv_kernel = 7
            self.max_fc_channel = 16384
