# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
 Copyright(c) 2017 by Digital Media Professionals Inc.
 All rights reserved.
------------------------------------------------------------
"""

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ParseError(Error):
    """Exception raised for CNN parse error."""
    def __init__(self, message):
        self.message = message

class ConvertError(Error):
    def __init__(self, message):
        self.message = message