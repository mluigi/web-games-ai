# File for testing functions

import os

if not os.path.exists("chkpoints"):
    os.mkdir("chkpoints")
    os.makedirs("chkpoints/prev")
    os.makedirs("chkpoints/lts")
