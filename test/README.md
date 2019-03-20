# Overview of Test

This test compares an output from a converted network with its reference output.


# How to Test

Before pushing your branch to the one on Github, please apply the DMP Tool to models under `./model/` and commit the generated.
Why do we need commit the generated is because an AI FPGA Module is not able to run the DMP Tool.

To run a test, please execute `./test.sh` on an AI FPGA Module.


# How to Update Reference Output

To update refernce outputs, run `./test.sh save_reference`.
This saves outputs from a converted network as reference outputs.
