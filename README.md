# Llama 3 NPU Test

this is a command line script to send prompts to a Llama3 model running on the NPU and measures the overall performance.
It works on Windows and uses the utility `powercfg` to generate a battery report before and after each generation, to keep
track of battery consumption

## Usage
Launch the script from the command line with the param `-h` or `--help` to see the supported options

If you want to read prompts from a file, pass the its path to the command line parameter `prompts_file`. If you don't
pass it it will read prompts from the command line. Type `exit` to exit

